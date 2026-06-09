"""Per-agent (IPPO) PPO update steps.

Each agent owns its own actor (LoRA + value head) and runs this update
independently. MAPPO is achieved by *adding* a shared ``CentralizedCritic``
(in ``rl_layer/centralized_critic.py``) and passing
``value_loss_enabled=False`` to ``action_level_ppo_step`` so the per-agent
value head is bypassed.

Action-level:  optimises log π(action | prompt) where π is the LLM's own
sequence-log-probability over the candidate action strings (constrained
generation — see ``RLLayer._score_actions``). ``action_level_ppo_step``
runs the PPO loss + per-transition gradient accumulation in-function and
returns only the info dict; the caller in ``ppo_update.py`` does
zero_grad → call → clip → step around it.

Token-level:   optimises the full token-level log-likelihood of the generated
response (only triggered by the agent's learning-belief mechanism).
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_layer.trajectory_buffer import Transition

logger = logging.getLogger(__name__)


# ── Helpers ──

def _normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)


# ── Action-level PPO update ──

def action_level_ppo_step(
    rl_layer,                       # RLLayer instance (owns LoRA model + value head + scorer)
    batch: List[Transition],
    clip_eps: float,
    value_clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    device: torch.device,
    max_length: int = 512,          # accepted for API symmetry; ignored (rl_layer carries cfg)
    value_loss_enabled: bool = True,
    scaler=None,                    # torch.amp.GradScaler the caller is using
) -> dict:
    """Per-transition gradient-accumulating PPO mini-batch step.

    Returns only the diagnostic ``info`` dict — backward + gradient
    accumulation happen INSIDE this function, per transition.

    Why per-transition: the constrained-generation actor (``_score_actions``)
    runs ONE full LLM forward per transition. The previous batched version
    kept all those forward graphs alive until a single end-of-mini-batch
    backward, with peak memory growing linearly in ``mini_batch_size``. On
    the 9B + 22-candidate × ~514-token-prompt scoring path that gives
    ~4 GB of retained ``out.logits`` × N graphs and OOMs even a 45 GB L40
    by the first PPO update. Doing forward → loss_term/N → backward each
    iteration releases the graph immediately; peak stays at a single
    forward regardless of how large the mini-batch is.

    Math equivalence: PPO's mean-reduction loss
        L = (1/N) Σ_i [ −min(surr1_i, surr2_i) + value_coef·v_loss_i
                       − entropy_coef·H_i ]
    Gradient via per-transition backward of (loss_term_i / N):
        ∂L/∂θ = Σ_i ∂(loss_term_i / N)/∂θ
    so accumulating gradients across iterations yields the exact same
    parameter update as the batched form. Adv-stat diagnostics are
    computed on stored buffer scalars (no autograd needed).

    A transition whose stored canonical action_idx is NOT in the current
    candidate set is dropped (e.g. checkpoint resumed after
    mask_slot_actions toggled). When every transition is dropped we
    return an empty info dict without calling backward.
    """

    candidate_set = rl_layer._candidate_actions
    full_to_cand = rl_layer._full_idx_to_cand_idx

    # ── Pre-filter dropped transitions BEFORE any forward ──
    kept: List[Tuple[Transition, int]] = []
    for tr in batch:
        cand_idx = full_to_cand.get(tr.action_idx, -1)
        if cand_idx >= 0:
            kept.append((tr, cand_idx))

    # Batch-level advantage statistics (no autograd; just numpy on stored scalars).
    advantages_stats = torch.tensor(
        [tr.advantage for tr, _ in kept],
        dtype=torch.float32, device=device,
    ) if kept else None

    if not kept:
        return {
            "policy_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
            "clip_frac": 0.0, "ratio_max": 1.0, "ratio_mean": 1.0,
            "adv_mean": 0.0, "adv_std": 0.0, "adv_min": 0.0, "adv_max": 0.0,
            "frac_pos_advantage": 0.5, "value_loss": 0.0,
            "n_kept": 0, "n_dropped": len(batch),
        }

    N_kept = len(kept)

    # ── Per-transition forward → loss → backward (graph released each iter) ──
    policy_loss_sum = 0.0
    entropy_sum     = 0.0
    approx_kl_sum   = 0.0
    clip_count      = 0
    ratio_values: List[float] = []
    value_loss_sum  = 0.0
    n_nonfinite     = 0

    for tr, cand_idx in kept:
        cand_logp, pooled = rl_layer._score_actions(
            tr.prompt_text, candidate_set, with_grad=True,
        )
        if not torch.isfinite(cand_logp).all():
            n_nonfinite += 1
            continue
        dist = torch.distributions.Categorical(logits=cand_logp)
        idx_t = torch.tensor(cand_idx, device=device)
        new_log_prob = dist.log_prob(idx_t).to(torch.float32)
        entropy_term = dist.entropy().to(torch.float32)

        old_log_prob = torch.tensor(
            tr.old_log_prob, dtype=torch.float32, device=device,
        )
        advantage = torch.tensor(
            tr.advantage, dtype=torch.float32, device=device,
        )

        # PPO clipped surrogate — scalar per transition.
        ratio = (new_log_prob - old_log_prob).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        p_loss = -torch.min(surr1, surr2)

        if value_loss_enabled:
            returns_t   = torch.tensor(tr.returns,   dtype=torch.float32, device=device)
            old_value_t = torch.tensor(tr.old_value, dtype=torch.float32, device=device)
            new_value = rl_layer.value_head(pooled).squeeze(-1).to(torch.float32)
            value_clipped = old_value_t + torch.clamp(
                new_value - old_value_t, -value_clip_eps, value_clip_eps,
            )
            v_loss1 = F.mse_loss(new_value, returns_t, reduction="none").squeeze()
            v_loss2 = F.mse_loss(value_clipped, returns_t, reduction="none").squeeze()
            # PPO2 value clipping uses MAX, not min — see centralized_critic.py.
            v_loss  = torch.max(v_loss1, v_loss2)
            loss_term = (p_loss + value_coef * v_loss - entropy_coef * entropy_term) / N_kept
            value_loss_sum += float(v_loss.item())
        else:
            loss_term = (p_loss - entropy_coef * entropy_term) / N_kept

        # ── Backward releases the forward graph ──
        if scaler is not None:
            scaler.scale(loss_term).backward()
        else:
            loss_term.backward()

        # Diagnostics — scalars only, no autograd-connected tensors retained.
        r_val = float(ratio.item())
        ratio_values.append(r_val)
        policy_loss_sum += float(p_loss.item())
        entropy_sum     += float(entropy_term.item())
        approx_kl_sum   += float((old_log_prob - new_log_prob).item())
        if abs(r_val - 1.0) > clip_eps:
            clip_count += 1

    if n_nonfinite:
        logger.warning(
            "action_level_ppo_step: %d/%d transitions had non-finite logits "
            "and were skipped (model may be diverging — check grad norms).",
            n_nonfinite, N_kept,
        )

    if not ratio_values:
        return {
            "policy_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
            "clip_frac": 0.0, "ratio_max": 1.0, "ratio_mean": 1.0,
            "adv_mean": 0.0, "adv_std": 0.0, "adv_min": 0.0, "adv_max": 0.0,
            "frac_pos_advantage": 0.5, "value_loss": 0.0,
            "n_kept": 0, "n_dropped": len(batch), "n_nonfinite": n_nonfinite,
        }

    # ── Build info dict (means over the kept mini-batch) ──
    info = {
        "policy_loss": policy_loss_sum / N_kept,
        "entropy":     entropy_sum / N_kept,
        "approx_kl":   approx_kl_sum / N_kept,
        "clip_frac":   clip_count / N_kept,
        "ratio_max":   max(ratio_values),
        "ratio_mean":  sum(ratio_values) / N_kept,
        "adv_mean":    float(advantages_stats.mean().item()),
        "adv_std":     float(advantages_stats.std().item()) if N_kept > 1 else 0.0,
        "adv_min":     float(advantages_stats.min().item()),
        "adv_max":     float(advantages_stats.max().item()),
        "frac_pos_advantage": float((advantages_stats > 0).float().mean().item()),
        "n_kept":      N_kept,
        "n_dropped":   len(batch) - N_kept,
        "n_nonfinite": n_nonfinite,
        "value_loss":  value_loss_sum / N_kept if value_loss_enabled else 0.0,
    }
    return info


# ── Token-level PPO update (for self-triggered fine-tuning) ──

def token_level_ppo_step(
    model: nn.Module,
    tokenizer,
    batch: List[Transition],
    clip_eps: float,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """Token-level PPO over full generated sequences.

    Here we treat the entire prompt+response as a sequence and compute
    per-token log-probs.  The reward is assigned at the sequence level
    and distributed uniformly across response tokens.
    """
    prompts = [t.prompt_text for t in batch]
    rewards = torch.tensor([t.reward for t in batch],
                           dtype=torch.float32, device=device)
    old_log_probs_seq = torch.tensor([t.old_log_prob for t in batch],
                                     dtype=torch.float32, device=device)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    outputs = model(**enc, labels=enc.input_ids)
    logits = outputs.logits[:, :-1, :]  # (B, L-1, V)
    targets = enc.input_ids[:, 1:]  # (B, L-1)
    mask = enc.attention_mask[:, 1:].float()  # (B, L-1)

    per_token_log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(logits.size(0), logits.size(1))  # (B, L-1)

    seq_lengths = mask.sum(dim=1).clamp(min=1).float()  # (B,) avoid div-by-zero
    seq_log_probs = (per_token_log_probs * mask).sum(dim=1) / seq_lengths  # (B,) normalized

    # PPO ratio (sequence-level)
    ratio = (seq_log_probs - old_log_probs_seq).exp()
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    info = {
        "token_policy_loss": loss.item(),
        "approx_kl": (old_log_probs_seq - seq_log_probs).mean().item(),
    }
    return loss, info
