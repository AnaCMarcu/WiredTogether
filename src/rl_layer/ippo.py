"""Per-agent (IPPO) PPO update steps.

Each agent owns its own actor (LoRA + action head) and value head, and runs
this update independently. MAPPO is achieved by *adding* a shared
``CentralizedCritic`` (in ``rl_layer/centralized_critic.py``) and passing
``value_loss_enabled=False`` to ``action_level_ppo_step`` so the per-agent
value head is bypassed.

Action-level:  optimises log π(action | prompt) for the discrete Craftium
actions using a classification head on the LLM's last hidden state.

Token-level:   optimises the full token-level log-likelihood of the generated
response (only triggered by the agent's learning-belief mechanism).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_layer.trajectory_buffer import Transition


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
) -> Tuple[torch.Tensor, dict]:
    """Single PPO mini-batch update. Returns scalar loss + info dict.

    Surgical-swap version (no ActionHead, no action_mask tensor): the policy
    is the LLM's own sequence-log-prob over the CANDIDATE action strings,
    computed by ``rl_layer._score_actions``. The candidate set IS the mask
    (membership = allowed). For each transition we:

      1. Score the FULL candidate set via _score_actions (with gradients) so
         we get a proper softmax normaliser.
      2. Build Categorical(logits=cand_log_probs) and take log_prob at the
         stored action's INDEX WITHIN THE CANDIDATE SET (the buffer stores
         the canonical index; we round-trip via rl_layer._full_idx_to_cand_idx).
      3. ratio = exp(new − old), then the EXACT existing surr1/surr2/clip
         policy loss + entropy + diagnostics dict.

    When ``value_loss_enabled`` is False (MAPPO mode with a centralised critic),
    the per-agent value head is bypassed entirely. The critic is updated
    separately by ``CentralizedCritic.update``.

    A transition whose stored canonical action_idx is NOT in the current
    candidate set (e.g. checkpoint resumed after mask_slot_actions toggled)
    is dropped from the batch with a warning rather than crashing.
    """

    candidate_set = rl_layer._candidate_actions
    full_to_cand = rl_layer._full_idx_to_cand_idx

    cand_idxs_in_set: List[int] = []   # one per surviving transition
    kept_transitions: List[Transition] = []
    new_log_probs_list = []
    entropy_terms = []
    pooled_list = []   # only populated when value_loss_enabled

    for tr in batch:
        cand_idx = full_to_cand.get(tr.action_idx, -1)
        if cand_idx < 0:
            # Action no longer in the candidate set — skip rather than
            # let it poison the loss. Unusual; only happens on a
            # mask_slot_actions config change between ckpt save and load.
            continue
        cand_logp, pooled = rl_layer._score_actions(
            tr.prompt_text, candidate_set, with_grad=True,
        )
        dist = torch.distributions.Categorical(logits=cand_logp)
        idx_t = torch.tensor(cand_idx, device=device)
        new_log_probs_list.append(dist.log_prob(idx_t))
        entropy_terms.append(dist.entropy())
        if value_loss_enabled:
            pooled_list.append(pooled)
        kept_transitions.append(tr)
        cand_idxs_in_set.append(cand_idx)

    if not kept_transitions:
        # Degenerate mini-batch (all transitions filtered). Return a
        # zero-grad loss + empty info so the outer loop can no-op
        # gracefully without aborting the whole update.
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {
            "policy_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
            "clip_frac": 0.0, "ratio_max": 1.0, "ratio_mean": 1.0,
            "adv_mean": 0.0, "adv_std": 0.0, "adv_min": 0.0, "adv_max": 0.0,
            "frac_pos_advantage": 0.5, "value_loss": 0.0,
            "n_kept": 0, "n_dropped": len(batch),
        }

    old_log_probs = torch.tensor(
        [t.old_log_prob for t in kept_transitions],
        dtype=torch.float32, device=device,
    )
    advantages = torch.tensor(
        [t.advantage for t in kept_transitions],
        dtype=torch.float32, device=device,
    )

    new_log_probs = torch.stack(new_log_probs_list).to(torch.float32)  # (B,)
    entropy = torch.stack(entropy_terms).mean()

    # ── PPO clipped policy loss (IDENTICAL math to the previous version) ──
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Diagnostics for the audit (section 5): ratio distribution + advantage
    # sign balance. After per-rollout normalisation in compute_gae() the
    # advantage mean should be ~0; if frac_pos drifts far from 0.5 the
    # advantages are degenerate. ratio_max > 2.0 means the data is heavily
    # off-policy by the time we update.
    with torch.no_grad():
        _ratio_diff = (ratio - 1.0).abs()
        ratio_max  = float(ratio.max().item())
        ratio_mean = float(ratio.mean().item())
        adv_mean   = float(advantages.mean().item())
        adv_std    = float(advantages.std().item())
        adv_min    = float(advantages.min().item())
        adv_max    = float(advantages.max().item())
        frac_pos_advantage = float((advantages > 0).float().mean().item())

    info = {
        "policy_loss": policy_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": (old_log_probs - new_log_probs).mean().item(),
        "clip_frac": (_ratio_diff > clip_eps).float().mean().item(),
        "ratio_max":  ratio_max,
        "ratio_mean": ratio_mean,
        "adv_mean":   adv_mean,
        "adv_std":    adv_std,
        "adv_min":    adv_min,
        "adv_max":    adv_max,
        "frac_pos_advantage": frac_pos_advantage,
        "n_kept":     len(kept_transitions),
        "n_dropped":  len(batch) - len(kept_transitions),
    }

    if value_loss_enabled:
        returns = torch.tensor(
            [t.returns for t in kept_transitions],
            dtype=torch.float32, device=device,
        )
        old_values = torch.tensor(
            [t.old_value for t in kept_transitions],
            dtype=torch.float32, device=device,
        )
        # pooled_list has one (1, H) tensor per kept transition; stack to (B, H).
        pooled_batch = torch.cat(pooled_list, dim=0)  # (B, H)
        new_values = rl_layer.value_head(pooled_batch).squeeze(-1)  # (B,)
        value_clipped = old_values + torch.clamp(
            new_values - old_values, -value_clip_eps, value_clip_eps
        )
        v_loss1 = F.mse_loss(new_values, returns, reduction="none")
        v_loss2 = F.mse_loss(value_clipped, returns, reduction="none")
        # PPO2 value clipping uses MAX, not min (see centralized_critic.py
        # for the rationale). Active in independent-critic mode only.
        value_loss = torch.max(v_loss1, v_loss2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        info["value_loss"] = value_loss.item()
    else:
        loss = policy_loss - entropy_coef * entropy
        info["value_loss"] = 0.0

    return loss, info


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
    # Per-token NLL → sum per sequence → mean across batch
    # outputs.loss is already mean over tokens; we need per-sequence
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
    # Use mean-std normalization of rewards as a REINFORCE baseline (not raw _normalize,
    # which would be reward-scaling without a value baseline — same issue as normalizing
    # raw rewards and calling them advantages).
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    info = {
        "token_policy_loss": loss.item(),
        "approx_kl": (old_log_probs_seq - seq_log_probs).mean().item(),
    }
    return loss, info
