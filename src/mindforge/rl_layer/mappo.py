"""MAPPO (Multi-Agent PPO) update logic.

Action-level:  optimises log π(action | prompt) for 23 discrete Craftium
actions using a classification head on the LLM's last hidden state.

Token-level:   optimises the full token-level log-likelihood of the generated
response (only triggered by the agent's learning-belief mechanism).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_layer.trajectory_buffer import Transition


# ── Helpers ──

def _normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)


# ── Action-level PPO update ──

def action_level_ppo_step(
    model: nn.Module,               # base LM (with LoRA active)
    action_head: nn.Module,         # linear: hidden_size → n_actions
    value_head: nn.Module,          # linear: hidden_size → 1
    tokenizer,
    batch: List[Transition],
    clip_eps: float,
    value_clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    device: torch.device,
    max_length: int = 512,
) -> Tuple[torch.Tensor, dict]:
    """Single PPO mini-batch update.  Returns scalar loss + info dict."""

    prompts = [t.prompt_text for t in batch]
    action_idxs = torch.tensor([t.action_idx for t in batch], device=device)
    old_log_probs = torch.tensor([t.old_log_prob for t in batch],
                                 dtype=torch.float32, device=device)
    advantages = torch.tensor([t.advantage for t in batch],
                              dtype=torch.float32, device=device)
    returns = torch.tensor([t.returns for t in batch],
                           dtype=torch.float32, device=device)
    old_values = torch.tensor([t.old_value for t in batch],
                              dtype=torch.float32, device=device)

    # Advantages are already normalized over the full rollout in compute_gae().
    # Do NOT normalize here — per-mini-batch normalization would destroy the
    # signal about which parts of the rollout were better than others.

    # Tokenize prompts — cap at max_length to bound activation memory.
    # At 512 tokens the RL prompt fits comfortably; model_max_length (32768)
    # would make the padded batch 64× larger and OOM during backprop.
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Forward pass – get last hidden state
    outputs = model(**enc, output_hidden_states=True)
    # Use last token of each sequence (like causal LM pooling)
    seq_lengths = (enc.attention_mask.sum(dim=1) - 1).to(device)  # (B,)
    last_hidden = outputs.hidden_states[-1]  # (B, L, H)
    batch_idx = torch.arange(last_hidden.size(0), device=device)
    pooled = last_hidden[batch_idx, seq_lengths].float()  # (B, H) — upcast to fp32 to prevent NaN from fp16 overflow

    # Action head → policy distribution over discrete actions
    action_logits = action_head(pooled)  # (B, n_actions)
    action_dist = torch.distributions.Categorical(logits=action_logits)
    new_log_probs = action_dist.log_prob(action_idxs)
    entropy = action_dist.entropy().mean()

    # Value head
    new_values = value_head(pooled).squeeze(-1)  # (B,)

    # ── PPO clipped policy loss ──
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # ── Clipped value loss (element-wise max, then reduce) ──
    value_clipped = old_values + torch.clamp(
        new_values - old_values, -value_clip_eps, value_clip_eps
    )
    v_loss1 = F.mse_loss(new_values, returns, reduction="none")
    v_loss2 = F.mse_loss(value_clipped, returns, reduction="none")
    value_loss = torch.min(v_loss1, v_loss2).mean()

    # ── Total loss ──
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    info = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": (old_log_probs - new_log_probs).mean().item(),
        "clip_frac": ((ratio - 1.0).abs() > clip_eps).float().mean().item(),
    }
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
    mask = enc.attention_mask[:, 1:]  # (B, L-1)

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
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    info = {
        "token_policy_loss": loss.item(),
        "approx_kl": (old_log_probs_seq - seq_log_probs).mean().item(),
    }
    return loss, info
