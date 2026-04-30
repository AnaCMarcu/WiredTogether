"""PPO update loop for RLLayer — extracted from RLLayer.update() for readability.

This module owns:
- last-value bootstrap (centralised critic OR per-agent value head)
- GAE invocation
- social-replay collection from neighbour buffers
- entropy annealing
- the PPO mini-batch loop (delegates to ``rl_layer.ippo.action_level_ppo_step``)

The class method ``RLLayer.update`` is now a thin wrapper around ``run_ppo_update``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from rl_layer.ippo import action_level_ppo_step

if TYPE_CHECKING:
    from rl_layer.rl_layer import RLLayer
    from rl_layer.trajectory_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


def run_ppo_update(
    rl: "RLLayer",
    neighbour_buffers: Optional[Dict[int, "RolloutBuffer"]] = None,
    hebbian_graph=None,
) -> Dict:
    """Run a full PPO update over ``rl.buffer`` and return an info dict.

    Mutates ``rl`` (clears its buffer, increments _update_count, optimizer step).
    Returns {} if RL is disabled or the buffer is empty.
    """
    if not rl.config.enabled or not rl.buffer.ready:
        return {}

    # Release any fragmented CUDA cache before the backward pass.
    torch.cuda.empty_cache()
    rl.model.train()

    last_value = _bootstrap_last_value(rl)
    rl.buffer.compute_gae(
        rl.config.gamma, rl.config.gae_lambda, last_value,
        use_global_value=rl._use_centralized,
    )

    social_transitions = _collect_social_replay(rl, neighbour_buffers, hebbian_graph)
    entropy_coef = _anneal_entropy(rl)

    all_info: Dict = {}
    # GradScaler off: model weights are FP16/BF16 directly so unscaling would fail.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for _ in range(rl.config.ppo_epochs):
        for batch in rl.buffer.sample_batches(
            rl.config.mini_batch_size, extra_transitions=social_transitions,
        ):
            with torch.amp.autocast(rl._device.type, dtype=rl._dtype):
                loss, info = action_level_ppo_step(
                    model=rl.model,
                    action_head=rl.action_head,
                    value_head=rl.value_head,
                    tokenizer=rl.tokenizer,
                    batch=batch,
                    clip_eps=rl.config.clip_eps,
                    value_clip_eps=rl.config.value_clip_eps,
                    entropy_coef=entropy_coef,
                    value_coef=rl.config.value_coef,
                    device=rl._device,
                    max_length=rl.config.rl_prompt_max_tokens,
                    value_loss_enabled=not rl._use_centralized,
                )
            rl.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(rl.optimizer)
            nn.utils.clip_grad_norm_(rl.model.parameters(), rl.config.max_grad_norm)
            scaler.step(rl.optimizer)
            scaler.update()
            all_info = info  # keep last batch info

    rl._update_count += 1
    rl.buffer.clear()
    all_info["entropy_coef"] = entropy_coef

    logger.info(
        "RLLayer agent %d update #%d: %s",
        rl.agent_id, rl._update_count, all_info,
    )
    return all_info


# ─── Helpers ──────────────────────────────────────────────────────────

def _bootstrap_last_value(rl: "RLLayer") -> float:
    """V(s_T) for GAE bootstrap. Uses centralised critic when active."""
    if len(rl.buffer) == 0:
        return 0.0
    last_tr = rl.buffer.get_all()[-1]
    if last_tr.done:
        return 0.0
    if rl._use_centralized and last_tr.joint_state is not None:
        return float(rl.centralized_critic.evaluate(last_tr.joint_state))
    if rl._use_centralized:
        return 0.0  # centralised but no joint_state stored — should not happen
    with torch.no_grad():
        pooled = rl._encode_prompt(last_tr.prompt_text)
        return rl.value_head(pooled).squeeze(-1).item()


def _collect_social_replay(rl: "RLLayer", neighbour_buffers, hebbian_graph):
    """Sample neighbour transitions weighted by Hebbian bonds (Eq. 7)."""
    social_transitions = []
    if not neighbour_buffers or hebbian_graph is None:
        return social_transitions

    buffer_sizes = {aid: len(buf) for aid, buf in neighbour_buffers.items()}
    max_id = max(buffer_sizes.keys()) + 1 if buffer_sizes else 0
    sizes_list = [buffer_sizes.get(i, 0) for i in range(max_id)]
    indices = hebbian_graph.get_social_replay_indices(
        agent_i=rl.agent_id,
        buffer_sizes=sizes_list,
        rho=hebbian_graph.config.social_replay_rho,
    )
    for buf_idx, agent_j in indices:
        buf_j = neighbour_buffers.get(agent_j)
        if buf_j is None:
            continue
        all_j = buf_j.get_all()
        if buf_idx < len(all_j):
            social_transitions.append(all_j[buf_idx])

    if social_transitions:
        logger.info(
            "RLLayer agent %d: social replay — %d neighbour transitions from %d agents",
            rl.agent_id, len(social_transitions),
            len({j for _, j in indices}),
        )
    return social_transitions


def _anneal_entropy(rl: "RLLayer") -> float:
    """Linearly decay entropy_coef across the configured update window."""
    cfg = rl.config
    if cfg.entropy_anneal_steps <= 0:
        return cfg.entropy_coef
    progress = min(rl._update_count / cfg.entropy_anneal_steps, 1.0)
    return cfg.entropy_start + progress * (cfg.entropy_end - cfg.entropy_start)
