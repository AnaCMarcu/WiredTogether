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
    _first_batch_checked = False
    for epoch_i in range(rl.config.ppo_epochs):
        for batch in rl.buffer.sample_batches(
            rl.config.mini_batch_size, extra_transitions=social_transitions,
        ):
            rl.optimizer.zero_grad()
            with torch.amp.autocast(rl._device.type, dtype=rl._dtype):
                info = action_level_ppo_step(
                    rl_layer=rl,
                    batch=batch,
                    clip_eps=rl.config.clip_eps,
                    value_clip_eps=rl.config.value_clip_eps,
                    entropy_coef=entropy_coef,
                    value_coef=rl.config.value_coef,
                    device=rl._device,
                    max_length=rl.config.rl_prompt_max_tokens,
                    value_loss_enabled=not rl._use_centralized,
                    scaler=scaler,
                )
            # Ratio sanity check assumes on-policy data: with social replay
            # mixed in, first-epoch ratios are legitimately π_i/π_j ≠ 1, so
            # the check would cry "tokenization inconsistent" spuriously.
            if (epoch_i == 0 and not _first_batch_checked
                    and not social_transitions and info.get("n_kept", 0) > 0):
                _first_batch_checked = True
                r_mean = float(info.get("ratio_mean", 1.0))
                if abs(r_mean - 1.0) > 0.05:
                    logger.warning(
                        "RLLayer agent %d first-epoch mean(ratio)=%.4f deviates "
                        "from 1.0 — tokenization/scoring may be inconsistent "
                        "between select_action and action_level_ppo_step. "
                        "Verify _candidate_token_ids match the candidate "
                        "ordering used at sample time.",
                        rl.agent_id, r_mean,
                    )
                else:
                    logger.info(
                        "RLLayer agent %d first-epoch mean(ratio)=%.4f (OK)",
                        rl.agent_id, r_mean,
                    )
            if info.get("n_kept", 0) > 0:
                scaler.unscale_(rl.optimizer)
                total_norm = nn.utils.clip_grad_norm_(
                    rl.model.parameters(), rl.config.max_grad_norm,
                )
                if torch.isfinite(total_norm):
                    scaler.step(rl.optimizer)
                else:
                    logger.warning(
                        "RLLayer agent %d update #%d: non-finite grad norm "
                        "(%s) — skipping optimizer step to avoid corrupting "
                        "weights", rl.agent_id, rl._update_count + 1, total_norm,
                    )
                    rl.optimizer.zero_grad(set_to_none=True)
                scaler.update()
            all_info = info  # keep last batch info

    rl._update_count += 1
    rl.buffer.clear()
    all_info["entropy_coef"] = entropy_coef
    # How many bond-weighted neighbour transitions entered this update's
    # pool (0 when replay is off) — lands in record_rl_update/wandb so the
    # analysis can verify the mixture actually fired per update.
    all_info["social_replay_n"] = len(social_transitions)

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
    """Sample neighbour transitions weighted by Hebbian bonds (Eq. 7).

    Two correctness properties this function is responsible for:

    **Advantages are computed here, on a copy.** Agents update in sequence
    within the same step, so at the moment agent i samples, a neighbour's
    buffer has NOT had its own ``compute_gae`` run yet (its update comes
    later — or its buffer was already cleared by an earlier update). The
    dataclass-default ``advantage=0.0`` would contribute zero policy
    gradient but a spurious entropy/value term. So each selected
    neighbour's transitions are deep-copied and GAE is computed on the
    copy, with the same γ/λ and the same centralised/independent baseline
    switch the owner would use; the neighbour's real buffer is untouched.
    (Historical note: before 2026-08-20 this function also passed a sizes
    list that put 0 at agent_i's own slot, so ``int(ρ·own_size)`` was
    always 0 and social replay silently never fired — in ANY run.)

    **Off-policy correction is PPO's own ratio.** A neighbour transition
    stores log π_j(a|s) as ``old_log_prob``; ``action_level_ppo_step``
    re-scores the prompt under agent i's adapter, so its ratio
    exp(log π_i − log π_j) IS the importance weight π_i/π_j, and the PPO
    clip bounds its variance — shared-experience actor-critic (SEAC,
    Christianos et al. 2020) with a clipped estimator instead of a raw
    IS weight. This is the correction the old "disabled until IS
    correction is added" note in hebbian/config.py asked for.
    """
    social_transitions = []
    if not neighbour_buffers or hebbian_graph is None:
        return social_transitions

    # Full per-agent sizes INCLUDING our own buffer at agent_id's slot —
    # get_social_replay_indices scales the neighbour count by own_size.
    buffer_sizes = {aid: len(buf) for aid, buf in neighbour_buffers.items()}
    buffer_sizes[rl.agent_id] = len(rl.buffer)
    sizes_list = [buffer_sizes.get(i, 0) for i in range(max(buffer_sizes) + 1)]
    indices = hebbian_graph.get_social_replay_indices(
        agent_i=rl.agent_id,
        buffer_sizes=sizes_list,
        rho=hebbian_graph.config.social_replay_rho,
    )
    if not indices:
        return social_transitions

    # GAE-on-copy, once per neighbour that was actually selected.
    gaed: Dict[int, list] = {}
    for agent_j in {j for _, j in indices}:
        buf_j = neighbour_buffers.get(agent_j)
        if buf_j is None or len(buf_j) == 0:
            continue
        gaed[agent_j] = _gae_on_copy(rl, buf_j)

    for buf_idx, agent_j in indices:
        all_j = gaed.get(agent_j)
        if all_j is not None and buf_idx < len(all_j):
            social_transitions.append(all_j[buf_idx])

    if social_transitions:
        logger.info(
            "RLLayer agent %d: social replay — %d neighbour transitions "
            "from %d agents (rho=%.2f, own=%d)",
            rl.agent_id, len(social_transitions), len(gaed),
            hebbian_graph.config.social_replay_rho, len(rl.buffer),
        )
    return social_transitions


def _gae_on_copy(rl: "RLLayer", buf_j) -> list:
    """Deep-copy a neighbour buffer's transitions and compute GAE on the copy.

    Baseline switch mirrors the owner's own update: centralised mode uses
    ``old_value_global`` (the shared critic's V_global, identical for all
    agents at a step) with the critic re-evaluating the last joint state as
    bootstrap; independent mode uses the neighbour's stored ``old_value``
    (its value head's V(s) at selection time) throughout — including as the
    bootstrap, matching the codebase's V(s_T) ≈ V(s_{T-1}) approximation in
    ``_bootstrap_last_value`` without needing the neighbour's value head.
    """
    import copy

    from rl_layer.trajectory_buffer import RolloutBuffer

    tmp = RolloutBuffer(max_size=buf_j.max_size)
    tmp._buf = copy.deepcopy(buf_j.get_all())
    last = tmp._buf[-1]
    if last.done:
        last_value = 0.0
    elif rl._use_centralized and last.joint_state is not None:
        last_value = float(rl.centralized_critic.evaluate(last.joint_state))
    elif rl._use_centralized and last.old_value_global is not None:
        last_value = last.old_value_global
    else:
        last_value = last.old_value
    tmp.compute_gae(
        rl.config.gamma, rl.config.gae_lambda, last_value,
        use_global_value=rl._use_centralized,
    )
    return tmp.get_all()


def _anneal_entropy(rl: "RLLayer") -> float:
    """Linearly decay entropy_coef across the configured update window."""
    cfg = rl.config
    if cfg.entropy_anneal_steps <= 0:
        return cfg.entropy_coef
    progress = min(rl._update_count / cfg.entropy_anneal_steps, 1.0)
    return cfg.entropy_start + progress * (cfg.entropy_end - cfg.entropy_start)
