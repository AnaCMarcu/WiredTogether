"""GRPOTrainer — clipped surrogate + KL update on the ``grpo_policy`` LoRA.

One ``step()`` =

    1. ``sampler.sample_group()``                              → G trajectories + tensors
    2. ``verifier.score_group(trajectories)``                  → G scalar rewards
    3. Build ``ScoredTrajectory`` list + ``GroupBuffer.add_group``  → advantages
    4. For each scored trajectory:
        a. Recompute ``new_logprobs`` under the policy adapter (with grad)
        b. ``ratio = exp(new_logprobs - old_logprobs)``        per token
        c. ``surrogate = min(ratio·A, clip(ratio,1±ε)·A)``     per token, with A constant per trajectory
        d. ``kl = ReferencePolicy.compute_kl(prompt, tokens, policy_logprobs=new_logprobs)`` per token
        e. ``loss_traj = -mean(surrogate) + β·mean(kl)``
    5. Mean across the group, backward, optimizer.step
    6. Return a metrics dict (used by ``metrics_grpo`` / Tensorboard / JSONL)

This module imports torch eagerly. It is only useful on HPC. The local
dev env can syntax-check it but cannot run it.

Stage-4b cross-agent borrowing: ``ScoredTrajectory.origin_agent`` flags
borrowed samples. The trainer currently treats them identically to own
samples (the clipped surrogate naturally bounds off-policy bias — see
``docs/rlvr_grpo_plan.md`` §5.4 Stage 4b). The truncated-IS variant
(Option 4b-ii) is a Stage-4 follow-up not yet implemented.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import random
from dataclasses import replace

import numpy as np

from rlvr.grpo_buffer import (
    GroupBuffer,
    PerAgentTrajectoryBuffer,
    ScoredTrajectory,
    group_relative_advantage,
)
from rlvr.metrics_grpo import (
    GRPOStepMetrics,
    _hebbian_step_stats,
    _milestone_stats,
)
from rlvr.rollout_sampler import (
    MultiAgentRolloutSampler,
    RolloutSampler,
    RolloutTensors,
)
from rlvr.trajectory import GRPOTrajectory
from rlvr.verifier import FiveChambersVerifier

if TYPE_CHECKING:
    import torch

    from rlvr.reference_policy import GRPOLanguageModel, ReferencePolicy

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Hyperparameters for one GRPO training run."""

    clip_epsilon: float = 0.2
    """PPO/GRPO clip range (1±ε on the policy ratio)."""

    kl_coefficient: float = 0.05
    """β in ``loss = -surrogate + β · KL``. RLVR-most sensitive knob."""

    learning_rate: float = 5e-6
    """LoRA-adapter learning rate. The legacy stack uses 1e-4 on a wider
    LoRA; 5e-6 is the plan default for GRPO's tighter clipped updates."""

    n_per_group: int = 4
    """Forwarded to ``SamplerConfig.n_per_group``; the trainer assumes the
    sampler enforces this."""

    total_steps: int = 1000
    """Number of GRPO ``step()`` calls to make in ``train()``."""

    checkpoint_interval: int = 100
    """Save the policy adapter every K steps."""

    log_interval: int = 10
    """Print one metrics line every K steps."""

    grad_clip_norm: float = 1.0
    """Global gradient norm clip — protects against the rare exploding
    ratio that escapes the surrogate clip."""

    team_reward: bool = False
    """Stage 3 flag: ``True`` = one shared reward per joint trajectory;
    ``False`` = per-agent reward. Stage 2 uses False (single-agent)."""

    hebbian_group_composition: bool = False
    """Stage 4b toggle. When ``True``, each trained agent's group of G is
    assembled by mixing own-buffer samples with teammate-buffer samples,
    weighted by W̄. Requires a ``HebbianGRPOBridge`` on the trainer.
    Forces ``team_reward=False`` semantics — borrowing is per-agent."""

    hebbian_borrow_fraction: float = 0.25
    """Fraction of agent i's group that comes from teammate buffers.
    Default 0.25 → K = ⌈0.75·G⌉ own + ⌈0.25·G⌉ borrowed. Bounded by
    teammate buffer availability; falls back to own when teammates empty
    (e.g. early in training)."""

    hebbian_buffer_capacity: int = 64
    """Per-agent FIFO capacity. Bounds memory and limits how stale a
    borrowed trajectory's logprobs can become."""

    hebbian_snapshot_interval: int = 10
    """How often (in GRPO steps) to write a graph snapshot to
    ``hebbian_snapshots.jsonl``. Set to 0 to disable snapshots entirely.
    No-op when the Hebbian bridge is disabled regardless of this value.
    Each snapshot is small (~1 KB for N=3 agents); 10 keeps the sidecar
    ≪ the per-step metrics jsonl."""


class GRPOTrainer:
    def __init__(
        self,
        config: GRPOConfig,
        model: "GRPOLanguageModel",
        reference: "ReferencePolicy",
        verifier: FiveChambersVerifier,
        sampler: RolloutSampler | MultiAgentRolloutSampler,
        checkpoint_dir: Path | str | None = None,
        hebbian_bridge=None,
        rng_seed: int | None = None,
    ):
        import torch

        self.config = config
        self.model = model
        self.reference = reference
        self.verifier = verifier
        self.sampler = sampler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.hebbian_bridge = hebbian_bridge
        self.step_idx = 0
        self._rng = random.Random(rng_seed)
        self._np_rng = np.random.default_rng(rng_seed)

        # Stage 4b: per-agent FIFO buffer for cross-agent borrowing. Constructed
        # unconditionally — cheap when unused. Only populated and read when
        # ``config.hebbian_group_composition`` is on.
        self.per_agent_buffer = PerAgentTrajectoryBuffer(
            capacity_per_agent=config.hebbian_buffer_capacity,
        )

        # Optimizer over the policy adapter parameters only. Reference
        # adapter is frozen (requires_grad=False) so its params would be
        # silently ignored even if included — but filter for clarity.
        trainable = [p for p in self.model.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "No trainable parameters in the policy adapter. Check that "
                "ReferencePolicy correctly froze only the reference adapter."
            )
        self.optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
        self.buffer = GroupBuffer(group_size=config.n_per_group)

        # §A.2: time-to-first-milestone tracker. Pre-populate with every
        # known milestone id as ``None`` so the sidecar JSON has a stable
        # schema (T5 reads from it). Unknown milestones that fire during
        # training get added on the fly.
        self._first_fire: dict[str, int | None] = _init_first_fire()

    # ──── one update step ───────────────────────────────────────────

    def step(self) -> GRPOStepMetrics:
        """Sample a group, score, compute loss, take one optimizer step.

        Dispatches on sampler type:
          * ``RolloutSampler`` → single-agent (Stage 2) path
          * ``MultiAgentRolloutSampler`` → multi-agent (Stage 3) path,
            with 3A vs 3B determined by ``self.config.team_reward``.
        """
        self.model.set_active_adapter(self.model.config.policy_adapter)
        if isinstance(self.sampler, MultiAgentRolloutSampler):
            return self._step_multi_agent()
        return self._step_single_agent()

    def _step_single_agent(self) -> GRPOStepMetrics:
        """Stage-2 single-agent step. Unchanged from the original ``step``."""
        rollouts: list[tuple[GRPOTrajectory, RolloutTensors]] = self.sampler.sample_group()
        trajectories = [t for t, _ in rollouts]
        tensors_list = [te for _, te in rollouts]

        rewards = self.verifier.score_group(trajectories)

        scored = [
            ScoredTrajectory(
                trajectory=traj,
                reward=reward,
                prompt_text=tensors.prompt_text,
                response_tokens=tensors.response_tokens,
                response_logprobs=tensors.response_logprobs,
            )
            for traj, reward, tensors in zip(trajectories, rewards, tensors_list)
        ]
        self.buffer.add_group(scored)
        metrics = self._update(self.buffer.get_minibatch(self.config.n_per_group))
        self.step_idx += 1
        return metrics

    def _step_multi_agent(self) -> GRPOStepMetrics:
        """Stage-3 (vanilla) or Stage-4b (Hebbian-composed) multi-agent step.

        Dispatches on ``config.hebbian_group_composition``:
          * False → Stage-3 path via ``assemble_multi_agent_batch``
          * True  → Stage-4b path: borrows trajectories from teammate
                    buffers, recomputes advantages per agent's mixed group
        """
        if self.config.hebbian_group_composition:
            return self._step_multi_agent_composed()

        joints = self.sampler.sample_joint_group()
        scored = assemble_multi_agent_batch(
            joints, self.verifier, team_reward=self.config.team_reward
        )
        metrics = self._update(scored)
        self.step_idx += 1
        return metrics

    def _step_multi_agent_composed(self) -> GRPOStepMetrics:
        """Stage-4b. Thin wrapper over ``assemble_composed_multi_agent_batch``."""
        joints = self.sampler.sample_joint_group()
        batch = assemble_composed_multi_agent_batch(
            joints=joints,
            verifier=self.verifier,
            per_agent_buffer=self.per_agent_buffer,
            config=self.config,
            hebbian_bridge=self.hebbian_bridge,
            rng=self._rng,
            np_rng=self._np_rng,
        )
        metrics = self._update(batch)
        self.step_idx += 1
        return metrics

    def _update(self, batch: list[ScoredTrajectory]) -> GRPOStepMetrics:
        """Inner update on one minibatch (== full group in Stage 2).

        Gradients are accumulated **per-sample** (one backward call per
        trajectory, scaled by 1/n) so peak activation memory is one
        forward graph at a time, not the whole batch. Without this, a
        2B LoRA model with G×n_trained_agents trajectories and long
        Craftium prompts OOMs an 80 GiB A100.
        """
        import torch

        self.optimizer.zero_grad()
        # Make sure the active adapter is the policy adapter — ReferencePolicy
        # toggles it internally and restores, but be defensive.
        self.model.set_active_adapter(self.model.config.policy_adapter)

        eps = self.config.clip_epsilon
        valid = [s for s in batch
                 if s.response_tokens is not None and s.response_logprobs is not None]
        skipped = len(batch) - len(valid)
        if skipped:
            logger.warning(
                "ScoredTrajectory missing tensors; skipping %d in loss", skipped)

        n_valid = len(valid)
        if n_valid == 0:
            return GRPOStepMetrics(
                step=self.step_idx, group_size=len(batch),
                group_mean_reward=float(sum(s.reward for s in batch)
                                        / max(len(batch), 1)),
                group_reward_std=0.0, advantage_mean_abs=0.0,
                surrogate_loss=0.0, kl_loss=0.0, total_loss=0.0,
                fraction_clipped=0.0, grad_norm=0.0,
                **_milestone_stats(batch),
                **_hebbian_step_stats(self.hebbian_bridge),
            )

        sum_surrogate = 0.0
        sum_kl = 0.0
        clipped_counts: list[int] = []
        total_counts: list[int] = []

        for s in valid:
            new_logprobs = self.model.logprobs(s.prompt_text, s.response_tokens)
            old_logprobs = s.response_logprobs.detach().to(new_logprobs.device)
            ratio = torch.exp(new_logprobs - old_logprobs)
            advantage = torch.tensor(s.advantage, device=new_logprobs.device)

            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantage
            per_token_surrogate = torch.minimum(unclipped, clipped)
            surrogate = per_token_surrogate.mean()

            # KL — reference logprobs computed under the reference adapter
            # inside compute_kl, which restores the policy adapter on exit.
            kl_per_token = self.reference.compute_kl(
                s.prompt_text, s.response_tokens,
                policy_logprobs=new_logprobs,
            )
            kl = kl_per_token.mean()

            # Divide by n_valid so accumulated grads equal the original
            # batch-mean loss exactly: Σ_i d/dθ((-surr_i + β·kl_i)/N)
            #                         = d/dθ((-mean(surr) + β·mean(kl))).
            sample_loss = (-surrogate + self.config.kl_coefficient * kl) / n_valid
            sample_loss.backward()

            sum_surrogate += float(surrogate.detach().item())
            sum_kl += float(kl.detach().item())

            with torch.no_grad():
                was_clipped = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).sum().item()
                clipped_counts.append(int(was_clipped))
                total_counts.append(int(ratio.numel()))

            # Drop refs to the graph so CUDA caching allocator can reclaim
            # activations before the next sample's forward pass.
            del new_logprobs, kl_per_token, ratio, surrogate, kl, sample_loss

        surrogate_loss = -sum_surrogate / n_valid
        kl_loss = sum_kl / n_valid
        total_loss = surrogate_loss + self.config.kl_coefficient * kl_loss

        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.model.parameters() if p.requires_grad),
            max_norm=self.config.grad_clip_norm,
        )
        self.optimizer.step()

        rewards = [s.reward for s in batch]
        advantages = [s.advantage for s in batch]
        return GRPOStepMetrics(
            step=self.step_idx,
            group_size=len(batch),
            group_mean_reward=float(sum(rewards) / len(rewards)),
            group_reward_std=float(_std(rewards)),
            advantage_mean_abs=float(sum(abs(a) for a in advantages) / len(advantages)),
            surrogate_loss=float(surrogate_loss),
            kl_loss=float(kl_loss),
            total_loss=float(total_loss),
            fraction_clipped=(sum(clipped_counts) / sum(total_counts)
                              if sum(total_counts) > 0 else 0.0),
            grad_norm=float(grad_norm),
            **_milestone_stats(batch),
            **_hebbian_step_stats(self.hebbian_bridge),
        )

    # ──── outer loop ────────────────────────────────────────────────

    def train(self, total_steps: int | None = None,
              metrics_path: Path | str | None = None) -> None:
        """Run ``total_steps`` GRPO updates.

        If ``metrics_path`` is set, every ``step()`` appends one JSON record
        to that file (one line per step). This is the data ``compare_modes.py``
        reads for the headline thesis figure.

        After the loop, writes ``time_to_first.json`` next to ``metrics_path``
        — a ``{milestone_id: first_step_index | None}`` map feeding the T5
        sample-efficiency table.
        """
        import json as _json

        n = total_steps if total_steps is not None else self.config.total_steps
        metrics_path = Path(metrics_path) if metrics_path else None
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # §A.3: hebbian_snapshots.jsonl sidecar — written next to metrics_path
        # when the bridge is active and the interval is positive.
        hebbian_snap_path: Path | None = None
        if (metrics_path is not None
                and self.hebbian_bridge is not None
                and self.config.hebbian_snapshot_interval > 0):
            hebbian_snap_path = metrics_path.parent / "hebbian_snapshots.jsonl"

        for _ in range(n):
            metrics = self.step()
            if metrics_path is not None:
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(_json.dumps(metrics.as_dict()) + "\n")
            _record_first_fires(self._first_fire, metrics)
            if (hebbian_snap_path is not None
                    and self.step_idx % self.config.hebbian_snapshot_interval == 0):
                snap = self.hebbian_bridge.snapshot(self.step_idx)
                with hebbian_snap_path.open("a", encoding="utf-8") as f:
                    f.write(_json.dumps(snap) + "\n")
            if self.step_idx % self.config.log_interval == 0:
                logger.info(
                    "grpo step=%d reward=%.2f±%.2f surrogate=%.4f kl=%.4f "
                    "clipped=%.1f%% grad=%.2f milestone_rate=%.2f borrowed=%.2f",
                    metrics.step, metrics.group_mean_reward,
                    metrics.group_reward_std, metrics.surrogate_loss,
                    metrics.kl_loss, 100 * metrics.fraction_clipped,
                    metrics.grad_norm, metrics.milestone_fire_rate,
                    metrics.borrowed_fraction,
                )
            if (self.checkpoint_dir is not None
                    and self.step_idx % self.config.checkpoint_interval == 0):
                self.save_checkpoint()

        if metrics_path is not None:
            _write_time_to_first(self._first_fire, metrics_path.parent)

    def save_checkpoint(self) -> Path | None:
        """Write the policy adapter to ``checkpoint_dir/step_<N>/``."""
        if self.checkpoint_dir is None:
            return None
        target = self.checkpoint_dir / f"step_{self.step_idx:06d}"
        target.mkdir(parents=True, exist_ok=True)
        # peft's save_pretrained saves only the active adapter by default.
        self.model.set_active_adapter(self.model.config.policy_adapter)
        self.model.model.save_pretrained(str(target))
        return target


# ──── helpers ──────────────────────────────────────────────────────────


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _init_first_fire() -> dict[str, int | None]:
    """Pre-populate the time-to-first-milestone tracker with every known
    milestone id (from ``MILESTONE_TRACK``) mapped to ``None``.

    Returns ``{}`` in dev environments where ``mindforge.agent_modules`` is
    unavailable — the tracker then only records milestones that actually
    fire during training, which keeps the trainer importable without the
    legacy stack.
    """
    try:
        from mindforge.agent_modules.craftium_metric import MILESTONE_TRACK
        return {mid: None for mid in MILESTONE_TRACK}
    except ImportError:
        return {}


def _record_first_fires(
    first_fire: dict[str, int | None],
    metrics: GRPOStepMetrics,
) -> None:
    """Mutate ``first_fire`` in place: record the first GRPO step at which
    each milestone fires.

    Idempotent on a fixed ``first_fire`` snapshot — re-calling with the
    same metrics doesn't overwrite an existing first-fire entry. A
    milestone fires when its count in ``metrics.milestone_fires_by_id`` is
    positive; entries with count 0 (or missing) leave ``first_fire``
    untouched.

    Unknown milestones (not pre-populated by ``_init_first_fire``) are
    added on the fly so the legacy ``MILESTONE_TRACK`` is not the hard
    schema — if a Lua mod fires something new, we still record it.
    """
    for mid, count in metrics.milestone_fires_by_id.items():
        if count <= 0:
            continue
        if first_fire.get(mid) is None:
            first_fire[mid] = metrics.step


def _write_time_to_first(
    first_fire: dict[str, int | None],
    parent: Path,
) -> Path | None:
    """Write ``first_fire`` to ``parent/time_to_first.json``.

    No-op (returns None) if ``first_fire`` is empty — happens in dev envs
    where ``MILESTONE_TRACK`` couldn't be imported and no milestones fired.
    Idempotent on a stable ``first_fire`` (sorted keys for diff-friendly
    output).
    """
    import json as _json

    if not first_fire:
        return None
    parent.mkdir(parents=True, exist_ok=True)
    sidecar = parent / "time_to_first.json"
    with sidecar.open("w", encoding="utf-8") as f:
        _json.dump(first_fire, f, indent=2, sort_keys=True)
    return sidecar


def assemble_composed_multi_agent_batch(
    joints: list,
    verifier: FiveChambersVerifier,
    per_agent_buffer: PerAgentTrajectoryBuffer,
    config: GRPOConfig,
    hebbian_bridge,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> list[ScoredTrajectory]:
    """Stage-4b batch assembly. Hebbian-weighted group composition.

    Side effect: each per-agent ScoredTrajectory built from ``joints`` is
    pushed onto ``per_agent_buffer`` (so future steps can borrow it).

    Per-agent borrow rules:
      * K = ⌈(1 - ``hebbian_borrow_fraction``) · G⌉ own + (G - K) borrowed
      * Borrowed trajectories carry ``origin_agent = teammate_id``
      * Group-relative advantage is computed within agent_i's *mixed* group
        — borrowed rewards influence the normalisation alongside own rewards
    """
    import math

    scores = verifier.score_joint_group(joints, team_reward=False)

    own_pool: dict[int, list[ScoredTrajectory]] = {}
    for joint, joint_scores in zip(joints, scores):
        for aid, (traj, tensors) in joint.per_agent.items():
            s = ScoredTrajectory(
                trajectory=traj,
                reward=float(joint_scores[aid]),
                advantage=0.0,
                prompt_text=tensors.prompt_text,
                response_tokens=tensors.response_tokens,
                response_logprobs=tensors.response_logprobs,
                origin_agent=None,
            )
            per_agent_buffer.add(s)
            own_pool.setdefault(aid, []).append(s)

    G = config.n_per_group
    # If the Hebbian bridge is unavailable, fall back to all-own (Stage-3
    # behaviour). Otherwise apply the borrow-fraction split.
    bridge_active = (
        hebbian_bridge is not None and hebbian_bridge.is_enabled()
    )
    if bridge_active:
        K = int(math.ceil((1.0 - config.hebbian_borrow_fraction) * G))
        K = max(0, min(K, G))
    else:
        K = G
    borrow_target = G - K

    batch: list[ScoredTrajectory] = []
    for aid in own_pool:
        own_samples = _draw_own(per_agent_buffer, aid, K, own_pool[aid], rng=rng)
        borrowed = _draw_borrowed(
            per_agent_buffer, hebbian_bridge, aid, borrow_target,
            rng=rng, np_rng=np_rng,
        )
        # Tag every sample with the borrower's id so callers can filter
        # the flat batch by "agent_i's group" — distinct from each item's
        # ``trajectory.agent_id`` (which differs for borrowed samples).
        # Use replace() to avoid mutating buffered objects — own_samples
        # are buffer references that may be re-sampled later by other agents.
        own_samples = [replace(s, owning_agent_id=aid) for s in own_samples]
        borrowed = [replace(s, owning_agent_id=aid) for s in borrowed]
        group = own_samples + borrowed
        if not group:
            continue
        rewards = np.array([s.reward for s in group], dtype=np.float64)
        advantages = group_relative_advantage(rewards)
        for s, a in zip(group, advantages):
            s.advantage = float(a)
        batch.extend(group)
    return batch


def _draw_own(
    buffer: PerAgentTrajectoryBuffer,
    agent_id: int,
    k: int,
    fallback: list[ScoredTrajectory],
    rng: random.Random,
) -> list[ScoredTrajectory]:
    if k <= 0:
        return []
    samples = buffer.sample(agent_id, k, rng=rng)
    if len(samples) < k:
        samples.extend(fallback[: k - len(samples)])
    return samples[:k]


def _draw_borrowed(
    buffer: PerAgentTrajectoryBuffer,
    hebbian_bridge,
    agent_id: int,
    n: int,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> list[ScoredTrajectory]:
    """W̄-weighted borrowing from teammate buffers.

    Fallbacks:
      * Missing / disabled bridge → no borrowed samples
      * Teammate buffer empty for a sampled teammate → that draw is skipped
      * W̄ all zero → uniform over teammates (early in training before any
        bonds form)
    """
    if n <= 0 or hebbian_bridge is None or not hebbian_bridge.is_enabled():
        return []
    w_bar = hebbian_bridge.normalized_weights(agent_id)
    n_agents = len(w_bar)
    teammates = [j for j in range(n_agents) if j != agent_id]
    if not teammates:
        return []

    out: list[ScoredTrajectory] = []
    for _ in range(n):
        w_team = np.array([w_bar[j] for j in teammates], dtype=np.float64)
        if w_team.sum() <= 1e-9:
            teammate = int(np_rng.choice(teammates))
        else:
            probs = w_team / w_team.sum()
            teammate = int(np_rng.choice(teammates, p=probs))
        samples = buffer.sample(teammate, 1, rng=rng)
        if not samples:
            continue
        out.append(replace(samples[0], origin_agent=teammate))
    return out


def assemble_multi_agent_batch(
    joints: list,
    verifier: FiveChambersVerifier,
    team_reward: bool,
) -> list[ScoredTrajectory]:
    """Score G joint rollouts, compute advantages, build the combined
    ``ScoredTrajectory`` batch the trainer's ``_update`` consumes.

    Pure-python — no torch, no model — so the dispatch logic is testable
    without the HPC stack. Used by ``GRPOTrainer._step_multi_agent``.

    3A mode (``team_reward=True``): G team-summed rewards → G group-relative
    advantages → each broadcast to every trained agent in its joint.
    Output length: ``G * N``.

    3B mode (``team_reward=False``): per-agent rewards → N independent
    group-relative normalisations (each over G rewards) → per-(joint, agent)
    advantage. Output length: ``G * N``.
    """
    scores = verifier.score_joint_group(joints, team_reward=team_reward)
    scored: list[ScoredTrajectory] = []

    if team_reward:
        team_rewards = np.array(scores, dtype=np.float64)
        advantages = group_relative_advantage(team_rewards)
        for joint, team_r, advantage in zip(joints, scores, advantages):
            for aid, (traj, tensors) in joint.per_agent.items():
                scored.append(ScoredTrajectory(
                    trajectory=traj,
                    reward=float(team_r),
                    advantage=float(advantage),
                    prompt_text=tensors.prompt_text,
                    response_tokens=tensors.response_tokens,
                    response_logprobs=tensors.response_logprobs,
                    owning_agent_id=aid,
                ))
        return scored

    # 3B
    per_agent_rewards: dict[int, list[float]] = {}
    for joint_scores in scores:
        for aid, r in joint_scores.items():
            per_agent_rewards.setdefault(aid, []).append(float(r))

    per_agent_advantage: dict[int, list[float]] = {
        aid: [
            float(a) for a in group_relative_advantage(np.array(rs, dtype=np.float64))
        ]
        for aid, rs in per_agent_rewards.items()
    }

    for joint_idx, (joint, joint_scores) in enumerate(zip(joints, scores)):
        for aid, (traj, tensors) in joint.per_agent.items():
            scored.append(ScoredTrajectory(
                trajectory=traj,
                reward=float(joint_scores[aid]),
                advantage=per_agent_advantage[aid][joint_idx],
                prompt_text=tensors.prompt_text,
                response_tokens=tensors.response_tokens,
                response_logprobs=tensors.response_logprobs,
                owning_agent_id=aid,
            ))
    return scored
