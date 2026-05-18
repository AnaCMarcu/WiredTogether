"""ScoredTrajectory and GroupBuffer — storage + group-relative advantages.

A ``GroupBuffer`` collects one "group" of G trajectories (assembled by a
``RolloutSampler`` from the same equivalence class), scores them with the
verifier, and computes the group-relative advantage:

    A_i = (r_i - mean(r)) / (std(r) + ε)

This advantage replaces the value-function baseline that PPO would use.
No critic, no GAE.

The advantage math is pure numpy and fully testable without torch. Torch
tensors only appear in ``response_tokens`` / ``response_logprobs``, which
the trainer fills in during the rollout. The buffer is happy to hold
``None`` for those fields if the caller is doing a dry run.

Stage-4b note: ``ScoredTrajectory.origin_agent`` is the cross-agent
borrowing tag — None for own-trajectories, teammate id for borrowed.
The trainer uses this to pick the right ``π_old`` for the surrogate ratio
(see ``docs/rlvr_grpo_plan.md`` §5.4 Stage 4b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rlvr.trajectory import GRPOTrajectory

if TYPE_CHECKING:
    import torch


_ADVANTAGE_EPSILON = 1e-8


@dataclass
class ScoredTrajectory:
    """A trajectory plus its scalar reward, group-relative advantage, and
    the prompt / response tensors needed by the trainer.

    The dataclass is mutable (no ``frozen=True``) because ``advantage`` is
    populated *after* construction via ``GroupBuffer.add_group``.
    """

    trajectory: GRPOTrajectory
    reward: float
    advantage: float = 0.0
    prompt_text: str = ""
    response_tokens: "torch.Tensor | None" = None
    response_logprobs: "torch.Tensor | None" = None
    origin_agent: int | None = None
    """``None`` if collected by this trajectory's own agent; teammate id if
    borrowed (Stage 4b). The trainer dispatches off this field to pick
    the right ``π_old`` for the surrogate ratio."""

    owning_agent_id: int | None = None
    """Which trained agent's GRPO group this sample belongs to. Identical
    to ``trajectory.agent_id`` for own samples; equal to the *borrower's*
    agent_id (and differs from ``trajectory.agent_id``) for Stage-4b
    borrowed samples. ``None`` until the assembler tags it.
    """


@dataclass
class GroupBuffer:
    """Holds one group of G trajectories. Advantages are normalised within
    the group on ``add_group``. After that the buffer is read-only until
    ``reset()``.

    Stage 2 uses one buffer per agent per update; Stage 3+ extends to a
    list of buffers (one per training agent).
    """

    group_size: int
    items: list[ScoredTrajectory] = field(default_factory=list)

    def add_group(
        self,
        scored: list[ScoredTrajectory],
    ) -> None:
        """Replace any existing contents with ``scored``, compute and assign
        group-relative advantages in place.

        ``len(scored)`` must equal ``self.group_size``.
        """
        if len(scored) != self.group_size:
            raise ValueError(
                f"GroupBuffer expected {self.group_size} scored trajectories, "
                f"got {len(scored)}"
            )
        rewards = np.array([s.reward for s in scored], dtype=np.float64)
        advantages = group_relative_advantage(rewards)
        for s, a in zip(scored, advantages):
            s.advantage = float(a)
        self.items = list(scored)

    def get_minibatch(self, batch_size: int) -> list[ScoredTrajectory]:
        """Return up to ``batch_size`` items, in stored order.

        Stage-2 caller uses ``batch_size == group_size`` (whole-group
        minibatching); later stages may sub-sample.
        """
        return list(self.items[:batch_size])

    def reset(self) -> None:
        self.items = []

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class PerAgentTrajectoryBuffer:
    """FIFO ring buffer of recent ``ScoredTrajectory`` per agent.

    Used by Stage-4b Hebbian-weighted group composition: agent i's GRPO
    group of size G is assembled by sampling K from its own buffer and
    G-K from teammate buffers, with teammate selection weighted by W̄
    (see ``docs/rlvr_grpo_plan.md`` §5.4 Stage 4b).

    Each per-agent FIFO is capped at ``capacity_per_agent`` items — old
    trajectories drop off the back as new ones arrive. Capacity bounds
    memory and limits how stale the off-policy bias can become.
    """

    capacity_per_agent: int = 64

    def __post_init__(self):
        from collections import deque
        self._buffers: dict[int, "deque[ScoredTrajectory]"] = {}
        # Stash the constructor so we don't import deque at every add().
        self._deque = deque

    def add(self, scored: ScoredTrajectory) -> None:
        aid = scored.trajectory.agent_id
        buf = self._buffers.get(aid)
        if buf is None:
            buf = self._deque(maxlen=self.capacity_per_agent)
            self._buffers[aid] = buf
        buf.append(scored)

    def sample(self, agent_id: int, n: int, rng=None) -> list[ScoredTrajectory]:
        """Sample ``n`` trajectories from agent_id's buffer with replacement.

        Returns fewer than ``n`` only if the buffer is empty.
        """
        import random as _random
        buf = self._buffers.get(agent_id)
        if not buf or n <= 0:
            return []
        r = rng if rng is not None else _random.Random()
        return [r.choice(buf) for _ in range(n)]

    def count(self, agent_id: int) -> int:
        return len(self._buffers.get(agent_id, ()))

    def reset(self) -> None:
        self._buffers.clear()


def group_relative_advantage(rewards: np.ndarray) -> np.ndarray:
    """Compute ``A_i = (r_i - mean) / (std + ε)`` across the group.

    Pure function. Numpy. Testable without torch. ``rewards`` must be 1-D.

    When every reward is identical, ``std == 0`` and the formula degenerates
    to all zeros (no signal to update on). This is the expected behaviour —
    if every trajectory got the same score, GRPO has no within-group
    information to act on. The ε in the denominator only guards against
    division by zero, not against degenerate signal.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {rewards.shape}")
    if rewards.size == 0:
        return rewards.copy()
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    return (rewards - mean) / (std + _ADVANTAGE_EPSILON)
