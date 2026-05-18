"""Tests for ``rlvr.rollout_sampler.RolloutSampler``.

Stub env + stub policy — no torch, no real LLM, no real Craftium. These
tests exercise the sampling logic (grouping, termination, prompt_id
bucketing) in isolation. Real-env integration is tested only on HPC.
"""

from __future__ import annotations

import pytest

from rlvr.rollout_sampler import (
    Policy,
    RolloutEnv,
    RolloutSampler,
    RolloutTensors,
    SamplerConfig,
)
from rlvr.trajectory import GRPOTrajectory


class _ScriptedEnv:
    """A stub env that fires ``m17_switch_pressed`` on env step ``fire_at``
    if ``fire`` is True. Reset places the agent at a configurable position.
    """

    def __init__(self, *, fire: bool = False, fire_at: int = 3,
                 start_position=(0.0, 0.0, 0.0), chamber: str = "ch3"):
        self.fire = fire
        self.fire_at = fire_at
        self.start_position = start_position
        self.chamber = chamber
        self._t = 0

    def reset(self):
        self._t = 0
        return {}, {"chamber": self.chamber, "position": self.start_position}

    def step(self, action):
        self._t += 1
        info = {"chamber": self.chamber, "position": self.start_position}
        if self.fire and self._t == self.fire_at:
            info["milestone_events"] = [{
                "step": self._t,
                "agent_id": 0,
                "milestone_id": "m17_switch_pressed",
            }]
        return {}, 0.0, False, info


class _DyingEnv(_ScriptedEnv):
    """Emits a death event at step ``die_at``."""

    def __init__(self, *, die_at: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.die_at = die_at

    def step(self, action):
        obs, r, done, info = super().step(action)
        if self._t == self.die_at:
            info["events"] = [{"step": self._t, "type": "death", "agent_id": 0}]
        return obs, r, done, info


class _RotatingPositionEnv(_ScriptedEnv):
    """Each reset returns a different starting position, cycling through a list."""

    def __init__(self, positions, **kwargs):
        super().__init__(**kwargs)
        self.positions = positions
        self._reset_count = 0

    def reset(self):
        pos = self.positions[self._reset_count % len(self.positions)]
        self._reset_count += 1
        self._t = 0
        return {}, {"chamber": self.chamber, "position": pos}


class _NopPolicy:
    """Emits a constant action with no tensors. Logprobs are None."""

    def act(self, observation, info):
        return ({"action": "nop", "communication_target": None, "thoughts": ""},
                RolloutTensors(prompt_text="prompt"))


# ──── prompt-id bucketing ───────────────────────────────────────────────


def test_prompt_id_buckets_nearby_positions_together():
    sampler = RolloutSampler(
        env=_ScriptedEnv(),
        policy=_NopPolicy(),
        config=SamplerConfig(position_bucket_size=2.0),
    )
    id_a = sampler._make_prompt_id("ch3", (0.4, 0.0, 0.7))
    id_b = sampler._make_prompt_id("ch3", (0.6, 0.0, 0.8))
    assert id_a == id_b


def test_prompt_id_separates_distant_positions():
    sampler = RolloutSampler(
        env=_ScriptedEnv(),
        policy=_NopPolicy(),
        config=SamplerConfig(position_bucket_size=2.0),
    )
    id_a = sampler._make_prompt_id("ch3", (0.0, 0.0, 0.0))
    id_b = sampler._make_prompt_id("ch3", (10.0, 0.0, 0.0))
    assert id_a != id_b


def test_prompt_id_separates_different_chambers():
    sampler = RolloutSampler(
        env=_ScriptedEnv(),
        policy=_NopPolicy(),
        config=SamplerConfig(),
    )
    id_a = sampler._make_prompt_id("ch3", (0.0, 0.0, 0.0))
    id_b = sampler._make_prompt_id("ch4", (0.0, 0.0, 0.0))
    assert id_a != id_b


# ──── horizon termination ───────────────────────────────────────────────


def test_horizon_is_hard_cap():
    sampler = RolloutSampler(
        env=_ScriptedEnv(fire=False),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=1, horizon=5),
    )
    group = sampler.sample_group()
    traj, _ = group[0]
    assert traj.n_steps() == 5
    assert traj.termination_reason == "horizon"


def test_milestone_fires_early_termination():
    sampler = RolloutSampler(
        env=_ScriptedEnv(fire=True, fire_at=3),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=1, horizon=20),
    )
    group = sampler.sample_group()
    traj, _ = group[0]
    assert traj.termination_reason == "milestone_fired"
    assert traj.n_steps() == 3   # terminated on the firing step


def test_death_causes_early_termination():
    sampler = RolloutSampler(
        env=_DyingEnv(die_at=4),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=1, horizon=20),
    )
    group = sampler.sample_group()
    traj, _ = group[0]
    assert traj.termination_reason == "death"


# ──── group assembly ────────────────────────────────────────────────────


def test_group_assembles_when_bucket_full():
    """Same-position rollouts all share a prompt_id → bucket fills in G rollouts."""
    sampler = RolloutSampler(
        env=_ScriptedEnv(),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=3, horizon=5),
    )
    group = sampler.sample_group()
    assert len(group) == 3
    pids = {traj.prompt_id for traj, _ in group}
    assert len(pids) == 1   # all share one bucket


def test_distant_buckets_collected_in_parallel():
    """When position alternates between two distant buckets, the sampler
    needs 2*G rollouts to fill one bucket. Verifies the streaming logic
    handles partial-bucket retention across calls.
    """
    positions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)] * 4   # alternating
    sampler = RolloutSampler(
        env=_RotatingPositionEnv(positions=positions),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=2, horizon=5),
    )
    # The first 2 rollouts land in different buckets → no group emitted yet.
    # The 3rd and 4th rollouts complete each bucket. sample_group emits
    # whichever bucket fills first.
    group = sampler.sample_group()
    assert len(group) == 2
    pids = {traj.prompt_id for traj, _ in group}
    assert len(pids) == 1


def test_group_emits_distinct_trajectories():
    """The same group's trajectories share a prompt_id but are otherwise
    independent rollouts (different ``start_step``s)."""
    sampler = RolloutSampler(
        env=_ScriptedEnv(),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=4, horizon=3),
    )
    group = sampler.sample_group()
    start_steps = sorted(traj.start_step for traj, _ in group)
    assert len(set(start_steps)) == 4   # all distinct
    # And they cover contiguous step windows of length 3.
    assert start_steps == [0, 3, 6, 9]


# ──── safety ───────────────────────────────────────────────────────────


def test_max_resets_safety_bound():
    """If no bucket can fill within max_resets_per_group, raise rather than
    spin forever."""
    positions = [(2 * i, 0.0, 0.0) for i in range(100)]  # all distinct buckets
    sampler = RolloutSampler(
        env=_RotatingPositionEnv(positions=positions),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=2, horizon=2, max_resets_per_group=10),
    )
    with pytest.raises(RuntimeError, match="did not fill any bucket"):
        sampler.sample_group()


# ──── tensors are forwarded ─────────────────────────────────────────────


def test_rollout_tensors_carry_prompt_text():
    """The sampler concatenates per-step prompt text. With H=3, accumulated
    prompt text is the per-step text × 3.
    """
    sampler = RolloutSampler(
        env=_ScriptedEnv(),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=1, horizon=3),
    )
    group = sampler.sample_group()
    _, tensors = group[0]
    assert tensors.prompt_text == "prompt" * 3


def test_env_outputs_aligned_with_actions():
    sampler = RolloutSampler(
        env=_ScriptedEnv(),
        policy=_NopPolicy(),
        config=SamplerConfig(n_per_group=1, horizon=4),
    )
    group = sampler.sample_group()
    traj, _ = group[0]
    assert len(traj.actions) == len(traj.env_outputs)
