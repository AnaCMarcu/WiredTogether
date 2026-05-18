"""Tests for Stage-4b Hebbian-weighted group composition.

Covers:
    * ``PerAgentTrajectoryBuffer`` (add/sample/count/capacity)
    * ``assemble_composed_multi_agent_batch`` end-to-end with stubs
    * ``_draw_borrowed`` fallbacks (no bridge, empty teammate buffer,
      all-zero W̄)
    * ``origin_agent`` tagging on borrowed samples

The full ``GRPOTrainer._step_multi_agent_composed`` needs torch (for the
``_update`` backward pass) — that's HPC-only. These tests cover the
pure-python assembly that drives it.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from rlvr.grpo_buffer import PerAgentTrajectoryBuffer, ScoredTrajectory
from rlvr.grpo_trainer import (
    GRPOConfig,
    _draw_borrowed,
    _draw_own,
    assemble_composed_multi_agent_batch,
)
from rlvr.rollout_sampler import JointRollout, RolloutTensors
from rlvr.trajectory import GRPOTrajectory
from rlvr.verifier import FiveChambersVerifier, VerifierConfig


# ──── PerAgentTrajectoryBuffer ──────────────────────────────────────────


def _scored(agent_id: int, reward: float = 1.0) -> ScoredTrajectory:
    return ScoredTrajectory(
        trajectory=GRPOTrajectory(
            prompt_id="p", agent_id=agent_id, chamber="ch3",
            start_step=0, end_step=2,
        ),
        reward=reward,
    )


def test_buffer_isolates_per_agent():
    buf = PerAgentTrajectoryBuffer(capacity_per_agent=10)
    buf.add(_scored(0))
    buf.add(_scored(1))
    buf.add(_scored(1))
    assert buf.count(0) == 1
    assert buf.count(1) == 2
    assert buf.count(2) == 0


def test_buffer_capacity_drops_oldest():
    buf = PerAgentTrajectoryBuffer(capacity_per_agent=2)
    s_old, s_mid, s_new = _scored(0, 0.1), _scored(0, 0.5), _scored(0, 0.9)
    buf.add(s_old)
    buf.add(s_mid)
    buf.add(s_new)
    assert buf.count(0) == 2
    samples = buf.sample(0, 100)   # over-sample to see what's in
    rewards = {s.reward for s in samples}
    assert 0.1 not in rewards    # the oldest got dropped
    assert rewards.issubset({0.5, 0.9})


def test_buffer_sample_empty_returns_empty():
    buf = PerAgentTrajectoryBuffer()
    assert buf.sample(0, 5) == []


def test_buffer_reset():
    buf = PerAgentTrajectoryBuffer()
    buf.add(_scored(0))
    buf.add(_scored(1))
    buf.reset()
    assert buf.count(0) == 0
    assert buf.count(1) == 0


# ──── _draw_own ─────────────────────────────────────────────────────────


def test_draw_own_uses_buffer_when_full():
    buf = PerAgentTrajectoryBuffer()
    for _ in range(5):
        buf.add(_scored(0))
    rng = random.Random(0)
    out = _draw_own(buf, agent_id=0, k=3, fallback=[], rng=rng)
    assert len(out) == 3
    assert all(s.trajectory.agent_id == 0 for s in out)


def test_draw_own_falls_back_when_buffer_undersized():
    buf = PerAgentTrajectoryBuffer()
    # Only 1 item in buffer
    buf.add(_scored(0))
    fallback = [_scored(0) for _ in range(5)]
    rng = random.Random(0)
    out = _draw_own(buf, agent_id=0, k=4, fallback=fallback, rng=rng)
    assert len(out) == 4


# ──── _draw_borrowed ───────────────────────────────────────────────────


class _FakeBridge:
    def __init__(self, w_matrix: np.ndarray, enabled: bool = True):
        self._w = w_matrix
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def normalized_weights(self, agent_id: int) -> np.ndarray:
        row = self._w[agent_id].astype(np.float64)
        return row


def test_draw_borrowed_zero_when_bridge_disabled():
    buf = PerAgentTrajectoryBuffer()
    buf.add(_scored(1))
    buf.add(_scored(2))
    bridge = _FakeBridge(np.eye(3), enabled=False)
    out = _draw_borrowed(
        buf, bridge, agent_id=0, n=3,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    assert out == []


def test_draw_borrowed_zero_when_no_bridge():
    buf = PerAgentTrajectoryBuffer()
    out = _draw_borrowed(
        buf, None, agent_id=0, n=3,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    assert out == []


def test_draw_borrowed_skips_empty_teammate_buffers():
    buf = PerAgentTrajectoryBuffer()
    # Only agent 1 has anything in their buffer
    buf.add(_scored(1))
    # W̄[0,:] = uniform over teammates (1 and 2)
    w = np.array([[0.0, 0.5, 0.5],
                  [0.5, 0.0, 0.5],
                  [0.5, 0.5, 0.0]])
    bridge = _FakeBridge(w)
    out = _draw_borrowed(
        buf, bridge, agent_id=0, n=10,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    # Out length depends on how often np_rng picks teammate 1 vs 2.
    # Every successful draw came from teammate 1 (only one with samples).
    assert all(s.origin_agent == 1 for s in out)


def test_draw_borrowed_uniform_fallback_when_w_zero():
    buf = PerAgentTrajectoryBuffer()
    buf.add(_scored(1))
    buf.add(_scored(2))
    w = np.zeros((3, 3))   # no bonds yet
    bridge = _FakeBridge(w)
    out = _draw_borrowed(
        buf, bridge, agent_id=0, n=20,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    origins = {s.origin_agent for s in out}
    assert origins == {1, 2}   # both teammates sampled at least once


def test_draw_borrowed_weighted_by_w_bar():
    """W̄[0, :] = [0, 0.95, 0.05] → almost all borrows come from agent 1."""
    buf = PerAgentTrajectoryBuffer()
    for _ in range(20):
        buf.add(_scored(1))
        buf.add(_scored(2))
    w = np.array([[0.0, 0.95, 0.05],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    bridge = _FakeBridge(w)
    out = _draw_borrowed(
        buf, bridge, agent_id=0, n=200,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    from_1 = sum(1 for s in out if s.origin_agent == 1)
    from_2 = sum(1 for s in out if s.origin_agent == 2)
    # Expect ~190 / ~10 with some sampling variance.
    assert from_1 > from_2 * 5


def test_draw_borrowed_tags_origin_agent():
    buf = PerAgentTrajectoryBuffer()
    buf.add(_scored(1))
    w = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    bridge = _FakeBridge(w)
    out = _draw_borrowed(
        buf, bridge, agent_id=0, n=3,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    assert len(out) == 3
    assert all(s.origin_agent == 1 for s in out)
    # Original buffer items have origin_agent = None — replace() must not
    # mutate the buffered objects.
    buffered = buf.sample(1, 1)[0]
    assert buffered.origin_agent is None


# ──── assemble_composed_multi_agent_batch (end-to-end) ────────────────


def _joint(milestones_by_agent: dict[int, str | None]) -> JointRollout:
    per_agent = {}
    for aid, mid in milestones_by_agent.items():
        milestones = []
        if mid:
            milestones = [{"step": 1, "agent_id": aid, "milestone_id": mid}]
        traj = GRPOTrajectory(
            prompt_id=f"p{aid}", agent_id=aid, chamber="ch3",
            start_step=0, end_step=2, actions=[], env_outputs=[],
            milestone_events=milestones, event_log=[],
            termination_reason="horizon",
        )
        per_agent[aid] = (traj, RolloutTensors())
    return JointRollout(per_agent=per_agent)


def _verifier() -> FiveChambersVerifier:
    return FiveChambersVerifier(VerifierConfig(
        use_format_reward=False, use_alive_bonus=False,
    ))


def test_assemble_composed_falls_back_without_bridge():
    """No Hebbian bridge → no borrowing, all samples are own."""
    joints = [_joint({0: "m17_switch_pressed", 1: None, 2: None}) for _ in range(4)]
    buf = PerAgentTrajectoryBuffer()
    config = GRPOConfig(
        n_per_group=4, hebbian_group_composition=True,
        hebbian_borrow_fraction=0.5,
    )
    batch = assemble_composed_multi_agent_batch(
        joints=joints, verifier=_verifier(),
        per_agent_buffer=buf, config=config,
        hebbian_bridge=None,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    # 3 trained agents × G=4 = 12 items (all own — borrowed is empty
    # because bridge is None).
    assert len(batch) == 12
    assert all(s.origin_agent is None for s in batch)


def test_assemble_composed_with_bridge_borrows():
    joints = [_joint({0: None, 1: "m17_switch_pressed", 2: None}) for _ in range(4)]
    buf = PerAgentTrajectoryBuffer()
    # Pre-seed teammate buffers so borrowing succeeds immediately.
    for aid in (0, 1, 2):
        for _ in range(8):
            buf.add(_scored(aid))

    w = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    bridge = _FakeBridge(w)
    config = GRPOConfig(
        n_per_group=4, hebbian_group_composition=True,
        hebbian_borrow_fraction=0.5,   # 2 own + 2 borrowed per agent
    )
    batch = assemble_composed_multi_agent_batch(
        joints=joints, verifier=_verifier(),
        per_agent_buffer=buf, config=config,
        hebbian_bridge=bridge,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    # 3 trained agents × G=4 = 12 items total.
    assert len(batch) == 12
    # Filter by owning_agent_id (the borrower), not trajectory.agent_id.
    a0_group = [s for s in batch if s.owning_agent_id == 0]
    assert len(a0_group) == 4
    borrowed_in_a0 = [s for s in a0_group if s.origin_agent is not None]
    # W̄[0, :] = [0, 1, 0] → every borrowed item in agent 0's group came from teammate 1.
    assert len(borrowed_in_a0) == 2
    assert all(s.origin_agent == 1 for s in borrowed_in_a0)
    # And those borrowed items' original trajectory was agent 1's.
    assert all(s.trajectory.agent_id == 1 for s in borrowed_in_a0)


def test_assemble_composed_advantages_sum_to_zero_per_agent_group():
    """Per-agent group is normalised within itself → advantages sum to ~0
    across agent_i's own + borrowed mix. Group filter is owning_agent_id."""
    joints = [_joint({0: "m17_switch_pressed" if i == 0 else None})
              for i in range(4)]
    buf = PerAgentTrajectoryBuffer()
    config = GRPOConfig(
        n_per_group=4, hebbian_group_composition=True,
        hebbian_borrow_fraction=0.0,   # all-own
    )
    batch = assemble_composed_multi_agent_batch(
        joints=joints, verifier=_verifier(),
        per_agent_buffer=buf, config=config,
        hebbian_bridge=None,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    # Only agent 0 is in the joints — its group has 4 trajectories.
    by_group = {}
    for s in batch:
        by_group.setdefault(s.owning_agent_id, []).append(s)
    for aid, group in by_group.items():
        adv_sum = sum(s.advantage for s in group)
        assert abs(adv_sum) < 1e-6, f"group {aid}: {adv_sum}"


def test_assemble_composed_pushes_to_buffer():
    """Side effect: new joints' trajectories land in per_agent_buffer for
    future borrowing."""
    joints = [_joint({0: None, 1: None}) for _ in range(3)]
    buf = PerAgentTrajectoryBuffer()
    config = GRPOConfig(
        n_per_group=3, hebbian_group_composition=True,
        hebbian_borrow_fraction=0.0,
    )
    assemble_composed_multi_agent_batch(
        joints=joints, verifier=_verifier(),
        per_agent_buffer=buf, config=config,
        hebbian_bridge=None,
        rng=random.Random(0), np_rng=np.random.default_rng(0),
    )
    assert buf.count(0) == 3
    assert buf.count(1) == 3
