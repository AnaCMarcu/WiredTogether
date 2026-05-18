"""Tests for ``rlvr.grpo_buffer``.

The advantage math is pure numpy. These tests do NOT need torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlvr.grpo_buffer import (
    GroupBuffer,
    ScoredTrajectory,
    group_relative_advantage,
)
from rlvr.trajectory import GRPOTrajectory


def _scored(reward: float, agent_id: int = 0) -> ScoredTrajectory:
    return ScoredTrajectory(
        trajectory=GRPOTrajectory(
            prompt_id="p",
            agent_id=agent_id,
            chamber="ch3",
            start_step=0,
            end_step=4,
        ),
        reward=reward,
    )


# ──── advantage math ────────────────────────────────────────────────────


def test_advantage_zero_mean_within_group():
    """Group-relative advantages always sum to ~0."""
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    a = group_relative_advantage(rewards)
    assert a.sum() == pytest.approx(0.0, abs=1e-6)


def test_advantage_polarisation():
    """``[1, 1, 0, 0]`` → positives ~ +1, negatives ~ -1. The plan's
    Stage-2 §5.2 task 7 spot test.
    """
    rewards = np.array([1.0, 1.0, 0.0, 0.0])
    a = group_relative_advantage(rewards)
    assert a[0] == pytest.approx(1.0, abs=1e-3)
    assert a[1] == pytest.approx(1.0, abs=1e-3)
    assert a[2] == pytest.approx(-1.0, abs=1e-3)
    assert a[3] == pytest.approx(-1.0, abs=1e-3)


def test_advantage_zero_when_all_equal():
    """Identical rewards → no within-group signal."""
    rewards = np.array([5.0, 5.0, 5.0, 5.0])
    a = group_relative_advantage(rewards)
    assert np.allclose(a, 0.0, atol=1e-6)


def test_advantage_rejects_2d():
    with pytest.raises(ValueError):
        group_relative_advantage(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_advantage_empty_returns_empty():
    a = group_relative_advantage(np.array([]))
    assert a.size == 0


# ──── buffer behavior ───────────────────────────────────────────────────


def test_buffer_add_group_assigns_advantages():
    buf = GroupBuffer(group_size=4)
    scored = [_scored(r) for r in (1.0, 1.0, 0.0, 0.0)]
    buf.add_group(scored)
    assert len(buf) == 4
    advs = [s.advantage for s in buf.items]
    assert advs[0] == pytest.approx(1.0, abs=1e-3)
    assert advs[2] == pytest.approx(-1.0, abs=1e-3)


def test_buffer_rejects_wrong_size():
    buf = GroupBuffer(group_size=4)
    with pytest.raises(ValueError):
        buf.add_group([_scored(1.0), _scored(0.0)])


def test_buffer_get_minibatch_returns_in_order():
    buf = GroupBuffer(group_size=4)
    scored = [_scored(r, agent_id=i) for i, r in enumerate([1.0, 2.0, 3.0, 4.0])]
    buf.add_group(scored)
    mb = buf.get_minibatch(2)
    assert len(mb) == 2
    assert mb[0].trajectory.agent_id == 0
    assert mb[1].trajectory.agent_id == 1


def test_buffer_reset():
    buf = GroupBuffer(group_size=2)
    buf.add_group([_scored(1.0), _scored(0.0)])
    assert len(buf) == 2
    buf.reset()
    assert len(buf) == 0


def test_scored_trajectory_origin_agent_defaults_none():
    """Stage-4b borrowing tag is None by default — own-trajectory."""
    s = _scored(1.0)
    assert s.origin_agent is None


def test_buffer_overwrites_previous_group():
    """A second ``add_group`` replaces the first, doesn't append."""
    buf = GroupBuffer(group_size=2)
    buf.add_group([_scored(1.0), _scored(0.0)])
    buf.add_group([_scored(3.0), _scored(7.0)])
    rewards_in_buf = [s.reward for s in buf.items]
    assert rewards_in_buf == [3.0, 7.0]
