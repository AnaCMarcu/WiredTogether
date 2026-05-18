"""Tests for ``_milestone_stats``: aggregates milestone counts, borrowed
fraction, and per-agent breakdowns for the metrics stream.
"""

from __future__ import annotations

from rlvr.grpo_buffer import ScoredTrajectory
from rlvr.grpo_trainer import _milestone_stats
from rlvr.trajectory import GRPOTrajectory


def _scored(
    *,
    agent_id: int,
    owning: int | None = None,
    origin: int | None = None,
    reward: float = 0.0,
    fires: int = 0,
) -> ScoredTrajectory:
    milestones = [
        {"step": i, "agent_id": agent_id, "milestone_id": f"m{i}"}
        for i in range(fires)
    ]
    return ScoredTrajectory(
        trajectory=GRPOTrajectory(
            prompt_id="p", agent_id=agent_id, chamber="ch3",
            start_step=0, end_step=2,
            milestone_events=milestones,
        ),
        reward=reward,
        owning_agent_id=owning,
        origin_agent=origin,
    )


def test_empty_batch_returns_zeros():
    out = _milestone_stats([])
    assert out["milestone_fires"] == 0
    assert out["milestone_fire_rate"] == 0.0
    assert out["borrowed_fraction"] == 0.0
    assert out["per_agent_reward"] == {}


def test_per_agent_reward_groups_by_owning_agent():
    batch = [
        _scored(agent_id=0, owning=0, reward=10.0),
        _scored(agent_id=0, owning=0, reward=20.0),
        _scored(agent_id=1, owning=1, reward=5.0),
    ]
    out = _milestone_stats(batch)
    assert out["per_agent_reward"] == {"0": 15.0, "1": 5.0}


def test_borrowed_fraction_uses_origin_agent():
    batch = [
        _scored(agent_id=0, owning=0, origin=None),
        _scored(agent_id=1, owning=0, origin=1),    # borrowed by 0 from 1
        _scored(agent_id=2, owning=0, origin=2),    # borrowed by 0 from 2
        _scored(agent_id=0, owning=0, origin=None),
    ]
    out = _milestone_stats(batch)
    assert out["borrowed_fraction"] == 0.5


def test_per_agent_attributes_to_borrower_not_origin():
    """A trajectory with ``trajectory.agent_id=1`` borrowed by agent 0
    contributes to agent 0's per_agent_reward, not agent 1's."""
    batch = [
        _scored(agent_id=1, owning=0, origin=1, reward=42.0),
    ]
    out = _milestone_stats(batch)
    assert out["per_agent_reward"] == {"0": 42.0}


def test_single_agent_path_falls_back_to_trajectory_agent_id():
    """When ``owning_agent_id`` is None (Stage-2 path), per-agent grouping
    uses ``trajectory.agent_id`` instead."""
    batch = [
        _scored(agent_id=0, owning=None, reward=10.0),
        _scored(agent_id=0, owning=None, reward=20.0),
    ]
    out = _milestone_stats(batch)
    assert out["per_agent_reward"] == {"0": 15.0}


def test_milestone_counts_sum_per_trajectory():
    batch = [
        _scored(agent_id=0, owning=0, fires=2),   # 2 milestones
        _scored(agent_id=0, owning=0, fires=0),   # 0 milestones
        _scored(agent_id=1, owning=1, fires=1),   # 1 milestone
    ]
    out = _milestone_stats(batch)
    # Total milestone events
    assert out["milestone_fires"] == 3
    # Fraction of trajectories with ≥ 1 fire = 2/3
    assert out["milestone_fire_rate"] == 2 / 3


def test_per_agent_milestone_rate():
    batch = [
        _scored(agent_id=0, owning=0, fires=1),
        _scored(agent_id=0, owning=0, fires=0),
        _scored(agent_id=1, owning=1, fires=1),
        _scored(agent_id=1, owning=1, fires=1),
    ]
    out = _milestone_stats(batch)
    # Agent 0: 1/2 trajectories with a fire. Agent 1: 2/2.
    assert out["per_agent_milestone_rate"] == {"0": 0.5, "1": 1.0}
