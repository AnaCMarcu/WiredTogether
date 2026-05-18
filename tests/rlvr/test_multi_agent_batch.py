"""Tests for ``assemble_multi_agent_batch`` (Stage 3 trainer dispatch).

The full ``GRPOTrainer._step_multi_agent`` runs end-to-end only on HPC.
This module covers the pure-python scoring + advantage assembly.
"""

from __future__ import annotations

import pytest

from rlvr.grpo_trainer import assemble_multi_agent_batch
from rlvr.rollout_sampler import JointRollout, RolloutTensors
from rlvr.trajectory import GRPOTrajectory
from rlvr.verifier import FiveChambersVerifier, VerifierConfig


def _traj(agent_id: int, milestone_id: str | None = None) -> GRPOTrajectory:
    milestones = []
    if milestone_id:
        milestones = [{"step": 1, "agent_id": agent_id, "milestone_id": milestone_id}]
    return GRPOTrajectory(
        prompt_id=f"p{agent_id}",
        agent_id=agent_id,
        chamber="ch3",
        start_step=0,
        end_step=2,
        actions=[],
        env_outputs=[],
        milestone_events=milestones,
        event_log=[],
        termination_reason="horizon",
    )


def _joint(milestones_by_agent: dict[int, str | None]) -> JointRollout:
    per_agent = {
        aid: (_traj(aid, mid), RolloutTensors())
        for aid, mid in milestones_by_agent.items()
    }
    return JointRollout(per_agent=per_agent)


def _verifier() -> FiveChambersVerifier:
    return FiveChambersVerifier(VerifierConfig(
        use_format_reward=False, use_alive_bonus=False,
    ))


# ──── 3A team reward ────────────────────────────────────────────────────


def test_team_reward_advantage_broadcast_across_agents():
    """Two joints, each with two trained agents. Team reward differs across
    joints → group-relative gives ±1; each is broadcast to both agents."""
    joints = [
        _joint({0: "m17_switch_pressed", 1: None}),  # team reward 40
        _joint({0: None, 1: None}),                  # team reward 0
    ]
    batch = assemble_multi_agent_batch(joints, _verifier(), team_reward=True)
    assert len(batch) == 4  # 2 joints × 2 agents

    # Joint 0's two agents share advantage +1; joint 1's two share -1.
    by_joint = [[s for s in batch if s.reward == r] for r in (40.0, 0.0)]
    assert all(s.advantage == pytest.approx(1.0, abs=1e-3) for s in by_joint[0])
    assert all(s.advantage == pytest.approx(-1.0, abs=1e-3) for s in by_joint[1])


def test_team_reward_each_agent_carries_team_reward():
    joints = [_joint({0: "m17_switch_pressed", 1: "m18_door_opened"})]
    batch = assemble_multi_agent_batch(joints, _verifier(), team_reward=True)
    assert len(batch) == 2
    # Both agents see the team reward = 40 + 60 = 100.
    assert all(s.reward == 100.0 for s in batch)


# ──── 3B per-agent reward ───────────────────────────────────────────────


def test_per_agent_advantage_normalised_per_agent():
    """Two joints, two agents.
    Agent 0 rewards across joints: [40, 0]  → advantages ±1
    Agent 1 rewards across joints: [0, 60]  → advantages ∓1
    Each agent's advantage is independent of the other.
    """
    joints = [
        _joint({0: "m17_switch_pressed", 1: None}),
        _joint({0: None, 1: "m18_door_opened"}),
    ]
    batch = assemble_multi_agent_batch(joints, _verifier(), team_reward=False)
    assert len(batch) == 4

    by_agent_joint = {(s.trajectory.agent_id, s.trajectory.start_step): s
                      for s in batch}
    # Agent 0 in joint 0 (rewarded) → +adv. Agent 0 in joint 1 → -adv.
    s_a0_j0 = next(s for s in batch
                   if s.trajectory.agent_id == 0 and s.reward == 40.0)
    s_a0_j1 = next(s for s in batch
                   if s.trajectory.agent_id == 0 and s.reward == 0.0)
    assert s_a0_j0.advantage > 0
    assert s_a0_j1.advantage < 0
    # Agent 1's group has same shape but with opposite assignment.
    s_a1_j0 = next(s for s in batch
                   if s.trajectory.agent_id == 1 and s.reward == 0.0)
    s_a1_j1 = next(s for s in batch
                   if s.trajectory.agent_id == 1 and s.reward == 60.0)
    assert s_a1_j0.advantage < 0
    assert s_a1_j1.advantage > 0


def test_per_agent_batch_size_is_joints_times_agents():
    joints = [_joint({0: None, 1: None, 2: None}) for _ in range(4)]
    batch = assemble_multi_agent_batch(joints, _verifier(), team_reward=False)
    assert len(batch) == 12   # 4 × 3


# ──── ScoredTrajectory carries the right trajectory ────────────────────


def test_each_scored_carries_correct_trajectory():
    """Use distinct-reward milestones so we can match score→trajectory."""
    joints = [
        _joint({0: "m17_switch_pressed", 1: None}),     # agent 0: 40
        _joint({0: None, 1: "m22_all_mobs_killed"}),    # agent 1: 150
    ]
    batch = assemble_multi_agent_batch(joints, _verifier(), team_reward=False)
    s_40 = next(s for s in batch if s.reward == 40.0)
    s_150 = next(s for s in batch if s.reward == 150.0)
    assert any(me["milestone_id"] == "m17_switch_pressed"
               for me in s_40.trajectory.milestone_events)
    assert any(me["milestone_id"] == "m22_all_mobs_killed"
               for me in s_150.trajectory.milestone_events)


# ──── default-zero std degeneracy ──────────────────────────────────────


def test_team_reward_all_zero_gives_zero_advantages():
    """When every joint scores the same team reward, group-relative gives
    ~0 advantages everywhere — no within-group signal."""
    joints = [_joint({0: None, 1: None}) for _ in range(3)]
    batch = assemble_multi_agent_batch(joints, _verifier(), team_reward=True)
    assert all(abs(s.advantage) < 1e-6 for s in batch)
