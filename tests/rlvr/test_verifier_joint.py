"""Tests for ``FiveChambersVerifier.score_joint_group`` (Stage 3 multi-agent).

Covers 3A team-reward and 3B per-agent reward modes. Hebbian diffusion is
exercised here with a hand-rolled fake graph; the real
``HebbianSocialGraph`` integration test lives in
``test_hebbian_grpo_bridge.py`` (Stage 4).
"""

from __future__ import annotations

import pytest

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


def _verifier(**overrides) -> FiveChambersVerifier:
    defaults = dict(use_format_reward=False, use_alive_bonus=False)
    defaults.update(overrides)
    return FiveChambersVerifier(VerifierConfig(**defaults))


# ──── 3B: per-agent rewards ──────────────────────────────────────────────


def test_per_agent_returns_dict_per_joint():
    joints = [_joint({0: "m17_switch_pressed", 1: None, 2: "m_comm_ch3"})]
    out = _verifier().score_joint_group(joints, team_reward=False)
    assert len(out) == 1
    assert out[0] == {0: 40.0, 1: 0.0, 2: 30.0}


def test_per_agent_handles_multiple_joints():
    joints = [
        _joint({0: "m17_switch_pressed", 1: None}),
        _joint({0: None, 1: "m18_door_opened"}),
    ]
    out = _verifier().score_joint_group(joints, team_reward=False)
    assert out[0] == {0: 40.0, 1: 0.0}
    assert out[1] == {0: 0.0, 1: 60.0}


# ──── 3A: team rewards ─────────────────────────────────────────────────


def test_team_reward_sums_within_joint():
    joints = [_joint({0: "m17_switch_pressed", 1: "m18_door_opened"})]
    out = _verifier().score_joint_group(joints, team_reward=True)
    assert out == [100.0]   # 40 + 60


def test_team_reward_one_per_joint():
    joints = [
        _joint({0: "m17_switch_pressed", 1: None}),
        _joint({0: None, 1: None}),
    ]
    out = _verifier().score_joint_group(joints, team_reward=True)
    assert out == [40.0, 0.0]


# ──── Hebbian diffusion (Stage 4a integration point) ──────────────────


class _FakeHebbian:
    """Identity-graph Hebbian. ``diffuse_rewards`` returns inputs unchanged
    so we can test the wiring without the real ``HebbianSocialGraph``.
    """

    class _Config:
        def __init__(self, n):
            self.num_agents = n

    def __init__(self, n: int = 3):
        self.config = self._Config(n)
        self.last_call: list | None = None

    def diffuse_rewards(self, ordered: list[float]) -> list[float]:
        self.last_call = list(ordered)
        return list(ordered)


def test_diffusion_disabled_returns_raw_per_agent():
    joints = [_joint({0: "m17_switch_pressed", 1: None, 2: None})]
    fake = _FakeHebbian(n=3)
    v = _verifier(hebbian_reward_diffusion=False)
    v.hebbian = fake
    out = v.score_joint_group(joints, team_reward=False)
    assert fake.last_call is None
    assert out[0] == {0: 40.0, 1: 0.0, 2: 0.0}


def test_diffusion_enabled_invokes_hebbian():
    joints = [_joint({0: "m17_switch_pressed", 1: None, 2: None})]
    fake = _FakeHebbian(n=3)
    v = _verifier(hebbian_reward_diffusion=True)
    v.hebbian = fake
    v.score_joint_group(joints, team_reward=False)
    assert fake.last_call == [40.0, 0.0, 0.0]


def test_diffusion_skipped_in_team_reward_mode():
    """Team reward sums first; diffusion is per-agent and doesn't apply."""
    joints = [_joint({0: "m17_switch_pressed", 1: None, 2: None})]
    fake = _FakeHebbian(n=3)
    v = _verifier(hebbian_reward_diffusion=True)
    v.hebbian = fake
    v.score_joint_group(joints, team_reward=True)
    assert fake.last_call is None
