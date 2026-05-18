"""Tests for ``MultiAgentRolloutSampler`` (Stage 3)."""

from __future__ import annotations

import pytest

from rlvr.rollout_sampler import (
    JointRollout,
    MultiAgentRolloutSampler,
    MultiAgentSamplerConfig,
    RolloutTensors,
)


class _MultiAgentEnv:
    """3-agent stub. Each agent starts at its own configurable position;
    chamber is shared. Fires ``m17_switch_pressed`` for ``firing_agent`` at
    step ``fire_at`` if ``fire`` is True.
    """

    def __init__(
        self,
        *,
        positions=None,
        chamber: str = "ch3",
        fire: bool = False,
        fire_at: int = 3,
        firing_agent: int = 0,
    ):
        self.positions = positions or {0: (0.0, 0.0, 0.0),
                                        1: (10.0, 0.0, 10.0),
                                        2: (20.0, 0.0, 20.0)}
        self.chamber = chamber
        self.fire = fire
        self.fire_at = fire_at
        self.firing_agent = firing_agent
        self._t = 0

    def reset(self):
        self._t = 0
        obs = {aid: {} for aid in self.positions}
        info = {aid: {"chamber": self.chamber, "position": pos}
                for aid, pos in self.positions.items()}
        return obs, info

    def step(self, actions):
        self._t += 1
        obs = {aid: {} for aid in self.positions}
        rewards = {aid: 0.0 for aid in self.positions}
        done = {aid: False for aid in self.positions}
        # Milestone events appear on the firing agent's info only.
        info = {aid: {"chamber": self.chamber, "position": self.positions[aid]}
                for aid in self.positions}
        if self.fire and self._t == self.fire_at:
            info[self.firing_agent]["milestone_events"] = [{
                "step": self._t,
                "agent_id": self.firing_agent,
                "milestone_id": "m17_switch_pressed",
            }]
        return obs, rewards, done, info


class _NopPolicy:
    def act(self, observation, info):
        return ({"action": "nop", "communication_target": None, "thoughts": ""},
                RolloutTensors(prompt_text="P"))


def _make_sampler(*, trained=(0, 1, 2), G=2, H=5, env=None, fire=False):
    env = env or _MultiAgentEnv(fire=fire)
    config = MultiAgentSamplerConfig(
        n_per_group=G, horizon=H, num_agents=3, trained_agents=trained,
    )
    return MultiAgentRolloutSampler(env=env, policy=_NopPolicy(), config=config)


# ──── per-agent trajectories ────────────────────────────────────────────


def test_joint_rollout_contains_trajectory_per_trained_agent():
    sampler = _make_sampler(trained=(0, 1, 2), G=1, H=3)
    group = sampler.sample_joint_group()
    assert len(group) == 1
    rollout = group[0]
    assert isinstance(rollout, JointRollout)
    assert rollout.trained_agent_ids == [0, 1, 2]


def test_scenery_agents_excluded_from_rollout():
    sampler = _make_sampler(trained=(0,), G=1, H=3)
    group = sampler.sample_joint_group()
    assert group[0].trained_agent_ids == [0]


def test_each_agents_trajectory_has_distinct_prompt_id():
    """Three trained agents starting at distinct positions → 3 distinct
    per-agent ``prompt_id``s within a single joint rollout."""
    sampler = _make_sampler(trained=(0, 1, 2), G=1, H=3)
    group = sampler.sample_joint_group()
    rollout = group[0]
    pids = {rollout.per_agent[aid][0].prompt_id for aid in rollout.trained_agent_ids}
    assert len(pids) == 3


def test_joint_prompt_id_combines_per_agent_keys():
    sampler = _make_sampler(trained=(0, 1, 2), G=1, H=3)
    group = sampler.sample_joint_group()
    rollout = group[0]
    expected_parts = [rollout.per_agent[aid][0].prompt_id
                      for aid in rollout.trained_agent_ids]
    assert rollout.joint_prompt_id == "|".join(expected_parts)


# ──── grouping ──────────────────────────────────────────────────────────


def test_joint_group_assembles_with_same_positions():
    """Identical starting positions every reset → same joint_prompt_id →
    bucket fills in G rollouts."""
    sampler = _make_sampler(trained=(0, 1), G=3, H=2)
    group = sampler.sample_joint_group()
    assert len(group) == 3
    joint_pids = {r.joint_prompt_id for r in group}
    assert len(joint_pids) == 1


# ──── early termination ────────────────────────────────────────────────


def test_milestone_fire_triggers_early_termination():
    env = _MultiAgentEnv(fire=True, fire_at=2, firing_agent=0)
    sampler = _make_sampler(trained=(0, 1, 2), G=1, H=10, env=env)
    group = sampler.sample_joint_group()
    rollout = group[0]
    # Every per-agent trajectory shares the joint termination reason.
    for aid in rollout.trained_agent_ids:
        assert rollout.per_agent[aid][0].termination_reason == "milestone_fired"
        assert rollout.per_agent[aid][0].n_steps() == 2


def test_milestone_event_visible_on_all_agent_trajectories():
    """The aggregated milestone events are attached to every trained
    agent's trajectory; the verifier filters per agent via ``agent_id``."""
    env = _MultiAgentEnv(fire=True, fire_at=2, firing_agent=1)
    sampler = _make_sampler(trained=(0, 1, 2), G=1, H=10, env=env)
    rollout = sampler.sample_joint_group()[0]
    for aid in rollout.trained_agent_ids:
        traj, _ = rollout.per_agent[aid]
        assert any(me["milestone_id"] == "m17_switch_pressed"
                   for me in traj.milestone_events)


# ──── safety bound ─────────────────────────────────────────────────────


def test_max_resets_safety_bound():
    """Distinct positions every reset → buckets stay tiny; safety raises."""

    class _ShiftingPos:
        def __init__(self):
            self.t = 0
            self.reset_count = 0

        def reset(self):
            self.t = 0
            self.reset_count += 1
            pos = (self.reset_count * 10.0, 0.0, 0.0)
            return ({0: {}, 1: {}, 2: {}},
                    {a: {"chamber": "ch3", "position": pos} for a in (0, 1, 2)})

        def step(self, actions):
            self.t += 1
            return ({a: {} for a in (0, 1, 2)},
                    {a: 0.0 for a in (0, 1, 2)},
                    {a: False for a in (0, 1, 2)},
                    {a: {"chamber": "ch3",
                          "position": (self.reset_count * 10.0, 0.0, 0.0)}
                     for a in (0, 1, 2)})

    sampler = MultiAgentRolloutSampler(
        env=_ShiftingPos(),
        policy=_NopPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=2, horizon=2, num_agents=3,
            trained_agents=(0, 1, 2),
            max_resets_per_group=5,
        ),
    )
    with pytest.raises(RuntimeError, match="did not fill any bucket"):
        sampler.sample_joint_group()
