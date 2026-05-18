"""Tests for ``rlvr.verifier.FiveChambersVerifier``.

Covers:
    * milestone scoring from each stream (Lua / Python) + dedup
    * format reward summation + weighting
    * alive bonus (presence and absence of death events)
    * idempotence
    * ``score_group`` matches per-trajectory scoring when diffusion off
    * ``explain`` decomposition has stable keys

Hebbian diffusion in ``score_group`` is exercised indirectly here (via the
"diffusion off → identity" guard) and properly in
``test_hebbian_grpo_bridge.py`` once that bridge lands in Stage 4.
"""

from __future__ import annotations

import pytest

from rlvr.trajectory import GRPOTrajectory
from rlvr.verifier import FiveChambersVerifier, VerifierConfig


def _traj(
    *,
    agent_id: int = 0,
    start: int = 0,
    end: int = 4,
    actions: list[dict] | None = None,
    milestone_events: list[dict] | None = None,
    event_log: list[dict] | None = None,
) -> GRPOTrajectory:
    if actions is None:
        actions = [{"action": "forward",
                    "communication_target": None,
                    "thoughts": ""}] * (end - start + 1)
    return GRPOTrajectory(
        prompt_id="p",
        agent_id=agent_id,
        chamber="ch3",
        start_step=start,
        end_step=end,
        actions=actions,
        env_outputs=[],
        milestone_events=milestone_events or [],
        event_log=event_log or [],
        termination_reason="horizon",
    )


# ──── milestone scoring ──────────────────────────────────────────────────


def test_lua_milestone_credits_matching_agent():
    traj = _traj(
        milestone_events=[
            {"step": 2, "agent_id": 0, "milestone_id": "m17_switch_pressed"},
        ],
    )
    v = FiveChambersVerifier(VerifierConfig(use_format_reward=False, use_alive_bonus=False))
    parts = v.explain(traj)
    assert parts["milestone"] == 40.0


def test_python_milestone_credits_via_contributors():
    traj = _traj(
        event_log=[
            {"step": 3, "type": "milestone", "id": "m17_switch_pressed",
             "contributors": ["agent_0", "agent_1"]},
        ],
    )
    v = FiveChambersVerifier(VerifierConfig(use_format_reward=False, use_alive_bonus=False))
    assert v.explain(traj)["milestone"] == 40.0


def test_milestone_skipped_for_other_agent():
    traj = _traj(
        agent_id=0,
        milestone_events=[{"step": 1, "agent_id": 1, "milestone_id": "m17_switch_pressed"}],
        event_log=[{"step": 2, "type": "milestone", "id": "m17_switch_pressed",
                    "contributors": ["agent_2"]}],
    )
    v = FiveChambersVerifier(VerifierConfig(use_format_reward=False, use_alive_bonus=False))
    assert v.explain(traj)["milestone"] == 0.0


def test_milestone_dedupe_across_streams():
    """Same (step, milestone_id) appearing in both streams counts once."""
    traj = _traj(
        agent_id=0,
        milestone_events=[{"step": 5, "agent_id": 0, "milestone_id": "m22_all_mobs_killed"}],
        event_log=[{"step": 5, "type": "milestone", "id": "m22_all_mobs_killed",
                    "contributors": ["agent_0"]}],
    )
    v = FiveChambersVerifier(VerifierConfig(use_format_reward=False, use_alive_bonus=False))
    assert v.explain(traj)["milestone"] == 150.0  # not 300.0


def test_unknown_milestone_ignored():
    traj = _traj(
        milestone_events=[{"step": 1, "agent_id": 0, "milestone_id": "m99_fake"}],
    )
    v = FiveChambersVerifier(VerifierConfig(use_format_reward=False, use_alive_bonus=False))
    assert v.explain(traj)["milestone"] == 0.0


def test_communication_milestone_credited():
    traj = _traj(
        event_log=[{"step": 1, "type": "comm_milestone", "id": "m_comm_ch3",
                    "contributors": ["agent_0"]}],
    )
    v = FiveChambersVerifier(VerifierConfig(use_format_reward=False, use_alive_bonus=False))
    assert v.explain(traj)["milestone"] == 30.0


# ──── format reward ─────────────────────────────────────────────────────


def test_format_reward_sums_per_step():
    actions = [
        {"action": "forward", "communication_target": None, "thoughts": ""},  # 1.0
        {"action": "forward"},                                                  # 0.5
        {"action": "fly"},                                                      # 0.0 (invalid)
    ]
    traj = _traj(actions=actions, end=2)
    cfg = VerifierConfig(use_milestone_rewards=False, use_alive_bonus=False,
                         format_reward_weight=1.0)
    parts = FiveChambersVerifier(cfg).explain(traj)
    assert parts["format"] == 1.5  # 1.0 + 0.5 + 0.0


def test_format_reward_respects_weight():
    actions = [{"action": "forward", "communication_target": None, "thoughts": ""}] * 5
    traj = _traj(actions=actions, end=4)
    cfg = VerifierConfig(use_milestone_rewards=False, use_alive_bonus=False,
                         format_reward_weight=0.1)
    parts = FiveChambersVerifier(cfg).explain(traj)
    assert parts["format"] == pytest.approx(0.5)  # 5 * 1.0 * 0.1


def test_format_disabled_returns_zero():
    actions = [{"action": "forward", "communication_target": None, "thoughts": ""}]
    traj = _traj(actions=actions, end=0)
    cfg = VerifierConfig(use_milestone_rewards=False, use_alive_bonus=False,
                         use_format_reward=False)
    assert FiveChambersVerifier(cfg).explain(traj)["format"] == 0.0


# ──── alive bonus ───────────────────────────────────────────────────────


def test_alive_bonus_when_no_death_event():
    traj = _traj()
    v = FiveChambersVerifier(VerifierConfig(use_milestone_rewards=False,
                                             use_format_reward=False,
                                             alive_bonus_amount=7.0))
    assert v.explain(traj)["alive"] == 7.0


def test_alive_bonus_drops_on_death_agent_id():
    traj = _traj(event_log=[{"step": 3, "type": "death", "agent_id": 0}])
    v = FiveChambersVerifier(VerifierConfig(use_milestone_rewards=False,
                                             use_format_reward=False))
    assert v.explain(traj)["alive"] == 0.0


def test_alive_bonus_drops_on_death_via_contributors():
    traj = _traj(event_log=[{"step": 3, "type": "agent_died",
                              "contributors": ["agent_0"]}])
    v = FiveChambersVerifier(VerifierConfig(use_milestone_rewards=False,
                                             use_format_reward=False))
    assert v.explain(traj)["alive"] == 0.0


def test_other_agents_death_doesnt_drop_bonus():
    traj = _traj(agent_id=0, event_log=[{"step": 3, "type": "death", "agent_id": 1}])
    v = FiveChambersVerifier(VerifierConfig(use_milestone_rewards=False,
                                             use_format_reward=False))
    assert v.explain(traj)["alive"] > 0.0


# ──── purity / idempotence ──────────────────────────────────────────────


def test_idempotent_scoring():
    traj = _traj(
        milestone_events=[{"step": 1, "agent_id": 0, "milestone_id": "m17_switch_pressed"}],
    )
    v = FiveChambersVerifier(VerifierConfig())
    a = v.score(traj)
    b = v.score(traj)
    assert a == b


def test_explain_keys_stable_when_components_off():
    traj = _traj()
    v = FiveChambersVerifier(VerifierConfig(
        use_milestone_rewards=False,
        use_format_reward=False,
        use_alive_bonus=False,
    ))
    parts = v.explain(traj)
    assert set(parts.keys()) == {"milestone", "format", "alive"}
    assert all(value == 0.0 for value in parts.values())


def test_score_equals_sum_of_explain():
    traj = _traj(
        milestone_events=[{"step": 1, "agent_id": 0, "milestone_id": "m17_switch_pressed"}],
    )
    v = FiveChambersVerifier(VerifierConfig())
    parts = v.explain(traj)
    assert v.score(traj) == sum(parts.values())


# ──── score_group ───────────────────────────────────────────────────────


def test_score_group_without_diffusion_equals_per_trajectory():
    trajectories = [
        _traj(agent_id=0, milestone_events=[
            {"step": 1, "agent_id": 0, "milestone_id": "m17_switch_pressed"}]),
        _traj(agent_id=1),
        _traj(agent_id=2, milestone_events=[
            {"step": 1, "agent_id": 2, "milestone_id": "m_comm_ch1"}]),
    ]
    v = FiveChambersVerifier(VerifierConfig())
    expected = [v.score(t) for t in trajectories]
    assert v.score_group(trajectories) == expected


def test_well_formed_strictly_higher_than_partial():
    """Two trajectories differing only in JSON validity — well-formed wins."""
    well = [{"action": "forward", "communication_target": None, "thoughts": "x"}] * 3
    partial = [{"action": "forward"}] * 3
    t_well = _traj(actions=well, end=2)
    t_partial = _traj(actions=partial, end=2)
    cfg = VerifierConfig(use_milestone_rewards=False, use_alive_bonus=False,
                         format_reward_weight=1.0)
    v = FiveChambersVerifier(cfg)
    assert v.score(t_well) > v.score(t_partial)
