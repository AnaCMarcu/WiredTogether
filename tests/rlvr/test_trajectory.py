"""Tests for the ``GRPOTrajectory`` dataclass."""

from __future__ import annotations

import pytest

from rlvr.trajectory import GRPOTrajectory


def _make(start: int = 0, end: int = 4, **overrides) -> GRPOTrajectory:
    base = dict(
        prompt_id="ch3:near_switch",
        agent_id=0,
        chamber="ch3",
        start_step=start,
        end_step=end,
        actions=[{"action": "forward"}] * (end - start + 1),
        env_outputs=[{"chamber": "ch3"}] * (end - start + 1),
        milestone_events=[],
        event_log=[],
        termination_reason="horizon",
    )
    base.update(overrides)
    return GRPOTrajectory(**base)


def test_n_steps_is_inclusive():
    traj = _make(start=10, end=14)
    assert traj.n_steps() == 5


def test_frozen_prevents_mutation():
    traj = _make()
    with pytest.raises(Exception):
        # FrozenInstanceError in py3.11+; broader Exception keeps the test
        # robust to dataclass internal changes.
        traj.agent_id = 99  # type: ignore[misc]


def test_value_equality():
    a = _make()
    b = _make()
    assert a == b


def test_unequal_when_actions_differ():
    a = _make()
    b = _make(actions=[{"action": "dig"}] * 5)
    assert a != b


def test_default_lists_are_independent():
    """Regression: ``field(default_factory=list)`` must give each instance
    its own list. If we accidentally share a default list across instances,
    appending to one would mutate them all — but trajectories are frozen,
    so the actual failure mode is different: mutation through the field's
    object reference would still bleed. Guard against the shared-default
    bug regardless.
    """
    a = GRPOTrajectory(prompt_id="p", agent_id=0, chamber="ch1",
                       start_step=0, end_step=0)
    b = GRPOTrajectory(prompt_id="p", agent_id=0, chamber="ch1",
                       start_step=0, end_step=0)
    assert a.actions is not b.actions
    assert a.milestone_events is not b.milestone_events
