"""Tests for the ``EpisodeLogger.register_callback`` API added in Stage 1.

The callback is the Stage-1 passive-observer integration point: ``rlvr``
listens to step / event / finalize hooks to reconstruct trajectories the
verifier will score (see ``docs/rlvr_grpo_plan.md`` §5.1).

These tests guard the API contract — they do NOT exercise the real
``PassiveLoggerCallback`` (that lives in ``rlvr.passive_logger`` and gets its
own tests when implemented).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindforge.env.episode_logger import EpisodeLogger, EpisodeLoggerCallback


class _RecordingCallback:
    """Captures every hook call so the test can assert on the sequence."""

    def __init__(self):
        self.steps: list = []
        self.events: list = []
        self.finals: list = []

    def on_step(self, step, positions, actions, messages,
                task_rewards, comm_rewards, infos):
        self.steps.append({
            "step": step,
            "positions": positions,
            "actions": actions,
            "messages": messages,
            "task_rewards": task_rewards,
            "comm_rewards": comm_rewards,
            "infos": infos,
        })

    def on_event(self, event):
        self.events.append(event)

    def on_finalize(self, summary):
        self.finals.append(summary)


class _PartialCallback:
    """Implements only on_event — verifies missing hooks are silently skipped."""

    def __init__(self):
        self.events: list = []

    def on_event(self, event):
        self.events.append(event)


class _BrokenCallback:
    """Raises in every hook — verifies failures are swallowed."""

    def on_step(self, *args, **kwargs):
        raise RuntimeError("boom-step")

    def on_event(self, *args, **kwargs):
        raise RuntimeError("boom-event")

    def on_finalize(self, *args, **kwargs):
        raise RuntimeError("boom-finalize")


def test_callback_receives_all_hooks(tmp_path: Path):
    cb = _RecordingCallback()
    elog = EpisodeLogger(tmp_path, episode=0)
    elog.register_callback(cb)

    elog.log_step(
        step=1,
        positions={0: (1.0, 2.0, 3.0)},
        actions={0: "dig"},
        messages={0: "hi"},
        task_rewards={0: 0.5},
        comm_rewards={0: 0.0},
        infos={"chambers": {0: "ch1"}, "hp": {0: 20}, "wielded": {0: "pickaxe"}},
    )
    elog.log_event({"step": 1, "type": "milestone", "milestone_id": "m17_switch_pressed"})
    elog.finalize({"return": 42.0})

    assert len(cb.steps) == 1
    assert cb.steps[0]["step"] == 1
    assert cb.steps[0]["actions"] == {0: "dig"}
    assert cb.events == [{"step": 1, "type": "milestone", "milestone_id": "m17_switch_pressed"}]
    assert cb.finals == [{"return": 42.0}]


def test_partial_callback_skips_missing_hooks(tmp_path: Path):
    cb = _PartialCallback()
    elog = EpisodeLogger(tmp_path, episode=0)
    elog.register_callback(cb)

    elog.log_step(1, {0: (0.0, 0.0, 0.0)}, {0: "nop"}, {0: ""}, {0: 0.0}, {0: 0.0}, {})
    elog.log_event({"step": 1, "type": "milestone"})
    elog.finalize({})

    # on_event was called; on_step / on_finalize were silently skipped because
    # the callback doesn't implement them.
    assert cb.events == [{"step": 1, "type": "milestone"}]


def test_broken_callback_does_not_kill_logger(tmp_path: Path, caplog):
    cb = _BrokenCallback()
    elog = EpisodeLogger(tmp_path, episode=0)
    elog.register_callback(cb)

    # Each of these should complete despite the callback raising.
    elog.log_step(1, {0: (0.0, 0.0, 0.0)}, {0: "nop"}, {0: ""}, {0: 0.0}, {0: 0.0}, {})
    elog.log_event({"step": 1, "type": "milestone"})
    elog.finalize({})

    # The on-disk log must still be intact.
    event_lines = (tmp_path / "episodes/ep_0000/event_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert event_lines == ['{"step": 1, "type": "milestone"}']

    summary = json.loads((tmp_path / "episodes/ep_0000/summary.json").read_text(encoding="utf-8"))
    assert summary == {}


def test_multiple_callbacks_fire_in_registration_order(tmp_path: Path):
    order: list[str] = []

    class A:
        def on_event(self, event):
            order.append("A")

    class B:
        def on_event(self, event):
            order.append("B")

    elog = EpisodeLogger(tmp_path, episode=0)
    elog.register_callback(A())
    elog.register_callback(B())
    elog.log_event({"type": "test"})
    elog.finalize({})

    assert order == ["A", "B"]


def test_protocol_is_runtime_checkable():
    """``EpisodeLoggerCallback`` is a runtime-checkable Protocol — duck-typed
    classes that implement the three methods satisfy ``isinstance``.
    """
    assert isinstance(_RecordingCallback(), EpisodeLoggerCallback)
    # _PartialCallback is missing on_step and on_finalize — not a full match.
    assert not isinstance(_PartialCallback(), EpisodeLoggerCallback)


def test_no_callbacks_is_fine(tmp_path: Path):
    """Backwards-compatibility: no callbacks registered → behaves exactly as
    before. This is the regression check for the legacy MAPPO path.
    """
    elog = EpisodeLogger(tmp_path, episode=0)
    elog.log_step(1, {0: (0.0, 0.0, 0.0)}, {0: "nop"}, {0: ""}, {0: 0.0}, {0: 0.0}, {})
    elog.log_event({"step": 1, "type": "milestone"})
    elog.finalize({"return": 0.0})
    # Files should still exist and be well-formed.
    assert (tmp_path / "episodes/ep_0000/event_log.jsonl").exists()
    assert (tmp_path / "episodes/ep_0000/summary.json").exists()
