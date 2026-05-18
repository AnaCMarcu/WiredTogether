"""Tests for ``rlvr.passive_logger``.

End-to-end: register the callback on an EpisodeLogger, fire a few step /
event / finalize calls, read the resulting JSONL trajectory file, score
it with the verifier. This is the Stage-1 verification path that the plan
prescribes — once on real data, this is the gate; here, on synthetic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mindforge.env.episode_logger import EpisodeLogger
from rlvr.passive_logger import (
    PassiveLoggerCallback,
    attach,
    attach_if_enabled,
    load_trajectories,
    trajectory_from_jsonable,
    trajectory_to_jsonable,
)
from rlvr.trajectory import GRPOTrajectory
from rlvr.verifier import FiveChambersVerifier, VerifierConfig


def _drive(elog: EpisodeLogger):
    """Push enough through the logger to produce per-agent trajectories."""
    elog.log_step(
        step=1,
        positions={0: (0.0, 0.0, 0.0), 1: (1.0, 1.0, 1.0)},
        actions={0: "forward", 1: "dig"},
        messages={0: "", 1: "hi"},
        task_rewards={0: 0.0, 1: 0.0},
        comm_rewards={0: 0.0, 1: 0.0},
        infos={"chambers": {0: "ch3", 1: "ch3"},
               "hp": {0: 20, 1: 20},
               "wielded": {0: "", 1: "pickaxe"}},
    )
    elog.log_event({
        "step": 2,
        "type": "milestone",
        "id": "m17_switch_pressed",
        "contributors": ["agent_0"],
    })
    elog.log_step(
        step=2,
        positions={0: (0.0, 0.0, 1.0), 1: (1.0, 1.0, 2.0)},
        actions={0: "dig", 1: "dig"},
        messages={0: "", 1: ""},
        task_rewards={0: 40.0, 1: 0.0},
        comm_rewards={0: 0.0, 1: 0.0},
        infos={"chambers": {0: "ch3", 1: "ch3"}},
    )
    elog.finalize({"return": 40.0})


def test_passive_logger_writes_one_trajectory_per_agent(tmp_path: Path):
    elog = EpisodeLogger(tmp_path, episode=0)
    out = tmp_path / "grpo_trajectories.jsonl"
    elog.register_callback(PassiveLoggerCallback(out))

    _drive(elog)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    trajectories = [json.loads(line) for line in lines]
    agent_ids = sorted(t["agent_id"] for t in trajectories)
    assert agent_ids == [0, 1]


def test_trajectories_roundtrip_through_jsonable(tmp_path: Path):
    elog = EpisodeLogger(tmp_path, episode=0)
    out = tmp_path / "grpo_trajectories.jsonl"
    elog.register_callback(PassiveLoggerCallback(out))
    _drive(elog)

    loaded = load_trajectories(out)
    assert len(loaded) == 2
    assert all(isinstance(t, GRPOTrajectory) for t in loaded)


def test_loaded_trajectory_is_verifier_ready(tmp_path: Path):
    """End-to-end: passive observer → JSONL → load → score."""
    elog = EpisodeLogger(tmp_path, episode=0)
    out = tmp_path / "grpo_trajectories.jsonl"
    elog.register_callback(PassiveLoggerCallback(out))
    _drive(elog)

    trajectories = load_trajectories(out)
    by_agent = {t.agent_id: t for t in trajectories}

    v = FiveChambersVerifier(VerifierConfig())
    parts_0 = v.explain(by_agent[0])
    parts_1 = v.explain(by_agent[1])

    # Agent 0 fired the switch milestone → 40.0 from milestones.
    assert parts_0["milestone"] == 40.0
    # Agent 1 did not — no milestone credit.
    assert parts_1["milestone"] == 0.0
    # Both agents are alive (no death events) → equal alive bonus.
    assert parts_0["alive"] == parts_1["alive"] == 5.0
    # Format reward is small but nonzero — actions wrapped as {"action": str}
    # score 0.5 per step (optional fields absent).
    assert 0.0 < parts_0["format"] < 1.0


def test_chamber_inferred_from_first_step(tmp_path: Path):
    elog = EpisodeLogger(tmp_path, episode=0)
    out = tmp_path / "grpo_trajectories.jsonl"
    elog.register_callback(PassiveLoggerCallback(out))
    _drive(elog)

    trajectories = load_trajectories(out)
    assert all(t.chamber == "ch3" for t in trajectories)


def test_event_log_attached_to_every_trajectory(tmp_path: Path):
    """All events seen during the episode are attached to every agent's
    trajectory. The verifier filters per agent via contributors. This is
    intentionally permissive — Stage 2's sampler will tighten the window.
    """
    elog = EpisodeLogger(tmp_path, episode=0)
    out = tmp_path / "grpo_trajectories.jsonl"
    elog.register_callback(PassiveLoggerCallback(out))
    _drive(elog)

    trajectories = load_trajectories(out)
    for t in trajectories:
        assert any(ev.get("id") == "m17_switch_pressed" for ev in t.event_log)


def test_attach_helper(tmp_path: Path):
    elog = EpisodeLogger(tmp_path, episode=0)
    cb = attach(elog, tmp_path)
    assert isinstance(cb, PassiveLoggerCallback)
    assert cb.output_path == tmp_path / "grpo_trajectories.jsonl"


def test_attach_if_enabled_respects_env_var(tmp_path: Path, monkeypatch):
    elog = EpisodeLogger(tmp_path, episode=0)

    monkeypatch.delenv("RLVR_PASSIVE_LOG", raising=False)
    assert attach_if_enabled(elog, tmp_path) is None

    monkeypatch.setenv("RLVR_PASSIVE_LOG", "1")
    cb = attach_if_enabled(elog, tmp_path)
    assert isinstance(cb, PassiveLoggerCallback)


def test_buffer_resets_between_episodes(tmp_path: Path):
    """Finalize must clear internal state — otherwise episode N+1 includes
    episode N's steps and we double-emit."""
    elog1 = EpisodeLogger(tmp_path, episode=0)
    out = tmp_path / "grpo_trajectories.jsonl"
    cb = PassiveLoggerCallback(out)
    elog1.register_callback(cb)
    _drive(elog1)

    # Re-use the same callback on a fresh logger.
    elog2 = EpisodeLogger(tmp_path, episode=1)
    elog2.register_callback(cb)
    _drive(elog2)

    lines = out.read_text(encoding="utf-8").splitlines()
    # 2 agents × 2 episodes = 4 trajectories total.
    assert len(lines) == 4


def test_to_jsonable_roundtrip():
    traj = GRPOTrajectory(
        prompt_id="p",
        agent_id=2,
        chamber="ch5",
        start_step=10,
        end_step=15,
        actions=[{"action": "dig"}],
        env_outputs=[{"hp": 20}],
        milestone_events=[{"step": 12, "agent_id": 2, "milestone_id": "m27_boss_defeated"}],
        event_log=[{"step": 11, "type": "milestone"}],
        termination_reason="death",
    )
    roundtripped = trajectory_from_jsonable(trajectory_to_jsonable(traj))
    assert roundtripped == traj


def test_numpy_position_coerced_to_list(tmp_path: Path):
    """Positions arriving as numpy arrays must serialize as lists."""
    np = pytest.importorskip("numpy")
    elog = EpisodeLogger(tmp_path, episode=0)
    out = tmp_path / "grpo_trajectories.jsonl"
    elog.register_callback(PassiveLoggerCallback(out))

    elog.log_step(
        step=1,
        positions={0: np.array([1.0, 2.0, 3.0])},
        actions={0: "forward"},
        messages={0: ""},
        task_rewards={0: 0.0},
        comm_rewards={0: 0.0},
        infos={},
    )
    elog.finalize({})

    trajectories = load_trajectories(out)
    assert trajectories[0].env_outputs[0]["position"] == [1.0, 2.0, 3.0]
