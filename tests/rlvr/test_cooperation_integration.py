"""Tests for §A.4: ``MultiAgentRolloutSampler`` cooperation-metric integration.

Covers:
    * ``_load_cooperation_metric`` returns the class or None
    * ``_coop_observe`` adapts GRPO action dicts to the legacy observe API
    * ``_append_coop_summary`` writes JSONL with the right metadata
    * end-to-end: ``JointRollout.coop_summary`` is populated and the
      ``episode_summary.jsonl`` sidecar accumulates one line per joint
    * dev-env safety: when ``CooperationMetric`` is unimportable, the
      sampler still works (just skips coop tracking)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.rollout_sampler import (
    JointRollout,
    MultiAgentRolloutSampler,
    MultiAgentSamplerConfig,
    RolloutTensors,
    _append_coop_summary,
    _coerce_coop_value,
    _coop_observe,
    _load_cooperation_metric,
)


# ──── _load_cooperation_metric ──────────────────────────────────────────


def test_load_cooperation_metric_returns_class_when_available():
    """``mindforge.env.cooperation_metric`` IS importable in this env
    (no torch / no gymnasium dependency on it)."""
    cls = _load_cooperation_metric()
    assert cls is not None
    assert cls.__name__ == "CooperationMetric"


# ──── _coerce_coop_value ────────────────────────────────────────────────


def test_coerce_coop_value_passes_primitives():
    assert _coerce_coop_value(None) is None
    assert _coerce_coop_value(42) == 42
    assert _coerce_coop_value(3.14) == 3.14
    assert _coerce_coop_value("hi") == "hi"
    assert _coerce_coop_value(True) is True


def test_coerce_coop_value_coerces_numpy_arrays():
    import numpy as np
    assert _coerce_coop_value(np.array([1, 2, 3])) == [1, 2, 3]
    assert _coerce_coop_value(np.array([[0.5, 1.0]])) == [[0.5, 1.0]]


def test_coerce_coop_value_recurses_into_dicts():
    import numpy as np
    out = _coerce_coop_value({"x": np.array([1, 2]), "y": [np.float32(0.5)]})
    assert out == {"x": [1, 2], "y": [0.5]}


def test_coerce_coop_value_handles_tuples():
    assert _coerce_coop_value((1, 2, 3)) == [1, 2, 3]


# ──── _append_coop_summary ──────────────────────────────────────────────


def test_append_coop_summary_writes_jsonl_record(tmp_path: Path):
    path = tmp_path / "episode_summary.jsonl"
    summary = {
        "final_step": 50,
        "cooperation_score": 0.42,
        "carry_imbalance": 1.3,
    }
    _append_coop_summary(
        path, summary,
        start_step=0, end_step=50,
        chamber="ch3", termination_reason="milestone_fired",
    )
    text = path.read_text(encoding="utf-8").strip()
    record = json.loads(text)
    assert record["rollout_start_step"] == 0
    assert record["rollout_end_step"] == 50
    assert record["chamber"] == "ch3"
    assert record["termination_reason"] == "milestone_fired"
    assert record["cooperation_score"] == 0.42
    assert record["final_step"] == 50


def test_append_coop_summary_appends_multiple(tmp_path: Path):
    path = tmp_path / "episode_summary.jsonl"
    _append_coop_summary(path, {"x": 1}, start_step=0, end_step=10,
                          chamber="ch1", termination_reason="horizon")
    _append_coop_summary(path, {"x": 2}, start_step=10, end_step=20,
                          chamber="ch3", termination_reason="death")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["x"] == 1
    assert json.loads(lines[1])["x"] == 2


# ──── _coop_observe wiring ──────────────────────────────────────────────


class _RecordingCoop:
    """Stand-in for CooperationMetric — records observe_* calls."""

    def __init__(self):
        self.step_calls: list[dict] = []
        self.milestone_calls: list[dict] = []

    def observe_step(self, step, positions, actions, messages, task_rewards, infos=None):
        self.step_calls.append({
            "step": step, "positions": positions, "actions": actions,
            "messages": messages, "task_rewards": task_rewards, "infos": infos,
        })

    def observe_milestone(self, step, milestone_id, contributors):
        self.milestone_calls.append({
            "step": step, "milestone_id": milestone_id,
            "contributors": contributors,
        })


def test_coop_observe_translates_action_dicts_to_name_strings():
    coop = _RecordingCoop()
    actions = {
        0: {"action": "dig", "communication_target": None, "thoughts": "x"},
        1: {"action": "forward", "communication_target": 2, "thoughts": "hi"},
    }
    info = {0: {"position": (0., 0., 0.)}, 1: {"position": (1., 0., 0.)}}
    _coop_observe(
        coop, step=5,
        actions_this_step=actions,
        info_by_agent=info,
        rewards_by_agent={0: 0.0, 1: 1.0},
        new_milestones=[],
        seen_milestones=set(),
        n_agents=2,
    )
    assert len(coop.step_calls) == 1
    call = coop.step_calls[0]
    assert call["step"] == 5
    assert call["actions"] == {0: "dig", 1: "forward"}
    assert call["positions"] == {0: (0., 0., 0.), 1: (1., 0., 0.)}


def test_coop_observe_passes_thoughts_when_comm_target_set():
    coop = _RecordingCoop()
    actions = {
        0: {"action": "dig", "communication_target": None, "thoughts": "x"},
        1: {"action": "forward", "communication_target": 2, "thoughts": "team plan"},
    }
    _coop_observe(coop, step=0,
                  actions_this_step=actions, info_by_agent={},
                  rewards_by_agent={}, new_milestones=[],
                  seen_milestones=set(), n_agents=2)
    msgs = coop.step_calls[0]["messages"]
    # Agent 0 has no comm target → no message.
    assert 0 not in msgs or msgs[0] == ""
    # Agent 1's thoughts get forwarded.
    assert msgs[1] == "team plan"


def test_coop_observe_rejects_bool_comm_target():
    """``communication_target=True`` is a bug — guard against it leaking
    into observe_step messages."""
    coop = _RecordingCoop()
    actions = {0: {"action": "dig", "communication_target": True,
                   "thoughts": "x"}}
    _coop_observe(coop, step=0,
                  actions_this_step=actions, info_by_agent={},
                  rewards_by_agent={}, new_milestones=[],
                  seen_milestones=set(), n_agents=1)
    msgs = coop.step_calls[0]["messages"]
    assert msgs == {} or msgs.get(0, "") == ""


def test_coop_observe_forwards_new_milestones():
    coop = _RecordingCoop()
    seen: set = set()
    _coop_observe(
        coop, step=10,
        actions_this_step={0: {"action": "dig"}},
        info_by_agent={},
        rewards_by_agent={},
        new_milestones=[
            {"step": 10, "agent_id": 0, "milestone_id": "m17_switch_pressed"},
        ],
        seen_milestones=seen,
        n_agents=1,
    )
    assert len(coop.milestone_calls) == 1
    call = coop.milestone_calls[0]
    assert call["milestone_id"] == "m17_switch_pressed"
    assert call["contributors"] == ["agent_0"]
    # Seen set is mutated so a re-fire isn't double-counted.
    assert (10, "m17_switch_pressed") in seen


def test_coop_observe_dedups_milestones_via_seen_set():
    coop = _RecordingCoop()
    seen: set = set()
    ev = {"step": 10, "agent_id": 0, "milestone_id": "m17_switch_pressed"}
    _coop_observe(coop, step=10, actions_this_step={},
                  info_by_agent={}, rewards_by_agent={},
                  new_milestones=[ev], seen_milestones=seen, n_agents=1)
    _coop_observe(coop, step=11, actions_this_step={},
                  info_by_agent={}, rewards_by_agent={},
                  new_milestones=[ev], seen_milestones=seen, n_agents=1)
    # Only the first call records — the second sees it in `seen`.
    assert len(coop.milestone_calls) == 1


# ──── end-to-end sampler + JSONL sidecar ───────────────────────────────


class _MultiAgentEnvWithMilestone:
    """Minimal multi-agent env that fires one milestone for agent 0 at
    step ``fire_at`` so cooperation tracking has something to record.
    """

    def __init__(self, fire_at: int = 3):
        self.fire_at = fire_at
        self._t = 0

    def reset(self):
        self._t = 0
        obs = {0: {}, 1: {}}
        info = {
            0: {"chamber": "ch3", "position": (0.0, 0.0, 0.0)},
            1: {"chamber": "ch3", "position": (10.0, 0.0, 10.0)},
        }
        return obs, info

    def step(self, actions):
        self._t += 1
        obs = {0: {}, 1: {}}
        rewards = {0: 0.1, 1: 0.0}
        done = {0: False, 1: False}
        info = {0: {"chamber": "ch3", "position": (0.0, 0.0, 0.0)},
                1: {"chamber": "ch3", "position": (10.0, 0.0, 10.0)}}
        if self._t == self.fire_at:
            info[0]["milestone_events"] = [{
                "step": self._t, "agent_id": 0,
                "milestone_id": "m17_switch_pressed",
            }]
        return obs, rewards, done, info


class _NopPolicy:
    def act(self, observation, info):
        return ({"action": "dig", "communication_target": 1, "thoughts": "go"},
                RolloutTensors(prompt_text="P"))


def test_sampler_attaches_coop_summary_to_joint_rollout(tmp_path: Path):
    sampler = MultiAgentRolloutSampler(
        env=_MultiAgentEnvWithMilestone(),
        policy=_NopPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=1, horizon=5, num_agents=2,
            trained_agents=(0, 1),
        ),
        episode_summary_path=None,        # in-memory only
        collect_cooperation_metrics=True,
    )
    group = sampler.sample_joint_group()
    assert isinstance(group[0], JointRollout)
    assert group[0].coop_summary is not None
    summary = group[0].coop_summary
    # The summary's schema is set by CooperationMetric; spot-check a key.
    assert "cooperation_score" in summary
    # The milestone we fired should be in the log. ``CooperationMetric``
    # stores the id under the key ``milestone`` (not ``milestone_id``).
    mlog = summary.get("milestone_log", [])
    assert any(entry.get("milestone") == "m17_switch_pressed"
               for entry in mlog)


def test_sampler_writes_episode_summary_jsonl(tmp_path: Path):
    path = tmp_path / "episode_summary.jsonl"
    sampler = MultiAgentRolloutSampler(
        env=_MultiAgentEnvWithMilestone(),
        policy=_NopPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=2, horizon=5, num_agents=2,
            trained_agents=(0, 1),
        ),
        episode_summary_path=path,
        collect_cooperation_metrics=True,
    )
    sampler.sample_joint_group()
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    # 2 joints filled the group → 2 lines.
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert "rollout_start_step" in record
    assert "rollout_end_step" in record
    assert record["chamber"] == "ch3"
    assert "cooperation_score" in record


def test_sampler_with_collect_off_doesnt_track(tmp_path: Path):
    sampler = MultiAgentRolloutSampler(
        env=_MultiAgentEnvWithMilestone(),
        policy=_NopPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=1, horizon=3, num_agents=2,
            trained_agents=(0, 1),
        ),
        collect_cooperation_metrics=False,
    )
    group = sampler.sample_joint_group()
    assert group[0].coop_summary is None


def test_sampler_without_path_doesnt_write_sidecar(tmp_path: Path):
    sampler = MultiAgentRolloutSampler(
        env=_MultiAgentEnvWithMilestone(),
        policy=_NopPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=1, horizon=3, num_agents=2,
            trained_agents=(0, 1),
        ),
        episode_summary_path=None,
        collect_cooperation_metrics=True,
    )
    sampler.sample_joint_group()
    # No sidecar file because path is None — coop_summary is still set
    # on the JointRollout (in-memory).
    assert not (tmp_path / "episode_summary.jsonl").exists()


def test_sampler_creates_parent_directory(tmp_path: Path):
    """If the parent of episode_summary_path doesn't exist, the sampler
    creates it (matches the trainer's metrics_path behavior)."""
    nested = tmp_path / "runs" / "grpo" / "G4" / "seed_0"
    path = nested / "episode_summary.jsonl"
    assert not nested.exists()
    MultiAgentRolloutSampler(
        env=_MultiAgentEnvWithMilestone(),
        policy=_NopPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=1, horizon=2, num_agents=2,
            trained_agents=(0, 1),
        ),
        episode_summary_path=path,
    )
    assert nested.exists()
