"""Tests for the §A.1 extensions: ``GRPOStepMetrics`` chamber/milestone/Hebbian
fields, ``_milestone_stats`` chamber-and-id breakdowns, ``_hebbian_step_stats``.

These cover the new logging surfaces the thesis tables T2 (per-chamber) and
T5 (steps-to-first-milestone) read from. The existing trainer behavior is
covered by ``test_grpo_trainer.py``; this file targets only the new fields.
"""

from __future__ import annotations

import pytest

from rlvr.grpo_buffer import ScoredTrajectory
from rlvr.metrics_grpo import (
    CHAMBERS,
    GRPOStepMetrics,
    _hebbian_step_stats,
    _milestone_stats,
)
from rlvr.trajectory import GRPOTrajectory


def _scored(
    *,
    agent_id: int,
    owning: int | None = None,
    origin: int | None = None,
    reward: float = 0.0,
    milestone_ids: list[str] | None = None,
) -> ScoredTrajectory:
    """Build a ScoredTrajectory with the given milestone IDs as fired events.

    ``milestone_ids`` is the list of ``milestone_id`` strings to embed in the
    trajectory's ``milestone_events`` list. Each gets a placeholder step.
    """
    milestone_ids = milestone_ids or []
    events = [
        {"step": i, "agent_id": agent_id, "milestone_id": mid}
        for i, mid in enumerate(milestone_ids)
    ]
    return ScoredTrajectory(
        trajectory=GRPOTrajectory(
            prompt_id="p", agent_id=agent_id, chamber="ch3",
            start_step=0, end_step=2,
            milestone_events=events,
        ),
        reward=reward,
        owning_agent_id=owning,
        origin_agent=origin,
    )


# ──── chamber breakdown ─────────────────────────────────────────────────


def test_chambers_are_canonical_six():
    """The CHAMBERS tuple is the schema for T2 — keep it stable."""
    assert CHAMBERS == (
        "ch1_solo", "ch2_anvils", "ch3_switches",
        "ch4_combat", "ch5_boss", "communication",
    )


def test_per_chamber_breakdown_attributes_correctly():
    batch = [
        _scored(agent_id=0, owning=0,
                milestone_ids=["m17_switch_pressed", "m_comm_ch3"]),
        _scored(agent_id=1, owning=1,
                milestone_ids=["m22_all_mobs_killed"]),
    ]
    out = _milestone_stats(batch)
    chambers = out["milestone_fires_by_chamber"]
    assert chambers["ch3_switches"] == 1     # m17
    assert chambers["communication"] == 1    # m_comm_ch3
    assert chambers["ch4_combat"] == 1       # m22
    # Unfired chambers stay at zero.
    assert chambers["ch1_solo"] == 0
    assert chambers["ch2_anvils"] == 0
    assert chambers["ch5_boss"] == 0


def test_empty_batch_returns_zero_chambers():
    out = _milestone_stats([])
    chambers = out["milestone_fires_by_chamber"]
    assert set(chambers.keys()) == set(CHAMBERS)
    assert all(v == 0 for v in chambers.values())


def test_unknown_milestone_id_is_counted_per_id_but_skips_chamber():
    """A milestone not in MILESTONE_TRACK appears in the by_id dict (so it's
    visible for debug) but doesn't bump any chamber count."""
    batch = [_scored(agent_id=0, owning=0,
                     milestone_ids=["m17_switch_pressed", "m_fake_xyz"])]
    out = _milestone_stats(batch)
    assert out["milestone_fires_by_id"]["m_fake_xyz"] == 1
    assert out["milestone_fires_by_chamber"]["ch3_switches"] == 1
    # No spurious chamber bumps from the unknown id.
    assert sum(out["milestone_fires_by_chamber"].values()) == 1


def test_per_id_is_sparse():
    """Only milestones that actually fired appear in the by_id dict."""
    batch = [_scored(agent_id=0, owning=0,
                     milestone_ids=["m1_move_5", "m1_move_5"])]
    out = _milestone_stats(batch)
    # Just the one fired milestone, count 2.
    assert out["milestone_fires_by_id"] == {"m1_move_5": 2}


def test_per_id_aggregates_across_trajectories():
    batch = [
        _scored(agent_id=0, owning=0, milestone_ids=["m17_switch_pressed"]),
        _scored(agent_id=1, owning=1, milestone_ids=["m17_switch_pressed",
                                                      "m18_door_opened"]),
    ]
    out = _milestone_stats(batch)
    by_id = out["milestone_fires_by_id"]
    assert by_id["m17_switch_pressed"] == 2
    assert by_id["m18_door_opened"] == 1


# ──── Hebbian step stats ────────────────────────────────────────────────


class _FakeGraphCfg:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.num_agents = 3


class _FakeGraph:
    """Mimics ``HebbianSocialGraph`` just enough for the step-stat helper."""

    def __init__(self, enabled: bool = True, metrics: dict | None = None,
                 raise_on_get: bool = False):
        self.config = _FakeGraphCfg(enabled)
        self._metrics = metrics or {}
        self._raise = raise_on_get

    def get_graph_metrics(self) -> dict:
        if self._raise:
            raise RuntimeError("simulated graph failure")
        return self._metrics


class _FakeBridge:
    def __init__(self, graph: _FakeGraph):
        self.graph = graph

    def is_enabled(self) -> bool:
        return self.graph.config.enabled


def test_hebbian_stats_zero_when_bridge_none():
    out = _hebbian_step_stats(None)
    assert out == {
        "hebbian_mean_bond": 0.0,
        "hebbian_sparsity": 0.0,
        "hebbian_modularity": 0.0,
    }


def test_hebbian_stats_zero_when_disabled():
    bridge = _FakeBridge(_FakeGraph(enabled=False))
    out = _hebbian_step_stats(bridge)
    assert all(v == 0.0 for v in out.values())


def test_hebbian_stats_forwards_graph_metrics():
    bridge = _FakeBridge(_FakeGraph(metrics={
        "mean_bond_strength": 0.42,
        "sparsity": 0.71,
        "modularity_proxy": 0.18,
    }))
    out = _hebbian_step_stats(bridge)
    assert out == {
        "hebbian_mean_bond": 0.42,
        "hebbian_sparsity": 0.71,
        "hebbian_modularity": 0.18,
    }


def test_hebbian_stats_handles_missing_keys_gracefully():
    """An older graph version that doesn't return all keys should yield 0s
    for the missing ones, not crash."""
    bridge = _FakeBridge(_FakeGraph(metrics={"mean_bond_strength": 0.3}))
    out = _hebbian_step_stats(bridge)
    assert out["hebbian_mean_bond"] == 0.3
    assert out["hebbian_sparsity"] == 0.0
    assert out["hebbian_modularity"] == 0.0


def test_hebbian_stats_swallows_exceptions():
    """A buggy ``get_graph_metrics`` must not kill the training step."""
    bridge = _FakeBridge(_FakeGraph(raise_on_get=True))
    out = _hebbian_step_stats(bridge)
    assert all(v == 0.0 for v in out.values())


# ──── GRPOStepMetrics dataclass surface ────────────────────────────────


def test_step_metrics_defaults_keep_jsonl_schema_stable():
    """A minimal construction (only required fields) should produce a dict
    with every field — older JSONL files still parse forward when reading.
    """
    m = GRPOStepMetrics(
        step=1, group_size=4,
        group_mean_reward=0.0, group_reward_std=0.0,
        advantage_mean_abs=0.0,
        surrogate_loss=0.0, kl_loss=0.0, total_loss=0.0,
        fraction_clipped=0.0, grad_norm=0.0,
    )
    d = m.as_dict()
    for key in (
        "milestone_fires", "milestone_fire_rate", "borrowed_fraction",
        "per_agent_reward", "per_agent_milestone_rate",
        "milestone_fires_by_chamber", "milestone_fires_by_id",
        "hebbian_mean_bond", "hebbian_sparsity", "hebbian_modularity",
    ):
        assert key in d, f"missing {key} in schema"


def test_step_metrics_kwargs_unpack_from_helpers():
    """``_milestone_stats`` and ``_hebbian_step_stats`` must produce kwargs
    that combine cleanly with the constructor — guards against drift."""
    batch = [_scored(agent_id=0, owning=0,
                     milestone_ids=["m1_move_5"], reward=10.0)]
    m = GRPOStepMetrics(
        step=42, group_size=1,
        group_mean_reward=10.0, group_reward_std=0.0,
        advantage_mean_abs=0.5,
        surrogate_loss=-0.1, kl_loss=0.05, total_loss=-0.05,
        fraction_clipped=0.0, grad_norm=0.4,
        **_milestone_stats(batch),
        **_hebbian_step_stats(None),
    )
    assert m.milestone_fires_by_chamber["ch1_solo"] == 1
    assert m.milestone_fires_by_id["m1_move_5"] == 1
    assert m.hebbian_mean_bond == 0.0


# ──── trainer.py re-export ──────────────────────────────────────────────


def test_trainer_reexports_metrics_grpo():
    """Backwards-compat: existing code paths import GRPOStepMetrics from
    ``rlvr.grpo_trainer``. The re-export keeps them working."""
    from rlvr.grpo_trainer import GRPOStepMetrics as TrainerExport

    assert TrainerExport is GRPOStepMetrics


def test_trainer_reexports_milestone_stats():
    from rlvr.grpo_trainer import _milestone_stats as TrainerExport

    assert TrainerExport is _milestone_stats
