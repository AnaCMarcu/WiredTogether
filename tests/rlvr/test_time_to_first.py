"""Tests for §A.2: time-to-first-milestone tracking + ``time_to_first.json``
sidecar.

Covers the three free helpers (``_init_first_fire``,
``_record_first_fires``, ``_write_time_to_first``) without needing to
instantiate a full ``GRPOTrainer`` (which requires the torch / PEFT stack).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.grpo_trainer import (
    _init_first_fire,
    _record_first_fires,
    _write_time_to_first,
)
from rlvr.metrics_grpo import GRPOStepMetrics


def _step(step: int, fires_by_id: dict[str, int] | None = None) -> GRPOStepMetrics:
    """Build a minimal GRPOStepMetrics with the given milestone fires."""
    return GRPOStepMetrics(
        step=step, group_size=4,
        group_mean_reward=0.0, group_reward_std=0.0,
        advantage_mean_abs=0.0,
        surrogate_loss=0.0, kl_loss=0.0, total_loss=0.0,
        fraction_clipped=0.0, grad_norm=0.0,
        milestone_fires_by_id=fires_by_id or {},
    )


# ──── _init_first_fire ──────────────────────────────────────────────────


def test_init_first_fire_has_known_milestones():
    """The pre-populated dict covers the 33 milestones from MILESTONE_TRACK
    (28 chamber-track + 5 communication). Each starts as None."""
    out = _init_first_fire()
    # The legacy module is importable in this env (test_reward_table passes).
    assert len(out) == 33
    # Spot-check a few known milestones.
    for mid in ("m1_move_5", "m17_switch_pressed", "m27_boss_defeated",
                "m_comm_ch1", "m_comm_ch5"):
        assert mid in out
        assert out[mid] is None


# ──── _record_first_fires ──────────────────────────────────────────────


def test_records_first_fire_step():
    first_fire = {"m17_switch_pressed": None, "m18_door_opened": None}
    _record_first_fires(first_fire,
                        _step(step=42, fires_by_id={"m17_switch_pressed": 1}))
    assert first_fire["m17_switch_pressed"] == 42
    assert first_fire["m18_door_opened"] is None


def test_subsequent_fires_dont_overwrite():
    """The 'first' in first-fire is load-bearing — only the earliest step
    is recorded, even when the same milestone fires again later."""
    first_fire = {"m17_switch_pressed": None}
    _record_first_fires(first_fire,
                        _step(step=10, fires_by_id={"m17_switch_pressed": 1}))
    _record_first_fires(first_fire,
                        _step(step=20, fires_by_id={"m17_switch_pressed": 3}))
    assert first_fire["m17_switch_pressed"] == 10


def test_zero_count_doesnt_record():
    first_fire = {"m17_switch_pressed": None}
    _record_first_fires(first_fire,
                        _step(step=10, fires_by_id={"m17_switch_pressed": 0}))
    assert first_fire["m17_switch_pressed"] is None


def test_unknown_milestone_added_on_the_fly():
    """A milestone fires that the pre-population didn't know about → added."""
    first_fire = {"m17_switch_pressed": None}
    _record_first_fires(first_fire,
                        _step(step=5, fires_by_id={"m_brand_new": 1}))
    assert first_fire["m_brand_new"] == 5
    assert first_fire["m17_switch_pressed"] is None


def test_multiple_milestones_in_one_step():
    first_fire = {"m17_switch_pressed": None, "m18_door_opened": None,
                  "m1_move_5": None}
    _record_first_fires(first_fire, _step(step=7, fires_by_id={
        "m17_switch_pressed": 1,
        "m18_door_opened": 1,
        "m_unknown": 1,
    }))
    assert first_fire["m17_switch_pressed"] == 7
    assert first_fire["m18_door_opened"] == 7
    assert first_fire["m_unknown"] == 7
    assert first_fire["m1_move_5"] is None


# ──── _write_time_to_first ─────────────────────────────────────────────


def test_writes_sidecar_with_sorted_keys(tmp_path: Path):
    first_fire = {"m17_switch_pressed": 42, "m1_move_5": 8, "m_unknown": 99}
    out = _write_time_to_first(first_fire, tmp_path)
    assert out == tmp_path / "time_to_first.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == first_fire
    # Sorted keys for diff-friendliness — first line of the JSON should
    # start with the smallest key (alphabetical).
    text = out.read_text(encoding="utf-8")
    lines = [l.strip().rstrip(",") for l in text.splitlines() if ":" in l]
    keys_in_order = [l.split('"')[1] for l in lines]
    assert keys_in_order == sorted(keys_in_order)


def test_empty_first_fire_writes_nothing(tmp_path: Path):
    """Dev-env fallback: when MILESTONE_TRACK can't be imported AND no
    milestone fires, ``first_fire`` stays empty → no sidecar written."""
    out = _write_time_to_first({}, tmp_path)
    assert out is None
    assert not (tmp_path / "time_to_first.json").exists()


def test_preserves_none_values(tmp_path: Path):
    """Milestones that never fired keep their None — Kaplan-Meier-style
    censoring is the consumer's job, not the writer's."""
    first_fire = {"m1_move_5": 5, "m27_boss_defeated": None}
    out = _write_time_to_first(first_fire, tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["m1_move_5"] == 5
    assert data["m27_boss_defeated"] is None


def test_idempotent_writes(tmp_path: Path):
    first_fire = {"m1_move_5": 5}
    _write_time_to_first(first_fire, tmp_path)
    first_pass = (tmp_path / "time_to_first.json").read_text(encoding="utf-8")
    _write_time_to_first(first_fire, tmp_path)
    second_pass = (tmp_path / "time_to_first.json").read_text(encoding="utf-8")
    assert first_pass == second_pass


def test_creates_parent_directory(tmp_path: Path):
    """If the parent doesn't exist yet, the writer should create it."""
    nested = tmp_path / "runs" / "G4" / "seed_0"
    assert not nested.exists()
    out = _write_time_to_first({"m1_move_5": 5}, nested)
    assert out is not None
    assert out.exists()


# ──── end-to-end through a simulated train loop ────────────────────────


def test_simulated_training_records_first_fires_correctly(tmp_path: Path):
    """Simulate a sequence of GRPO steps. Verify that the first-fire times
    in the sidecar match the earliest step where each milestone fired.
    """
    first_fire = _init_first_fire()

    # Step 5: m17 fires.
    _record_first_fires(first_fire,
                        _step(step=5, fires_by_id={"m17_switch_pressed": 1}))
    # Step 12: m17 fires again, m18 fires for the first time.
    _record_first_fires(first_fire, _step(step=12, fires_by_id={
        "m17_switch_pressed": 2, "m18_door_opened": 1,
    }))
    # Step 30: m1 fires (legacy chamber).
    _record_first_fires(first_fire, _step(step=30, fires_by_id={
        "m1_move_5": 5,
    }))

    out = _write_time_to_first(first_fire, tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["m17_switch_pressed"] == 5
    assert data["m18_door_opened"] == 12
    assert data["m1_move_5"] == 30
    # Milestones that never fired stay None.
    assert data["m27_boss_defeated"] is None
    assert data["m_comm_ch5"] is None
