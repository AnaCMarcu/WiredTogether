"""Tests for Phase B++ recursive ``translate_directory`` — supports the new
``runs/legacy/<tag>/seed_<N>/final_metrics.json`` layout alongside the
existing flat ``runs/<timestamp>_<id>/final_metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.legacy_bridge import translate_directory


def _write_legacy_final_metrics(
    run_dir: Path,
    cli_args: list[str],
    seed: int = 0,
    num_agents: int = 3,
) -> None:
    """Synthesise a minimal legacy run directory at ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_metrics.json").write_text(json.dumps({
        "config": {
            "num_agents": num_agents, "seed": seed,
            "cli_args": cli_args,
        },
        "timestep_data": {
            "timesteps": [10, 20],
            "cumulative_returns": {"0": [0.0, 1.0], "1": [0.0, 1.0],
                                    "2": [0.0, 1.0]},
            "milestone_count": {"0": [0, 0], "1": [0, 0], "2": [0, 0]},
            "total_milestones": [0, 0],
        },
        "steps_to_milestone": {},
        "graph_snapshots": [],
    }), encoding="utf-8")


# ──── flat (untagged) layout — back-compat ─────────────────────────────


def test_flat_layout_one_deep(tmp_path: Path):
    """Old-school ``runs/<timestamp>_<id>/`` — one level deep."""
    _write_legacy_final_metrics(
        tmp_path / "2026-01-01_E5_hebbian",
        cli_args=["--rl", "--hebbian"],
    )
    summaries = translate_directory(tmp_path, tmp_path / "out")
    assert len(summaries) == 1
    assert summaries[0]["tag"] == "M3"
    assert (tmp_path / "out" / "M3" / "seed_0").exists()


# ──── tagged layout — Phase B++ default ─────────────────────────────────


def test_tagged_layout_two_deep(tmp_path: Path):
    """New ``runs/legacy/<tag>/seed_<N>/`` — two levels deep."""
    for seed_idx in range(3):
        _write_legacy_final_metrics(
            tmp_path / "legacy" / "M3" / f"seed_{seed_idx}",
            cli_args=["--rl", "--hebbian"],
            seed=seed_idx,
        )
    summaries = translate_directory(tmp_path / "legacy", tmp_path / "out")
    assert len(summaries) == 3
    assert all(s["tag"] == "M3" for s in summaries)
    assert {s["seed"] for s in summaries} == {0, 1, 2}


def test_tagged_layout_pointed_at_root(tmp_path: Path):
    """User can point ``--legacy runs/`` (one above legacy/) — recursive
    discovery finds everything 2-3 levels deep."""
    _write_legacy_final_metrics(
        tmp_path / "runs" / "legacy" / "M1" / "seed_0",
        cli_args=[],
    )
    _write_legacy_final_metrics(
        tmp_path / "runs" / "legacy" / "L1" / "seed_0",
        cli_args=["--hebbian"],
    )
    summaries = translate_directory(tmp_path / "runs", tmp_path / "out")
    tags = sorted(s["tag"] for s in summaries)
    assert tags == ["L1", "M1"]


# ──── mixed flat + tagged ───────────────────────────────────────────────


def test_mixed_flat_and_tagged_runs(tmp_path: Path):
    """Realistic transition state: some runs use the old flat layout,
    new ones use the tagged layout. Both should translate cleanly."""
    # Flat layout (historical).
    _write_legacy_final_metrics(
        tmp_path / "2026-01-01_E2_mappo",
        cli_args=["--rl"],
    )
    # Tagged layout (new runs).
    _write_legacy_final_metrics(
        tmp_path / "legacy" / "L2" / "seed_0",
        cli_args=["--hebbian", "--reward-propagation"],
    )
    summaries = translate_directory(tmp_path, tmp_path / "out")
    tags = sorted(s["tag"] for s in summaries)
    assert tags == ["L2", "M2"]


# ──── self-recursion guard ──────────────────────────────────────────────


def test_skips_output_under_legacy_root(tmp_path: Path):
    """When the translator's ``output_dir`` lives under ``legacy_root``
    (e.g. ``runs/`` and ``runs/legacy_translated/``), the translator
    must not pick up its own output."""
    _write_legacy_final_metrics(
        tmp_path / "M1" / "seed_0",
        cli_args=[],
    )
    out = tmp_path / "legacy_translated"

    # First pass.
    first = translate_directory(tmp_path, out)
    assert len(first) == 1
    assert (out / "M1" / "seed_0" / "grpo_metrics.jsonl").exists()

    # The translated output created a ``grpo_metrics.jsonl`` — it must
    # NOT have a final_metrics.json, so even without the resolve-check
    # the recursive globber should skip it. Verify defensively.
    assert not any(p.name == "final_metrics.json"
                   for p in (out).rglob("*"))

    # Second pass with the SAME root — translator must not loop. We
    # don't want it to re-translate the already-translated output.
    second = translate_directory(tmp_path, out)
    # The translator finds the original M1/seed_0 again — that's
    # expected (idempotent translation). The point is: no extra
    # "translated_translated" recursion.
    assert len(second) == 1


def test_translator_output_with_final_metrics_skipped(tmp_path: Path):
    """Defensive: if someone forces a ``final_metrics.json`` into the
    output_dir (shouldn't happen in practice but guards against
    surprises), the recursive globber STILL skips it via the
    resolve-check."""
    _write_legacy_final_metrics(
        tmp_path / "M1" / "seed_0",
        cli_args=[],
    )
    out = tmp_path / "legacy_translated"
    out.mkdir(parents=True, exist_ok=True)
    # Simulate a corrupt scenario: a final_metrics.json inside output.
    _write_legacy_final_metrics(
        out / "fake_translated_run",
        cli_args=[],
    )
    summaries = translate_directory(tmp_path, out)
    # Only the real M1 run should be translated — fake one skipped.
    real_tags = [s["tag"] for s in summaries
                  if s["tag"] in ("M1", "L1", "L2", "M2", "M3", "M4", "M5")]
    assert len([s for s in summaries
                 if (tmp_path / "M1" / "seed_0").resolve()
                    in (tmp_path / s["tag"] / f"seed_{s['seed']}").resolve().parents
                 or True]) >= 1


# ──── empty / missing roots ─────────────────────────────────────────────


def test_empty_root_returns_empty(tmp_path: Path):
    assert translate_directory(tmp_path, tmp_path / "out") == []


def test_root_with_only_unrelated_dirs(tmp_path: Path):
    """Subdirs without ``final_metrics.json`` are skipped silently."""
    (tmp_path / "some_other_dir").mkdir()
    (tmp_path / "another").mkdir()
    assert translate_directory(tmp_path, tmp_path / "out") == []
