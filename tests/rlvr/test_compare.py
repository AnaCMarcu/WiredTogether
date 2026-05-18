"""Tests for ``rlvr.compare``: metrics loading, summarisation, plotting.

End-to-end: write a synthetic ``grpo_metrics.jsonl``, load it, summarise,
generate a plot. Verifies the full Stage-6 path without needing matplotlib
to actually render to a screen (Agg backend).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.compare import (
    RunMetrics,
    generate_plots,
    load_runs,
    rolling_mean,
    save_summary,
    summarize_runs,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _synthetic_records(n: int = 20, fire_rate: float = 0.5) -> list[dict]:
    out = []
    for i in range(n):
        out.append({
            "step": i + 1,
            "group_size": 4,
            "group_mean_reward": 10.0 + i * 0.5,
            "group_reward_std": 1.0,
            "advantage_mean_abs": 0.3,
            "surrogate_loss": -0.01 - i * 0.001,
            "kl_loss": 0.05,
            "total_loss": -0.005,
            "fraction_clipped": 0.1,
            "grad_norm": 0.4,
            "milestone_fires": int(i * 0.3),
            "milestone_fire_rate": fire_rate + i * 0.01,
            "borrowed_fraction": 0.2 if i % 2 == 0 else 0.0,
        })
    return out


# ──── load_runs ─────────────────────────────────────────────────────────


def test_load_runs_reads_jsonl(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    _write_jsonl(path, _synthetic_records(5))
    runs = load_runs([path])
    assert len(runs) == 1
    run = runs[0]
    assert isinstance(run, RunMetrics)
    assert run.label == "run"   # filename stem
    assert len(run.records) == 5


def test_load_runs_custom_labels(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    _write_jsonl(path, _synthetic_records(3))
    runs = load_runs([path], labels=["mylabel"])
    assert runs[0].label == "mylabel"


def test_load_runs_label_count_mismatch_raises(tmp_path: Path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    _write_jsonl(p1, [])
    _write_jsonl(p2, [])
    with pytest.raises(ValueError):
        load_runs([p1, p2], labels=["only_one"])


# ──── RunMetrics accessors ─────────────────────────────────────────────


def test_run_series_default_zero(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    _write_jsonl(path, [{"step": 1}])  # missing many keys
    run = load_runs([path])[0]
    assert run.series("milestone_fire_rate") == [0.0]


def test_run_final_window_mean(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    records = [{"step": i, "group_mean_reward": float(i)} for i in range(10)]
    _write_jsonl(path, records)
    run = load_runs([path])[0]
    # Trailing 3 of [0..9] is [7, 8, 9] → mean 8.
    assert run.final_window_mean("group_mean_reward", window=3) == 8.0


# ──── rolling_mean ─────────────────────────────────────────────────────


def test_rolling_mean_passthrough_when_window_small():
    assert rolling_mean([1.0, 2.0, 3.0], window=1) == [1.0, 2.0, 3.0]


def test_rolling_mean_smoothes():
    # Window 2: [(1)/1, (1+3)/2, (3+5)/2, (5+7)/2] = [1, 2, 4, 6]
    out = rolling_mean([1.0, 3.0, 5.0, 7.0], window=2)
    assert out == [1.0, 2.0, 4.0, 6.0]


# ──── summarize_runs ──────────────────────────────────────────────────


def test_summarize_two_runs(tmp_path: Path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    _write_jsonl(p1, _synthetic_records(20, fire_rate=0.5))
    _write_jsonl(p2, _synthetic_records(20, fire_rate=0.8))
    runs = load_runs([p1, p2])
    summary = summarize_runs(runs, window=5)
    assert set(summary.keys()) == {"a", "b"}
    # Run b has higher fire_rate baseline → higher final fire rate.
    assert summary["b"]["final_milestone_fire_rate"] > summary["a"]["final_milestone_fire_rate"]


def test_save_summary_writes_valid_json(tmp_path: Path):
    summary = {"a": {"final_milestone_fire_rate": 0.5}}
    out = tmp_path / "nested/summary.json"
    save_summary(summary, out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["a"]["final_milestone_fire_rate"] == 0.5


# ──── generate_plots (matplotlib end-to-end) ──────────────────────────


def test_generate_plots_produces_expected_figures(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, _synthetic_records(20))
    runs = load_runs([p])
    figures = generate_plots(runs, window=5)
    # The four headline series plots are always produced.
    for key in ("group_mean_reward", "milestone_fire_rate", "kl_loss",
                "fraction_clipped", "final_milestone_rate_bar"):
        assert key in figures
    # Borrowed fraction was nonzero in synthetic records → plot included.
    assert "borrowed_fraction" in figures


def test_generate_plots_skips_borrowed_when_all_zero(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    records = _synthetic_records(10)
    for r in records:
        r["borrowed_fraction"] = 0.0
    _write_jsonl(p, records)
    runs = load_runs([p])
    figures = generate_plots(runs, window=3)
    assert "borrowed_fraction" not in figures


def test_generate_plots_can_save(tmp_path: Path):
    """End-to-end save round-trip."""
    p = tmp_path / "x.jsonl"
    _write_jsonl(p, _synthetic_records(8))
    runs = load_runs([p])
    figures = generate_plots(runs, window=3)
    for name, fig in figures.items():
        out_path = tmp_path / f"{name}.png"
        fig.savefig(out_path, dpi=80)
        assert out_path.exists() and out_path.stat().st_size > 0
