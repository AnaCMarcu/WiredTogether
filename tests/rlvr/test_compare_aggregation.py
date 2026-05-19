"""Tests for §B aggregation + significance layer in ``rlvr.compare``.

Covers:
    * ``RunMetrics.per_chamber_series`` / ``final_window_mean_per_chamber``
    * sidecar loaders (``load_time_to_first`` / ``load_coop_summaries`` /
      ``load_hebbian_snapshots``)
    * ``bootstrap_ci`` / ``final_window_ci``
    * ``aggregate_seeds`` (per-chamber + cooperation + time-to-first)
    * ``paired_bootstrap_delta`` / ``_wilcoxon_p`` / ``compare_ablations``

All synthetic — no HPC artifacts needed. Each test writes fake JSONL +
sidecars to ``tmp_path`` and reads back through the public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlvr.compare import (
    AblationSummary,
    PairwiseComparison,
    RunMetrics,
    _aggregate_time_to_first,
    _wilcoxon_p,
    aggregate_seeds,
    bootstrap_ci,
    compare_ablations,
    final_window_ci,
    load_coop_summaries,
    load_hebbian_snapshots,
    load_runs,
    load_time_to_first,
    paired_bootstrap_delta,
)


# ──── synthetic-run helpers ─────────────────────────────────────────────


def _write_grpo_jsonl(
    path: Path,
    n_steps: int,
    fire_rate: float = 0.5,
    reward_base: float = 10.0,
    chamber_fires: dict[str, int] | None = None,
) -> Path:
    """Write a synthetic grpo_metrics.jsonl with the new §A.1 schema."""
    chamber_fires = chamber_fires or {"ch3_switches": 1}
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_steps):
            record = {
                "step": i + 1,
                "group_size": 4,
                "group_mean_reward": reward_base + i * 0.1,
                "group_reward_std": 1.0,
                "advantage_mean_abs": 0.3,
                "surrogate_loss": -0.01,
                "kl_loss": 0.05,
                "total_loss": -0.005,
                "fraction_clipped": 0.1,
                "grad_norm": 0.4,
                "milestone_fires": sum(chamber_fires.values()),
                "milestone_fire_rate": fire_rate,
                "borrowed_fraction": 0.0,
                "per_agent_reward": {"0": reward_base},
                "per_agent_milestone_rate": {"0": fire_rate},
                "milestone_fires_by_chamber": chamber_fires,
                "milestone_fires_by_id": {},
                "hebbian_mean_bond": 0.0,
                "hebbian_sparsity": 0.0,
                "hebbian_modularity": 0.0,
            }
            f.write(json.dumps(record) + "\n")
    return path


def _write_sidecars(
    run_dir: Path,
    time_to_first: dict | None = None,
    coop_records: list[dict] | None = None,
    hebbian_snapshots: list[dict] | None = None,
):
    if time_to_first is not None:
        (run_dir / "time_to_first.json").write_text(
            json.dumps(time_to_first), encoding="utf-8",
        )
    if coop_records is not None:
        path = run_dir / "episode_summary.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in coop_records:
                f.write(json.dumps(r) + "\n")
    if hebbian_snapshots is not None:
        path = run_dir / "hebbian_snapshots.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in hebbian_snapshots:
                f.write(json.dumps(s) + "\n")


# ──── per_chamber_series ────────────────────────────────────────────────


def test_per_chamber_series_reads_nested_field(tmp_path: Path):
    path = _write_grpo_jsonl(
        tmp_path / "run.jsonl", n_steps=5,
        chamber_fires={"ch3_switches": 2, "ch1_solo": 1},
    )
    run = load_runs([path])[0]
    s_ch3 = run.per_chamber_series("ch3_switches")
    assert s_ch3 == [2.0] * 5
    s_ch1 = run.per_chamber_series("ch1_solo")
    assert s_ch1 == [1.0] * 5
    s_ch5 = run.per_chamber_series("ch5_boss")
    assert s_ch5 == [0.0] * 5   # not present → defaults to 0


def test_final_window_mean_per_chamber(tmp_path: Path):
    """Last 3 steps' average per-chamber fires."""
    path = _write_grpo_jsonl(
        tmp_path / "run.jsonl", n_steps=10,
        chamber_fires={"ch3_switches": 3},
    )
    run = load_runs([path])[0]
    assert run.final_window_mean_per_chamber("ch3_switches", window=3) == 3.0


# ──── sidecar loaders ───────────────────────────────────────────────────


def test_load_time_to_first(tmp_path: Path):
    _write_sidecars(tmp_path, time_to_first={"m1_move_5": 5,
                                               "m17_switch_pressed": None})
    grpo_path = tmp_path / "grpo_metrics.jsonl"
    grpo_path.write_text("")   # placeholder so the parent resolves
    data = load_time_to_first(grpo_path)
    assert data == {"m1_move_5": 5, "m17_switch_pressed": None}


def test_load_time_to_first_missing_returns_empty(tmp_path: Path):
    grpo_path = tmp_path / "grpo_metrics.jsonl"
    grpo_path.write_text("")
    assert load_time_to_first(grpo_path) == {}


def test_load_coop_summaries(tmp_path: Path):
    records = [
        {"cooperation_score": 0.5, "carry_imbalance": 1.0},
        {"cooperation_score": 0.7, "carry_imbalance": 0.5},
    ]
    _write_sidecars(tmp_path, coop_records=records)
    grpo_path = tmp_path / "grpo_metrics.jsonl"
    grpo_path.write_text("")
    out = load_coop_summaries(grpo_path)
    assert out == records


def test_load_hebbian_snapshots(tmp_path: Path):
    snapshots = [
        {"step": 10, "mean_bond_strength": 0.3},
        {"step": 20, "mean_bond_strength": 0.5},
    ]
    _write_sidecars(tmp_path, hebbian_snapshots=snapshots)
    grpo_path = tmp_path / "grpo_metrics.jsonl"
    grpo_path.write_text("")
    out = load_hebbian_snapshots(grpo_path)
    assert out == snapshots


# ──── bootstrap_ci ──────────────────────────────────────────────────────


def test_bootstrap_ci_point_estimate_matches_statistic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    point, _, _ = bootstrap_ci(values, n_resamples=1000, seed=0)
    assert point == 3.0   # median


def test_bootstrap_ci_singleton_returns_point():
    point, lo, hi = bootstrap_ci([5.0], n_resamples=1000, seed=0)
    assert point == lo == hi == 5.0


def test_bootstrap_ci_empty_returns_zeros():
    point, lo, hi = bootstrap_ci([], n_resamples=1000, seed=0)
    assert point == lo == hi == 0.0


def test_bootstrap_ci_reproducible():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_ci(values, n_resamples=500, seed=42)
    b = bootstrap_ci(values, n_resamples=500, seed=42)
    assert a == b


def test_bootstrap_ci_brackets_point():
    """For symmetric data, the point should sit inside the CI."""
    values = list(np.linspace(0, 10, 21))
    point, lo, hi = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert lo <= point <= hi


# ──── final_window_ci ──────────────────────────────────────────────────


def test_final_window_ci_per_seed(tmp_path: Path):
    """Builds CI over per-seed final-window means."""
    paths = []
    for i in range(5):
        p = tmp_path / f"seed_{i}" / "grpo_metrics.jsonl"
        p.parent.mkdir(parents=True)
        _write_grpo_jsonl(p, n_steps=20, fire_rate=0.4 + 0.05 * i)
        paths.append(p)
    runs = load_runs(paths)
    out = final_window_ci(runs, "milestone_fire_rate", window=10, bootstrap=1000)
    assert out["n"] == 5
    # Median should fall in the bootstrap CI.
    assert out["ci_low"] <= out["median"] <= out["ci_high"]
    # Raw seed values are preserved for paired comparisons.
    assert len(out["raw"]) == 5


def test_final_window_ci_empty():
    out = final_window_ci([], "milestone_fire_rate", window=10)
    assert out["n"] == 0
    assert out["raw"] == []


# ──── _aggregate_time_to_first (Kaplan-Meier-style) ────────────────────


def test_time_to_first_aggregation_uncensored(tmp_path: Path):
    """All seeds fired the milestone → median is the median of those steps."""
    paths = []
    for seed_idx, fire_step in enumerate([10, 20, 30, 40, 50]):
        p = tmp_path / f"seed_{seed_idx}" / "grpo_metrics.jsonl"
        p.parent.mkdir(parents=True)
        _write_grpo_jsonl(p, n_steps=10)
        _write_sidecars(p.parent, time_to_first={"m1_move_5": fire_step})
        paths.append(p)
    runs = load_runs(paths)
    out = _aggregate_time_to_first(runs)
    assert out["m1_move_5"]["median"] == 30.0
    assert out["m1_move_5"]["n_fired"] == 5
    assert out["m1_move_5"]["n_total"] == 5


def test_time_to_first_aggregation_majority_censored(tmp_path: Path):
    """3 of 5 seeds never fired → median is None."""
    paths = []
    for seed_idx, fire_step in enumerate([10, 20, None, None, None]):
        p = tmp_path / f"seed_{seed_idx}" / "grpo_metrics.jsonl"
        p.parent.mkdir(parents=True)
        _write_grpo_jsonl(p, n_steps=10)
        _write_sidecars(p.parent, time_to_first={"m1_move_5": fire_step})
        paths.append(p)
    runs = load_runs(paths)
    out = _aggregate_time_to_first(runs)
    assert out["m1_move_5"]["median"] is None
    assert out["m1_move_5"]["n_fired"] == 2
    assert out["m1_move_5"]["n_total"] == 5


def test_time_to_first_majority_fired(tmp_path: Path):
    """3 of 5 fired (≥50%) → median computed from the firing seeds."""
    paths = []
    for seed_idx, step in enumerate([10, 20, 30, None, None]):
        p = tmp_path / f"seed_{seed_idx}" / "grpo_metrics.jsonl"
        p.parent.mkdir(parents=True)
        _write_grpo_jsonl(p, n_steps=10)
        _write_sidecars(p.parent, time_to_first={"m1_move_5": step})
        paths.append(p)
    runs = load_runs(paths)
    out = _aggregate_time_to_first(runs)
    # 3 fired with median 20.
    assert out["m1_move_5"]["median"] == 20.0
    assert out["m1_move_5"]["n_fired"] == 3


# ──── aggregate_seeds end-to-end ───────────────────────────────────────


def test_aggregate_seeds_produces_ablation_summary(tmp_path: Path):
    paths = []
    for i in range(5):
        seed_dir = tmp_path / f"seed_{i}"
        seed_dir.mkdir()
        p = seed_dir / "grpo_metrics.jsonl"
        _write_grpo_jsonl(p, n_steps=10, fire_rate=0.4 + 0.05 * i,
                           chamber_fires={"ch3_switches": 2})
        _write_sidecars(
            seed_dir,
            time_to_first={"m17_switch_pressed": 5 + i},
            coop_records=[{"cooperation_score": 0.5 + 0.1 * i,
                           "communication_efficacy": 0.4}],
        )
        paths.append(p)
    runs = load_runs(paths)
    summary = aggregate_seeds(runs, label="G4", window=5, bootstrap=500)

    assert isinstance(summary, AblationSummary)
    assert summary.label == "G4"
    assert summary.n_seeds == 5
    assert "milestone_fire_rate" in summary.final_metrics
    assert summary.per_chamber["ch3_switches"]["median"] == 2.0
    assert "cooperation_score" in summary.cooperation
    # Bootstrap median of [0.5, 0.6, 0.7, 0.8, 0.9] = 0.7.
    assert summary.cooperation["cooperation_score"]["median"] == pytest.approx(
        0.7, abs=1e-6,
    )
    # Time-to-first median of [5, 6, 7, 8, 9] = 7.
    assert summary.time_to_first["m17_switch_pressed"]["median"] == 7.0


def test_aggregate_seeds_serialisable(tmp_path: Path):
    """``AblationSummary.as_dict`` must produce JSON-serializable output."""
    paths = []
    for i in range(3):
        p = tmp_path / f"seed_{i}" / "grpo_metrics.jsonl"
        p.parent.mkdir(parents=True)
        _write_grpo_jsonl(p, n_steps=5)
        paths.append(p)
    summary = aggregate_seeds(load_runs(paths), label="G2",
                               window=3, bootstrap=200)
    text = json.dumps(summary.as_dict())
    assert json.loads(text)["label"] == "G2"


# ──── paired_bootstrap_delta ───────────────────────────────────────────


def test_paired_bootstrap_delta_known_separation():
    """Pairs where a > b consistently → CI excludes 0."""
    a = [0.7, 0.8, 0.75, 0.85, 0.82, 0.78, 0.83]
    b = [0.5, 0.55, 0.52, 0.58, 0.51, 0.54, 0.57]
    delta, lo, hi = paired_bootstrap_delta(a, b, n_resamples=2000, seed=0)
    assert delta > 0
    assert lo > 0   # CI excludes 0 → significant


def test_paired_bootstrap_delta_no_separation():
    a = [0.5, 0.6, 0.4, 0.55, 0.45]
    b = [0.6, 0.5, 0.5, 0.45, 0.55]   # similar distribution
    delta, lo, hi = paired_bootstrap_delta(a, b, n_resamples=2000, seed=0)
    # No reliable separation expected.
    assert lo < 0 < hi or abs(delta) < 0.1


def test_paired_bootstrap_delta_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_delta([1, 2, 3], [1, 2], n_resamples=10, seed=0)


def test_paired_bootstrap_delta_singleton():
    delta, lo, hi = paired_bootstrap_delta([5.0], [3.0], n_resamples=10, seed=0)
    assert delta == lo == hi == 2.0


# ──── _wilcoxon_p ──────────────────────────────────────────────────────


def test_wilcoxon_p_too_few_samples_returns_none():
    """n<5 → underpowered → return None per spec."""
    assert _wilcoxon_p([1, 2, 3], [4, 5, 6]) is None
    assert _wilcoxon_p([1, 2, 3, 4], [5, 6, 7, 8]) is None


def test_wilcoxon_p_clear_difference_significant():
    """5 paired observations with consistent a > b → p < 0.1."""
    a = [10, 12, 14, 11, 15]
    b = [1, 2, 3, 4, 5]
    p = _wilcoxon_p(a, b)
    assert p is not None
    # Exact Wilcoxon for n=5 fully-ordered has p = 0.0625 (one tail) ×2 = 0.0625
    # for two-sided. Look for p ≤ 0.1.
    assert p <= 0.1


def test_wilcoxon_p_all_zero_deltas_returns_none():
    """When every pair is identical, Wilcoxon is undefined — guard."""
    a = [1, 2, 3, 4, 5]
    p = _wilcoxon_p(a, a)
    assert p is None


# ──── compare_ablations ────────────────────────────────────────────────


def test_compare_ablations_detects_significant_improvement(tmp_path: Path):
    """G4 consistently beats G2 on milestone_fire_rate → flagged significant."""
    g2_paths = []
    g4_paths = []
    for i in range(7):
        for label, rate, paths in (("G2", 0.30, g2_paths),
                                    ("G4", 0.70, g4_paths)):
            run_dir = tmp_path / label / f"seed_{i}"
            run_dir.mkdir(parents=True)
            p = run_dir / "grpo_metrics.jsonl"
            _write_grpo_jsonl(p, n_steps=20,
                               fire_rate=rate + 0.02 * i)   # small per-seed noise
            paths.append(p)
    ablations = {
        "G2": load_runs(g2_paths),
        "G4": load_runs(g4_paths),
    }
    comps = compare_ablations(
        ablations, baseline="G2",
        metrics=("milestone_fire_rate",),
        window=10, bootstrap=1000,
    )
    assert len(comps) == 1
    c = comps[0]
    assert c.method == "G4"
    assert c.baseline == "G2"
    assert c.n == 7
    assert c.delta_median > 0
    assert c.significant_bootstrap
    # Wilcoxon p should be small for clear separation at n=7.
    assert c.wilcoxon_p is not None
    assert c.wilcoxon_p < 0.1


def test_compare_ablations_rejects_unknown_baseline(tmp_path: Path):
    ablations = {"G2": []}
    with pytest.raises(ValueError):
        compare_ablations(ablations, baseline="G99")


def test_compare_ablations_skips_baseline_self_pair(tmp_path: Path):
    """Method == baseline → no comparison emitted."""
    p = tmp_path / "seed_0" / "grpo_metrics.jsonl"
    p.parent.mkdir()
    _write_grpo_jsonl(p, n_steps=5)
    ablations = {"G2": load_runs([p])}
    comps = compare_ablations(ablations, baseline="G2",
                               metrics=("milestone_fire_rate",))
    assert comps == []


def test_compare_ablations_truncates_to_min_length(tmp_path: Path):
    """When method has fewer seeds than baseline, only the first N pairs
    are used and n in the result reflects this."""
    g2_paths = []
    g4_paths = []
    for i in range(5):
        for label, paths in (("G2", g2_paths), ("G4", g4_paths)):
            run_dir = tmp_path / label / f"seed_{i}"
            run_dir.mkdir(parents=True)
            p = run_dir / "grpo_metrics.jsonl"
            _write_grpo_jsonl(p, n_steps=5)
            paths.append(p)
    # G4 only has 3 seeds (one crashed).
    g4_paths = g4_paths[:3]
    ablations = {
        "G2": load_runs(g2_paths),
        "G4": load_runs(g4_paths),
    }
    comps = compare_ablations(ablations, baseline="G2",
                               metrics=("milestone_fire_rate",))
    assert comps[0].n == 3
    # Wilcoxon disabled at n<5.
    assert comps[0].wilcoxon_p is None
