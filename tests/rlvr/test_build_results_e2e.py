"""End-to-end test for the §B + §6 pipeline.

Generates a synthetic ``runs/grpo/<tag>/seed_<N>/`` tree (G2 baseline +
G4 full-Hebbian, 5 seeds each), runs ``scripts/build_results.py``, and
verifies the output tree has the expected structure + parseable content.

This is the local smoke that replaces last session's
``tests/rlvr/_smoke_gen_metrics.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_results.py"


# ──── synthesizers ──────────────────────────────────────────────────────


def _write_step_jsonl(
    path: Path,
    n_steps: int,
    fire_rate: float,
    reward_base: float = 10.0,
    chamber_fires: dict[str, int] | None = None,
    hebbian_mean_bond: float = 0.0,
) -> None:
    """One synthetic grpo_metrics.jsonl line per GRPO step.

    Includes all §A.1 extension fields so the consumer's per-chamber
    aggregation has data to work with.
    """
    chamber_fires = chamber_fires or {"ch3_switches": 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_steps):
            f.write(json.dumps({
                "step": i + 1,
                "group_size": 4,
                "group_mean_reward": reward_base + i * 0.05,
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
                "milestone_fires_by_id": {"m17_switch_pressed": chamber_fires.get(
                    "ch3_switches", 0)},
                "hebbian_mean_bond": hebbian_mean_bond,
                "hebbian_sparsity": 0.5,
                "hebbian_modularity": 0.1,
            }) + "\n")


def _write_sidecars(
    seed_dir: Path,
    time_to_first: dict | None = None,
    coop_summaries: list[dict] | None = None,
    hebbian_snapshots: list[dict] | None = None,
) -> None:
    if time_to_first is not None:
        (seed_dir / "time_to_first.json").write_text(
            json.dumps(time_to_first), encoding="utf-8")
    if coop_summaries is not None:
        with (seed_dir / "episode_summary.jsonl").open("w", encoding="utf-8") as f:
            for r in coop_summaries:
                f.write(json.dumps(r) + "\n")
    if hebbian_snapshots is not None:
        with (seed_dir / "hebbian_snapshots.jsonl").open("w", encoding="utf-8") as f:
            for s in hebbian_snapshots:
                f.write(json.dumps(s) + "\n")


def _build_synthetic_tree(root: Path, n_seeds: int = 5) -> None:
    """G2 baseline (low fire rate) vs G4 full Hebbian (high fire rate)."""
    for seed in range(n_seeds):
        # G2 baseline.
        g2_dir = root / "G2" / f"seed_{seed}"
        _write_step_jsonl(
            g2_dir / "grpo_metrics.jsonl",
            n_steps=30, fire_rate=0.30 + 0.01 * seed,
            chamber_fires={"ch1_solo": 2, "ch3_switches": 0},
        )
        _write_sidecars(
            g2_dir,
            time_to_first={"m1_move_5": 5 + seed, "m17_switch_pressed": None,
                            "m19_all_in_communal": None,
                            "m22_all_mobs_killed": None},
            coop_summaries=[{"cooperation_score": 0.3 + 0.01 * seed,
                              "communication_efficacy": 0.2,
                              "carry_imbalance": 1.5,
                              "ch4_damage_gini": 0.2,
                              "ch5_damage_gini": 0.2}],
        )

        # G4 full Hebbian.
        g4_dir = root / "G4" / f"seed_{seed}"
        _write_step_jsonl(
            g4_dir / "grpo_metrics.jsonl",
            n_steps=30, fire_rate=0.70 + 0.01 * seed,
            chamber_fires={"ch1_solo": 3, "ch3_switches": 2,
                            "ch4_combat": 1},
            hebbian_mean_bond=0.4 + 0.02 * seed,
        )
        _write_sidecars(
            g4_dir,
            time_to_first={"m1_move_5": 3 + seed,
                            "m17_switch_pressed": 12 + seed,
                            "m19_all_in_communal": 20 + seed,
                            "m22_all_mobs_killed": None},
            coop_summaries=[{"cooperation_score": 0.6 + 0.02 * seed,
                              "communication_efficacy": 0.6,
                              "carry_imbalance": 0.8,
                              "ch4_damage_gini": 0.4,
                              "ch5_damage_gini": 0.3}],
            hebbian_snapshots=[
                {"step": 10, "enabled": True,
                 "mean_bond_strength": 0.1 + 0.05 * seed, "sparsity": 0.7,
                 "modularity_proxy": 0.1,
                 "top_3_pairs": [], "per_agent_out_strength": [0.1, 0.1, 0.1],
                 "W": [[0.0, 0.1, 0.2],
                       [0.1, 0.0, 0.3],
                       [0.2, 0.3, 0.0]],
                 "ltd_heatmap": [[0.0]*3]*3},
                {"step": 20, "enabled": True,
                 "mean_bond_strength": 0.3 + 0.05 * seed, "sparsity": 0.4,
                 "modularity_proxy": 0.2,
                 "top_3_pairs": [], "per_agent_out_strength": [0.3, 0.3, 0.3],
                 "W": [[0.0, 0.3, 0.5],
                       [0.3, 0.0, 0.6],
                       [0.5, 0.6, 0.0]],
                 "ltd_heatmap": [[0.0]*3]*3},
            ],
        )


# ──── end-to-end ────────────────────────────────────────────────────────


def test_build_results_end_to_end(tmp_path: Path):
    """Synthesise 2 ablations × 5 seeds, run build_results.py, verify outputs."""
    runs_root = tmp_path / "runs" / "grpo"
    runs_root.mkdir(parents=True)
    out_root = tmp_path / "results"

    _build_synthetic_tree(runs_root, n_seeds=5)

    # Invoke as a subprocess so the CLI argparse runs end-to-end.
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT),
         "--grpo", str(runs_root),
         "--out", str(out_root),
         "--ablations", "G2,G4",
         "--baseline", "G2",
         "--bootstrap", "500",        # small for fast tests
         "--window", "10",
         "--rolling-window", "5"],
        env={**_inherit_env(), "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, check=False,
    )

    # Surface stderr if we failed for any reason.
    assert result.returncode == 0, (
        f"build_results.py failed: stdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}"
    )

    # ─── per-ablation summaries ─────────────────────────────────────────

    for tag in ("G2", "G4"):
        sjson = out_root / "per_ablation" / tag / "summary.json"
        assert sjson.exists(), f"missing {sjson}"
        data = json.loads(sjson.read_text(encoding="utf-8"))
        assert data["label"] == tag
        assert data["n_seeds"] == 5
        assert "milestone_fire_rate" in data["final_metrics"]
        assert "ch3_switches" in data["per_chamber"]
        assert "cooperation_score" in data["cooperation"]
        assert "m17_switch_pressed" in data["time_to_first"]

    # G4 should beat G2 on milestone_fire_rate.
    g2_rate = json.loads(
        (out_root / "per_ablation" / "G2" / "summary.json").read_text(
            encoding="utf-8")
    )["final_metrics"]["milestone_fire_rate"]["median"]
    g4_rate = json.loads(
        (out_root / "per_ablation" / "G4" / "summary.json").read_text(
            encoding="utf-8")
    )["final_metrics"]["milestone_fire_rate"]["median"]
    assert g4_rate > g2_rate, f"expected G4 > G2; got {g4_rate} vs {g2_rate}"

    # ─── pairwise comparisons ───────────────────────────────────────────

    comps_json = out_root / "cross_ablation" / "comparisons.json"
    assert comps_json.exists()
    comps = json.loads(comps_json.read_text(encoding="utf-8"))
    # One method (G4) × 2 metrics (fire_rate + group_reward) = 2 records.
    assert len(comps) == 2
    fire_comp = next(c for c in comps if c["metric"] == "milestone_fire_rate")
    assert fire_comp["method"] == "G4"
    assert fire_comp["baseline"] == "G2"
    assert fire_comp["n"] == 5
    assert fire_comp["delta_median"] > 0
    assert fire_comp["significant_bootstrap"]
    # n=5 → Wilcoxon usable.
    assert fire_comp["wilcoxon_p"] is not None
    assert fire_comp["wilcoxon_p"] < 0.1

    # ─── tables ─────────────────────────────────────────────────────────

    tables_dir = out_root / "tables"
    for name in ("T1_headline", "T2_per_chamber", "T3_hebbian_axis",
                 "T4_coop_comm", "T5_sample_efficiency"):
        md_path = tables_dir / f"{name}.md"
        tex_path = tables_dir / f"{name}.tex"
        assert md_path.exists(), f"missing {md_path}"
        assert tex_path.exists(), f"missing {tex_path}"
        # Sanity-check content non-empty.
        assert md_path.read_text(encoding="utf-8").strip()
        assert tex_path.read_text(encoding="utf-8").strip()

    # T1 should mention both methods.
    t1 = (tables_dir / "T1_headline.md").read_text(encoding="utf-8")
    assert "G2" in t1
    assert "G4" in t1

    # T5 should show G4's fired milestones and dashes for never-fired.
    t5 = (tables_dir / "T5_sample_efficiency.md").read_text(encoding="utf-8")
    assert "m19_all_in_communal" in t5

    # ─── plots ──────────────────────────────────────────────────────────

    plots_dir = out_root / "cross_ablation" / "plots"
    assert plots_dir.exists()
    expected_plots = (
        "headline.png", "learning_curves.png", "per_chamber_bars.png",
        "hebbian_axis_decomposition.png", "bond_strength_evolution.png",
    )
    for name in expected_plots:
        p = plots_dir / name
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 0, f"{p} is empty"


def test_build_results_no_plots_skips_matplotlib(tmp_path: Path):
    """--no-plots produces tables + summary but no PNGs."""
    runs_root = tmp_path / "runs" / "grpo"
    runs_root.mkdir(parents=True)
    out_root = tmp_path / "results"
    _build_synthetic_tree(runs_root, n_seeds=3)

    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT),
         "--grpo", str(runs_root),
         "--out", str(out_root),
         "--ablations", "G2,G4",
         "--baseline", "G2",
         "--bootstrap", "100",
         "--window", "10",
         "--no-plots", "--no-latex"],
        env={**_inherit_env(), "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr

    # Tables (md only) + summaries exist.
    assert (out_root / "tables" / "T1_headline.md").exists()
    assert not (out_root / "tables" / "T1_headline.tex").exists()
    assert (out_root / "per_ablation" / "G2" / "summary.json").exists()
    # No plots directory or no PNGs.
    plots_dir = out_root / "cross_ablation" / "plots"
    if plots_dir.exists():
        assert not any(plots_dir.glob("*.png"))


def test_build_results_missing_ablation_warns_continues(tmp_path: Path):
    """Asking for an ablation that has no seeds doesn't crash — it skips
    that ablation and processes the others. Verifies graceful degradation."""
    runs_root = tmp_path / "runs" / "grpo"
    runs_root.mkdir(parents=True)
    out_root = tmp_path / "results"
    _build_synthetic_tree(runs_root, n_seeds=3)

    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT),
         "--grpo", str(runs_root),
         "--out", str(out_root),
         "--ablations", "G2,G4,G99_missing",
         "--baseline", "G2",
         "--bootstrap", "100",
         "--window", "5"],
        env={**_inherit_env(), "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    # G99 has no summary; G2 and G4 do.
    assert (out_root / "per_ablation" / "G2" / "summary.json").exists()
    assert (out_root / "per_ablation" / "G4" / "summary.json").exists()
    assert not (out_root / "per_ablation" / "G99_missing").exists()


# ──── helpers ──────────────────────────────────────────────────────────


def _inherit_env() -> dict:
    """Minimal env dict for subprocess — pass through PATH and python-related
    vars so poetry's interpreter is usable.
    """
    import os
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE",
            "POETRY_VIRTUALENVS_IN_PROJECT", "VIRTUAL_ENV")
    out = {}
    for k in keep:
        v = os.environ.get(k)
        if v is not None:
            out[k] = v
    return out
