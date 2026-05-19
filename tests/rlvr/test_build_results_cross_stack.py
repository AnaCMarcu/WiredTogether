"""End-to-end test for the Phase B+ cross-stack pipeline.

Synthesises one legacy run (M3 = MAPPO + Hebbian) and one GRPO run
(G4 = full Hebbian) under a shared run-tree, drives ``build_results.py``
with both ``--grpo`` and ``--legacy`` flags, and verifies that the
output tables and plots show both rows.

This is the Phase B+ counterpart to ``test_build_results_e2e.py``
(which covers GRPO-only). Together they verify the full thesis grid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_results.py"


# ──── synthesisers ──────────────────────────────────────────────────────


def _write_grpo_run(
    seed_dir: Path,
    n_steps: int = 20,
    fire_rate: float = 0.6,
    chamber_fires: dict[str, int] | None = None,
) -> None:
    """Write one GRPO seed's worth of artifacts (4 sidecars)."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    chamber_fires = chamber_fires or {"ch1_solo": 2, "ch3_switches": 1}
    with (seed_dir / "grpo_metrics.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n_steps):
            f.write(json.dumps({
                "step": i + 1, "group_size": 3,
                "group_mean_reward": 5.0 + i * 0.2,
                "group_reward_std": 1.0, "advantage_mean_abs": 0.4,
                "surrogate_loss": -0.01, "kl_loss": 0.05,
                "total_loss": -0.005, "fraction_clipped": 0.12,
                "grad_norm": 0.5,
                "milestone_fires": sum(chamber_fires.values()),
                "milestone_fire_rate": fire_rate,
                "borrowed_fraction": 0.25,
                "per_agent_reward": {"0": 5.0, "1": 5.0, "2": 5.0},
                "per_agent_milestone_rate": {"0": fire_rate, "1": fire_rate, "2": fire_rate},
                "milestone_fires_by_chamber": chamber_fires,
                "milestone_fires_by_id": {"m17_switch_pressed": chamber_fires.get(
                    "ch3_switches", 0)},
                "hebbian_mean_bond": 0.35,
                "hebbian_sparsity": 0.5,
                "hebbian_modularity": 0.15,
            }) + "\n")
    (seed_dir / "time_to_first.json").write_text(json.dumps({
        "m1_move_5": 3, "m17_switch_pressed": 12,
        "m19_all_in_communal": 18, "m22_all_mobs_killed": None,
    }), encoding="utf-8")
    with (seed_dir / "episode_summary.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "cooperation_score": 0.55,
            "communication_efficacy": 0.6,
            "carry_imbalance": 0.9,
            "ch4_damage_gini": 0.3, "ch5_damage_gini": 0.25,
        }) + "\n")
    with (seed_dir / "hebbian_snapshots.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 10, "enabled": True,
                             "mean_bond_strength": 0.25,
                             "sparsity": 0.6, "modularity_proxy": 0.1,
                             "top_3_pairs": [], "per_agent_out_strength": [0.2]*3,
                             "W": [[0.0]*3]*3, "ltd_heatmap": [[0.0]*3]*3}) + "\n")


def _write_legacy_run(
    run_dir: Path,
    cli_args: list[str],
    seed: int,
    num_agents: int = 3,
    cumulative_reward_curve: list[float] | None = None,
    milestones_total_curve: list[int] | None = None,
) -> None:
    """Write a legacy ``final_metrics.json`` + one episode_summary file.

    Shaped to mirror what ``craftium_metric.save_run_metrics`` produces.
    """
    snapshots = [10, 20, 30, 40, 50]
    cumulative_reward_curve = cumulative_reward_curve or [1.0, 3.0, 6.0, 10.0, 15.0]
    milestones_total_curve = milestones_total_curve or [0, 1, 3, 5, 8]

    run_dir.mkdir(parents=True, exist_ok=True)
    cum_returns = {
        str(a): [v + a * 0.5 for v in cumulative_reward_curve]
        for a in range(num_agents)
    }
    milestone_count = {
        str(a): [v // num_agents for v in milestones_total_curve]
        for a in range(num_agents)
    }
    final_metrics = {
        "config": {
            "num_agents": num_agents, "seed": seed,
            "cli_args": cli_args,
            "experiment_id": "synthetic",
        },
        "cumulative_returns": [cum_returns[str(a)][-1]
                                for a in range(num_agents)],
        "timestep_data": {
            "timesteps": snapshots,
            "cumulative_returns": cum_returns,
            "milestone_count": milestone_count,
            "total_milestones": milestones_total_curve,
        },
        "steps_to_milestone": {
            "ch1_solo": {"m1_move_5": 5, "m2_dig_3_any": 14},
            "ch3_switches": {"m17_switch_pressed": 27,
                              "m18_door_opened": None,
                              "m19_all_in_communal": None},
            "ch4_combat": {"m22_all_mobs_killed": None},
        },
        "graph_snapshots": [
            {"step": 10, "mean_bond_strength": 0.15, "sparsity": 0.8},
            {"step": 30, "mean_bond_strength": 0.35, "sparsity": 0.5},
        ],
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics), encoding="utf-8",
    )
    ep_dir = run_dir / "episodes" / "ep_0001"
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode_summary.json").write_text(
        json.dumps({
            "cooperation_score": 0.38,
            "communication_efficacy": 0.45,
            "carry_imbalance": 1.4,
            "ch4_damage_gini": 0.35, "ch5_damage_gini": 0.3,
        }),
        encoding="utf-8",
    )


# ──── helpers ──────────────────────────────────────────────────────────


def _inherit_env() -> dict:
    import os
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE",
            "POETRY_VIRTUALENVS_IN_PROJECT", "VIRTUAL_ENV")
    return {k: v for k, v in ((k, os.environ.get(k)) for k in keep) if v is not None}


def _run_build_results(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), *args],
        env={**_inherit_env(), "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, check=False,
    )


# ──── end-to-end ────────────────────────────────────────────────────────


def test_build_results_ingests_grpo_and_legacy_together(tmp_path: Path):
    """Synthesise 3 seeds of G4 (GRPO) + 3 seeds of M3 (legacy MAPPO+Heb),
    drive ``build_results.py`` with both ``--grpo`` and ``--legacy``,
    verify the unified output tree.
    """
    grpo_root = tmp_path / "runs" / "grpo"
    legacy_root = tmp_path / "runs" / "legacy"
    out_root = tmp_path / "results"

    # G4 (GRPO + full Hebbian) — higher fire rate.
    for seed_idx in range(3):
        _write_grpo_run(
            grpo_root / "G4" / f"seed_{seed_idx}",
            n_steps=20, fire_rate=0.65 + 0.02 * seed_idx,
        )

    # M3 (MAPPO + Hebbian) — legacy schema. Three runs in legacy/.
    for seed_idx in range(3):
        _write_legacy_run(
            legacy_root / f"E5_seed_{seed_idx}",
            cli_args=["--rl", "--hebbian"],
            seed=seed_idx,
            cumulative_reward_curve=[0.5, 1.5, 3.5, 6.0, 9.5],
            milestones_total_curve=[0, 1, 2, 3, 5],
        )

    result = _run_build_results([
        "--grpo", str(grpo_root),
        "--legacy", str(legacy_root),
        "--out", str(out_root),
        "--ablations", "M3,G4",
        "--baseline", "M3",
        "--bootstrap", "200",
        "--window", "5",
        "--rolling-window", "3",
    ])
    assert result.returncode == 0, (
        f"build_results.py failed: stdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}"
    )

    # Translator output landed at <out>/legacy_translated/.
    assert (out_root / "legacy_translated" / "M3").is_dir()
    assert (out_root / "legacy_translated" / "M3" / "seed_0").is_dir()

    # Per-ablation summaries.
    for tag in ("G4", "M3"):
        sjson = out_root / "per_ablation" / tag / "summary.json"
        assert sjson.exists(), f"missing {sjson}"
        data = json.loads(sjson.read_text(encoding="utf-8"))
        assert data["label"] == tag
        assert data["n_seeds"] == 3

    # Tables contain BOTH methods.
    t1 = (out_root / "tables" / "T1_headline.md").read_text(encoding="utf-8")
    assert "M3" in t1
    assert "G4" in t1

    # Pairwise comparisons: G4 vs M3 should appear in comparisons.json.
    comps = json.loads(
        (out_root / "cross_ablation" / "comparisons.json")
        .read_text(encoding="utf-8")
    )
    assert any(c["method"] == "G4" and c["baseline"] == "M3" for c in comps)

    # Cross-stack grouped plot exists.
    cross_plot = out_root / "cross_ablation" / "plots" / "cross_stack_grouped.png"
    assert cross_plot.exists()
    assert cross_plot.stat().st_size > 0


def test_build_results_grpo_only_unchanged(tmp_path: Path):
    """Regression: when ``--legacy`` isn't passed, behaviour matches the
    Phase A pipeline (no translation step, no legacy_translated/ dir)."""
    grpo_root = tmp_path / "runs" / "grpo"
    out_root = tmp_path / "results"

    for seed_idx in range(3):
        _write_grpo_run(
            grpo_root / "G4" / f"seed_{seed_idx}",
            n_steps=10, fire_rate=0.6,
        )

    result = _run_build_results([
        "--grpo", str(grpo_root),
        "--out", str(out_root),
        "--ablations", "G4",
        "--baseline", "G4",
        "--bootstrap", "100",
        "--window", "5",
    ])
    assert result.returncode == 0, result.stderr
    assert not (out_root / "legacy_translated").exists()
    assert (out_root / "per_ablation" / "G4" / "summary.json").exists()


def test_build_results_legacy_only(tmp_path: Path):
    """Mirror: when ``--grpo`` isn't passed, only legacy translation
    happens. Useful for thesis-side reanalysis of historical runs."""
    legacy_root = tmp_path / "runs" / "legacy"
    out_root = tmp_path / "results"

    for seed_idx in range(3):
        _write_legacy_run(
            legacy_root / f"E5_seed_{seed_idx}",
            cli_args=["--rl", "--hebbian"], seed=seed_idx,
        )

    result = _run_build_results([
        "--legacy", str(legacy_root),
        "--out", str(out_root),
        "--ablations", "M3",
        "--baseline", "M3",
        "--bootstrap", "100",
        "--window", "3",
    ])
    assert result.returncode == 0, result.stderr
    assert (out_root / "per_ablation" / "M3" / "summary.json").exists()
    assert (out_root / "legacy_translated" / "M3").is_dir()


def test_build_results_rejects_no_inputs(tmp_path: Path):
    """Neither --grpo nor --legacy → exit code 2, descriptive error."""
    result = _run_build_results([
        "--out", str(tmp_path / "out"),
        "--ablations", "G4",
        "--baseline", "G4",
        "--bootstrap", "100",
    ])
    assert result.returncode != 0
    assert "must pass at least one of --grpo / --legacy" in result.stderr.lower()


def test_build_results_handles_multiple_legacy_tags(tmp_path: Path):
    """Translator auto-tags each legacy run by its cli_args.

    Tree: legacy/ contains 1 M1 run + 1 L1 run + 1 M3 run.
    After translation: legacy_translated/M1/seed_*, M3/seed_*, L1/seed_*.
    """
    legacy_root = tmp_path / "runs" / "legacy"
    out_root = tmp_path / "results"

    _write_legacy_run(legacy_root / "plain_seed_0", cli_args=[], seed=0)
    _write_legacy_run(legacy_root / "heb_seed_0",
                       cli_args=["--hebbian"], seed=0)
    _write_legacy_run(legacy_root / "mappo_heb_seed_0",
                       cli_args=["--rl", "--hebbian"], seed=0)

    result = _run_build_results([
        "--legacy", str(legacy_root),
        "--out", str(out_root),
        "--ablations", "M1,L1,M3",
        "--baseline", "M1",
        "--bootstrap", "100",
        "--window", "3",
    ])
    assert result.returncode == 0, result.stderr
    for tag in ("M1", "L1", "M3"):
        assert (out_root / "legacy_translated" / tag / "seed_0").is_dir()
        assert (out_root / "per_ablation" / tag / "summary.json").exists()
