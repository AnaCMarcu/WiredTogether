"""Tests for the §B2 cross-stack grouped plot helpers.

Covers ``_stack_family``, ``_has_hebbian``, and end-to-end production
of the ``cross_stack_grouped`` figure via ``generate_cross_ablation_plots``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr.compare import (
    _has_hebbian,
    _stack_family,
    aggregate_seeds,
    generate_cross_ablation_plots,
    load_runs,
)


# ──── _stack_family truth table ────────────────────────────────────────


@pytest.mark.parametrize("tag, expected", [
    # GRPO family — solid line
    ("G2", "grpo"),
    ("G2b", "grpo"),
    ("G3a", "grpo"),
    ("G3b", "grpo"),
    ("G4", "grpo"),
    # legacy-RL — dashed
    ("M2", "legacy_rl"),
    ("M3", "legacy_rl"),
    ("M4", "legacy_rl"),
    ("M5", "legacy_rl"),
    # LLM-only — dotted
    ("M1", "llm_only"),
    ("L1", "llm_only"),
    ("L2", "llm_only"),
])
def test_stack_family(tag, expected):
    assert _stack_family(tag) == expected


def test_unknown_tag_defaults_to_grpo():
    """An unrecognised tag → solid line (conservative default)."""
    assert _stack_family("X99") == "grpo"


# ──── _has_hebbian truth table ─────────────────────────────────────────


@pytest.mark.parametrize("tag, has", [
    # Hebbian active.
    ("L1", True),
    ("L2", True),
    ("M3", True),
    ("M5", True),
    ("G3a", True),
    ("G3b", True),
    ("G4", True),
    # No Hebbian.
    ("M1", False),
    ("M2", False),
    ("M4", False),
    ("G2", False),
    ("G2b", False),
])
def test_has_hebbian(tag, has):
    assert _has_hebbian(tag) is has


# ──── end-to-end: cross_stack_grouped figure rendered ─────────────────


def _write_synthetic_run(
    path: Path, n_steps: int = 20, fire_rate: float = 0.3,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_steps):
            f.write(json.dumps({
                "step": i + 1, "group_size": 3,
                "group_mean_reward": 5.0 + i * 0.1,
                "milestone_fire_rate": fire_rate + i * 0.005,
                "milestone_fires": i // 4,
                "milestone_fires_by_chamber": {"ch1_solo": 1},
            }) + "\n")


def test_cross_stack_plot_renders_with_all_11_methods(tmp_path: Path):
    """11-method mixed-stack ablation → ``cross_stack_grouped`` figure
    produced + matplotlib doesn't crash on the legend / palette."""
    ablations = {}
    summaries = {}
    rates = {
        "M1": 0.10, "L1": 0.25, "L2": 0.35,
        "M2": 0.40, "M3": 0.50, "M4": 0.42, "M5": 0.52,
        "G2": 0.48, "G2b": 0.51, "G3a": 0.59, "G3b": 0.57, "G4": 0.71,
    }
    for tag, base in rates.items():
        seed_paths = []
        for seed_idx in range(3):
            p = tmp_path / tag / f"seed_{seed_idx}" / "grpo_metrics.jsonl"
            _write_synthetic_run(p, n_steps=15, fire_rate=base)
            seed_paths.append(p)
        ablations[tag] = load_runs(seed_paths)
        summaries[tag] = aggregate_seeds(
            ablations[tag], label=tag, window=5, bootstrap=200,
        )

    figures = generate_cross_ablation_plots(
        ablations, summaries, window=5,
    )
    assert "cross_stack_grouped" in figures

    # Persist to disk and verify it's non-empty.
    out = tmp_path / "plot.png"
    figures["cross_stack_grouped"].savefig(out, dpi=80)
    assert out.stat().st_size > 0


def test_cross_stack_plot_skips_empty_ablations(tmp_path: Path):
    """A tag in the dict with zero seeds → silently skipped, no crash."""
    p = tmp_path / "G4" / "seed_0" / "grpo_metrics.jsonl"
    _write_synthetic_run(p, n_steps=10)
    ablations = {
        "G4": load_runs([p]),
        "M99_empty": [],   # in dict but no seeds
    }
    summaries = {
        "G4": aggregate_seeds(ablations["G4"], label="G4",
                               window=3, bootstrap=100),
        "M99_empty": aggregate_seeds([], label="M99_empty",
                                      window=3, bootstrap=100),
    }
    figures = generate_cross_ablation_plots(
        ablations, summaries, window=3,
    )
    assert "cross_stack_grouped" in figures
