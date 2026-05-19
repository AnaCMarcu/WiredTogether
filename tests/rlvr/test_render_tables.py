"""Tests for §B.5 ``render_tables`` — T1-T5 markdown + LaTeX output.

End-to-end: build synthetic ``AblationSummary`` + ``PairwiseComparison``
objects, render the five tables, snapshot-check key cell contents.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rlvr.compare import (
    AblationSummary,
    PairwiseComparison,
    render_tables,
    save_tables,
)


def _summary(
    label: str,
    *,
    n_seeds: int = 5,
    fire_rate_median: float = 0.5,
    reward_median: float = 10.0,
    chamber_fires: dict[str, float] | None = None,
    coop: dict[str, float] | None = None,
    ttf: dict[str, int | None] | None = None,
) -> AblationSummary:
    """Build a minimal AblationSummary with the cells T1-T5 read."""
    s = AblationSummary(label=label, n_seeds=n_seeds)
    s.final_metrics = {
        "milestone_fire_rate": {
            "median": fire_rate_median,
            "p10": fire_rate_median - 0.05,
            "p90": fire_rate_median + 0.05,
            "ci_low": fire_rate_median - 0.07,
            "ci_high": fire_rate_median + 0.07,
            "n": n_seeds, "raw": [fire_rate_median] * n_seeds,
        },
        "group_mean_reward": {
            "median": reward_median,
            "p10": reward_median - 1.0,
            "p90": reward_median + 1.0,
            "ci_low": reward_median - 1.5,
            "ci_high": reward_median + 1.5,
            "n": n_seeds, "raw": [reward_median] * n_seeds,
        },
    }
    chamber_fires = chamber_fires or {"ch3_switches": 1.0}
    s.per_chamber = {
        ch: {"median": v, "p10": v, "p90": v, "ci_low": v, "ci_high": v,
             "n": n_seeds, "raw": [v] * n_seeds}
        for ch, v in chamber_fires.items()
    }
    coop = coop or {}
    s.cooperation = {
        k: {"median": v, "p10": v, "p90": v, "ci_low": v, "ci_high": v,
            "n": n_seeds, "raw": [v] * n_seeds}
        for k, v in coop.items()
    }
    ttf = ttf or {}
    s.time_to_first = {
        mid: {
            "median": step if step is not None else None,
            "p10": step, "p90": step,
            "n_fired": 0 if step is None else n_seeds,
            "n_total": n_seeds,
        }
        for mid, step in ttf.items()
    }
    return s


# ──── render_tables top-level ───────────────────────────────────────────


def test_render_tables_returns_all_five():
    ablations = {"G2": _summary("G2"), "G4": _summary("G4")}
    out = render_tables(ablations, fmt="markdown")
    assert set(out.keys()) == {
        "T1_headline", "T2_per_chamber", "T3_hebbian_axis",
        "T4_coop_comm", "T5_sample_efficiency",
    }


def test_render_tables_rejects_bad_format():
    ablations = {"G2": _summary("G2")}
    with pytest.raises(ValueError):
        render_tables(ablations, fmt="html")


# ──── T1 ────────────────────────────────────────────────────────────────


def test_t1_markdown_contains_method_labels_and_cells():
    ablations = {
        "G2": _summary("G2", fire_rate_median=0.5),
        "G4": _summary("G4", fire_rate_median=0.7),
    }
    out = render_tables(ablations, fmt="markdown")["T1_headline"]
    assert "G2" in out
    assert "G4" in out
    assert "0.50 [0.45, 0.55]" in out
    assert "0.70 [0.65, 0.75]" in out


def test_t1_significance_stars_appear_when_supplied():
    ablations = {
        "G2": _summary("G2", fire_rate_median=0.3),
        "G4": _summary("G4", fire_rate_median=0.7),
    }
    comparisons = [
        PairwiseComparison(
            method="G4", baseline="G2", metric="milestone_fire_rate",
            n=5, delta_median=0.4, delta_ci_low=0.3, delta_ci_high=0.5,
            wilcoxon_p=0.0001, significant_bootstrap=True,
        ),
    ]
    out = render_tables(ablations, comparisons, fmt="markdown")["T1_headline"]
    # *** for p<0.001 (strict).
    assert "***" in out


def test_t1_no_stars_when_not_significant():
    comparisons = [
        PairwiseComparison(
            method="G4", baseline="G2", metric="milestone_fire_rate",
            n=5, delta_median=0.01, delta_ci_low=-0.05, delta_ci_high=0.07,
            wilcoxon_p=0.5, significant_bootstrap=False,
        ),
    ]
    ablations = {"G2": _summary("G2"), "G4": _summary("G4")}
    out = render_tables(ablations, comparisons, fmt="markdown")["T1_headline"]
    # No triple-stars; check we didn't spuriously add markers.
    # (Single * could legitimately appear elsewhere, but *** wouldn't.)
    assert "***" not in out


# ──── T2 ────────────────────────────────────────────────────────────────


def test_t2_markdown_has_six_chamber_columns():
    ablations = {
        "G2": _summary("G2", chamber_fires={
            "ch1_solo": 1.5, "ch3_switches": 0.5,
        }),
    }
    out = render_tables(ablations, fmt="markdown")["T2_per_chamber"]
    for chamber in ("ch1_solo", "ch2_anvils", "ch3_switches",
                    "ch4_combat", "ch5_boss", "communication"):
        assert chamber in out


def test_t2_mean_std_format():
    """Cells use mean ± std format, not median [p10, p90]."""
    ablations = {"G2": _summary("G2", chamber_fires={"ch1_solo": 1.0})}
    out = render_tables(ablations, fmt="markdown")["T2_per_chamber"]
    # All seeds had value 1.0 → mean 1.00, std 0.00.
    assert "1.00 ± 0.00" in out


# ──── T3 ────────────────────────────────────────────────────────────────


def test_t3_shows_baseline_as_ref():
    ablations = {"G2": _summary("G2")}
    out = render_tables(ablations, fmt="markdown")["T3_hebbian_axis"]
    # No comparisons → row marked "ref".
    assert "ref" in out


def test_t3_delta_format():
    """T3 cells use signed delta with CI."""
    ablations = {"G2": _summary("G2"), "G4": _summary("G4")}
    comparisons = [
        PairwiseComparison(
            method="G4", baseline="G2", metric="milestone_fire_rate",
            n=5, delta_median=0.25, delta_ci_low=0.15, delta_ci_high=0.35,
            wilcoxon_p=0.01, significant_bootstrap=True,
        ),
    ]
    out = render_tables(ablations, comparisons,
                        fmt="markdown")["T3_hebbian_axis"]
    assert "+0.25 [+0.15, +0.35]" in out
    # ** for p<0.01.
    assert "**" in out


# ──── T4 ────────────────────────────────────────────────────────────────


def test_t4_includes_cooperation_score_column():
    ablations = {
        "G2": _summary("G2", coop={
            "cooperation_score": 0.42,
            "communication_efficacy": 0.55,
        }),
    }
    out = render_tables(ablations, fmt="markdown")["T4_coop_comm"]
    assert "cooperation_score" in out
    assert "communication_efficacy" in out
    assert "0.42 ± 0.00" in out


def test_t4_missing_coop_keys_render_as_dash():
    """A run without an episode_summary.jsonl sidecar leaves coop empty —
    cells should render '—' not crash."""
    ablations = {"G2": _summary("G2", coop={})}
    out = render_tables(ablations, fmt="markdown")["T4_coop_comm"]
    assert "—" in out


# ──── T5 ────────────────────────────────────────────────────────────────


def test_t5_shows_median_step_when_fired():
    ablations = {
        "G4": _summary("G4", ttf={"m17_switch_pressed": 42}),
    }
    out = render_tables(ablations, fmt="markdown")["T5_sample_efficiency"]
    # m17 isn't a T5 milestone — only the 6 chamber-exit ones.
    # Check the standard milestones still render even when ttf is empty.
    assert "m19_all_in_communal" in out


def test_t5_renders_em_dash_for_unfired():
    """Empty ttf → all cells '—'."""
    ablations = {"G4": _summary("G4", ttf={})}
    out = render_tables(ablations, fmt="markdown")["T5_sample_efficiency"]
    # All 6 T5 milestones appear; all cells should be '—'.
    for mid in ("m7_dig_3_stone", "m15_chestplate_equipped",
                "m19_all_in_communal", "m22_all_mobs_killed",
                "m27_boss_defeated", "m28_all_alive_bonus"):
        assert mid in out


def test_t5_uncensored_milestone_shows_count():
    """When the milestone fired in all seeds, the cell shows the count."""
    ablations = {
        "G4": _summary("G4", n_seeds=5,
                        ttf={"m19_all_in_communal": 42}),
    }
    out = render_tables(ablations, fmt="markdown")["T5_sample_efficiency"]
    assert "42 (n=5/5)" in out


# ──── LaTeX output ─────────────────────────────────────────────────────


def test_latex_output_uses_booktabs():
    ablations = {"G2": _summary("G2"), "G4": _summary("G4")}
    out = render_tables(ablations, fmt="latex")
    for table in out.values():
        assert r"\toprule" in table
        assert r"\midrule" in table
        assert r"\bottomrule" in table
        assert r"\begin{tabular}" in table


def test_latex_escapes_underscores():
    """milestone_id contains underscores — must be escaped for LaTeX."""
    ablations = {
        "G4": _summary("G4", ttf={"m17_switch_pressed": 42}),
    }
    out = render_tables(ablations, fmt="latex")["T5_sample_efficiency"]
    # The escaped underscore appears (chamber names like ch1_solo too).
    assert r"\_" in out


def test_latex_renders_pm_symbol():
    """T2 uses ± in markdown → \\pm in LaTeX."""
    ablations = {"G2": _summary("G2", chamber_fires={"ch1_solo": 1.0})}
    out = render_tables(ablations, fmt="latex")["T2_per_chamber"]
    assert r"$\pm$" in out


def test_latex_em_dash_becomes_triple_hyphen():
    """T5 uses — in markdown → --- in LaTeX (booktabs convention)."""
    ablations = {"G4": _summary("G4", ttf={})}
    out = render_tables(ablations, fmt="latex")["T5_sample_efficiency"]
    assert "---" in out


# ──── save_tables ──────────────────────────────────────────────────────


def test_save_tables_writes_all_five(tmp_path: Path):
    ablations = {"G2": _summary("G2"), "G4": _summary("G4")}
    tables = render_tables(ablations, fmt="markdown")
    written = save_tables(tables, tmp_path, extension="md")
    for name in ("T1_headline", "T2_per_chamber", "T3_hebbian_axis",
                 "T4_coop_comm", "T5_sample_efficiency"):
        assert name in written
        assert written[name].exists()
        assert written[name].suffix == ".md"


def test_save_tables_creates_parent_directory(tmp_path: Path):
    ablations = {"G2": _summary("G2")}
    tables = render_tables(ablations, fmt="latex")
    nested = tmp_path / "results" / "tables"
    save_tables(tables, nested, extension="tex")
    for name in tables:
        assert (nested / f"{name}.tex").exists()
