"""Comparison utilities for GRPO ablation runs.

Loads one or more ``grpo_metrics.jsonl`` files and produces aggregate plots
that answer the thesis question: *does GRPO + Hebbian outperform vanilla
GRPO, and does that beat the MAPPO baseline?*

The CLI lives in ``scripts/compare_modes.py``; the testable logic lives
here as pure functions so plotting can be exercised without spinning up
the env.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class RunMetrics:
    """One GRPO run's metric stream loaded from JSONL."""

    label: str
    path: Path
    records: list[dict] = field(default_factory=list)

    @property
    def steps(self) -> list[int]:
        return [r.get("step", i) for i, r in enumerate(self.records)]

    def series(self, key: str) -> list[float]:
        """Per-step series for ``key``. Missing values default to 0.0."""
        return [float(r.get(key, 0.0)) for r in self.records]

    def total(self, key: str) -> float:
        return float(sum(self.series(key)))

    def mean(self, key: str) -> float:
        s = self.series(key)
        return float(sum(s) / len(s)) if s else 0.0

    def final_window_mean(self, key: str, window: int = 50) -> float:
        s = self.series(key)
        if not s:
            return 0.0
        tail = s[-min(window, len(s)):]
        return float(sum(tail) / len(tail))

    # ──── §B.1: per-chamber accessor ────────────────────────────────

    def per_chamber_series(self, chamber: str) -> list[float]:
        """Per-step count of milestone fires in ``chamber``.

        Reads the ``milestone_fires_by_chamber`` field on each step record
        (added in §A.1). Missing entries default to 0 — older JSONL files
        without the field cleanly read as zero.

        T2 (per-chamber fire rate) is computed via
        ``final_window_mean_per_chamber()`` which averages this series
        over the trailing window.
        """
        return [
            float(r.get("milestone_fires_by_chamber", {}).get(chamber, 0))
            for r in self.records
        ]

    def final_window_mean_per_chamber(
        self, chamber: str, window: int = 50,
    ) -> float:
        """Trailing-window mean of per-step chamber fires.

        Interpretation: fires per group-step in this chamber during the
        final ``window`` GRPO steps. Different chambers have different
        difficulty floors — compare across *methods within a chamber*,
        not across chambers.
        """
        s = self.per_chamber_series(chamber)
        if not s:
            return 0.0
        tail = s[-min(window, len(s)):]
        return float(sum(tail) / len(tail))


def load_runs(
    paths: Iterable[Path | str],
    labels: Iterable[str] | None = None,
) -> list[RunMetrics]:
    paths = [Path(p) for p in paths]
    if labels is None:
        labels = [p.stem for p in paths]
    labels = list(labels)
    if len(labels) != len(paths):
        raise ValueError(
            f"got {len(paths)} paths but {len(labels)} labels"
        )
    out: list[RunMetrics] = []
    for path, label in zip(paths, labels):
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        out.append(RunMetrics(label=label, path=path, records=records))
    return out


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Trailing rolling mean. ``window <= 1`` returns the input unchanged."""
    if window <= 1 or len(values) <= 1:
        return list(values)
    out: list[float] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
            out.append(running / window)
        else:
            out.append(running / (i + 1))
    return out


def summarize_runs(runs: list[RunMetrics], window: int = 50) -> dict:
    """Per-run aggregate stats. Output is JSON-safe.

    ``window`` is the trailing window used for the "final" mean — i.e. the
    quality of the policy at the end of training, not the average across
    the whole run.
    """
    summary = {}
    for run in runs:
        summary[run.label] = {
            "n_steps": len(run.records),
            "total_milestone_fires": run.total("milestone_fires"),
            "mean_milestone_fire_rate": run.mean("milestone_fire_rate"),
            "final_milestone_fire_rate": run.final_window_mean(
                "milestone_fire_rate", window=window),
            "final_group_mean_reward": run.final_window_mean(
                "group_mean_reward", window=window),
            "final_kl_loss": run.final_window_mean("kl_loss", window=window),
            "final_fraction_clipped": run.final_window_mean(
                "fraction_clipped", window=window),
            "final_borrowed_fraction": run.final_window_mean(
                "borrowed_fraction", window=window),
        }
    return summary


def generate_plots(runs: list[RunMetrics], window: int = 20):
    """Produce a dict of ``{plot_name: matplotlib.Figure}``.

    The matplotlib import is local so importing ``rlvr.compare`` doesn't
    require matplotlib for the summarization paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, "plt.Figure"] = {}

    metrics_to_plot = [
        ("group_mean_reward", "Group-mean reward"),
        ("milestone_fire_rate", "Milestone-fire rate"),
        ("kl_loss", "KL loss"),
        ("fraction_clipped", "Fraction clipped"),
    ]

    for key, title in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        for run in runs:
            xs = run.steps
            ys = rolling_mean(run.series(key), window=window)
            ax.plot(xs, ys, label=run.label)
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(title)
        ax.set_title(f"{title} (rolling window = {window})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        figures[key] = fig

    # Borrowed-fraction plot is only meaningful when ≥ 1 run had borrowing on.
    if any(run.total("borrowed_fraction") > 0 for run in runs):
        fig, ax = plt.subplots(figsize=(8, 4))
        for run in runs:
            xs = run.steps
            ys = rolling_mean(run.series("borrowed_fraction"), window=window)
            ax.plot(xs, ys, label=run.label)
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("Borrowed fraction")
        ax.set_title(f"Stage-4b borrowed fraction (rolling window = {window})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        figures["borrowed_fraction"] = fig

    # Headline bar chart: final-window mean per run.
    summary = summarize_runs(runs, window=max(window, 50))
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(runs)), 4))
    labels = list(summary.keys())
    final_rates = [summary[lbl]["final_milestone_fire_rate"] for lbl in labels]
    ax.bar(labels, final_rates)
    ax.set_ylabel("Milestone-fire rate (final window)")
    ax.set_title("End-of-training milestone-fire rate by run")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    figures["final_milestone_rate_bar"] = fig

    return figures


def save_summary(summary: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────
#  §B sidecar loaders + statistical layer
# ─────────────────────────────────────────────────────────────────────────


def load_time_to_first(grpo_metrics_path: Path | str) -> dict:
    """Load the ``time_to_first.json`` sidecar next to a ``grpo_metrics.jsonl``.

    Returns ``{}`` when the sidecar is missing — older runs that pre-date
    §A.2 don't have this file. T5 (sample efficiency) then can't be
    computed for those runs, which is handled in ``aggregate_seeds``.
    """
    sidecar = Path(grpo_metrics_path).parent / "time_to_first.json"
    if not sidecar.exists():
        return {}
    with sidecar.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_coop_summaries(grpo_metrics_path: Path | str) -> list[dict]:
    """Load the ``episode_summary.jsonl`` sidecar (one record per joint
    rollout). Returns ``[]`` when the sidecar is missing.
    """
    sidecar = Path(grpo_metrics_path).parent / "episode_summary.jsonl"
    if not sidecar.exists():
        return []
    out = []
    with sidecar.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_hebbian_snapshots(grpo_metrics_path: Path | str) -> list[dict]:
    """Load the ``hebbian_snapshots.jsonl`` sidecar (one per K GRPO steps).
    Returns ``[]`` when the sidecar is missing.
    """
    sidecar = Path(grpo_metrics_path).parent / "hebbian_snapshots.jsonl"
    if not sidecar.exists():
        return []
    out = []
    with sidecar.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ──── §B.2 bootstrap CIs ────────────────────────────────────────────────


def bootstrap_ci(
    values: list[float],
    statistic=None,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI on ``statistic`` of ``values``.

    Returns ``(point_estimate, ci_low, ci_high)``. Defaults to median;
    pass ``statistic=numpy.mean`` for a mean CI.

    With ``len(values) <= 1``, returns ``(point, point, point)`` — there's
    no resampling variance to estimate. The aggregation layer treats this
    as "CI not informative" and reports just the point estimate in tables.
    """
    import numpy as np

    if statistic is None:
        statistic = np.median
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    point = float(statistic(arr))
    if arr.size <= 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    samples = arr[indices]
    stats = np.array([statistic(row) for row in samples])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)


def final_window_ci(
    runs: list[RunMetrics],
    key: str,
    window: int = 50,
    bootstrap: int = 10000,
    seed: int = 0,
) -> dict:
    """Cross-seed bootstrap CI on a final-window metric.

    Each run contributes one number — the final-window mean of ``key``.
    Bootstrap over those N seed-numbers to estimate the CI of the
    central tendency.

    Returns a dict ``{median, p10, p90, ci_low, ci_high, n, raw}`` where
    ``raw`` is the per-seed list (for paired comparisons downstream).
    """
    import numpy as np

    if not runs:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "n": 0, "raw": []}
    raw = [r.final_window_mean(key, window=window) for r in runs]
    arr = np.asarray(raw, dtype=np.float64)
    median, ci_low, ci_high = bootstrap_ci(raw, n_resamples=bootstrap, seed=seed)
    p10, p90 = (float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.9))) \
        if arr.size > 1 else (median, median)
    return {
        "median": median,
        "p10": p10,
        "p90": p90,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(arr.size),
        "raw": [float(v) for v in arr],
    }


# ──── §B.3 per-ablation aggregation ────────────────────────────────────


@dataclass
class AblationSummary:
    """One ablation tag's aggregate stats across N seeds. Feeds T1-T5."""

    label: str
    n_seeds: int
    final_metrics: dict = field(default_factory=dict)
    """``{metric_name: {median, p10, p90, ci_low, ci_high, raw}}``.
    Standard keys: ``milestone_fire_rate``, ``group_mean_reward``,
    ``kl_loss``, ``fraction_clipped``, ``borrowed_fraction``."""

    per_chamber: dict = field(default_factory=dict)
    """``{chamber_name: {median, p10, p90, ci_low, ci_high, raw}}``.
    Per-chamber final-window mean fires-per-trajectory."""

    cooperation: dict = field(default_factory=dict)
    """``{coop_metric_name: {median, p10, p90, ci_low, ci_high, raw}}``
    aggregated across joint rollouts within each seed, then bootstrapped
    across seeds. Standard keys: ``cooperation_score``,
    ``communication_efficacy``, ``carry_imbalance``,
    ``ch4_damage_gini``, ``ch5_damage_gini``."""

    time_to_first: dict = field(default_factory=dict)
    """``{milestone_id: {median_step | None, n_fired, n_total, p10, p90}}``.
    Right-censored: when fewer than 50% of seeds reach a milestone, the
    median is reported as None — the table renderer prints '—'."""

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n_seeds": self.n_seeds,
            "final_metrics": self.final_metrics,
            "per_chamber": self.per_chamber,
            "cooperation": self.cooperation,
            "time_to_first": self.time_to_first,
        }


def aggregate_seeds(
    runs: list[RunMetrics],
    label: str,
    window: int = 50,
    bootstrap: int = 10000,
    seed: int = 0,
) -> AblationSummary:
    """Aggregate N seeds of one ablation into an ``AblationSummary``.

    Reads three sidecars per run (when present):
      * ``time_to_first.json`` → T5
      * ``episode_summary.jsonl`` → T4 cooperation block
      * (``hebbian_snapshots.jsonl`` loaded by callers when needed for
        the bond-evolution plot; not aggregated here)
    """
    from rlvr.metrics_grpo import CHAMBERS

    summary = AblationSummary(label=label, n_seeds=len(runs))
    if not runs:
        return summary

    # Final-window scalar metrics.
    for key in (
        "milestone_fire_rate", "group_mean_reward",
        "kl_loss", "fraction_clipped", "borrowed_fraction",
        "advantage_mean_abs", "surrogate_loss", "total_loss", "grad_norm",
    ):
        summary.final_metrics[key] = final_window_ci(
            runs, key, window=window, bootstrap=bootstrap, seed=seed,
        )

    # Per-chamber.
    for chamber in CHAMBERS:
        raw = [r.final_window_mean_per_chamber(chamber, window=window)
               for r in runs]
        median, ci_low, ci_high = bootstrap_ci(
            raw, n_resamples=bootstrap, seed=seed,
        )
        import numpy as np
        arr = np.asarray(raw, dtype=np.float64)
        p10, p90 = ((float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.9)))
                    if arr.size > 1 else (median, median))
        summary.per_chamber[chamber] = {
            "median": median, "p10": p10, "p90": p90,
            "ci_low": ci_low, "ci_high": ci_high,
            "n": len(raw), "raw": [float(v) for v in raw],
        }

    # Cooperation: collapse all joint-rollout summaries per seed into a
    # mean, then bootstrap across seeds.
    coop_keys = (
        "cooperation_score", "communication_efficacy", "carry_imbalance",
        "ch4_damage_gini", "ch5_damage_gini",
        "proximity_events", "co_action_events", "joint_dig_events",
    )
    for key in coop_keys:
        per_seed_means = []
        for r in runs:
            coops = load_coop_summaries(r.path)
            if not coops:
                continue
            values = [float(c.get(key, 0.0)) for c in coops
                      if c.get(key) is not None]
            if values:
                per_seed_means.append(sum(values) / len(values))
        if not per_seed_means:
            continue
        median, ci_low, ci_high = bootstrap_ci(
            per_seed_means, n_resamples=bootstrap, seed=seed,
        )
        import numpy as np
        arr = np.asarray(per_seed_means, dtype=np.float64)
        p10, p90 = ((float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.9)))
                    if arr.size > 1 else (median, median))
        summary.cooperation[key] = {
            "median": median, "p10": p10, "p90": p90,
            "ci_low": ci_low, "ci_high": ci_high,
            "n": int(arr.size), "raw": [float(v) for v in arr],
        }

    # Time-to-first-milestone — Kaplan-Meier-style censoring.
    summary.time_to_first = _aggregate_time_to_first(runs)

    return summary


def _aggregate_time_to_first(runs: list[RunMetrics]) -> dict:
    """Per-milestone aggregation with right-censoring.

    For each milestone, collect first-fire step across seeds. None values
    (never fired in that seed) count as censored.

    Output: ``{mid: {median, n_fired, n_total, p10, p90}}``.
    When ``n_fired / n_total < 0.5``, ``median`` is ``None`` (the renderer
    prints '—').
    """
    import numpy as np

    per_mid: dict[str, list[int | None]] = {}
    for r in runs:
        ttf = load_time_to_first(r.path)
        for mid, step in ttf.items():
            per_mid.setdefault(mid, []).append(step)

    out: dict[str, dict] = {}
    for mid, steps in per_mid.items():
        n_total = len(steps)
        fired = [s for s in steps if s is not None]
        n_fired = len(fired)
        if n_fired < n_total / 2:
            median = None
            p10 = None
            p90 = None
        else:
            arr = np.asarray(fired, dtype=np.float64)
            median = float(np.median(arr))
            p10 = float(np.quantile(arr, 0.1)) if arr.size > 1 else median
            p90 = float(np.quantile(arr, 0.9)) if arr.size > 1 else median
        out[mid] = {
            "median": median,
            "p10": p10,
            "p90": p90,
            "n_fired": n_fired,
            "n_total": n_total,
        }
    return out


# ──── §B.4 pairwise significance ───────────────────────────────────────


@dataclass
class PairwiseComparison:
    """One method-vs-baseline comparison on one metric.

    Stars are derived from the bootstrap CI (excludes 0 → significant)
    AND optionally cross-checked with Wilcoxon signed-rank at n≥5.
    """

    method: str
    baseline: str
    metric: str
    n: int
    delta_median: float
    """median of the per-seed deltas ``a_i - b_i``."""
    delta_ci_low: float
    delta_ci_high: float
    wilcoxon_p: float | None
    """Two-sided exact Wilcoxon p-value. ``None`` when n<5 (test is
    underpowered to be meaningful) or scipy unavailable."""
    significant_bootstrap: bool
    """True iff the 95% bootstrap CI on the delta excludes 0."""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def paired_bootstrap_delta(
    a: list[float],
    b: list[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap on per-pair deltas ``a[i] - b[i]``.

    Returns ``(median_delta, ci_low, ci_high)``. ``a`` and ``b`` must be
    the same length and seed-aligned.
    """
    import numpy as np

    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal-length lists "
                         f"(got {len(a)} vs {len(b)})")
    deltas = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if deltas.size == 0:
        return 0.0, 0.0, 0.0
    median_delta = float(np.median(deltas))
    if deltas.size == 1:
        return median_delta, median_delta, median_delta

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(n_resamples, deltas.size))
    resamples = deltas[indices]
    resample_medians = np.median(resamples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(resample_medians, [alpha, 1.0 - alpha])
    return median_delta, float(lo), float(hi)


def _wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    """Two-sided exact Wilcoxon signed-rank p-value, or None when
    ``len(a) < 5`` (too underpowered to report), all deltas are zero
    (test is undefined), or scipy is unavailable.
    """
    if len(a) < 5 or len(a) != len(b):
        return None
    # All-zero deltas → test is undefined. scipy returns NaN-on-divide
    # with a RuntimeWarning rather than raising, so check explicitly.
    if all(x == y for x, y in zip(a, b)):
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover
        return None
    try:
        # mode="exact" for small n; falls back automatically when scipy
        # decides the asymptotic approximation is safer.
        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p = float(result.pvalue)
        # Final guard: NaN can still slip through some scipy versions.
        if p != p:   # NaN check
            return None
        return p
    except (ValueError, ZeroDivisionError):
        return None


def compare_ablations(
    ablations: dict[str, list[RunMetrics]],
    baseline: str,
    metrics: tuple[str, ...] = (
        "milestone_fire_rate",
        "group_mean_reward",
    ),
    window: int = 50,
    bootstrap: int = 10000,
    seed: int = 0,
) -> list[PairwiseComparison]:
    """Pairwise method-vs-baseline comparisons on the given metrics.

    Seed alignment assumption: ``ablations[method]`` and
    ``ablations[baseline]`` must be the same length and ordered so
    ``ablations[method][i]`` and ``ablations[baseline][i]`` share an
    env seed. The HPC ``submit_grpo_ablation.sh`` script enforces this
    by submitting all variants with the same ``--array=0-N``.

    When method seeds and baseline seeds have different counts (e.g. one
    array task failed), the comparison is truncated to the shortest list
    and ``n`` reflects the actually-paired count.
    """
    if baseline not in ablations:
        raise ValueError(f"baseline {baseline!r} not in ablations {list(ablations)}")
    if not isinstance(metrics, (tuple, list)):
        raise TypeError("metrics must be a tuple or list")

    out: list[PairwiseComparison] = []
    baseline_runs = ablations[baseline]
    for method, runs in ablations.items():
        if method == baseline:
            continue
        for metric in metrics:
            a_vals = [r.final_window_mean(metric, window=window) for r in runs]
            b_vals = [r.final_window_mean(metric, window=window) for r in baseline_runs]
            n = min(len(a_vals), len(b_vals))
            a_vals = a_vals[:n]
            b_vals = b_vals[:n]
            if n == 0:
                continue
            delta, ci_low, ci_high = paired_bootstrap_delta(
                a_vals, b_vals, n_resamples=bootstrap, seed=seed,
            )
            p_value = _wilcoxon_p(a_vals, b_vals)
            significant = (ci_low > 0.0) or (ci_high < 0.0)
            out.append(PairwiseComparison(
                method=method, baseline=baseline, metric=metric,
                n=n,
                delta_median=delta, delta_ci_low=ci_low, delta_ci_high=ci_high,
                wilcoxon_p=p_value,
                significant_bootstrap=significant,
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────
#  §B.5 table rendering — markdown + LaTeX, no pandas
# ─────────────────────────────────────────────────────────────────────────


# Milestones surfaced by T5 (sample efficiency). One chamber-exit / capstone
# per chamber so the table has a tractable number of rows.
_T5_MILESTONES = (
    "m7_dig_3_stone",
    "m15_chestplate_equipped",
    "m19_all_in_communal",
    "m22_all_mobs_killed",
    "m27_boss_defeated",
    "m28_all_alive_bonus",
)

# Cooperation columns in T4.
_T4_COLUMNS = (
    "cooperation_score",
    "communication_efficacy",
    "carry_imbalance",
    "ch4_damage_gini",
    "ch5_damage_gini",
)

# Primary metrics in T1.
_T1_COLUMNS = (
    "milestone_fire_rate",
    "group_mean_reward",
)


def render_tables(
    ablations: dict[str, AblationSummary],
    comparisons: list[PairwiseComparison] | None = None,
    fmt: str = "markdown",
) -> dict[str, str]:
    """Render T1-T5 thesis tables.

    Returns ``{table_name: rendered_string}`` for T1_headline,
    T2_per_chamber, T3_hebbian_axis, T4_coop_comm, T5_sample_efficiency.

    ``fmt`` is one of ``"markdown"`` or ``"latex"``. LaTeX output uses
    ``booktabs`` (\\toprule / \\midrule / \\bottomrule) and is wrapped in
    a ``table`` float with caption + label so it can be ``\\input``ed
    directly into the thesis.

    ``comparisons`` is the output of ``compare_ablations``. When supplied,
    significance stars are appended to the corresponding cells in T1 and T3.
    """
    if fmt not in ("markdown", "latex"):
        raise ValueError(f"unsupported format: {fmt!r} — use 'markdown' or 'latex'")

    comp_lookup = _build_comp_lookup(comparisons or [])
    return {
        "T1_headline": _render_t1(ablations, comp_lookup, fmt),
        "T2_per_chamber": _render_t2(ablations, fmt),
        "T3_hebbian_axis": _render_t3(ablations, comp_lookup, fmt),
        "T4_coop_comm": _render_t4(ablations, fmt),
        "T5_sample_efficiency": _render_t5(ablations, fmt),
    }


def save_tables(tables: dict[str, str], out_dir: Path | str,
                extension: str = "md") -> dict[str, Path]:
    """Write each rendered table to ``out_dir/<name>.<ext>``.

    Returns the map of table name → path on disk for downstream tooling.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, text in tables.items():
        path = out_dir / f"{name}.{extension}"
        path.write_text(text, encoding="utf-8")
        written[name] = path
    return written


# ──── lookup helpers ───────────────────────────────────────────────────


def _build_comp_lookup(
    comparisons: list[PairwiseComparison],
) -> dict[tuple[str, str], PairwiseComparison]:
    return {(c.method, c.metric): c for c in comparisons}


def _stars(comp: "PairwiseComparison | None") -> str:
    """Significance markers from a pairwise comparison.

    ``*`` p<0.05, ``**`` p<0.01, ``***`` p<0.001 — derived from Wilcoxon
    when available, otherwise from bootstrap-CI significance only (then
    just ``*`` when CI excludes 0).
    """
    if comp is None or not comp.significant_bootstrap:
        return ""
    p = comp.wilcoxon_p
    if p is None:
        return "*"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_median_window(stats: dict | None) -> str:
    """Format a ``{median, p10, p90}`` cell as ``median [p10, p90]``."""
    if not stats:
        return "—"
    median = stats.get("median")
    if median is None:
        return "—"
    p10 = stats.get("p10", median)
    p90 = stats.get("p90", median)
    return f"{median:.2f} [{p10:.2f}, {p90:.2f}]"


def _fmt_mean_std(raw: list[float] | None) -> str:
    """Format ``mean ± std`` from per-seed raw values."""
    if not raw:
        return "—"
    import numpy as np
    arr = np.asarray(raw, dtype=np.float64)
    return f"{float(arr.mean()):.2f} ± {float(arr.std(ddof=0)):.2f}"


def _fmt_delta_ci(comp: "PairwiseComparison | None") -> str:
    """Format a paired-bootstrap delta with its CI."""
    if comp is None:
        return "—"
    return (f"{comp.delta_median:+.2f} "
            f"[{comp.delta_ci_low:+.2f}, {comp.delta_ci_high:+.2f}]"
            f"{_stars(comp)}")


# ──── markdown / latex formatters ──────────────────────────────────────


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown pipe table. Cells already formatted as strings."""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _latex_table(headers: list[str], rows: list[list[str]],
                 caption: str, label: str) -> str:
    """Render a ``booktabs`` LaTeX table. ``caption`` and ``label`` are
    inserted verbatim; the caller is responsible for escaping."""
    col_spec = "l" * 1 + "r" * (len(headers) - 1) if len(headers) > 1 else "l"
    head_row = " & ".join(_latex_escape(h) for h in headers) + r" \\"
    body_rows = [
        " & ".join(_latex_escape(c) for c in row) + r" \\"
        for row in rows
    ]
    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{" + col_spec + "}",
        r"    \toprule",
        "    " + head_row,
        r"    \midrule",
        *["    " + r for r in body_rows],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def _latex_escape(text: str) -> str:
    """Minimal LaTeX escaping for table cells.

    Handles the cases our cells actually produce: ``±``, ``—``, ``*``,
    underscore-suffixed metric names, square brackets in CIs. We leave
    user-provided caption/label strings alone.
    """
    if not isinstance(text, str):
        text = str(text)
    return (text
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("—", r"---")
            .replace("±", r"$\pm$"))


# ──── T1: headline ─────────────────────────────────────────────────────


def _render_t1(ablations: dict, comp_lookup: dict, fmt: str) -> str:
    headers = ["Method", "Final fire rate", "Group reward", "n"]
    rows = []
    for label, summary in ablations.items():
        cells = [label]
        for metric in _T1_COLUMNS:
            stats = summary.final_metrics.get(metric, {})
            base = _fmt_median_window(stats)
            comp = comp_lookup.get((label, metric))
            cells.append(base + _stars(comp))
        cells.append(str(summary.n_seeds))
        rows.append(cells)
    if fmt == "markdown":
        return f"# T1 — Headline comparison\n\n{_md_table(headers, rows)}"
    return _latex_table(headers, rows,
                        caption="Headline GRPO ablation comparison "
                                "(n seeds, bootstrap 95\\% CI).",
                        label="tab:t1_headline")


# ──── T2: per-chamber fire rate ────────────────────────────────────────


def _render_t2(ablations: dict, fmt: str) -> str:
    from rlvr.metrics_grpo import CHAMBERS
    headers = ["Method", *CHAMBERS]
    rows = []
    for label, summary in ablations.items():
        cells = [label]
        for chamber in CHAMBERS:
            stats = summary.per_chamber.get(chamber, {})
            cells.append(_fmt_mean_std(stats.get("raw")))
        rows.append(cells)
    if fmt == "markdown":
        return f"# T2 — Per-chamber fire rate (mean ± std across seeds)\n\n" \
               f"{_md_table(headers, rows)}"
    return _latex_table(headers, rows,
                        caption="Per-chamber milestone-fire rate, mean "
                                r"$\pm$ std across seeds.",
                        label="tab:t2_per_chamber")


# ──── T3: Hebbian-axis decomposition ──────────────────────────────────


def _render_t3(ablations: dict, comp_lookup: dict, fmt: str) -> str:
    headers = ["Ablation",
               "Δ fire rate vs baseline",
               "Δ group reward vs baseline",
               "n"]
    rows = []
    for label, summary in ablations.items():
        row = [label]
        for metric in _T1_COLUMNS:
            comp = comp_lookup.get((label, metric))
            row.append(_fmt_delta_ci(comp) if comp else "ref")
        row.append(str(summary.n_seeds))
        rows.append(row)
    if fmt == "markdown":
        return ("# T3 — Hebbian-axis additive contribution\n\n"
                "Each row shows the paired-bootstrap delta vs. the configured "
                "baseline. ``*`` p<0.05, ``**`` p<0.01, ``***`` p<0.001 "
                "(Wilcoxon signed-rank when n≥5, else bootstrap CI excludes 0).\n\n"
                + _md_table(headers, rows))
    return _latex_table(headers, rows,
                        caption="Hebbian-axis additive contribution. "
                                "Paired-bootstrap delta vs. the configured "
                                "baseline; stars from Wilcoxon when "
                                r"$n \geq 5$.",
                        label="tab:t3_hebbian_axis")


# ──── T4: cooperation & communication ─────────────────────────────────


def _render_t4(ablations: dict, fmt: str) -> str:
    headers = ["Method", *_T4_COLUMNS]
    rows = []
    for label, summary in ablations.items():
        row = [label]
        for key in _T4_COLUMNS:
            stats = summary.cooperation.get(key, {})
            row.append(_fmt_mean_std(stats.get("raw")))
        rows.append(row)
    if fmt == "markdown":
        return ("# T4 — Cooperation & communication\n\n"
                "Each cell aggregates per-seed means of the joint-rollout "
                "summaries (``episode_summary.jsonl``), then bootstrapped "
                "across seeds. Empty (—) when the sidecar wasn't written "
                "for that run.\n\n"
                + _md_table(headers, rows))
    return _latex_table(headers, rows,
                        caption="Cooperation and communication metrics "
                                "(per-seed means; mean $\\pm$ std).",
                        label="tab:t4_coop_comm")


# ──── T5: sample efficiency ──────────────────────────────────────────


def _render_t5(ablations: dict, fmt: str) -> str:
    headers = ["Milestone", *ablations.keys()]
    rows = []
    for mid in _T5_MILESTONES:
        row = [mid]
        for label in ablations.keys():
            tt = ablations[label].time_to_first.get(mid, {})
            median = tt.get("median")
            if median is None:
                row.append("—")
            else:
                n_fired = tt.get("n_fired", 0)
                n_total = tt.get("n_total", 0)
                row.append(f"{int(median)} (n={n_fired}/{n_total})")
        rows.append(row)
    if fmt == "markdown":
        return ("# T5 — Sample efficiency (median GRPO step to first fire)\n\n"
                "Right-censored: ``—`` indicates fewer than 50% of seeds "
                "reached this milestone. Cells show ``median (n_fired / n_total)``.\n\n"
                + _md_table(headers, rows))
    return _latex_table(headers, rows,
                        caption="Sample efficiency: median GRPO step at "
                                "which each milestone first fires. "
                                r"--- = $<50\%$ of seeds reached it.",
                        label="tab:t5_sample_efficiency")


# ─────────────────────────────────────────────────────────────────────────
#  §6 cross-ablation plots — the thesis figures
# ─────────────────────────────────────────────────────────────────────────


def _stack_family(tag: str) -> str:
    """Classify an ablation tag into a stack family.

    Used by ``cross_stack_grouped`` plotting to pick line styles —
    GRPO solid, legacy-RL dashed, LLM-only dotted. Unknown tags
    default to GRPO (the conservative choice — they'll show as solid).
    """
    if tag in ("M1", "L1", "L2"):
        return "llm_only"
    if tag in ("M2", "M3", "M4", "M5"):
        return "legacy_rl"
    # Anything else (G2, G2b, G3a, G3b, G4, custom tags) → GRPO family.
    return "grpo"


def _has_hebbian(tag: str) -> bool:
    """Whether the variant has Hebbian active anywhere (graph, prompt, RL).

    Used by ``cross_stack_grouped`` to color-group methods by Hebbian
    on/off. The thesis story is "does Hebbian help across stacks?" so
    color makes that comparison visually direct.
    """
    return tag in ("L1", "L2", "M3", "M5", "G3a", "G3b", "G4")


def generate_cross_ablation_plots(
    ablations: dict[str, list[RunMetrics]],
    summaries: dict[str, AblationSummary],
    window: int = 20,
):
    """Produce the thesis-side cross-ablation figures as a dict of
    ``{plot_name: matplotlib.Figure}``.

    Plots:
      * ``headline`` — 2-panel: learning curves + final-fire-rate bars
      * ``learning_curves`` — milestone-fire rate over GRPO steps,
        median across seeds + p10-p90 bands, one line per ablation
      * ``per_chamber_bars`` — grouped bars, x = chamber, color = method
      * ``hebbian_axis_decomposition`` — final fire rate per ablation
        with bootstrap error bars
      * ``bond_strength_evolution`` — mean bond + sparsity over GRPO steps
        for ablations whose runs have hebbian_snapshots.jsonl sidecars

    Skipped silently when there's no data (e.g. no Hebbian runs → no
    bond_strength_evolution).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures: dict[str, "plt.Figure"] = {}

    # ─── learning curves: median ± p10-p90 across seeds ────────────────

    figures["learning_curves"] = _plot_learning_curves(
        ablations, window=window,
    )

    # ─── Phase B+ §2: cross-stack grouped variant of learning_curves ──
    # Line styles by stack family (GRPO solid / legacy-RL dashed /
    # LLM-only dotted) + color by Hebbian on/off. With 11 rows the
    # plain learning_curves plot becomes unreadable; this groups
    # visually so the eye picks up the stack-family pattern first,
    # the Hebbian effect second.
    figures["cross_stack_grouped"] = _plot_cross_stack_grouped(
        ablations, window=window,
    )

    # ─── per-chamber bars ──────────────────────────────────────────────

    figures["per_chamber_bars"] = _plot_per_chamber_bars(summaries)

    # ─── Hebbian-axis decomposition ────────────────────────────────────

    figures["hebbian_axis_decomposition"] = _plot_hebbian_axis(summaries)

    # ─── bond strength evolution (Hebbian-only ablations) ──────────────

    bond_fig = _plot_bond_evolution(ablations)
    if bond_fig is not None:
        figures["bond_strength_evolution"] = bond_fig

    # ─── headline 2-panel composite ────────────────────────────────────

    figures["headline"] = _plot_headline(ablations, summaries, window=window)

    return figures


def _plot_learning_curves(
    ablations: dict[str, list[RunMetrics]],
    window: int = 20,
):
    """One line per ablation, median across seeds + p10-p90 band.

    Seeds within an ablation may have different step counts (rare but
    possible). We align on the shortest seed's length to avoid extrapolating.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, runs in ablations.items():
        if not runs:
            continue
        # Per-seed rolling series, aligned to shortest seed length.
        per_seed = [rolling_mean(r.series("milestone_fire_rate"), window=window)
                    for r in runs]
        if not per_seed:
            continue
        min_len = min(len(s) for s in per_seed)
        if min_len == 0:
            continue
        per_seed = [s[:min_len] for s in per_seed]
        arr = np.asarray(per_seed)
        steps = np.arange(1, min_len + 1)
        median = np.median(arr, axis=0)
        p10 = np.quantile(arr, 0.1, axis=0)
        p90 = np.quantile(arr, 0.9, axis=0)
        line, = ax.plot(steps, median, label=label)
        ax.fill_between(steps, p10, p90, alpha=0.2, color=line.get_color())
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Milestone-fire rate")
    ax.set_title(f"Learning curves (rolling window = {window}, "
                 "median ± p10-p90 across seeds)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig


def _plot_cross_stack_grouped(
    ablations: dict[str, list[RunMetrics]],
    window: int = 20,
):
    """Learning curves with two visual axes encoded in line style + colour:

      * line style → stack family (GRPO solid, legacy-RL dashed, LLM-only dotted)
      * line colour → Hebbian on / off (two palettes)

    With 11 methods, the plain ``_plot_learning_curves`` overflows the
    legend. This view groups visually so the reader sees the family
    pattern first and the Hebbian effect second. Each method still gets
    its own line (no within-family averaging).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    family_style = {"grpo": "-", "legacy_rl": "--", "llm_only": ":"}
    # Two palettes — Hebbian uses warm tones, no-Hebbian uses cool. Within
    # each, families are distinguished by line style.
    hebbian_palette = ["#d62728", "#ff7f0e", "#e377c2", "#bcbd22", "#8c564b"]
    plain_palette = ["#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#7f7f7f"]
    hebbian_used = 0
    plain_used = 0

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, runs in ablations.items():
        if not runs:
            continue
        per_seed = [rolling_mean(r.series("milestone_fire_rate"), window=window)
                    for r in runs]
        if not per_seed:
            continue
        min_len = min(len(s) for s in per_seed)
        if min_len == 0:
            continue
        per_seed = [s[:min_len] for s in per_seed]
        arr = np.asarray(per_seed)
        median = np.median(arr, axis=0)
        steps = np.arange(1, min_len + 1)

        style = family_style.get(_stack_family(label), "-")
        if _has_hebbian(label):
            color = hebbian_palette[hebbian_used % len(hebbian_palette)]
            hebbian_used += 1
        else:
            color = plain_palette[plain_used % len(plain_palette)]
            plain_used += 1
        ax.plot(steps, median, label=label, linestyle=style, color=color,
                linewidth=1.6)

    ax.set_xlabel("Training step (GRPO step or env-step snapshot)")
    ax.set_ylabel("Milestone-fire rate")
    ax.set_title(
        "Cross-stack learning curves "
        "(— GRPO, -- legacy-RL, ⋯ LLM-only; warm = Hebbian, cool = no Hebbian; "
        f"rolling window = {window})"
    )
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    return fig


def _plot_per_chamber_bars(summaries: dict[str, AblationSummary]):
    """Grouped bar chart: x = chamber, color = ablation."""
    import matplotlib.pyplot as plt
    import numpy as np
    from rlvr.metrics_grpo import CHAMBERS

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(CHAMBERS)), 4.5))
    n_methods = len(summaries)
    bar_w = 0.8 / max(n_methods, 1)
    xs = np.arange(len(CHAMBERS))

    for i, (label, summary) in enumerate(summaries.items()):
        medians = [summary.per_chamber.get(ch, {}).get("median", 0.0)
                   for ch in CHAMBERS]
        ci_lows = [summary.per_chamber.get(ch, {}).get("ci_low", m)
                   for ch, m in zip(CHAMBERS, medians)]
        ci_highs = [summary.per_chamber.get(ch, {}).get("ci_high", m)
                    for ch, m in zip(CHAMBERS, medians)]
        yerr_lo = [m - lo for m, lo in zip(medians, ci_lows)]
        yerr_hi = [hi - m for hi, m in zip(ci_highs, medians)]
        ax.bar(xs + i * bar_w, medians, bar_w, label=label,
               yerr=[yerr_lo, yerr_hi], capsize=2)

    ax.set_xticks(xs + bar_w * (n_methods - 1) / 2)
    ax.set_xticklabels(CHAMBERS, rotation=20, ha="right")
    ax.set_ylabel("Fires per group-step (final window)")
    ax.set_title("Per-chamber milestone-fire rate by ablation "
                 "(error bars: bootstrap 95% CI)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def _plot_hebbian_axis(summaries: dict[str, AblationSummary]):
    """One panel: ablation along x-axis, final fire rate on y-axis with
    bootstrap CIs. Shows whether the additive Hebbian axes pay off.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(summaries)), 4.5))
    labels = list(summaries.keys())
    medians = [summaries[lbl].final_metrics.get(
        "milestone_fire_rate", {}).get("median", 0.0) for lbl in labels]
    lows = [summaries[lbl].final_metrics.get(
        "milestone_fire_rate", {}).get("ci_low", m)
        for lbl, m in zip(labels, medians)]
    highs = [summaries[lbl].final_metrics.get(
        "milestone_fire_rate", {}).get("ci_high", m)
        for lbl, m in zip(labels, medians)]
    yerr_lo = [m - lo for m, lo in zip(medians, lows)]
    yerr_hi = [hi - m for hi, m in zip(highs, medians)]
    ax.errorbar(labels, medians, yerr=[yerr_lo, yerr_hi],
                fmt="o", capsize=4, markersize=8)
    ax.set_ylabel("Final-window milestone-fire rate")
    ax.set_title("Hebbian-axis decomposition (median + bootstrap 95% CI)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return fig


def _plot_bond_evolution(ablations: dict[str, list[RunMetrics]]):
    """Mean bond strength + sparsity over GRPO steps for Hebbian-enabled
    ablations. Returns None when no ablation has hebbian_snapshots.jsonl
    sidecars (e.g. only G2 baseline runs).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    has_any = False
    for runs in ablations.values():
        if any(load_hebbian_snapshots(r.path) for r in runs):
            has_any = True
            break
    if not has_any:
        return None

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for label, runs in ablations.items():
        per_seed_bond = []
        per_seed_sparsity = []
        steps_axis = None
        for r in runs:
            snaps = load_hebbian_snapshots(r.path)
            if not snaps:
                continue
            steps = [s["step"] for s in snaps if s.get("enabled")]
            bonds = [s.get("mean_bond_strength", 0.0)
                     for s in snaps if s.get("enabled")]
            sparsities = [s.get("sparsity", 0.0)
                          for s in snaps if s.get("enabled")]
            if not steps:
                continue
            if steps_axis is None:
                steps_axis = steps
            per_seed_bond.append(bonds)
            per_seed_sparsity.append(sparsities)
        if not per_seed_bond:
            continue
        # Align to shortest.
        min_len = min(len(s) for s in per_seed_bond)
        if min_len == 0:
            continue
        bond_arr = np.asarray([s[:min_len] for s in per_seed_bond])
        spar_arr = np.asarray([s[:min_len] for s in per_seed_sparsity])
        xs = np.asarray(steps_axis[:min_len])
        line_top, = ax_top.plot(xs, np.median(bond_arr, axis=0), label=label)
        ax_top.fill_between(xs,
                            np.quantile(bond_arr, 0.1, axis=0),
                            np.quantile(bond_arr, 0.9, axis=0),
                            alpha=0.2, color=line_top.get_color())
        ax_bottom.plot(xs, np.median(spar_arr, axis=0),
                       label=label, color=line_top.get_color())

    ax_top.set_ylabel("Mean bond strength")
    ax_top.set_title("Hebbian graph evolution (median + p10-p90 across seeds)")
    ax_top.legend(loc="best")
    ax_top.grid(True, alpha=0.3)
    ax_bottom.set_xlabel("GRPO step")
    ax_bottom.set_ylabel("Sparsity")
    ax_bottom.grid(True, alpha=0.3)
    return fig


def _plot_headline(
    ablations: dict[str, list[RunMetrics]],
    summaries: dict[str, AblationSummary],
    window: int = 20,
):
    """The 2-panel thesis headline: (A) learning curves, (B) final-rate bars.

    Designed to be read in 8 seconds — the thesis caption sentence ties
    the two panels into the headline claim.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=(3, 2))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A — learning curves.
    for label, runs in ablations.items():
        if not runs:
            continue
        per_seed = [rolling_mean(r.series("milestone_fire_rate"), window=window)
                    for r in runs]
        if not per_seed:
            continue
        min_len = min(len(s) for s in per_seed)
        if min_len == 0:
            continue
        arr = np.asarray([s[:min_len] for s in per_seed])
        steps = np.arange(1, min_len + 1)
        median = np.median(arr, axis=0)
        p10 = np.quantile(arr, 0.1, axis=0)
        p90 = np.quantile(arr, 0.9, axis=0)
        line, = ax_a.plot(steps, median, label=label, linewidth=2)
        ax_a.fill_between(steps, p10, p90, alpha=0.2, color=line.get_color())
    ax_a.set_xlabel("GRPO step")
    ax_a.set_ylabel("Milestone-fire rate")
    ax_a.set_title("(A) Learning curves (median ± p10-p90)")
    ax_a.legend(loc="best", fontsize=9)
    ax_a.grid(True, alpha=0.3)

    # Panel B — horizontal bars of final-window fire rate with bootstrap CI.
    items = sorted(
        summaries.items(),
        key=lambda lv: lv[1].final_metrics.get(
            "milestone_fire_rate", {}).get("median", 0.0),
    )
    labels = [k for k, _ in items]
    medians = [s.final_metrics.get("milestone_fire_rate", {}).get("median", 0.0)
               for _, s in items]
    lows = [s.final_metrics.get("milestone_fire_rate", {}).get("ci_low", m)
            for (_, s), m in zip(items, medians)]
    highs = [s.final_metrics.get("milestone_fire_rate", {}).get("ci_high", m)
             for (_, s), m in zip(items, medians)]
    xerr_lo = [m - lo for m, lo in zip(medians, lows)]
    xerr_hi = [hi - m for hi, m in zip(highs, medians)]
    ax_b.barh(labels, medians, xerr=[xerr_lo, xerr_hi], capsize=3)
    ax_b.set_xlabel("Final fire rate")
    ax_b.set_title("(B) End-of-training fire rate (95% CI)")
    ax_b.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig


def save_plots(plots: dict, out_dir: Path | str, dpi: int = 150) -> dict[str, Path]:
    """Write each figure to ``out_dir/<name>.png``. Returns map → paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, fig in plots.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written[name] = path
    return written
