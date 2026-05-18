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
