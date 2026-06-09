"""Plot training curves from parsed returns.csv."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV = Path(__file__).resolve().parent / "out" / "returns.csv"
OUT = Path(__file__).resolve().parent / "out"

COLORS = {
    "hebb_s":         "#1f77b4",
    "hebb_s_nocomm":  "#17becf",
    "ippo_baseline":  "#d62728",
    "seac":           "#2ca02c",
}
ORDER = ["hebb_s", "hebb_s_nocomm", "seac", "ippo_baseline"]


def smooth(y, window=5):
    if len(y) < window:
        return y
    k = np.ones(window) / window
    pad = window // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, k, mode="valid")[: len(y)]


def load():
    by_run = defaultdict(list)
    with CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                v = float(r["test_total_return_mean"])
                t = int(r["t_env"])
            except ValueError:
                continue
            by_run[(r["condition"], r["seed"])].append((t, v))
    for k in by_run:
        by_run[k].sort()
    return by_run


def aggregate(by_run, cond):
    runs = [v for (c, _), v in by_run.items() if c == cond]
    if not runs:
        return None, None, None
    common_t = sorted({t for run in runs for t, _ in run})
    common_t = np.array(common_t)
    mat = []
    for run in runs:
        ts = np.array([t for t, _ in run])
        vs = np.array([v for _, v in run])
        mat.append(np.interp(common_t, ts, vs))
    mat = np.array(mat)
    return common_t, mat.mean(axis=0), mat.std(axis=0, ddof=0)


def plot_learning_curves(by_run, smoothed=True, fname="learning_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in ORDER:
        t, mean, std = aggregate(by_run, cond)
        if t is None:
            continue
        if smoothed:
            mean_s, std_s = smooth(mean, 5), smooth(std, 5)
        else:
            mean_s, std_s = mean, std
        ax.plot(t / 1e6, mean_s, label=cond, color=COLORS[cond], linewidth=2)
        ax.fill_between(t / 1e6, mean_s - std_s, mean_s + std_s,
                        color=COLORS[cond], alpha=0.15)
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("test_total_return_mean")
    ax.set_title("Foraging-Comm-10x10-3p-3f-v3  (mean ± std across 5 seeds, smoothed)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    print(f"wrote {OUT / fname}")


def plot_per_seed_grid(by_run, fname="per_seed.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, cond in zip(axes.flat, ORDER):
        for (c, seed), rows in sorted(by_run.items()):
            if c != cond:
                continue
            ts = np.array([t for t, _ in rows]) / 1e6
            vs = smooth(np.array([v for _, v in rows]), 5)
            ax.plot(ts, vs, label=f"seed {seed}", linewidth=1.2, alpha=0.8)
        ax.set_title(cond)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("env steps (M)")
    for ax in axes[:, 0]:
        ax.set_ylabel("test_total_return_mean")
    fig.suptitle("Per-seed test return curves (smoothed)")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    print(f"wrote {OUT / fname}")


def plot_final_window_bars(by_run, fname="final_window_bars.png"):
    final = defaultdict(list)
    for (cond, _), rows in by_run.items():
        tail = [v for _, v in rows[-5:]]
        if tail:
            final[cond].append(np.mean(tail))
    fig, ax = plt.subplots(figsize=(7, 4))
    conds = [c for c in ORDER if c in final]
    means = [np.mean(final[c]) for c in conds]
    stds = [np.std(final[c], ddof=0) for c in conds]
    x = np.arange(len(conds))
    ax.bar(x, means, yerr=stds, color=[COLORS[c] for c in conds],
           alpha=0.8, capsize=5)
    for i, c in enumerate(conds):
        for v in final[c]:
            ax.scatter(i, v, color="black", s=15, zorder=3, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=15)
    ax.set_ylabel("test_total_return_mean  (last 5 evals)")
    ax.set_title("Final-window performance  (bars = mean±std, dots = per-seed)")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    print(f"wrote {OUT / fname}")


if __name__ == "__main__":
    by_run = load()
    plot_learning_curves(by_run)
    plot_per_seed_grid(by_run)
    plot_final_window_bars(by_run)
