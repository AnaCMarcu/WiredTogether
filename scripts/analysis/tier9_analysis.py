"""Tier-9 analysis: 3 variants × 5 seeds on Foraging-Comm-10x10-3p-3f-v3.

Computes paired Wilcoxon sign-rank tests on final-window means and emits
learning-curve + final-window-bar plots. Pre-registered comparisons:

  A) mappo_hebbian_r vs mappo_hebbian           — does diffusion help?
  B) mappo_hebbian_r vs mappo_hebbian_uniform_r — does Hebbian beat uniform?
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_logs import parse_log  # noqa

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("mappo_hebbian",           "baseline (no diffusion)",  "#1f77b4"),
    ("mappo_hebbian_r",         "Hebbian-weighted diffusion", "#d62728"),
    ("mappo_hebbian_uniform_r", "uniform diffusion (control)", "#2ca02c"),
]
SEEDS = list(range(5))


def get_return(kv):
    for k in ("test_total_return_mean", "test_return_mean"):
        if k in kv:
            return kv[k]
    return None


def load_run(variant: str, seed: int):
    path = LOG_DIR / f"{variant}_seed{seed}.log"
    rows = parse_log(path)
    return [(t, get_return(kv)) for t, kv in rows if get_return(kv) is not None]


def smooth(y, window=5):
    if len(y) < window:
        return y
    k = np.ones(window) / window
    pad = window // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, k, mode="valid")[: len(y)]


def main():
    # ── load all runs ──
    runs = {(v, s): load_run(v, s) for v, _, _ in VARIANTS for s in SEEDS}

    # ── per-variant per-seed final-window mean (last 5 evals) ──
    finals = {v: [] for v, _, _ in VARIANTS}
    for (v, s), rows in runs.items():
        tail = [r for _, r in rows[-5:]]
        finals[v].append(float(np.mean(tail)))

    print("=" * 72)
    print(f"{'variant':<28s}  {'mean':>8s}  {'std':>8s}  {'per-seed':>32s}")
    print("-" * 72)
    for v, desc, _ in VARIANTS:
        arr = np.array(finals[v])
        per = ", ".join(f"{x:.3f}" for x in arr)
        print(f"{v:<28s}  {arr.mean():>8.4f}  {arr.std(ddof=0):>8.4f}  [{per}]")
    print()

    # ── paired Wilcoxon (paired by seed) ──
    print("Paired Wilcoxon sign-rank (one-sided):")
    print("-" * 72)
    a = np.array(finals["mappo_hebbian_r"])
    b = np.array(finals["mappo_hebbian"])
    c = np.array(finals["mappo_hebbian_uniform_r"])

    for name, x, y, hyp in [
        ("A: mappo_hebbian_r > mappo_hebbian          ", a, b, "greater"),
        ("B: mappo_hebbian_r > mappo_hebbian_uniform_r", a, c, "greater"),
        ("(reverse A): mappo_hebbian > mappo_hebbian_r           ", b, a, "greater"),
        ("(reverse B): mappo_hebbian_uniform_r > mappo_hebbian_r ", c, a, "greater"),
        ("uniform_r > baseline (extra)                ", c, b, "greater"),
    ]:
        diffs = x - y
        try:
            stat, p = wilcoxon(diffs, alternative=hyp)
            print(f"  {name}  median delta={np.median(diffs):+.4f}  W={stat:.1f}  p={p:.4f}")
        except ValueError as e:
            print(f"  {name}  {e}")
    print()

    # ── write tidy CSV ──
    csv_path = OUT / "tier9_finals.csv"
    with csv_path.open("w") as f:
        f.write("variant,seed,final_window_mean,n_eval_points,peak\n")
        for v, _, _ in VARIANTS:
            for i, s in enumerate(SEEDS):
                rows = runs[(v, s)]
                tail = [r for _, r in rows[-5:]]
                peak = max(r for _, r in rows) if rows else float("nan")
                f.write(f"{v},{s},{np.mean(tail):.4f},{len(rows)},{peak:.4f}\n")
    print(f"wrote {csv_path}")

    # ── learning curves (mean ± std across seeds) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for v, desc, color in VARIANTS:
        all_t = sorted({t for s in SEEDS for t, _ in runs[(v, s)]})
        common_t = np.array(all_t)
        mat = []
        for s in SEEDS:
            ts = np.array([t for t, _ in runs[(v, s)]])
            vs = np.array([r for _, r in runs[(v, s)]])
            mat.append(np.interp(common_t, ts, vs))
        mat = np.array(mat)
        m, sd = smooth(mat.mean(0), 5), smooth(mat.std(0, ddof=0), 5)
        ax.plot(common_t / 1e6, m, label=desc, color=color, linewidth=2.2)
        ax.fill_between(common_t / 1e6, m - sd, m + sd, color=color, alpha=0.15)
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("test_total_return_mean")
    ax.set_title("Foraging-Comm-10x10-3p-3f-v3 — 3M steps, 5 seeds (smoothed)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    out_curve = OUT / "tier9_curves.png"
    fig.savefig(out_curve, dpi=140)
    print(f"wrote {out_curve}")

    # ── final-window bars with per-seed dots ──
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(VARIANTS))
    means = [np.mean(finals[v]) for v, _, _ in VARIANTS]
    stds = [np.std(finals[v], ddof=0) for v, _, _ in VARIANTS]
    colors = [c for _, _, c in VARIANTS]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.75, capsize=6,
           edgecolor="black", linewidth=0.6)
    for i, (v, _, _) in enumerate(VARIANTS):
        for val in finals[v]:
            ax.scatter(i, val, color="black", s=22, zorder=3, alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([d for _, d, _ in VARIANTS], rotation=10, ha="right")
    ax.set_ylabel("test_total_return_mean  (last 5 evals)")
    ax.set_title("Final-window performance  (bars = mean ± std, dots = per-seed)")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    out_bars = OUT / "tier9_bars.png"
    fig.savefig(out_bars, dpi=140)
    print(f"wrote {out_bars}")

    # ── paired diff plot (per seed: variant - baseline) ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base = np.array(finals["mappo_hebbian"])
    for v, desc, color in VARIANTS[1:]:
        diffs = np.array(finals[v]) - base
        for i, d in enumerate(diffs):
            ax.plot([i - 0.15 if "Hebbian" in desc else i + 0.15], [d],
                    "o", color=color, markersize=9, markeredgecolor="black")
        ax.bar(np.arange(5) + (-0.15 if "Hebbian" in desc else 0.15),
               diffs, width=0.25, color=color, alpha=0.4, label=desc)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("Δ final return vs. mappo_hebbian (baseline)")
    ax.set_title("Per-seed paired differences from baseline")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_paired = OUT / "tier9_paired.png"
    fig.savefig(out_paired, dpi=140)
    print(f"wrote {out_paired}")


if __name__ == "__main__":
    main()
