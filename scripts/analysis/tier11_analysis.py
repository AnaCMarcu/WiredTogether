"""Tier-11 analysis: 3 variants × 5 seeds on Foraging-Comm-10x10-3p-3f-coop-v3
(force_coop=True). Mirrors tier9_analysis.py with the new log path and
tier-9 reference values displayed for direct comparison.

Pre-registered comparisons (paired Wilcoxon sign-rank, one-sided):
  A) mappo_hebbian_r > mappo_hebbian           — does diffusion help under coop?
  B) mappo_hebbian_r > mappo_hebbian_uniform_r — does Hebbian beat uniform under coop?
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_logs import parse_log  # noqa

LOG_DIR = Path(__file__).resolve().parents[2] / "runs_from_daic" / "hebbian-marl" / "logs"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("mappo_hebbian",           "baseline (no diffusion)",       "#1f77b4"),
    ("mappo_hebbian_r",         "Hebbian-weighted diffusion",    "#d62728"),
    ("mappo_hebbian_uniform_r", "uniform diffusion (control)",   "#2ca02c"),
]
SEEDS = list(range(5))

# Tier-9 reference (Foraging-Comm-10x10-3p-3f-v3, mixed levels, 5 seeds, 3M).
TIER9_REF = {
    "mappo_hebbian":           (0.8546, 0.0713),
    "mappo_hebbian_r":         (0.7226, 0.0293),
    "mappo_hebbian_uniform_r": (0.9024, 0.0395),
}


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
    runs = {(v, s): load_run(v, s) for v, _, _ in VARIANTS for s in SEEDS}

    # ── per-variant per-seed final-window mean (last 5 evals) ──
    finals = {v: [] for v, _, _ in VARIANTS}
    for (v, s), rows in runs.items():
        if not rows:
            print(f"WARNING: empty rows for {v}_seed{s}")
            continue
        tail = [r for _, r in rows[-5:]]
        finals[v].append(float(np.mean(tail)))

    print("=" * 80)
    print(f"{'variant':<28s}  {'mean':>8s}  {'std':>8s}  {'per-seed':>36s}")
    print("-" * 80)
    for v, desc, _ in VARIANTS:
        arr = np.array(finals[v])
        per = ", ".join(f"{x:.3f}" for x in arr)
        ref_m, ref_s = TIER9_REF[v]
        print(f"{v:<28s}  {arr.mean():>8.4f}  {arr.std(ddof=0):>8.4f}  [{per}]")
        print(f"  tier-9 ref:                {ref_m:>8.4f}  {ref_s:>8.4f}   "
              f"delta = {arr.mean()-ref_m:+.4f}")
    print()

    # ── paired Wilcoxon ──
    print("Paired Wilcoxon sign-rank (one-sided), tier-11:")
    print("-" * 80)
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
    csv_path = OUT / "tier11_finals.csv"
    with csv_path.open("w") as f:
        f.write("variant,seed,final_window_mean,n_eval_points,peak\n")
        for v, _, _ in VARIANTS:
            for s in SEEDS:
                rows = runs[(v, s)]
                tail = [r for _, r in rows[-5:]]
                peak = max(r for _, r in rows) if rows else float("nan")
                f.write(f"{v},{s},{np.mean(tail):.4f},{len(rows)},{peak:.4f}\n")
    print(f"wrote {csv_path}")

    # ── learning curves (mean ± std) ──
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
    ax.set_title("Foraging-Comm-10x10-3p-3f-COOP-v3 — 3M steps, 5 seeds (smoothed)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    out_curve = OUT / "tier11_curves.png"
    fig.savefig(out_curve, dpi=140)
    print(f"wrote {out_curve}")

    # ── final-window bars, with tier-9 reference dots overlaid ──
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(VARIANTS))
    means_11 = [np.mean(finals[v]) for v, _, _ in VARIANTS]
    stds_11 = [np.std(finals[v], ddof=0) for v, _, _ in VARIANTS]
    means_9 = [TIER9_REF[v][0] for v, _, _ in VARIANTS]
    colors = [c for _, _, c in VARIANTS]
    width = 0.35
    ax.bar(x - width/2, means_11, width, yerr=stds_11, color=colors, alpha=0.85,
           capsize=5, edgecolor="black", linewidth=0.6, label="tier 11 (coop)")
    ax.bar(x + width/2, means_9, width, color=colors, alpha=0.35,
           hatch="//", edgecolor="black", linewidth=0.6, label="tier 9 (non-coop) ref")
    for i, (v, _, _) in enumerate(VARIANTS):
        for val in finals[v]:
            ax.scatter(i - width/2, val, color="black", s=18, zorder=3, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([d for _, d, _ in VARIANTS], rotation=10, ha="right")
    ax.set_ylabel("test_total_return_mean  (last 5 evals)")
    ax.set_title("Tier 11 (coop) vs. Tier 9 (non-coop) — final-window performance")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    out_bars = OUT / "tier11_bars.png"
    fig.savefig(out_bars, dpi=140)
    print(f"wrote {out_bars}")

    # ── signal-channel usage check ──
    # mappo_hebbian_r and mappo_hebbian_uniform_r both run HebbianParallelRunner
    # with hebbian.enabled=True, so they log hebbian/signal_total.
    print()
    print("Signal channel usage (hebbian/signal_total) trajectory  -- ")
    print("seed 0 of mappo_hebbian_r as a representative example:")
    print("-" * 80)
    import re
    path = LOG_DIR / "mappo_hebbian_r_seed0.log"
    pat = re.compile(r"hebbian/signal_total:\s+([\d.]+)")
    sigs = []
    with path.open() as f:
        for line in f:
            m = pat.search(line)
            if m:
                sigs.append(float(m.group(1)))
    if sigs:
        n = len(sigs)
        idx = [0, n // 10, n // 4, n // 2, 3 * n // 4, n - 1]
        for i in idx:
            print(f"  snapshot #{i:>3d}/{n}: signal_total = {sigs[i]:.1f}")


if __name__ == "__main__":
    main()
