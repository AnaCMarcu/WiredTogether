"""Tier-9 bond-structure analysis: descriptive look at the learned Hebbian
social graph in `mappo_hebbian_r` vs the (tracked-but-ignored) graph in
`mappo_hebbian_uniform_r`.

For each run we load the bonds.jsonl trajectory and compute:
  - W mean / max / sparsity / asymmetry over time
  - Final-state W matrix (averaged within variant)
  - Cross-seed consistency of bond structure
  - Correlation between bond magnitude and policy return
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BONDS_DIR = Path(__file__).resolve().parents[2] / "bonds"
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("mappo_hebbian_r",         "Hebbian-weighted (W_bar used)", "#d62728"),
    ("mappo_hebbian_uniform_r", "uniform diffusion (W_bar tracked but ignored)", "#2ca02c"),
]
SEEDS = list(range(5))


def load_trajectory(variant: str, seed: int):
    """Return list of snapshots; each is a dict with t_env, W (3x3 np.array),
    mean_bond_strength, sparsity, asymmetry_frob, out_strength."""
    path = BONDS_DIR / f"{variant}_seed{seed}" / f"seed_{seed}.jsonl"
    snaps = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            d["W"] = np.array(d["W"], dtype=np.float32)
            snaps.append(d)
    return snaps


def out_strength(W: np.ndarray) -> np.ndarray:
    """Sum of outgoing bonds per agent (excluding self)."""
    return (W.sum(axis=1) - np.diag(W))


def final_returns_by_seed(variant: str):
    """Read final-window test_total_return_mean from tier9_finals.csv."""
    path = OUT / "tier9_finals.csv"
    finals = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["variant"] == variant:
                finals[int(row["seed"])] = float(row["final_window_mean"])
    return finals


def main():
    # ── Load trajectories ──
    trajs = {(v, s): load_trajectory(v, s) for v, _, _ in VARIANTS for s in SEEDS}

    # ── 1. Summary table: mean / max / sparsity / asym at end ──
    print("=" * 88)
    print(f"{'variant':<28s} {'seed':>4s}  {'#snaps':>6s}  {'W_bar_mean[0]':>10s}  "
          f"{'W_bar_mean[end]':>11s}  {'sparsity[end]':>13s}  {'asym[end]':>9s}")
    print("-" * 88)
    for v, _, _ in VARIANTS:
        for s in SEEDS:
            snaps = trajs[(v, s)]
            first, last = snaps[0], snaps[-1]
            print(f"{v:<28s} {s:>4d}  {len(snaps):>6d}  "
                  f"{first['mean_bond_strength']:>10.4f}  "
                  f"{last['mean_bond_strength']:>11.4f}  "
                  f"{last['sparsity']:>13.4f}  "
                  f"{last['asymmetry_frob']:>9.4f}")
        print()

    # ── 2. Final W matrix (averaged within variant) ──
    print("Final W matrix (mean across 5 seeds):")
    print("-" * 88)
    for v, desc, _ in VARIANTS:
        W_finals = np.array([trajs[(v, s)][-1]["W"] for s in SEEDS])
        W_mean = W_finals.mean(axis=0)
        W_std = W_finals.std(axis=0, ddof=0)
        print(f"\n{v} ({desc}):")
        print(f"  mean across seeds:")
        for row in W_mean:
            print("   ", [f"{x:.4f}" for x in row])
        print(f"  std  across seeds:")
        for row in W_std:
            print("   ", [f"{x:.4f}" for x in row])
    print()

    # ── 3. Correlation: final W_bar magnitude vs policy return (per seed) ──
    print("Bond magnitude vs. final policy return (per seed):")
    print("-" * 88)
    for v, _, _ in VARIANTS:
        finals_pol = final_returns_by_seed(v)
        rows = []
        for s in SEEDS:
            W_end = trajs[(v, s)][-1]["W"]
            mean_bond = float(W_end.sum() / (W_end.shape[0] * (W_end.shape[1] - 1)))
            rows.append((s, mean_bond, finals_pol[s]))
        rows.sort(key=lambda r: r[1])
        print(f"\n{v}  (sorted by mean_bond):")
        print(f"  {'seed':>4s}  {'mean_bond':>10s}  {'final_return':>12s}")
        for s, b, r in rows:
            print(f"  {s:>4d}  {b:>10.4f}  {r:>12.4f}")
        bonds = np.array([r[1] for r in rows])
        rets = np.array([r[2] for r in rows])
        if bonds.std() > 0 and rets.std() > 0:
            corr = np.corrcoef(bonds, rets)[0, 1]
            print(f"  Pearson r (mean_bond, final_return) = {corr:+.3f}")
    print()

    # ── 4. Trajectory plot: W mean over t_env ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for v, desc, color in VARIANTS:
        for s in SEEDS:
            snaps = trajs[(v, s)]
            t = np.array([d["t_env"] for d in snaps]) / 1e6
            m = np.array([d["mean_bond_strength"] for d in snaps])
            ax.plot(t, m, color=color, alpha=0.35, linewidth=1.0)
        all_t = sorted({d["t_env"] for s in SEEDS for d in trajs[(v, s)]})
        common_t = np.array(all_t)
        mat = []
        for s in SEEDS:
            ts = np.array([d["t_env"] for d in trajs[(v, s)]])
            vs = np.array([d["mean_bond_strength"] for d in trajs[(v, s)]])
            mat.append(np.interp(common_t, ts, vs))
        mat = np.array(mat)
        ax.plot(common_t / 1e6, mat.mean(0), color=color, linewidth=2.5, label=desc)
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("mean bond strength W_bar")
    ax.set_title("Bond magnitude over training  (thin = per seed, thick = cross-seed mean)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_traj = OUT / "tier9_bonds_trajectory.png"
    fig.savefig(out_traj, dpi=140)
    print(f"wrote {out_traj}")

    # ── 5. Sparsity + asymmetry trajectories ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for v, desc, color in VARIANTS:
        all_t = sorted({d["t_env"] for s in SEEDS for d in trajs[(v, s)]})
        common_t = np.array(all_t)
        mat_sp, mat_as = [], []
        for s in SEEDS:
            snaps = trajs[(v, s)]
            ts = np.array([d["t_env"] for d in snaps])
            mat_sp.append(np.interp(common_t, ts, [d["sparsity"] for d in snaps]))
            mat_as.append(np.interp(common_t, ts, [d["asymmetry_frob"] for d in snaps]))
        mat_sp, mat_as = np.array(mat_sp), np.array(mat_as)
        ax1.plot(common_t / 1e6, mat_sp.mean(0), color=color, linewidth=2.2, label=desc)
        ax1.fill_between(common_t / 1e6, mat_sp.mean(0) - mat_sp.std(0),
                         mat_sp.mean(0) + mat_sp.std(0), color=color, alpha=0.15)
        ax2.plot(common_t / 1e6, mat_as.mean(0), color=color, linewidth=2.2, label=desc)
        ax2.fill_between(common_t / 1e6, mat_as.mean(0) - mat_as.std(0),
                         mat_as.mean(0) + mat_as.std(0), color=color, alpha=0.15)
    ax1.set_xlabel("env steps (M)"); ax1.set_ylabel("sparsity (fraction of bonds below threshold)")
    ax1.set_title("Sparsity over training"); ax1.grid(alpha=0.3); ax1.legend(loc="upper left")
    ax2.set_xlabel("env steps (M)"); ax2.set_ylabel("||W - W^T||_F  (Frobenius norm)")
    ax2.set_title("Asymmetry over training"); ax2.grid(alpha=0.3); ax2.legend(loc="upper right")
    fig.tight_layout()
    out_meta = OUT / "tier9_bonds_meta.png"
    fig.savefig(out_meta, dpi=140)
    print(f"wrote {out_meta}")

    # ── 6. Final-W heatmaps ──
    fig, axes = plt.subplots(2, 5, figsize=(13, 5))
    for col, s in enumerate(SEEDS):
        for row, (v, _, _) in enumerate(VARIANTS):
            ax = axes[row, col]
            W = trajs[(v, s)][-1]["W"]
            im = ax.imshow(W, cmap="viridis", vmin=0, vmax=max(0.1, W.max()))
            ax.set_xticks(range(W.shape[1])); ax.set_yticks(range(W.shape[0]))
            ax.set_title(f"{v.split('_')[-2] if v.endswith('_r') else v}  s{s}", fontsize=9)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                            color="white" if W[i, j] < 0.05 else "black", fontsize=7)
    axes[0, 0].set_ylabel("hebbian_r", fontsize=10)
    axes[1, 0].set_ylabel("uniform_r", fontsize=10)
    fig.suptitle("Final W matrices (t_env ≈ 3M) per seed and variant")
    fig.tight_layout()
    out_heat = OUT / "tier9_bonds_heatmaps.png"
    fig.savefig(out_heat, dpi=140)
    print(f"wrote {out_heat}")


if __name__ == "__main__":
    main()
