"""plot_results.py — turn ablation-grid runs into publication figures.

Loads sacred metrics under ``epymarl/results/sacred/`` and bond snapshots
under ``epymarl/results/bonds/`` and produces:

  - Episode-return curves (mean ± 95% CI per variant)
  - Bond evolution heatmap over time (for `hebb_s` / `hebb_rs` / `hebb_rsp`)
  - Bond asymmetry curve per Hebbian variant
  - Signal-count over training, per variant
  - Bond-comm correlation scatter for `hebb_s`
  - Paired-seed Wilcoxon signed-rank tests for the two headline claims:
      * `hebb_s` vs `seac`           — does Hebbian weighting beat uniform?
      * `hebb_s` vs `hebb_s_nocomm`  — does the comm term carry signal?

Outputs PNGs under ``results/plots/`` and prints p-values to stdout.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results-dir epymarl/results
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── data loading ──────────────────────────────────────────────────────


def _parse_run_name(run_name: str) -> Tuple[str, Optional[int]]:
    """Split `hebb_s_seed3` -> (`hebb_s`, 3). Returns (name, None) on failure."""
    m = re.match(r"^(.+)_seed(\d+)$", run_name)
    if not m:
        return run_name, None
    return m.group(1), int(m.group(2))


def load_metrics(results_dir: str) -> Dict[str, Dict[int, dict]]:
    """Return ``{variant: {seed: metrics_dict}}`` keyed by config name."""
    sacred_root = os.path.join(results_dir, "sacred")
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    if not os.path.isdir(sacred_root):
        return out

    # Sacred layout: results/sacred/{name}/{env_key}/{run_idx}/metrics.json
    for name_dir in sorted(os.listdir(sacred_root)):
        variant, seed = _parse_run_name(name_dir)
        for metrics_path in glob(
            os.path.join(sacred_root, name_dir, "*", "*", "metrics.json")
        ):
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if seed is None:
                # No seed in the name — use the run index as a stand-in
                run_idx = int(os.path.basename(os.path.dirname(metrics_path)))
                out[variant][run_idx] = m
            else:
                out[variant][seed] = m
    return out


def load_bonds(results_dir: str) -> Dict[str, Dict[int, List[dict]]]:
    """Return ``{variant: {seed: [bond_snapshot_dicts]}}``."""
    bonds_root = os.path.join(results_dir, "bonds")
    out: Dict[str, Dict[int, List[dict]]] = defaultdict(dict)
    if not os.path.isdir(bonds_root):
        return out

    for name_dir in sorted(os.listdir(bonds_root)):
        variant, _ = _parse_run_name(name_dir)
        for jsonl_path in glob(os.path.join(bonds_root, name_dir, "seed_*.jsonl")):
            seed_m = re.search(r"seed_(\d+)\.jsonl$", jsonl_path)
            if not seed_m:
                continue
            seed = int(seed_m.group(1))
            snapshots = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        snapshots.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if snapshots:
                out[variant][seed] = snapshots
    return out


# ── analysis ──────────────────────────────────────────────────────────


def _series(metrics: dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
    s = metrics.get(key)
    if not isinstance(s, dict) or not s.get("values"):
        return np.array([]), np.array([])
    return np.asarray(s["steps"]), np.asarray(s["values"], dtype=np.float64)


def _final_returns(metrics_by_seed: Dict[int, dict], window_fraction: float = 0.1) -> np.ndarray:
    """For each seed, compute the mean return over the final ``window_fraction``
    of training (default last 10%). Used for headline statistical tests."""
    out = []
    for seed in sorted(metrics_by_seed):
        steps, vals = _series(metrics_by_seed[seed], "total_return_mean")
        if vals.size == 0:
            # common_reward case might use return_mean
            steps, vals = _series(metrics_by_seed[seed], "return_mean")
        if vals.size == 0:
            continue
        n = max(1, int(window_fraction * vals.size))
        out.append(float(vals[-n:].mean()))
    return np.array(out)


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Return the two-sided p-value of a paired-seed Wilcoxon signed-rank test,
    or None if scipy is missing or the inputs are too short."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if a.size != b.size or a.size < 2:
        return None
    try:
        res = wilcoxon(a, b)
    except ValueError:
        return None
    return float(res.pvalue)


# ── plotting ──────────────────────────────────────────────────────────


def _mean_ci(values: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack ragged arrays by their common prefix length, then mean ± 95% CI."""
    if not values:
        return np.array([]), np.array([]), np.array([])
    L = min(v.size for v in values)
    if L == 0:
        return np.array([]), np.array([]), np.array([])
    stacked = np.stack([v[:L] for v in values], axis=0)  # (n_seeds, L)
    mean = stacked.mean(axis=0)
    if stacked.shape[0] >= 2:
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        ci = 1.96 * sem
    else:
        ci = np.zeros_like(mean)
    return mean, ci, np.arange(L)


def plot_return_curves(metrics_all: Dict[str, Dict[int, dict]], out_path: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(metrics_all):
        seeds_metrics = metrics_all[variant]
        value_seeds = []
        step_seeds = []
        for seed in sorted(seeds_metrics):
            steps, vals = _series(seeds_metrics[seed], "total_return_mean")
            if vals.size == 0:
                steps, vals = _series(seeds_metrics[seed], "return_mean")
            if vals.size == 0:
                continue
            value_seeds.append(vals)
            step_seeds.append(steps)
        if not value_seeds:
            continue
        mean, ci, _ = _mean_ci(value_seeds)
        # Use shortest step axis for x
        L = mean.size
        x = step_seeds[0][:L]
        ax.plot(x, mean, label=variant)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.15)
    ax.set_xlabel("env step")
    ax.set_ylabel("episode return")
    ax.set_title("Episode return — mean ± 95% CI across seeds")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bond_asymmetry(bonds_all: Dict[str, Dict[int, List[dict]]], out_path: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(bonds_all):
        seeds_bonds = bonds_all[variant]
        per_seed_series = []
        per_seed_steps = []
        for seed in sorted(seeds_bonds):
            snapshots = seeds_bonds[seed]
            asym = np.array([s.get("asymmetry_frob", 0.0) for s in snapshots])
            steps = np.array([s.get("t_env", i) for i, s in enumerate(snapshots)])
            if asym.size:
                per_seed_series.append(asym)
                per_seed_steps.append(steps)
        if not per_seed_series:
            continue
        mean, ci, _ = _mean_ci(per_seed_series)
        L = mean.size
        x = per_seed_steps[0][:L]
        ax.plot(x, mean, label=variant)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.15)
    ax.set_xlabel("env step")
    ax.set_ylabel(r"$\|W - W^\top\|_F$")
    ax.set_title("Bond asymmetry over training")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bond_heatmap(bonds: Dict[int, List[dict]], variant: str, out_path: str):
    """For one variant: average W across seeds at each snapshot, plot heatmap
    of final-time W."""
    import matplotlib.pyplot as plt

    final_Ws = []
    for seed in sorted(bonds):
        snapshots = bonds[seed]
        if not snapshots:
            continue
        last = snapshots[-1]
        W = np.array(last.get("W", []), dtype=np.float32)
        if W.size:
            final_Ws.append(W)
    if not final_Ws:
        return
    mean_W = np.mean(np.stack(final_Ws, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_W, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"{variant}: final W (mean over {len(final_Ws)} seeds)")
    ax.set_xlabel("target agent j")
    ax.set_ylabel("source agent i")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_signal_counts(metrics_all: Dict[str, Dict[int, dict]], out_path: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for variant in sorted(metrics_all):
        seeds_metrics = metrics_all[variant]
        per_seed_series = []
        per_seed_steps = []
        for seed in sorted(seeds_metrics):
            steps, vals = _series(seeds_metrics[seed], "hebbian/signal_total")
            if vals.size:
                per_seed_series.append(vals)
                per_seed_steps.append(steps)
        if not per_seed_series:
            continue
        any_data = True
        mean, ci, _ = _mean_ci(per_seed_series)
        L = mean.size
        x = per_seed_steps[0][:L]
        ax.plot(x, mean, label=variant)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.15)
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("env step")
    ax.set_ylabel("signal actions per episode (mean)")
    ax.set_title("Signal-action use over training")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="results",
        help="path containing the sacred/ and bonds/ subtrees",
    )
    parser.add_argument(
        "--out-dir", default="results/plots",
        help="where to write PNG figures",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_all = load_metrics(args.results_dir)
    bonds_all = load_bonds(args.results_dir)

    if not metrics_all:
        print(f"[plot_results] no metrics under {args.results_dir}/sacred — nothing to plot.")
        return

    print("[plot_results] variants:", sorted(metrics_all))

    # Figures
    plot_return_curves(metrics_all, os.path.join(args.out_dir, "return_curves.png"))
    if bonds_all:
        plot_bond_asymmetry(bonds_all, os.path.join(args.out_dir, "bond_asymmetry.png"))
        for variant in ("hebb_s", "hebb_rs", "hebb_rsp"):
            if variant in bonds_all:
                plot_bond_heatmap(
                    bonds_all[variant], variant,
                    os.path.join(args.out_dir, f"bond_heatmap_{variant}.png"),
                )
    plot_signal_counts(metrics_all, os.path.join(args.out_dir, "signal_counts.png"))

    # Headline statistical tests (failure-grace mechanism, tier 1).
    # All variants share the comm-augmented LBF action space; only the
    # Hebbian + sharing mechanism differs across arms.
    print()
    print("=== Headline paired-seed Wilcoxon tests (final 10% window) ===")
    headline_pairs = [
        ("hebb_s_grace",  "seac",                "claim 1: sharing+grace > uniform sharing"),
        ("hebb_s_grace",  "ippo_baseline",       "claim 2: sharing+grace > no-sharing floor"),
        ("hebb_s_grace",  "hebb_s_grace_nocomm", "claim 3: comm term carries signal in grace regime"),
        ("hebb_rs_grace", "seac",                "claim 4: full stack > uniform sharing"),
        ("hebb_rs_grace", "hebb_s_grace",        "claim 5: (a)+(b)+grace > (b)+grace alone"),
    ]
    for treat, base, label in headline_pairs:
        if treat not in metrics_all or base not in metrics_all:
            print(f"[skip] {label} — missing {treat!r} or {base!r}")
            continue
        a = _final_returns(metrics_all[treat])
        b = _final_returns(metrics_all[base])
        n = min(a.size, b.size)
        if n < 2:
            print(f"[skip] {label} — not enough paired seeds (n={n})")
            continue
        p = _paired_wilcoxon(a[:n], b[:n])
        print(f"{label}:")
        print(f"    {treat} ({a[:n].mean():+.4f}) vs {base} ({b[:n].mean():+.4f}) — n={n}, p={p}")

    print()
    print(f"[plot_results] wrote PNGs to {args.out_dir}/")


if __name__ == "__main__":
    main()
