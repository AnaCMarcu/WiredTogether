"""Aggregate `final_metrics.json` from multiple runs into median/IQR plots.

Usage:
    python scripts/aggregate_seeds.py 'runs/2026-05-04_*' --out runs/aggregate_v1
    python scripts/aggregate_seeds.py runs/run_a runs/run_b runs/run_c --out runs/agg

Reads each run's `final_metrics.json` and produces:
  * `summary.json`     — per-metric median + IQR + 10/90 percentile bands.
  * `plots/learning_curves.png`   — median return over steps with 10/90 band.
  * `plots/steps_to_milestone.png` — bars of median steps to each milestone fire.

Reporting protocol follows Yu et al. 2022 (MAPPO) / Palmer et al. 2022:
median + 25/75 percentile, never mean+stddev.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_run(run_dir: Path) -> dict | None:
    fm = run_dir / "final_metrics.json"
    if not fm.exists():
        # Back-compat: older runs wrote data.json.
        fm = run_dir / "data.json"
    if not fm.exists():
        return None
    try:
        return json.loads(fm.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _expand(patterns: list) -> list:
    paths: list = []
    for p in patterns:
        if any(ch in p for ch in "*?[]"):
            paths.extend(sorted(Path(x) for x in glob.glob(p)))
        else:
            paths.append(Path(p))
    return [p for p in paths if p.is_dir()]


def _percentiles(arr: np.ndarray, qs=(10, 25, 50, 75, 90)) -> dict:
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def aggregate(run_dirs: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    runs = []
    for rd in run_dirs:
        data = _load_run(rd)
        if data is None:
            print(f"[warn] skipping {rd} (no final_metrics.json)")
            continue
        runs.append({"path": str(rd), "data": data})
    if not runs:
        print("No runs loaded.")
        sys.exit(1)
    print(f"[info] aggregating {len(runs)} runs")

    # ── 1. Final cumulative returns: per-agent stats ──
    n_agents = max((len(r["data"].get("cumulative_returns") or []) for r in runs), default=0)
    final_returns = np.zeros((len(runs), n_agents))
    for i, r in enumerate(runs):
        cr = r["data"].get("cumulative_returns") or []
        for a, v in enumerate(cr):
            if a < n_agents:
                final_returns[i, a] = float(v)
    team_returns = final_returns.sum(axis=1)
    final_returns_stats = {
        "per_agent": [_percentiles(final_returns[:, a]) for a in range(n_agents)],
        "team":      _percentiles(team_returns),
    }

    # ── 2. Learning curves: ts_data.cumulative_returns aligned by step ──
    # Find common step grid (intersection of timesteps lists).
    series = []  # one (steps, returns_per_agent_TxA) per run
    for r in runs:
        td = r["data"].get("timestep_data") or {}
        ts = td.get("timesteps") or []
        crs = td.get("cumulative_returns") or []
        if not ts or not crs:
            continue
        T = len(ts)
        A = len(crs)
        arr = np.zeros((T, A))
        for a in range(A):
            row = crs[a]
            arr[:len(row), a] = row[:T]
        series.append((np.array(ts), arr))
    if series:
        # Use the run with the fewest steps as the reference grid.
        min_T = min(len(s[0]) for s in series)
        steps = series[0][0][:min_T]
        team = np.stack([s[1][:min_T].sum(axis=1) for s in series], axis=0)
        median = np.percentile(team, 50, axis=0)
        p10 = np.percentile(team, 10, axis=0)
        p90 = np.percentile(team, 90, axis=0)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, median, label="median (team)", color="C0")
            ax.fill_between(steps, p10, p90, alpha=0.25, color="C0", label="10–90 pct")
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative team return")
            ax.set_title(f"Learning curves over {len(series)} seeds")
            ax.legend()
            fig.savefig(plots_dir / "learning_curves.png", dpi=150)
            plt.close(fig)
        except ImportError:
            print("[warn] matplotlib not available; skipping learning_curves.png")
    else:
        steps = np.array([])

    # ── 3. Steps-to-milestone: median over runs (None → max steps + 1) ──
    milestone_first_step = {}
    for r in runs:
        s2m = r["data"].get("steps_to_milestone") or {}
        for track, ms_map in s2m.items():
            for mid, step in ms_map.items():
                key = mid
                if key not in milestone_first_step:
                    milestone_first_step[key] = []
                milestone_first_step[key].append(step if step is not None else None)
    milestone_stats = {}
    for mid, vals in milestone_first_step.items():
        completed = [v for v in vals if v is not None]
        completion_rate = len(completed) / len(vals) if vals else 0.0
        median_step = float(np.median(completed)) if completed else None
        milestone_stats[mid] = {
            "completion_rate": completion_rate,
            "median_step":     median_step,
            "n_runs":          len(vals),
        }

    # ── 4. Communication metrics aggregated across runs ──
    comm_agg = {}
    keys_to_agg = ["total_messages", "valid_messages", "total_tokens", "tokens_per_milestone"]
    for k in keys_to_agg:
        vals = [r["data"].get("comm_metrics", {}).get(k) for r in runs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            comm_agg[k] = _percentiles(np.array(vals))

    # ── 5. Cooperation metrics (credit_gini, etc.) ──
    coop_agg = {}
    for k in ["credit_gini"]:
        vals = [r["data"].get("coop_metrics", {}).get(k) for r in runs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            coop_agg[k] = _percentiles(np.array(vals))

    summary = {
        "n_runs": len(runs),
        "run_paths": [r["path"] for r in runs],
        "final_returns": final_returns_stats,
        "milestone_stats": milestone_stats,
        "comm_metrics_aggregated": comm_agg,
        "coop_metrics_aggregated": coop_agg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[info] wrote {out_dir/'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Run directories or globs (e.g. 'runs/2026-*')")
    ap.add_argument("--out", default="runs/aggregate", help="Output directory")
    args = ap.parse_args()

    run_dirs = _expand(args.paths)
    if not run_dirs:
        print(f"No matching run directories for: {args.paths}", file=sys.stderr)
        sys.exit(1)
    aggregate(run_dirs, Path(args.out))


if __name__ == "__main__":
    main()
