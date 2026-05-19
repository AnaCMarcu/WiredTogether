#!/usr/bin/env python3
"""plot_tier0.py - visualise the tier-0 diagnostic runs.

Generates three PNGs under results/plots/tier0/:

  - bonds.png    : mean off-diagonal W over t_env, one line per diagnostic
  - signals.png  : hebbian/signal_total per episode over t_env
  - returns.png  : return_mean over t_env

Partial runs (killed mid-flight) get a dashed line and "[partial]" in the
legend so you can tell apart "done but flat" from "incomplete data."

Usage:
    python scripts/plot_tier0.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EPYMARL_RESULTS = REPO_ROOT / "results"
MANIFEST = REPO_ROOT / "scripts" / "experiments.yaml"
OUT_DIR = REPO_ROOT / "results" / "plots" / "tier0"


def _latest_metrics(label: str, env_key: str) -> Optional[dict]:
    env_safe = env_key.replace(":", "_").replace("/", "_").replace("\\", "_")
    pattern = str(EPYMARL_RESULTS / "sacred" / label / env_safe / "*" / "metrics.json")
    cands = glob.glob(pattern)
    if not cands:
        return None
    latest = sorted(cands, key=lambda p: int(os.path.basename(os.path.dirname(p))))[-1]
    try:
        return json.loads(Path(latest).read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_bonds(label: str, seed: int) -> List[dict]:
    path = EPYMARL_RESULTS / "bonds" / label / f"seed_{seed}.jsonl"
    if not path.is_file():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _series(metrics: dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
    s = (metrics or {}).get(key)
    if not isinstance(s, dict) or not s.get("values"):
        return np.array([]), np.array([])
    return np.asarray(s["steps"]), np.asarray(s["values"], dtype=np.float64)


def _is_partial(t_env_last: int, t_max: int, tol: float = 0.95) -> bool:
    return t_env_last < t_max * tol


def main() -> int:
    import matplotlib.pyplot as plt

    data = yaml.safe_load(MANIFEST.read_text())
    defaults = data.get("defaults", {}) or {}
    tier0 = [e for e in data["experiments"] if e.get("tier") == 0]
    if not tier0:
        print("No tier-0 experiments in manifest.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather everything per diagnostic
    runs: List[dict] = []
    for exp in tier0:
        label = exp.get("label") or f'{exp["variant"]}_seed{exp.get("seeds", [0])[0]}'
        env_key = exp.get("env_key", defaults.get("env_key"))
        seed = exp.get("seeds", defaults.get("seeds", [0]))[0]
        t_max = int(exp.get("t_max", defaults.get("t_max", 5_000_000)))

        bonds = _load_bonds(label, seed)
        metrics = _latest_metrics(label, env_key) or {}

        bond_t = np.array([s["t_env"] for s in bonds]) if bonds else np.array([])
        bond_w = np.array([s["mean_bond_strength"] for s in bonds]) if bonds else np.array([])
        sig_t, sig_v = _series(metrics, "hebbian/signal_total")
        # total_return_mean: per-agent-reward variants (most hebb_* configs).
        # return_mean: shared-reward variants. Fall back.
        ret_t, ret_v = _series(metrics, "total_return_mean")
        if ret_v.size == 0:
            ret_t, ret_v = _series(metrics, "return_mean")

        last_t = int(bond_t.max()) if bond_t.size else 0
        partial = _is_partial(last_t, t_max)

        runs.append({
            "label": label,
            "env": env_key,
            "t_max": t_max,
            "partial": partial,
            "missing": not bonds,
            "bond_t": bond_t, "bond_w": bond_w,
            "sig_t": sig_t,   "sig_v": sig_v,
            "ret_t": ret_t,   "ret_v": ret_v,
        })

    def _plot(metric_t_key: str, metric_v_key: str, ylabel: str, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(9, 5))
        any_data = False
        for r in runs:
            t, v = r[metric_t_key], r[metric_v_key]
            if t.size == 0 or v.size == 0:
                continue
            any_data = True
            label = r["label"]
            if r["missing"]:
                label += " [missing]"
            elif r["partial"]:
                label += f" [partial: {int(t.max()):,}/{r['t_max']:,}]"
            style = "--" if r["partial"] else "-"
            ax.plot(t, v, style, label=label, linewidth=1.6)
        if not any_data:
            print(f"  [skip] {fname} - no data")
            plt.close(fig)
            return
        ax.set_xlabel("env step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out = OUT_DIR / fname
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")

    print(f"[plot_tier0] {len(runs)} runs, writing to {OUT_DIR.relative_to(REPO_ROOT)}/")

    _plot("bond_t", "bond_w",
          "mean off-diagonal W",
          "Tier-0 diagnostic: Hebbian bond evolution",
          "bonds.png")
    _plot("sig_t", "sig_v",
          "signal actions per episode",
          "Tier-0 diagnostic: signal-action rate",
          "signals.png")
    _plot("ret_t", "ret_v",
          "episode return (per-agent mean)",
          "Tier-0 diagnostic: return curve",
          "returns.png")

    # Status table
    print()
    print(f"{'label':30s} {'status':12s} {'last t_env':>12} / {'t_max':>9}")
    for r in runs:
        if r["missing"]:
            status = "MISSING"
        elif r["partial"]:
            status = "PARTIAL"
        else:
            status = "COMPLETE"
        last_t = int(r["bond_t"].max()) if r["bond_t"].size else 0
        print(f"{r['label']:30s} {status:12s} {last_t:>12,} / {r['t_max']:>9,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
