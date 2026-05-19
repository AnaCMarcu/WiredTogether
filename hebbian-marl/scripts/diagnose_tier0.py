#!/usr/bin/env python3
"""diagnose_tier0.py - interpret the tier-0 diagnostic runs.

Reads bonds/<label>/seed_<seed>.jsonl and the latest sacred metrics.json
for each tier-0 entry in scripts/experiments.yaml, prints a per-run
summary, and synthesises an overall interpretation against the three
hypotheses in HEBBIAN_MARL_PLAN.md section 1.3.

Usage:
    python scripts/diagnose_tier0.py
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


def _latest_metrics_path(label: str, env_key: str) -> Optional[Path]:
    env_safe = env_key.replace(":", "_").replace("/", "_").replace("\\", "_")
    pattern = str(EPYMARL_RESULTS / "sacred" / label / env_safe / "*" / "metrics.json")
    cands = glob.glob(pattern)
    if not cands:
        return None
    return Path(sorted(cands, key=lambda p: int(os.path.basename(os.path.dirname(p))))[-1])


def _bonds_path(label: str, seed: int) -> Path:
    return EPYMARL_RESULTS / "bonds" / label / f"seed_{seed}.jsonl"


def _load_bonds(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _series(metrics: dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
    s = metrics.get(key)
    if not isinstance(s, dict) or not s.get("values"):
        return np.array([]), np.array([])
    return np.asarray(s["steps"]), np.asarray(s["values"], dtype=np.float64)


def _late_mean(arr: np.ndarray, frac: float = 0.2) -> Optional[float]:
    if arr.size == 0:
        return None
    n = max(1, int(arr.size * frac))
    return float(arr[-n:].mean())


def analyse(label: str, env_key: str, seed: int) -> dict:
    bonds = _load_bonds(_bonds_path(label, seed))
    if not bonds:
        return {"label": label, "env": env_key, "status": "MISSING -- bonds file not found (run incomplete?)"}

    t_env = np.array([s["t_env"] for s in bonds])
    mean_w = np.array([s["mean_bond_strength"] for s in bonds])

    bond_init = float(mean_w[0])
    bond_peak = float(mean_w.max())
    bond_peak_t = int(t_env[np.argmax(mean_w)])
    bond_late = _late_mean(mean_w, 0.2)

    # Sacred metrics (signal rate, return)
    mpath = _latest_metrics_path(label, env_key)
    sig_late = None
    ret_late = None
    if mpath is not None:
        try:
            metrics = json.loads(mpath.read_text())
        except (json.JSONDecodeError, OSError):
            metrics = {}
        _, sig_vals = _series(metrics, "hebbian/signal_total")
        _, ret_vals = _series(metrics, "total_return_mean")
        if ret_vals.size == 0:
            _, ret_vals = _series(metrics, "return_mean")
        sig_late = _late_mean(sig_vals, 0.2)
        ret_late = _late_mean(ret_vals, 0.2)

    # Verdicts on the bond trajectory
    collapsed = bond_late is not None and bond_late < 0.02
    stable = bond_late is not None and bond_late >= 0.1
    recovered = (
        bond_late is not None
        and bond_peak >= 0.15
        and bond_late >= 0.1
        and np.min(mean_w[t_env > bond_peak_t]) < 0.05  # dipped between peak and end
    )

    if stable and not recovered:
        bond_verdict = "STABLE -- bonds grew and held"
    elif recovered:
        bond_verdict = "RECOVERED -- bonds dipped then climbed back"
    elif collapsed:
        bond_verdict = "COLLAPSED -- bonds decayed to ~0 and stayed there"
    else:
        bond_verdict = "AMBIGUOUS -- see trajectory"

    # Signal verdict
    if sig_late is None:
        sig_verdict = "no signal metric logged"
    elif sig_late < 0.5:
        sig_verdict = f"signals DEAD ({sig_late:.1f}/ep) -- PPO trained them out"
    elif sig_late < 3.0:
        sig_verdict = f"signals SPARSE ({sig_late:.1f}/ep)"
    else:
        sig_verdict = f"signals ALIVE ({sig_late:.1f}/ep)"

    return {
        "label": label,
        "env": env_key,
        "status": "OK",
        "bond_init": bond_init,
        "bond_peak": bond_peak,
        "bond_peak_t": bond_peak_t,
        "bond_late": bond_late,
        "bond_verdict": bond_verdict,
        "sig_late": sig_late,
        "sig_verdict": sig_verdict,
        "ret_late": ret_late,
        "collapsed": collapsed,
        "stable": stable,
        "recovered": recovered,
    }


def _hr(c: str = "-", n: int = 78) -> str:
    return c * n


def main() -> int:
    data = yaml.safe_load(MANIFEST.read_text())
    defaults = data.get("defaults", {}) or {}
    tier0 = [e for e in data["experiments"] if e.get("tier") == 0]

    if not tier0:
        print("No tier-0 experiments found in scripts/experiments.yaml.")
        return 1

    print(_hr("="))
    print(f"Tier-0 diagnostic interpretation  ({len(tier0)} runs)")
    print(_hr("="))

    results: Dict[str, dict] = {}
    for exp in tier0:
        label = exp.get("label") or f'{exp["variant"]}_seed{exp.get("seeds", [0])[0]}'
        env_key = exp.get("env_key", defaults.get("env_key"))
        seed = exp.get("seeds", defaults.get("seeds", [0]))[0]
        r = analyse(label, env_key, seed)
        results[label] = r

        print()
        print(f"[{label}]   variant={exp['variant']}  env={env_key}")
        if r["status"] != "OK":
            print(f"   {r['status']}")
            continue
        print(
            f"   bonds:   init={r['bond_init']:.3f}  peak={r['bond_peak']:.3f}@t={r['bond_peak_t']:>6}  "
            f"late={r['bond_late']:.4f}"
        )
        print(f"            {r['bond_verdict']}")
        sig_str = f"{r['sig_late']:.1f}/ep" if r['sig_late'] is not None else "n/a"
        ret_str = f"{r['ret_late']:.3f}" if r['ret_late'] is not None else "n/a"
        print(f"   signals: late={sig_str}   {r['sig_verdict']}")
        print(f"   return:  late={ret_str}")

    # ── Cross-run synthesis ──
    print()
    print(_hr("="))
    print("Overall interpretation")
    print(_hr("="))

    s = results.get("diag_hebb_s_500k", {})
    rd = results.get("diag_hebb_r_500k", {})
    rs = results.get("diag_hebb_rs_500k", {})
    big = results.get("diag_hebb_s_500k_bigenv", {})

    have_all = all(r.get("status") == "OK" for r in (s, rd, rs, big))
    if not have_all:
        missing = [
            name for name, r in [
                ("diag_hebb_s_500k", s),
                ("diag_hebb_r_500k", rd),
                ("diag_hebb_rs_500k", rs),
                ("diag_hebb_s_500k_bigenv", big),
            ]
            if r.get("status") != "OK"
        ]
        print(f"Some diagnostic runs are missing or failed: {missing}")
        print("Re-run them before drawing conclusions.")
        return 1

    s_ok = s["stable"] or s["recovered"]
    rd_ok = rd["stable"] or rd["recovered"]
    rs_ok = rs["stable"] or rs["recovered"]
    big_differs = (
        abs((big["bond_late"] or 0) - (s["bond_late"] or 0)) > 0.05
        or abs((big["sig_late"] or 0) - (s["sig_late"] or 0)) > 2.0
    )

    if s_ok:
        print("INTERPRETATION 1 (just needs more training time).")
        print("  hebb_s recovered with 10x the smoke budget.")
        print("  -> Proceed to tier 1. The smoke test was just too short.")
    elif (not s_ok) and (rd_ok or rs_ok):
        print("INTERPRETATION 2 -- refined: the SHARING feedback loop is the weak link.")
        print(f"  hebb_s:  {s['bond_verdict']}")
        print(f"  hebb_r:  {rd['bond_verdict']}")
        print(f"  hebb_rs: {rs['bond_verdict']}")
        print()
        if rs_ok:
            print("  -> hebb_rs survived. Pivot headline claim to `hebb_rs > seac` (a+b combined)")
            print("     and report `hebb_s > seac` as a null result with this diagnostic as")
            print("     supporting evidence that sharing alone cannot bootstrap signaling.")
        elif rd_ok:
            print("  -> Only the shorter (diffusion) loop closes on its own. Headline pivots")
            print("     to `hebb_r > mappo` if a positive claim is required, or report the")
            print("     null on the originally-intended `hebb_s > seac` comparison.")
    elif (not s_ok) and (not rd_ok) and (not rs_ok) and big_differs:
        print("INTERPRETATION 3 -- env mismatch.")
        print(f"  All loops collapsed on default env, but the bigger env's trajectory differs:")
        print(f"    default  late_W={s['bond_late']:.4f}  late_sig={s['sig_late']}")
        print(f"    bigenv   late_W={big['bond_late']:.4f}  late_sig={big['sig_late']}")
        print("  -> Consider rerunning the headline grid on the bigger env, or extend")
        print("     lbf_comm_wrapper to register a force_coop variant where joint loading")
        print("     is required (much stronger spatial co-activity signal).")
    else:
        print("INTERPRETATION 2 -- confirmed: the mechanism is structurally incomplete.")
        print("  No feedback path closed within 500k steps on either env.")
        print(f"    hebb_s:        {s['bond_verdict']}")
        print(f"    hebb_r:        {rd['bond_verdict']}")
        print(f"    hebb_rs:       {rs['bond_verdict']}")
        print(f"    hebb_s_bigenv: {big['bond_verdict']}")
        print()
        print("  -> The headline grid as currently designed will produce null results")
        print("     (W -> 0 means hebb_s == hebb_ippo). Do NOT run tier 1 yet.")
        print("     Options to reopen the design:")
        print("       a) Add an intrinsic reward for bond growth so PPO directly reinforces")
        print("          actions that lead to Hebbian co-activity.")
        print("       b) Drop the opportunity-cost design -- let signal coexist with movement.")
        print("       c) Curriculum: pretrain hebb_r (shorter loop) for N steps, then switch")
        print("          to hebb_s once bonds are established.")
        print("       d) Accept the null and write it up as the chapter's contribution:")
        print("          the design is principled, the diagnostic is honest, the result is")
        print("          informative regardless of sign.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
