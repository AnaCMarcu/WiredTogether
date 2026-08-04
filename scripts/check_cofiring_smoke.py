#!/usr/bin/env python3
"""Validate the Experiment-2 smoke runs against the pre-registered gates.

Usage (login node or locally, stdlib only -- no numpy/torch needed):

    python scripts/check_cofiring_smoke.py --runs-root runs/cofiring_smoke

Reads the three SMOKE=1 arms (prcoi, pri, anchor) and prints PASS / WARN /
FAIL per gate; exits 0 iff no gate FAILed, so it can guard the full-sweep
submission in a shell one-liner:

    python scripts/check_cofiring_smoke.py && MODEL_LLM=... WT_IMAGE=... \
        bash hpc/daic/experiments/submit_cofiring.sh

Gates (from the plan / launcher header):
  prcoi : run finished; act mix non-degenerate (no fresh act > 85%);
          all three channels appear in cofiring_events.jsonl; attribution
          total > 0; W finite/in-range.
  pri   : run finished; imitation requested; the proximity gate PASSED at
          least once and replay steps happened (never-passing gate = the arm
          is dead on arrival); abort rate < 1; W finite/in-range.
  anchor: run finished in LEGACY mode -- no choice sidecars on disk, empty
          social_act_metrics, and messages still flowed (comm counts > 0).

The 0.2–0.8 W-differentiation band is a FULL-RUN gate -- a 150-step smoke
barely moves W off its 0.1 warm start, so here W is only checked for sanity
(finite, [0,1], not saturated/collapsed) and reported for eyeballing.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ARMS = {
    "prcoi": "exp23_cofire_prcoi",
    "pri": "exp22_cofire_pri",
    "anchor": "exp27_cofire_anchor",
}
FRESH_ACTS = ("communicate", "observe", "imitate", "none")
ACT_MIX_CAP = 0.85

_results = []


def _report(level: str, arm: str, msg: str) -> None:
    _results.append((level, arm, msg))
    print(f"{level:<5} [{arm:<6}] {msg}")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_jsonl(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return


def _w_sanity(arm: str, run_dir: Path) -> None:
    snaps = list(_iter_jsonl(run_dir / "hebbian_snapshots.jsonl"))
    if not snaps or not snaps[-1].get("W"):
        _report("WARN", arm, "no hebbian_snapshots.jsonl W to check")
        return
    W = snaps[-1]["W"]
    off = [W[i][j] for i in range(len(W)) for j in range(len(W[i])) if i != j]
    bad = [w for w in off
           if not isinstance(w, (int, float)) or math.isnan(w)
           or w < 0.0 or w > 1.0]
    if bad:
        _report("FAIL", arm, f"W has NaN/out-of-range entries: {bad[:4]}")
        return
    mean = sum(off) / len(off)
    if mean > 0.95:
        _report("FAIL", arm, f"W saturated (mean {mean:.3f} > 0.95)")
    elif all(w < 0.005 for w in off):
        _report("FAIL", arm, "W fully collapsed (all off-diag < 0.005)")
    else:
        _report("PASS", arm,
                f"W sane: off-diag min/mean/max = "
                f"{min(off):.3f}/{mean:.3f}/{max(off):.3f} "
                f"(0.2-0.8 band is a full-run gate, not a smoke gate)")


def _finished(arm: str, run_dir: Path):
    fm = _load_json(run_dir / "final_metrics.json")
    if fm is None:
        _report("FAIL", arm, f"not finished: {run_dir / 'final_metrics.json'} "
                             f"missing/unreadable")
    else:
        _report("PASS", arm, "run finished (final_metrics.json present)")
    return fm


def check_prcoi(run_dir: Path) -> None:
    arm = "prcoi"
    fm = _finished(arm, run_dir)
    if fm is None:
        return
    sam = fm.get("social_act_metrics") or {}
    if not sam:
        _report("FAIL", arm, "social_act_metrics empty -- choice mode did not "
                             "engage (wrong flags?)")
        return

    acts = sam.get("act_counts") or {}
    fresh = {a: acts.get(a, 0) for a in FRESH_ACTS}
    total = sum(fresh.values())
    if total == 0:
        _report("FAIL", arm, "no social-act choices recorded at all")
    else:
        top_act, top_n = max(fresh.items(), key=lambda kv: kv[1])
        share = top_n / total
        line = " ".join(f"{a}={n} ({n / total:.0%})"
                        for a, n in fresh.items())
        if share > ACT_MIX_CAP:
            _report("FAIL", arm, f"act mix DEGENERATE -- {top_act} at "
                                 f"{share:.0%} (> {ACT_MIX_CAP:.0%}): {line}. "
                                 f"Rebalance the menu prompt before the sweep.")
        else:
            _report("PASS", arm, f"act mix non-degenerate: {line}")
        never = [a for a in ("communicate", "observe", "imitate")
                 if fresh[a] == 0]
        if never:
            _report("WARN", arm, f"act(s) never chosen: {', '.join(never)} -- "
                                 f"fine at 150 steps, watch in the full runs")

    chans = {"comm": 0, "obs": 0, "imit": 0}
    n_rows = 0
    for row in _iter_jsonl(run_dir / "cofiring_events.jsonl"):
        n_rows += 1
        for ch in chans:
            if float(row.get(f"c_{ch}", 0.0) or 0.0) > 0.0:
                chans[ch] += 1
    missing = [ch for ch, n in chans.items() if n == 0]
    if n_rows == 0:
        _report("FAIL", arm, "cofiring_events.jsonl empty -- no co-firing at all")
    elif missing:
        _report("FAIL", arm, f"channel(s) never fired in {n_rows} co-firing "
                             f"rows: {', '.join(missing)} "
                             f"(counts: {chans})")
    else:
        _report("PASS", arm, f"all channels fire: {chans} over {n_rows} rows")

    attr_total = float((sam.get("cofire_attribution") or {}).get("total") or 0.0)
    if attr_total > 0.0:
        _report("PASS", arm, f"growth attribution accumulated "
                             f"(total {attr_total:.4f})")
    else:
        _report("FAIL", arm, "cofire_attribution total is 0 -- no credited growth")

    _w_sanity(arm, run_dir)


def check_pri(run_dir: Path) -> None:
    arm = "pri"
    fm = _finished(arm, run_dir)
    if fm is None:
        return
    sam = fm.get("social_act_metrics") or {}
    if not sam:
        _report("FAIL", arm, "social_act_metrics empty -- choice mode did not "
                             "engage (wrong flags?)")
        return

    reqs = int(sam.get("imitation_requests") or 0)
    gate_failed = int(sam.get("imitation_gate_failed") or 0)
    replay_steps = int(sam.get("replay_steps") or 0)
    aborts = int(sam.get("imitation_aborts") or 0)

    if reqs == 0:
        _report("FAIL", arm, "the model never chose imitate -- menu prompt "
                             "needs rebalancing (imit is this arm's only act)")
    else:
        _report("PASS", arm, f"imitate requested {reqs}x "
                             f"(horizons: {sam.get('imitation_horizon_hist')})")

    if replay_steps > 0 and gate_failed < reqs:
        _report("PASS", arm, f"gate passed {reqs - gate_failed}/{reqs}, "
                             f"{replay_steps} replay steps executed, "
                             f"{aborts} aborts")
    elif reqs > 0:
        _report("FAIL", arm, f"the proximity gate NEVER passed "
                             f"({gate_failed}/{reqs} failed, {replay_steps} "
                             f"replay steps) -- pri is dead on arrival; "
                             f"consider a navigate-to-target macro before "
                             f"burning the sweep")

    if replay_steps > 0 and aborts >= replay_steps:
        _report("WARN", arm, f"high abort pressure: {aborts} aborts vs "
                             f"{replay_steps} replay steps")

    rs, nrs = replay_steps, int(sam.get("nonreplay_steps") or 0)
    if rs and nrs:
        rmean = sam.get("replay_reward_sum", 0.0) / rs
        nmean = sam.get("nonreplay_reward_sum", 0.0) / nrs
        _report("PASS", arm, f"imitation payoff readable: reward/step "
                             f"{rmean:+.3f} during replay vs {nmean:+.3f} "
                             f"baseline (informational)")

    _w_sanity(arm, run_dir)


def check_anchor(run_dir: Path) -> None:
    arm = "anchor"
    fm = _finished(arm, run_dir)
    if fm is None:
        return
    sam = fm.get("social_act_metrics", None)
    if sam:
        _report("FAIL", arm, f"legacy run has NON-EMPTY social_act_metrics "
                             f"({list(sam)[:4]}...) -- choice code leaked into "
                             f"legacy mode")
    else:
        _report("PASS", arm, "social_act_metrics empty (legacy purity)")

    leaked = [p.name for p in (run_dir / "social_acts.jsonl",
                               run_dir / "cofiring_events.jsonl") if p.exists()]
    if leaked:
        _report("FAIL", arm, f"choice sidecars exist in a LEGACY run: {leaked}")
    else:
        _report("PASS", arm, "no choice sidecars on disk (legacy purity)")

    comm_eps = fm.get("comm_count_per_episode") or []
    n_msgs = sum(sum(ep) for ep in comm_eps if isinstance(ep, list))
    if n_msgs > 0:
        _report("PASS", arm, f"messages still flow in legacy mode "
                             f"({n_msgs} comms recorded)")
    else:
        _report("FAIL", arm, "zero communications in the legacy anchor -- "
                             "enforced comm should message every step")

    _w_sanity(arm, run_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("runs/cofiring_smoke"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    checkers = {"prcoi": check_prcoi, "pri": check_pri, "anchor": check_anchor}
    print(f"== cofiring smoke gates: {args.runs_root} (seed_{args.seed}) ==")
    for arm, dirname in ARMS.items():
        run_dir = args.runs_root / dirname / f"seed_{args.seed}"
        if not run_dir.is_dir():
            _report("WARN", arm, f"run dir missing, skipped: {run_dir}")
            continue
        checkers[arm](run_dir)

    n_fail = sum(1 for lvl, _, _ in _results if lvl == "FAIL")
    n_warn = sum(1 for lvl, _, _ in _results if lvl == "WARN")
    n_pass = sum(1 for lvl, _, _ in _results if lvl == "PASS")
    print(f"== {n_pass} pass, {n_warn} warn, {n_fail} fail ==")
    if n_fail:
        print("DO NOT launch the full sweep -- fix the FAILs first.")
        return 1
    print("All gates green -- safe to launch the full sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
