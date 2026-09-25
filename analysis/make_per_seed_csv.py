#!/usr/bin/env python3
r"""make_per_seed_csv.py - per-seed and per-episode values behind the tables.

The paper's tables report mean +- POPULATION SD over POOLED EPISODES. That is
not a seed-level spread, and the two differ a lot here: three episodes of one
seed share a map, a seed and an agent population, and episode 2 of a run
inherits state from episode 1 (see the episode-reset note in the repo), so
episodes within a seed are not independent draws.

Resampling episodes would therefore be pseudoreplication - it would treat
n_seeds x 3 as the sample size and produce intervals roughly sqrt(3) too
narrow. The independent unit is the SEED. This script emits both levels so
the choice is explicit and checkable:

  per_episode.csv   one row per (condition, seed, episode) - the raw units
                    the tables pool. Episode index is within-run.
  per_seed.csv      one row per (condition, seed): the mean over that seed's
                    episodes. THIS is the unit to bootstrap or t-test over.
  per_seed_summary.csv
                    per condition: n_seeds, the seed-level mean, the
                    seed-level SD (sample, ddof=1) and SEM, next to the
                    pooled-episode population SD the tables print, so the
                    gap between the two is visible.

Metrics match the tables exactly (make_results' own definitions, imported
rather than reimplemented):
  task_return   decomposed team return per episode, excludes hebbian_diffuse
  milestone_pct 100 * non-comm milestones fired / 25
  coop_pct      100 * Ch2-Ch5 milestones fired / 17

Conditions cover every row of the RQ1 comparison (make_final_table.ROWS plus
the extended and RL trace-rule arms) and, with --cofire, the Experiment-2
suites.

Usage:
  python analysis/make_per_seed_csv.py --out paper_assets/per_seed
  python analysis/make_per_seed_csv.py --out paper_assets/per_seed --cofire
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

from paths import ASSETS, group  # noqa: F401  (also puts siblings on sys.path)

import make_final_table as mft
import make_results as MR
from make_final_table_extended import NEW_ROWS

from make_final_table_latex import EXTRA_ROWS, PANELS

# Exactly the conditions the paper prints. tab:final_comparison is the
# registry below; tab:bond_behaviour_rho re-uses five of its arms; and
# tab:cofire_main is the 3f suite. tab:transplant_main,
# tab:topology_horizon and tab:perception_by_model have their data rows
# commented out in results.tex, and the perception numbers come from the
# qualitative beliefs pipeline rather than from run metrics, so none of
# them has a per-seed row to emit here.
PAPER_KEYS = {key for _panel, blocks in PANELS for blk in blocks
              for _tag, _label, key in blk if key}

# Only the suite the paper reports. The other four co-firing suites
# (cofiring, cofiring_final, cofiring_bidi, cofiring_noreward) are the
# reward-modulated predecessors and appear nowhere in the paper.
COFIRE_SUITES = ["cofiring_bidi_3f"]


def seed_of(run) -> str:
    """load_runs stores the metrics-file path as _path (NOT _run_dir, which
    is what make_directive_timelines.load_run sets); reading the wrong key
    silently collapses every seed of a condition into one row."""
    m = re.search(r"seed_(\d+)", str(run.get("_path", "")))
    if not m:
        raise KeyError("cannot recover seed from run record: "
                       + str(sorted(run)[:6]))
    return m.group(1)


def per_episode_rows(label, arm_dir, runs):
    """One row per episode, using make_results' own metric definitions."""
    out = []
    for run in runs:
        sets_ = MR.episode_milestone_sets(run)
        task, decomposed = MR.episode_task_returns(run)
        for e, s in enumerate(sets_, 1):
            nc = sum(1 for m in s
                     if MR.MILESTONE_TRACK.get(m) not in MR.SOCIAL_ACT_TRACKS)
            out.append({
                "condition": label, "arm_dir": arm_dir, "seed": seed_of(run),
                "episode": e,
                "task_return": round(task[e - 1], 4) if e <= len(task) else "",
                "milestone_pct": round(100.0 * nc / MR.NONCOMM_MAX, 4),
                "coop_pct": round(100.0 * MR.coop_count(s) / MR.COOP_MAX, 4),
                "task_decomposed": int(decomposed),
            })
    return out


def collapse_to_seed(ep_rows):
    """Mean over each seed's episodes - the independent unit."""
    by = {}
    for r in ep_rows:
        by.setdefault((r["condition"], r["arm_dir"], r["seed"]), []).append(r)
    out = []
    for (cond, arm, seed), rs in by.items():
        row = {"condition": cond, "arm_dir": arm, "seed": seed,
               "n_episodes": len(rs)}
        for k in ("task_return", "milestone_pct", "coop_pct"):
            vals = [r[k] for r in rs if r[k] != ""]
            row[k] = round(st.fmean(vals), 4) if vals else ""
        out.append(row)
    return sorted(out, key=lambda r: (r["condition"], int(r["seed"])
                                      if r["seed"].isdigit() else 0))


def summarise(seed_rows, ep_rows):
    """Seed-level mean/SD/SEM beside the pooled-episode SD the tables print."""
    conds, out = {}, []
    for r in seed_rows:
        conds.setdefault(r["condition"], []).append(r)
    eps = {}
    for r in ep_rows:
        eps.setdefault(r["condition"], []).append(r)
    for cond, rs in conds.items():
        row = {"condition": cond, "n_seeds": len(rs),
               "n_episodes": sum(r["n_episodes"] for r in rs)}
        for k in ("task_return", "milestone_pct", "coop_pct"):
            sv = [r[k] for r in rs if r[k] != ""]
            ev = [r[k] for r in eps[cond] if r[k] != ""]
            row[f"{k}_seed_mean"] = round(st.fmean(sv), 3) if sv else ""
            # sample SD over seeds (ddof=1): the spread a CI should use
            row[f"{k}_seed_sd"] = (round(st.stdev(sv), 3) if len(sv) > 1
                                   else "")
            row[f"{k}_seed_sem"] = (round(st.stdev(sv) / len(sv) ** 0.5, 3)
                                    if len(sv) > 1 else "")
            # population SD over pooled episodes: what the tables print
            row[f"{k}_pooled_ep_sd"] = (round(st.pstdev(ev), 3) if ev else "")
        out.append(row)
    return sorted(out, key=lambda r: r["condition"])


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ASSETS / "per_seed")
    ap.add_argument("--cofire", action="store_true",
                    help="also emit the Experiment-2 co-firing suite "
                         "(cofiring_bidi_3f, the one the paper reports)")
    ap.add_argument("--all", action="store_true",
                    help="every registry condition, not just those printed "
                         "in the paper")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    rows = list(mft.ROWS) + list(NEW_ROWS) + list(EXTRA_ROWS)
    if not args.all:
        rows = [r for r in rows if r[0] in PAPER_KEYS]
        missing = PAPER_KEYS - {r[0] for r in rows}
        if missing:
            print(f"  [warn] paper rows with no registry entry: "
                  f"{sorted(missing)}", file=sys.stderr)
    if args.cofire:
        for suite in COFIRE_SUITES:
            root = group(suite)
            if not Path(root).is_dir():
                continue
            for arm in sorted(p.name for p in Path(root).iterdir()
                              if p.is_dir()):
                rows.append((f"{suite}/{arm}", arm, root, "", "", ""))

    ep_rows = []
    for name, d, root, *_ in rows:
        runs = MR.load_runs(Path(root), d, frozenset())
        if not runs:
            continue
        ep_rows += per_episode_rows(name, d, runs)

    seed_rows = collapse_to_seed(ep_rows)
    summary = summarise(seed_rows, ep_rows)

    write(args.out / "per_episode.csv", ep_rows,
          ["condition", "arm_dir", "seed", "episode", "task_return",
           "milestone_pct", "coop_pct", "task_decomposed"])
    write(args.out / "per_seed.csv", seed_rows,
          ["condition", "arm_dir", "seed", "n_episodes", "task_return",
           "milestone_pct", "coop_pct"])
    write(args.out / "per_seed_summary.csv", summary,
          ["condition", "n_seeds", "n_episodes"]
          + [f"{k}_{s}" for k in ("task_return", "milestone_pct", "coop_pct")
             for s in ("seed_mean", "seed_sd", "seed_sem", "pooled_ep_sd")])

    n_small = sum(1 for r in summary if r["n_seeds"] < 4)
    print(f"\n{len(summary)} conditions, {len(seed_rows)} seeds, "
          f"{len(ep_rows)} episodes")
    if n_small:
        print(f"[note] {n_small} conditions have fewer than 4 seeds; a "
              f"percentile bootstrap over 2-3 seeds has at most 3^3 = 27 "
              f"distinct resamples and its interval endpoints are the "
              f"observed min/max, so it is not informative there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
