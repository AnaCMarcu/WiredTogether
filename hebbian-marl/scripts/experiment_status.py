#!/usr/bin/env python3
"""experiment_status.py — print the state of every planned run.

Cross-references scripts/experiments.yaml against results/runs.jsonl:
each (variant, seed) is either PENDING, DONE, or FAIL.

Usage:
    python scripts/experiment_status.py
    python scripts/experiment_status.py --tier 1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_LOG = REPO_ROOT / "results" / "runs.jsonl"
MANIFEST = REPO_ROOT / "scripts" / "experiments.yaml"


def load_runs() -> dict:
    """Return the most recent record per label."""
    out: dict = {}
    if not RUNS_LOG.is_file():
        return out
    with RUNS_LOG.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["label"]] = r
    return out


def planned_runs(tier_filter):
    data = yaml.safe_load(MANIFEST.read_text())
    defaults = data.get("defaults", {}) or {}
    out = []
    for exp in data["experiments"]:
        tier = int(exp.get("tier", 99))
        if tier_filter is not None and tier != tier_filter:
            continue
        seeds = exp.get("seeds", defaults.get("seeds", [0]))
        label_override = exp.get("label")
        for s in seeds:
            label = label_override if label_override else f'{exp["variant"]}_seed{int(s)}'
            out.append((tier, label, exp.get("rationale", "")))
    out.sort()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", type=int, default=None)
    args = p.parse_args()

    runs = load_runs()
    plan = planned_runs(args.tier)

    by_tier = defaultdict(list)
    for tier, label, rationale in plan:
        rec = runs.get(label)
        if rec is None:
            status = "PENDING"
        elif rec.get("exit_code") == 0:
            status = f"DONE   ({rec.get('wall_seconds', '?')}s)"
        else:
            status = f"FAIL   (exit={rec.get('exit_code')})"
        by_tier[tier].append((label, status, rationale))

    for tier in sorted(by_tier):
        print(f"\n-- Tier {tier} --")
        for label, status, rationale in by_tier[tier]:
            print(f"  {label:35s}  {status:25s}  {rationale}")

    total = sum(len(v) for v in by_tier.values())
    done = sum(1 for v in by_tier.values() for _, s, _ in v if s.startswith("DONE"))
    fail = sum(1 for v in by_tier.values() for _, s, _ in v if s.startswith("FAIL"))
    pend = sum(1 for v in by_tier.values() for _, s, _ in v if s.startswith("PENDING"))
    print(f"\n[summary] {done}/{total} done, {fail} failed, {pend} pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
