"""Check whether returns trend upward during training (vs. flat-near-zero)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CSV = Path(__file__).resolve().parent / "out" / "returns.csv"

rows_by_run = defaultdict(list)
with CSV.open() as f:
    r = csv.DictReader(f)
    for row in r:
        key = (row["condition"], row["seed"])
        try:
            rows_by_run[key].append((int(row["t_env"]), float(row["test_total_return_mean"])))
        except ValueError:
            pass

by_cond = defaultdict(list)
for (cond, seed), rows in rows_by_run.items():
    rows.sort()
    by_cond[cond].append(rows)

print(f"{'condition':25s}  {'early(0-300k)':>14s}  {'mid(1-1.5M)':>14s}  {'late(2.5-3M)':>14s}  {'max-any-eval':>14s}")
for cond in sorted(by_cond):
    early, mid, late, peaks = [], [], [], []
    for rows in by_cond[cond]:
        e = [v for t, v in rows if t < 300_000]
        m = [v for t, v in rows if 1_000_000 <= t < 1_500_000]
        l = [v for t, v in rows if t >= 2_500_000]
        vals = [v for _, v in rows]
        if e: early.append(np.mean(e))
        if m: mid.append(np.mean(m))
        if l: late.append(np.mean(l))
        if vals: peaks.append(max(vals))

    def fmt(xs):
        a = np.array(xs)
        return f"{a.mean():.4f}±{a.std(ddof=0):.3f}"
    print(f"{cond:25s}  {fmt(early):>14s}  {fmt(mid):>14s}  {fmt(late):>14s}  {fmt(peaks):>14s}")
