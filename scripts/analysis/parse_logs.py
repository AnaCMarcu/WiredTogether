"""Parse epymarl log files and produce a comparison across conditions/seeds.

Reads `Recent Stats | t_env: N` blocks and extracts `test_total_return_mean`
and `total_return_mean`. Writes a tidy CSV and a per-condition summary.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECENT_STATS_RE = re.compile(r"Recent Stats \| t_env:\s+(\d+)")
KV_RE = re.compile(r"([a-zA-Z0-9_]+):\s*(-?\d+\.\d+)")
FILENAME_RE = re.compile(r"^(?P<cond>.+)_seed(?P<seed>\d+)\.log$")

KEYS = ("test_total_return_mean", "total_return_mean", "test_ep_length_mean")


def parse_log(path: Path):
    rows = []
    current_t = None
    pending = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RECENT_STATS_RE.search(line)
            if m:
                if current_t is not None and pending:
                    rows.append((current_t, pending))
                current_t = int(m.group(1))
                pending = {}
                continue
            if current_t is None:
                continue
            for k, v in KV_RE.findall(line):
                if k in KEYS and k not in pending:
                    pending[k] = float(v)
            if all(k in pending for k in KEYS):
                rows.append((current_t, pending))
                current_t = None
                pending = {}
    if current_t is not None and pending:
        rows.append((current_t, pending))
    return rows


def main():
    by_cond = defaultdict(list)
    for log in sorted(LOG_DIR.glob("*.log")):
        m = FILENAME_RE.match(log.name)
        if not m:
            continue
        cond, seed = m.group("cond"), int(m.group("seed"))
        rows = parse_log(log)
        by_cond[cond].append((seed, rows))
        print(f"{log.name}: {len(rows)} eval points  "
              f"final test_return={rows[-1][1].get('test_total_return_mean', float('nan')):.4f}"
              if rows else f"{log.name}: 0 eval points")

    csv_path = OUT_DIR / "returns.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("condition,seed,t_env,test_total_return_mean,total_return_mean,test_ep_length_mean\n")
        for cond, runs in by_cond.items():
            for seed, rows in runs:
                for t, kv in rows:
                    f.write(f"{cond},{seed},{t},"
                            f"{kv.get('test_total_return_mean', '')},"
                            f"{kv.get('total_return_mean', '')},"
                            f"{kv.get('test_ep_length_mean', '')}\n")
    print(f"\nwrote {csv_path}")

    print("\n=== Final-100k-window mean test_total_return (averaged over last ~5 eval points) ===")
    summary = []
    for cond in sorted(by_cond):
        finals = []
        for seed, rows in by_cond[cond]:
            tail = [kv.get("test_total_return_mean") for _, kv in rows[-5:]
                    if kv.get("test_total_return_mean") is not None]
            if tail:
                finals.append(np.mean(tail))
        if finals:
            arr = np.array(finals)
            summary.append((cond, arr.mean(), arr.std(ddof=0), len(arr), arr.tolist()))
    summary.sort(key=lambda x: -x[1])
    for cond, m, s, n, vals in summary:
        per_seed = ", ".join(f"{v:.4f}" for v in vals)
        print(f"  {cond:25s}  mean={m:.4f}  std={s:.4f}  n={n}   per-seed=[{per_seed}]")

    print("\n=== AUC (mean test_total_return across whole training) ===")
    for cond in sorted(by_cond):
        aucs = []
        for seed, rows in by_cond[cond]:
            vals = [kv.get("test_total_return_mean") for _, kv in rows
                    if kv.get("test_total_return_mean") is not None]
            if vals:
                aucs.append(np.mean(vals))
        if aucs:
            arr = np.array(aucs)
            print(f"  {cond:25s}  mean_auc={arr.mean():.4f}  std={arr.std(ddof=0):.4f}  n={len(arr)}")


if __name__ == "__main__":
    main()
