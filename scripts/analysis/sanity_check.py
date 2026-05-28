"""Analyse the tier-6 sanity runs against tier-1's collapse pattern."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

RECENT_RE = re.compile(r"Recent Stats \| t_env:\s+(\d+)")
KV_RE = re.compile(r"([a-zA-Z0-9_]+):\s*(-?\d+\.\d+)")

def parse_log(path: Path):
    rows = []
    cur_t = None
    pending = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RECENT_RE.search(line)
            if m:
                if cur_t is not None and pending:
                    rows.append((cur_t, pending))
                cur_t = int(m.group(1))
                pending = {}
                continue
            if cur_t is None:
                continue
            for k, v in KV_RE.findall(line):
                pending.setdefault(k, float(v))
    if cur_t is not None and pending:
        rows.append((cur_t, pending))
    return rows

RETURN_KEYS = ("test_total_return_mean", "test_return_mean")
def get_return(kv):
    for k in RETURN_KEYS:
        if k in kv:
            return kv[k]
    return None

SANITY_RUNS = [
    ("sanity_plain_easy",            "tier 6: ippo_baseline, plain lbf 8x8-2p-2f-coop, T=25"),
    ("sanity_comm_easy",             "tier 6: ippo_baseline, lbf_comm 8x8-2p-2f-v3, T=25"),
    ("sanity_horizon",               "tier 6: ippo_baseline, Foraging-Comm-10x10-3p-3f-v3, T=100"),
    ("sanity_mappo_plain_easy",      "tier 7: MAPPO, plain lbf 8x8-2p-2f-coop, T=25"),
    ("sanity_mappo_plain_hard",      "tier 7: MAPPO, plain lbf 10x10-3p-3f, T=50"),
    ("sanity_mappo_comm",            "tier 7: MAPPO + lbf_comm wrapper (crashed)"),
    ("sanity_mappo_hebbian_offflags","tier 8: mappo_hebbian, hebbian.enabled=False (runner sanity)"),
    ("sanity_mappo_hebbian_r",       "tier 8: mappo_hebbian_r, reward_diffusion=True (headline)"),
]


def windowed_mean(rows, lo, hi):
    vals = []
    for t, kv in rows:
        if lo <= t < hi:
            v = get_return(kv)
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    print(f"{'run':22s}  {'#evals':>6s}  {'early(<300k)':>13s}  {'mid(500-700k)':>14s}  "
          f"{'late(>=900k)':>13s}  {'peak':>7s}  {'final':>7s}")
    print("-" * 100)
    for label, desc in SANITY_RUNS:
        path = LOG_DIR / f"{label}.log"
        rows = parse_log(path)
        if not rows:
            print(f"{label:22s}  NO DATA")
            continue
        vals = [get_return(kv) for _, kv in rows if get_return(kv) is not None]
        peak = max(vals) if vals else float("nan")
        final = float(np.mean(vals[-5:])) if vals else float("nan")
        e = windowed_mean(rows, 0, 300_000)
        m = windowed_mean(rows, 500_000, 700_000)
        l = windowed_mean(rows, 900_000, 1_100_000)
        print(f"{label:22s}  {len(rows):>6d}  {e:>13.4f}  {m:>14.4f}  "
              f"{l:>13.4f}  {peak:>7.4f}  {final:>7.4f}")
        print(f"  {desc}")

    print("\nReference (tier-1 collapsed runs, final-window means):")
    print("  ippo_baseline 0.0065   seac 0.0115   hebb_s 0.0106   hebb_s_nocomm 0.0072")
    print("  (all four indistinguishable from random exploration noise)")


if __name__ == "__main__":
    main()
