#!/usr/bin/env python3
"""make_bond_asymmetry.py — is the learned social graph actually directed?

The Hebbian rule is asymmetric *by construction*: the modulator m[i,j] is
egocentric (it depends only on agent i's own advantage/reward), and the
obs/imit co-activity channels are directed (c_obs[i,j] fires when i observed
j, not the reverse). So W[i,j] and W[j,i] are free to diverge. This script
asks whether they empirically *do*, across every run in the experiment suite
that carries a Hebbian graph.

Per matrix W (N x N, zero diagonal) it computes, for every unordered pair
i < j:

    delta_ij  = W[i,j] - W[j,i]            signed directed-trust gap
    |delta|                                magnitude of the gap
    Wbar_ij   = (W[i,j] + W[j,i]) / 2      pair bond scale
    rel_ij    = |delta_ij| / Wbar_ij       gap as a fraction of bond scale

and per matrix:

    A_frob    = ||W - W^T||_F / ||W + W^T||_F   in [0, 1]; 0 = exactly
                symmetric, 1 = purely antisymmetric (one-way bonds)
    net_i     = sum_j (W[i,j] - W[j,i])         agent i's net outward trust

Sources, per run directory:
    hebbian_graph_final.json     end-of-run W (primary)
    hebbian_snapshots.jsonl      per-episode W (trajectory)

Outputs (default paper_assets/bond_asymmetry/):
    bond_asymmetry_pairs.csv     one row per (run, pair) at end of run
    bond_asymmetry_runs.csv      one row per run
    bond_asymmetry_arms.csv      one row per arm, aggregated over seeds
    bond_asymmetry_pairtests.csv per (arm, pair): mean delta over seeds + t-test
    bond_asymmetry_episodes.csv  one row per (run, episode)

Usage:
    python analysis/make_bond_asymmetry.py
    python analysis/make_bond_asymmetry.py --groups medium_runs cofiring_final
    python analysis/make_bond_asymmetry.py --include-smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from paths import ASSETS, RUNS  # noqa: E402  (also puts siblings on sys.path)

try:
    import make_results
    DIR_TO_LABEL = {d: lab for lab, d, _, _ in make_results.CONDITIONS}
except Exception:                                    # pragma: no cover
    DIR_TO_LABEL = {}

# Run groups that are smoke/probe rather than experiment arms. Dropped from
# the headline numbers unless asked for explicitly.
SMOKE_MARKERS = ("smoke", "probe", "orchestrator")

# Groups whose runs exist but are not comparable science. Skipped unless
# named explicitly in --groups.
VOID_GROUPS: set[str] = set()

# ── channel regime, keyed on (group, arm) ────────────────────────────────
# What can make W[i,j] differ from W[j,i] in a given arm:
#   frozen     W is a hardcoded topology, never updated.
#   symmetric  co-activity comes only from proximity and messages; c_spat is
#              an outer product and c_comm is written both ways.
#   mixed      a directed channel is available alongside the symmetric ones.
#   directed   the only social channel is directed (obs or imit), messaging
#              muted -- c_obs/c_imit are written on one side only.
#   bidi       same arms as directed/mixed, but --social-bidirectional makes
#              one obs/imit event credit BOTH directions (graph.py ~L498).
# Keyed on the pair because cofiring_bidi reuses the cofiring arm names with
# a different wiring rule -- they must never pool.
FROZEN_ARMS = {"exp09_llm_9b_allied_all", "exp10_llm_9b_allied_pair",
               "exp11_llm_9b_allied_none"}
DIRECTED_ARMS = {"exp21_cofire_pro", "exp22_cofire_pri"}
MIXED_ARMS = {"exp23_cofire_prcoi", "exp29_cofire_prco"}
BIDI_GROUPS = {"cofiring_bidi"}


def regime(group, arm):
    """Channel regime for one (group, arm) pair."""
    if arm in FROZEN_ARMS:
        return "frozen"
    if group in BIDI_GROUPS:
        return "bidi"
    if arm in DIRECTED_ARMS:
        return "directed"
    if arm in MIXED_ARMS:
        return "mixed"
    return "symmetric"


REGIME_ORDER = ["frozen", "symmetric", "mixed", "bidi", "directed"]

EPS = 1e-9
# Below this the pair has no bond to speak of and rel_ij is meaningless
# (0/0). The frozen No-bonds arm sits here by construction.
BOND_FLOOR = 1e-4


# -- matrix-level measures ------------------------------------------------


def pair_rows(W):
    """Yield (i, j, w_ij, w_ji, delta, absdelta, wbar, rel) for i < j."""
    N = W.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            w_ij, w_ji = float(W[i, j]), float(W[j, i])
            delta = w_ij - w_ji
            wbar = 0.5 * (w_ij + w_ji)
            rel = abs(delta) / wbar if wbar > BOND_FLOOR else float("nan")
            yield i, j, w_ij, w_ji, delta, abs(delta), wbar, rel


def matrix_stats(W):
    """Asymmetry summary for one weight matrix."""
    rows = list(pair_rows(W))
    absd = [r[5] for r in rows]
    rels = [r[7] for r in rows if not math.isnan(r[7])]
    off = W[~np.eye(W.shape[0], dtype=bool)]

    num = np.linalg.norm(W - W.T)
    den = np.linalg.norm(W + W.T)
    a_frob = float(num / den) if den > EPS else float("nan")

    net = (W.sum(axis=1) - W.sum(axis=0))          # out-strength - in-strength
    return {
        "n_agents": int(W.shape[0]),
        "n_pairs": len(rows),
        "mean_W": float(off.mean()) if off.size else float("nan"),
        "mean_abs_delta": float(np.mean(absd)) if absd else float("nan"),
        "max_abs_delta": float(np.max(absd)) if absd else float("nan"),
        "mean_rel_asym": float(np.mean(rels)) if rels else float("nan"),
        "max_rel_asym": float(np.max(rels)) if rels else float("nan"),
        "a_frob": a_frob,
        "max_net_trust": float(np.max(np.abs(net))) if net.size else float("nan"),
        "exactly_symmetric": bool(np.array_equal(W, W.T)),
    }


# -- run discovery / loading ----------------------------------------------


def load_W(path):
    """Load an N x N W from a hebbian_graph_final.json-style file."""
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, {}
    W = d.get("W")
    if not W:
        return None, {}
    try:
        arr = np.asarray(W, dtype=np.float64)
    except (TypeError, ValueError):
        return None, {}
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        return None, {}
    meta = {"mode": d.get("mode"), "frozen": bool(d.get("frozen", False)),
            "init_weight": d.get("init_weight"), "steps": d.get("_step_count")}
    return arr, meta


def load_snapshots(path):
    """Return [(episode, W, final_step, reward_total)] from the JSONL.

    Some runs were relaunched onto the same output dir (see the medium-run
    re-runs), which *appends* a second pass of episode records to this file
    -- e.g. episodes [1, 2, 1, 2, 3, 3]. hebbian_graph_final.json is
    overwritten by the last pass, so to stay consistent with it we keep the
    LAST record per episode index and sort. Returns (records, n_duplicates).
    """
    out = []
    try:
        text = path.read_text()
    except OSError:
        return out, 0
    for k, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        W = rec.get("W")
        if not W:
            continue
        try:
            arr = np.asarray(W, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
            continue
        out.append((int(rec.get("episode", k + 1)), arr,
                    rec.get("final_step"), rec.get("reward_total")))

    by_ep = {}
    for rec in out:                                  # last write wins
        by_ep[rec[0]] = rec
    return [by_ep[e] for e in sorted(by_ep)], len(out) - len(by_ep)


def discover(runs_root, groups=None, include_smoke=False):
    """Find every (group, arm, seed) run dir holding a Hebbian graph."""
    runs = []
    for group_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        group = group_dir.name
        if groups and group not in groups:
            continue
        smoke = any(m in group for m in SMOKE_MARKERS)
        if smoke and not include_smoke and not groups:
            continue
        if group in VOID_GROUPS and not groups:
            continue
        for arm_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(arm_dir.glob("seed_*")):
                final = seed_dir / "hebbian_graph_final.json"
                snaps = seed_dir / "hebbian_snapshots.jsonl"
                if not final.exists() and not snaps.exists():
                    continue
                try:
                    seed = int(seed_dir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                runs.append({
                    "group": group, "arm": arm_dir.name, "seed": seed,
                    "path": seed_dir, "smoke": smoke,
                    "label": DIR_TO_LABEL.get(arm_dir.name, arm_dir.name),
                    "final": final if final.exists() else None,
                    "snaps": snaps if snaps.exists() else None,
                })
    return runs


# -- aggregation ----------------------------------------------------------


def mean_sd(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan"), 0
    if len(xs) == 1:
        return xs[0], float("nan"), 1
    return st.mean(xs), st.stdev(xs), len(xs)


def collect(runs):
    """Build the per-pair, per-run and per-episode record tables."""
    pair_recs, run_recs, ep_recs = [], [], []

    for r in runs:
        W, meta = None, {}
        source = None
        if r["final"]:
            W, meta = load_W(r["final"])
            source = "hebbian_graph_final.json"
        snaps, n_dupes = load_snapshots(r["snaps"]) if r["snaps"] else ([], 0)
        if W is None and snaps:                      # fall back to last episode
            W = snaps[-1][1]
            source = "hebbian_snapshots.jsonl (last episode)"
        if W is None:
            continue
        # Provenance: the last episode snapshot should equal the final graph.
        final_matches = (
            bool(np.allclose(W, snaps[-1][1], atol=1e-6))
            if snaps and snaps[-1][1].shape == W.shape else None
        )

        base = {"group": r["group"], "arm": r["arm"], "label": r["label"],
                "seed": r["seed"], "smoke": r["smoke"],
                "regime": regime(r["group"], r["arm"])}
        ms = matrix_stats(W)
        run_recs.append({**base, **ms, "mode": meta.get("mode"),
                         "frozen": meta.get("frozen"),
                         "init_weight": meta.get("init_weight"),
                         "steps": meta.get("steps"),
                         "n_episodes": len(snaps),
                         "snapshot_dupes_dropped": n_dupes,
                         "final_matches_last_snapshot": final_matches,
                         "source": source})

        for i, j, w_ij, w_ji, d, ad, wbar, rel in pair_rows(W):
            pair_recs.append({**base, "i": i, "j": j,
                              "w_ij": w_ij, "w_ji": w_ji, "delta": d,
                              "abs_delta": ad, "w_mean": wbar,
                              "rel_asym": rel,
                              "stronger": (f"agent_{i}->agent_{j}" if d > 0
                                           else f"agent_{j}->agent_{i}"
                                           if d < 0 else "tie")})

        for ep, Wep, fstep, rtot in snaps:
            m = matrix_stats(Wep)
            ep_recs.append({**base, "episode": ep, "final_step": fstep,
                            "reward_total": rtot, **m})

    return pair_recs, run_recs, ep_recs


def arm_table(run_recs, pair_recs):
    """Aggregate run-level asymmetry per arm, over seeds."""
    by_arm = defaultdict(list)
    for r in run_recs:
        by_arm[(r["group"], r["arm"])].append(r)
    pairs_by_arm = defaultdict(list)
    for p in pair_recs:
        pairs_by_arm[(p["group"], p["arm"])].append(p)

    rows = []
    for key, rs in sorted(by_arm.items()):
        group, arm = key
        ps = pairs_by_arm[key]
        mad, mad_sd, n = mean_sd([r["mean_abs_delta"] for r in rs])
        rel, rel_sd, _ = mean_sd([r["mean_rel_asym"] for r in rs])
        frob, frob_sd, _ = mean_sd([r["a_frob"] for r in rs])
        mw, _, _ = mean_sd([r["mean_W"] for r in rs])
        rows.append({
            "group": group, "arm": arm, "label": rs[0]["label"],
            "regime": rs[0]["regime"],
            "n_seeds": n, "n_agents": rs[0]["n_agents"],
            "mode": rs[0]["mode"], "frozen": rs[0]["frozen"],
            "mean_W": mw,
            "mean_abs_delta": mad, "sd_abs_delta": mad_sd,
            "max_abs_delta": max((r["max_abs_delta"] for r in rs),
                                 default=float("nan")),
            "mean_rel_asym": rel, "sd_rel_asym": rel_sd,
            "max_rel_asym": max((r["max_rel_asym"] for r in rs),
                                default=float("nan")),
            "a_frob": frob, "sd_a_frob": frob_sd,
            "n_runs_exactly_symmetric": sum(r["exactly_symmetric"] for r in rs),
            "n_pairs_total": len(ps),
        })
    return rows


def pair_tests(pair_recs):
    """Per (arm, pair): is the signed gap reproducible across seeds?"""
    by = defaultdict(list)
    for p in pair_recs:
        by[(p["group"], p["arm"], p["i"], p["j"])].append(p)
    rows = []
    for (group, arm, i, j), ps in sorted(by.items()):
        deltas = [p["delta"] for p in ps]
        m, sd, n = mean_sd(deltas)
        t = pval = float("nan")
        if n >= 2 and sd == sd and sd > EPS:
            tt = stats.ttest_1samp(deltas, 0.0)
            t, pval = float(tt.statistic), float(tt.pvalue)
        elif n >= 2 and all(abs(d) < EPS for d in deltas):
            t, pval = 0.0, 1.0                       # frozen arm: delta == 0
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        rows.append({
            "group": group, "arm": arm, "label": ps[0]["label"],
            "regime": ps[0]["regime"],
            "pair": f"{i}-{j}", "n_seeds": n,
            "mean_delta": m, "sd_delta": sd,
            "mean_abs_delta": mean_sd([p["abs_delta"] for p in ps])[0],
            "mean_rel_asym": mean_sd([p["rel_asym"] for p in ps])[0],
            "t": t, "p": pval,
            "seeds_i_stronger": pos, "seeds_j_stronger": neg,
            "sign_consistency": max(pos, neg) / n if n else float("nan"),
        })
    return rows


def write_csv(path, rows, fields=None):
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=RUNS, type=Path)
    ap.add_argument("--groups", nargs="*", default=None,
                    help="restrict to these run groups (default: all non-smoke)")
    ap.add_argument("--include-smoke", action="store_true",
                    help="also include smoke/probe run groups")
    ap.add_argument("--out", default=ASSETS / "bond_asymmetry", type=Path)
    args = ap.parse_args()

    runs = discover(args.runs_root, args.groups, args.include_smoke)
    print(f"discovered {len(runs)} runs with a Hebbian graph", file=sys.stderr)
    pair_recs, run_recs, ep_recs = collect(runs)
    print(f"  usable: {len(run_recs)} runs, {len(pair_recs)} pairs, "
          f"{len(ep_recs)} episode snapshots", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "bond_asymmetry_pairs.csv", pair_recs)
    write_csv(args.out / "bond_asymmetry_runs.csv", run_recs)
    write_csv(args.out / "bond_asymmetry_episodes.csv", ep_recs)
    arms = arm_table(run_recs, pair_recs)
    write_csv(args.out / "bond_asymmetry_arms.csv", arms)
    tests = pair_tests(pair_recs)
    write_csv(args.out / "bond_asymmetry_pairtests.csv", tests)

    # Provenance sidecar: what was skipped and why, so the report can state
    # coverage without re-deriving it (and without going stale).
    ok = {(r["group"], r["arm"], r["seed"]) for r in run_recs}
    skipped = defaultdict(list)
    for r in runs:
        if (r["group"], r["arm"], r["seed"]) not in ok:
            skipped[r["group"]].append(r["arm"])
    all_groups = sorted(p.name for p in args.runs_root.iterdir() if p.is_dir())
    prov = {
        "discovered_runs": len(runs),
        "usable_runs": len(run_recs),
        "skipped_no_W": {g: sorted(set(a)) for g, a in sorted(skipped.items())},
        "skipped_no_W_total": len(runs) - len(run_recs),
        "included_groups": sorted({r["group"] for r in run_recs}),
        "excluded_smoke_groups": sorted(
            g for g in all_groups if any(m in g for m in SMOKE_MARKERS)),
        "excluded_void_groups": sorted(VOID_GROUPS & set(all_groups)),
    }
    (args.out / "bond_asymmetry_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    print(f"wrote 5 CSVs + provenance to {args.out}/", file=sys.stderr)
    return arms, tests


if __name__ == "__main__":
    main()
