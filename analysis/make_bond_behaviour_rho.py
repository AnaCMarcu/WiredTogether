#!/usr/bin/env python3
r"""make_bond_behaviour_rho.py - tab:bond_behaviour_rho, recomputed.

Spearman rho between the end-of-episode bond W_ij and the per-pair
interaction counts (messages / joint digs / proximity), for the
+Social plasticity (Hebbian 2.0, three-factor + trace) arms.

The earlier version of this table was computed on the +inst. arms. This
reads the plasticity arms instead, via make_final_table's run registry, so
the row set stays tied to tab:final_comparison.

Numbers read from episode_summary.json["cooperation_metrics"] (hebbian_W +
pair_interaction), NEVER from final_metrics.coop_metrics, which the
coop_eval nesting bug zeroes.

Two statistics per cell, as the paper prints them:
  - pooled rho over every pair-episode record of the condition
  - the mean +- sd of the per-seed rho

Usage:
  python analysis/make_bond_behaviour_rho.py
  python analysis/make_bond_behaviour_rho.py --arms trace --out ../paper_assets
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "qualitative"))

from paths import ASSETS  # noqa: F401,E402  (puts analysis/ siblings on path)

import make_final_table as mft  # noqa: E402
from make_final_table_extended import NEW_ROWS  # noqa: E402
from make_final_table_latex import EXTRA_ROWS  # noqa: E402
from qual_lib.bonds_behavior import _spearman  # noqa: E402

PLANES = [("messages", "Messages"),
          ("joint_dig", "Joint digging"),
          ("proximity", "Proximity")]

# (printed row label, registry key). The Gemma plasticity arm is listed
# twice because the registry and run_3f.py disagree about which directory
# is the Hebbian 2.0 anchor: new_exp_0_gemma_hebbian3f has
# hebbian_death_ltd=0 (three-factor WITHOUT death LTD, the trace precursor)
# while new_exp_0_gemma_si3f8 has 0.05 (the real Heb2.0 rule) but fewer
# seeds. Both are printed so the choice is explicit rather than inherited.
ARM_SETS = {
    "trace": [
        ("Qwen3.5-2B", "LLM-2B+Heb2.0"),
        ("Qwen3.5-9B", "LLM-9B+Heb2.0"),
        ("Gemma-E4B", "Gemma-E4B+Heb2.0"),
    ],
    "inst": [
        ("Qwen3.5-2B", "LLM-2B+Heb"),
        ("Qwen3.5-9B", "LLM-9B+Heb"),
        ("Gemma-E4B", "Gemma-E4B+Heb"),
    ],
}

# Extra directories not in the registry, addressed as "<root>::<dir>".
EXTRA_DIRS = {
    "Gemma-E4B si3f8": (
        "runs/compute/pareto_social_3f", "new_exp_0_gemma_si3f8"),
}


def pair_records(seed_dir: Path) -> list:
    """[(W_ij, {plane: count})] over every ordered pair of every episode."""
    out = []
    for ep_dir in sorted((seed_dir / "episodes").glob("ep_*")):
        path = ep_dir / "episode_summary.json"
        if not path.exists():
            continue
        try:
            coop = json.loads(path.read_text(encoding="utf-8")).get(
                "cooperation_metrics") or {}
        except (json.JSONDecodeError, OSError):
            continue
        W = coop.get("hebbian_W")
        tensor = coop.get("pair_interaction") or {}
        if not W:
            continue
        n = len(W)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                planes = {}
                for plane, _lbl in PLANES:
                    mat = tensor.get(plane)
                    if mat and i < len(mat) and j < len(mat[i]):
                        planes[plane] = float(mat[i][j])
                out.append((float(W[i][j]), planes))
    return out


def condition_stats(arm_dir: Path) -> dict:
    """Pooled and per-seed rho for one condition directory."""
    pooled, per_seed = [], []
    seeds = sorted(d for d in arm_dir.glob("seed_*") if d.is_dir())
    for seed_dir in seeds:
        recs = pair_records(seed_dir)
        if not recs:
            continue
        pooled.extend(recs)
        row = {}
        for plane, _lbl in PLANES:
            xs = [w for w, pl in recs if plane in pl]
            ys = [pl[plane] for _w, pl in recs if plane in pl]
            row[plane] = _spearman(xs, ys)
        per_seed.append(row)

    out = {"n": len(pooled), "n_seeds": len(per_seed)}
    for plane, _lbl in PLANES:
        xs = [w for w, pl in pooled if plane in pl]
        ys = [pl[plane] for _w, pl in pooled if plane in pl]
        out[plane] = {
            "pooled": _spearman(xs, ys),
            "seed_vals": [r[plane] for r in per_seed if r[plane] is not None],
        }
    return out


def fmt(stats: dict, plane: str) -> str:
    s = stats[plane]
    if s["pooled"] is None:
        return "--"
    vals = s["seed_vals"]
    if len(vals) >= 2:
        tail = " ($%+.2f \\pm %.2f$)" % (statistics.mean(vals),
                                         statistics.stdev(vals))
    elif vals:
        tail = " ($%+.2f$)" % vals[0]
    else:
        tail = ""
    return "$%+.2f$%s" % (s["pooled"], tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", choices=sorted(ARM_SETS), default="trace")
    ap.add_argument("--extra", action="store_true",
                    help="also print the alternative Gemma plasticity arm")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    registry = {n: (d, r) for n, d, r, *_ in
                list(mft.ROWS) + list(NEW_ROWS) + list(EXTRA_ROWS)}
    repo = Path(__file__).resolve().parent.parent

    targets = []
    for label, key in ARM_SETS[args.arms]:
        if key not in registry:
            print("  [skip] %s: %s not in registry" % (label, key),
                  file=sys.stderr)
            continue
        d, root = registry[key]
        targets.append((label, Path(root) / d))
    if args.extra:
        for label, (root, d) in EXTRA_DIRS.items():
            targets.append((label, repo / root / d))

    rows = []
    for label, arm_dir in targets:
        if not arm_dir.exists():
            print("  [skip] %s: %s missing" % (label, arm_dir),
                  file=sys.stderr)
            continue
        st = condition_stats(arm_dir)
        if not st["n"]:
            print("  [skip] %s: no pair records" % label, file=sys.stderr)
            continue
        rows.append((label, arm_dir.name, st))

    L = [r"\begin{tabular}{l c ccc}", r"\toprule",
         r"Condition & $n$ & Messages & Joint digging & Proximity \\",
         r"\midrule"]
    for label, _dirname, st in rows:
        L.append("%-11s & %d & %s & %s & %s \\\\" % (
            label, st["n"], fmt(st, "messages"),
            fmt(st, "joint_dig"), fmt(st, "proximity")))
    L += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(L) + "\n"
    print(tex)

    print("%% source directories and seed counts:", file=sys.stderr)
    for label, dirname, st in rows:
        print("%%   %-16s %-34s %d seeds, %d pair-episodes"
              % (label, dirname, st["n_seeds"], st["n"]), file=sys.stderr)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "bond_behaviour_rho_rows.tex").write_text(
            tex, encoding="utf-8")
        print("wrote %s/bond_behaviour_rho_rows.tex" % args.out,
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
