#!/usr/bin/env python3
"""make_beliefs_3f_view.py - one beliefs.csv for the Hebbian 2.0 perception figure.

make_pareto_perception_fig.py keys belief rows by run directory and maps those
to (size, arm). Two things stop it reading the three-factor arms directly:

  1. The same directory name means different rules in different groups.
     pareto_e2b_hebbian is reward_modulated under pareto_gemma4 and
     three_factor under pareto_gemma4_3f, and load_beliefs pools rows by key,
     so passing both belief files silently averages the two rules.
  2. The Hebbian 2.0 arms for E4B and Qwen are not named pareto_<size>_<arm>
     at all - they are new_exp_0_gemma_si3f8, exp34_* and exp35_* - so
     exp_to_size_arm does not recognise them and drops them without a word.

This writes a single merged file: baseline rows from the existing belief
tables (the base arms are rule-independent, so they need no re-parse) and
Hebbian rows from the three-factor tables only, every row rewritten to the
canonical pareto_<size>_<arm> key. Pass the result as the figure's sole
--beliefs argument and no pooling is possible.

Inputs are produced by analysis/qualitative/run_3f.py (parse + metrics).

Usage:
  python analysis/make_beliefs_3f_view.py --out paper_assets/perception_3f/beliefs_3f.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

Q = Path("analysis/qualitative")

# canonical key -> (belief table, run directory inside it)
SOURCES = {
    # base arms: rule-independent, reuse the tables that already exist
    "pareto_e2b_base":      (Q / "out_pareto/tables/beliefs.csv", "pareto_e2b_base"),
    "pareto_12b_base":      (Q / "out_pareto/tables/beliefs.csv", "pareto_12b_base"),
    "pareto_e4b_base":      (Q / "out_gemma/tables/beliefs.csv", "new_exp_0_gemma_base"),
    "pareto_qwen2b_base":   (Q / "out/tables/beliefs.csv", "exp01_llm_2b"),
    "pareto_qwen9b_base":   (Q / "out/tables/beliefs.csv", "exp02_llm_9b"),
    # Hebbian 2.0 arms: three_factor + signed death LTD, from run_3f.py
    "pareto_e2b_hebbian":   (Q / "out_3f_pareto/tables/beliefs.csv", "pareto_e2b_hebbian"),
    "pareto_12b_hebbian":   (Q / "out_3f_pareto/tables/beliefs.csv", "pareto_12b_hebbian"),
    # E4B: the results tables' Gemma-E4B+Heb2.0 arm. See --e4b-arm below.
    "pareto_e4b_hebbian":   (Q / "out_heb3f/tables/beliefs.csv", "new_exp_0_gemma_hebbian3f"),
    "pareto_qwen2b_hebbian": (Q / "out_3f_q2b/tables/beliefs.csv", "exp34_llm_2b_three_factor_fdecay"),
    "pareto_qwen9b_hebbian": (Q / "out_3f_q9b/tables/beliefs.csv", "exp35_llm_9b_three_factor_fdecay"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("paper_assets/perception_3f/beliefs_3f.csv"))
    # Which directory is the Gemma E4B coupled arm. "hebbian3f" is what
    # make_final_table_extended calls Gemma-E4B+Heb2.0 (6 seeds,
    # hebbian_death_ltd=0); "si3f8" is the social-interval anchor
    # (3 seeds, death_ltd=0.05). They are different runs, so the figures and
    # the results tables have to name the same one.
    ap.add_argument("--e4b-arm", choices=("hebbian3f", "si3f8"),
                    default="hebbian3f")
    args = ap.parse_args()
    if args.e4b_arm == "si3f8":
        SOURCES["pareto_e4b_hebbian"] = (
            Q / "out_3f_social/tables/beliefs.csv", "new_exp_0_gemma_si3f8")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cache, out_rows, fields = {}, [], None
    for key, (path, run_dir) in SOURCES.items():
        if path not in cache:
            if not path.is_file():
                print("  MISSING %s (run analysis/qualitative/run_3f.py first)"
                      % path, file=sys.stderr)
                cache[path] = []
            else:
                with open(path, newline="", encoding="utf-8") as fh:
                    rd = csv.DictReader(fh)
                    cache[path] = list(rd)
                    fields = fields or rd.fieldnames
        hits = [r for r in cache[path] if r["run"].split("/")[0] == run_dir]
        if not hits:
            print("  no rows for %s in %s" % (run_dir, path), file=sys.stderr)
            continue
        for r in hits:
            r = dict(r)
            r["run"] = "%s/%s" % (key, r["run"].split("/", 1)[1])
            out_rows.append(r)
        print("  %-24s <- %-34s %d rows" % (key, run_dir, len(hits)))

    if not out_rows:
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print("wrote %s (%d rows)" % (args.out, len(out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
