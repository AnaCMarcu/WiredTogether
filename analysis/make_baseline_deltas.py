#!/usr/bin/env python3
"""make_baseline_deltas.py — every coupling against its own baseline.

Raw rows in tab:final_comparison are hard to read because the arms have
different seed counts (2, 3, 4 and 6 across the table right now). A coupling
measured on seeds {42,123,456} compared against a baseline pooled over
{42,...,1213} mixes the treatment effect with which seeds happened to finish.

For every (baseline, coupling) pair this prints BOTH:

  all      each arm on all the seeds it has        (what Table 2 shows)
  matched  both arms restricted to the seeds they SHARE

The matched column is the one to quote. Where the two disagree, the gap is
seed composition, not the coupling.

Metrics are make_results' own (percentage denominators 25 / 17, decomposed
task return excluding hebbian_diffuse), so nothing can drift from the paper.

Usage:
  python analysis/make_baseline_deltas.py
  python analysis/make_baseline_deltas.py --out paper_assets/final_ext
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from paths import group  # noqa: F401  (also puts siblings on sys.path)

import make_final_table as mft
from make_results import COOP_MAX, NONCOMM_MAX, aggregate, load_runs
from make_final_table_extended import NEW_ROWS, DISPLAY

# label -> (dir, runs-root), from both registries.
ARMS = {name: (d, root) for name, d, root, *_ in list(mft.ROWS) + NEW_ROWS}

# (block title, baseline label, [coupling labels...])
COMPARISONS = [
    ("Qwen3.5-2B, frozen LLM", "LLM-2B",
     ["LLM-2B+Heb", "LLM-2B+3f-noLTD", "LLM-2B+Heb2.0"]),
    ("Qwen3.5-9B, frozen LLM", "LLM-9B",
     ["LLM-9B+Heb", "LLM-9B+3f-noLTD", "LLM-9B+Heb2.0"]),
    ("Gemma-E4B, frozen LLM", "Gemma-E4B",
     ["Gemma-E4B+Central Orch.", "Gemma-E4B+Heb",
      "Gemma-E4B+3f-noLTD", "Gemma-E4B+Heb2.0"]),
    ("Qwen3.5-2B, IPPO", "IPPO", ["IPPO+Heb", "IPPO+Heb+SR"]),
    ("Qwen3.5-2B, MAPPO", "MAPPO", ["MAPPO+Heb", "MAPPO+Heb+SR"]),
    ("Gemma-E4B, IPPO", "Gemma IPPO", ["Gemma IPPO+Heb", "Gemma IPPO+Heb+SR"]),
    ("Gemma-E4B, MAPPO", "Gemma MAPPO", ["Gemma MAPPO+Heb", "Gemma MAPPO+Heb+SR"]),
]


def seeds_of(label):
    d, root = ARMS[label]
    p = Path(root) / d
    if not p.is_dir():
        return set()
    return {f.parent.name[5:] for f in p.glob("seed_*/final_metrics.json")
            if re.fullmatch(r"seed_\d+", f.parent.name)}


def stats(label, keep=None):
    """Aggregate one arm, optionally restricted to a seed set."""
    d, root = ARMS[label]
    excl = frozenset()
    if keep is not None:
        excl = frozenset("%s/seed_%s" % (d, s) for s in seeds_of(label) - keep)
    runs = load_runs(Path(root), d, excl)
    if not runs:
        return None
    a = aggregate(runs)
    return {"task": a["task"][0],
            "ms": 100.0 * a["allms_nc"][0] / NONCOMM_MAX,
            "coop": 100.0 * a["coop"][0] / COOP_MAX,
            "n": a["n_runs"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    L = ["# Couplings vs their own baseline", "",
         "`all` = each arm on every seed it has. `matched` = both arms on the",
         "seeds they share — quote this one. Delta is coupling minus baseline;",
         "positive is better for all three metrics.", ""]

    for title, base, variants in COMPARISONS:
        b_all = stats(base)
        if b_all is None:
            L += ["## %s" % title, "", "*(baseline has no runs)*", ""]
            continue
        L += ["## %s" % title, "",
              "baseline `%s`: task %.0f, milest %.1f%%, coop %.1f%% (n=%d)"
              % (DISPLAY.get(base, base), b_all["task"], b_all["ms"],
                 b_all["coop"], b_all["n"]),
              "",
              "| coupling | basis | n | Δ task | Δ milest % | Δ coop % |",
              "|---|---|---|---|---|---|"]
        b_seeds = seeds_of(base)
        for v in variants:
            v_all = stats(v)
            if v_all is None:
                L.append("| %s | — | 0 | *(no runs)* | | |" % DISPLAY.get(v, v))
                continue
            L.append("| %s | all | %d | %+.0f | %+.1f | %+.1f |"
                     % (DISPLAY.get(v, v), v_all["n"],
                        v_all["task"] - b_all["task"],
                        v_all["ms"] - b_all["ms"],
                        v_all["coop"] - b_all["coop"]))
            shared = b_seeds & seeds_of(v)
            if shared and shared != b_seeds:
                bm, vm = stats(base, shared), stats(v, shared)
                if bm and vm:
                    L.append("| | **matched** (%s) | %d | **%+.0f** | **%+.1f** | **%+.1f** |"
                             % (",".join(sorted(shared, key=int)), vm["n"],
                                vm["task"] - bm["task"],
                                vm["ms"] - bm["ms"],
                                vm["coop"] - bm["coop"]))
        L.append("")

    txt = "\n".join(L)
    print(txt)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "baseline_deltas.md").write_text(txt + "\n", encoding="utf-8")
        print("\nwrote %s/baseline_deltas.md" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
