#!/usr/bin/env python3
r"""make_steps_table_pct.py - tab:steps_to_milestone, percentage variant.

Same table as make_steps_table_latex.py with three differences:

1. The ``+inst.`` (single-timescale) arms are dropped everywhere — the paper
   keeps only the base and ``+trace`` conditions.
2. The RL panel is included alongside the zero-shot panel, so the table
   covers every condition of tab:final_comparison that has runs.
3. The subscript is the PERCENTAGE of episodes that completed the milestone,
   not the raw count. Columns pool different numbers of seeds, so a raw
   count of 9 in a 9-episode column and 9 in an 18-episode column mean
   opposite things; a percentage is comparable across columns by
   construction.

Column definitions still come from make_final_table_latex.PANELS, so this
cannot disagree with the main table about which conditions exist.

Usage:
  python analysis/make_steps_table_pct.py --out paper_assets/final_ext
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paths import ASSETS  # noqa: F401  (also puts siblings on sys.path)

import make_final_table as mft
import make_results as MR
from make_final_table_extended import NEW_ROWS
from make_final_table_latex import EXTRA_ROWS, PANELS
from make_steps_table_latex import ROWS_META, SECTIONS

# Variant labels dropped from the table (the single-timescale ablation).
DROP_LABELS = ("+inst.",)

# Printed variant names, matching tab:final_comparison's row labels.
VARIANT_LABEL = {
    "+trace": r"$+$Social plasticity",
    "+Central Orch.": r"$+$Central orch.",
}


def column_specs():
    """[(key, printed header)] for every kept condition, in panel order.

    The header is the block's model name plus the variant suffix, so a
    rotated column head reads "Qwen3.5-2B IPPO +trace" rather than the bare
    "+trace" the main table can afford inside a grouped block.
    """
    out = []
    for _panel, blocks in PANELS:
        for blk in blocks:
            model = blk[0][1].strip()
            for _tag, lbl, key in blk:
                if key is None:
                    continue
                variant = lbl.replace(r"\ \ ", "").strip()
                if any(d in variant for d in DROP_LABELS):
                    continue
                head = (model if variant == model
                        else "%s %s" % (model, VARIANT_LABEL[variant]))
                out.append((key, head))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    registry = {n: (d, r) for n, d, r, *_ in
                list(mft.ROWS) + list(NEW_ROWS) + list(EXTRA_ROWS)}

    cols, meta = [], {}
    for key, head in column_specs():
        if key not in registry:
            print("  [skip] %s: not in registry" % key, file=sys.stderr)
            continue
        d, root = registry[key]
        runs = MR.load_runs(Path(root), d, frozenset())
        if not runs:
            print("  [skip] %s: no runs" % key, file=sys.stderr)
            continue
        a = MR.aggregate(runs)
        cols.append((key, head, a["steps"], a["n_eps"]))
        meta[key] = (a["n_runs"], a["n_eps"])

    L = [r"\begin{tabular}{l l *{%d}{c}}" % len(cols), r"\toprule",
         "ID & Milestone"]
    for _key, head, _steps, n_eps in cols:
        L.append(r" & \rotatebox{90}{%s ($n{=}%d$)~}" % (head, n_eps))
    L.append(r" \\")
    L.append(r"\midrule")
    ncol = len(cols) + 2
    for heading, mids in SECTIONS:
        L.append(r"\multicolumn{%d}{l}{\emph{%s}}\\" % (ncol, heading))
        for mid in mids:
            pid, name = ROWS_META[mid]
            cells = []
            for _key, _head, steps, n_eps in cols:
                med, n = steps.get(mid, (None, 0))
                if med is None or not n_eps:
                    cells.append("--")
                else:
                    pct = round(100.0 * n / n_eps)
                    cells.append(r"$%d_{%d\%%}$" % (round(med), pct))
            L.append("%-12s & %-22s & %s \\\\" % (pid, name, " & ".join(cells)))
        if (heading, mids) != SECTIONS[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(L) + "\n"
    print(tex)

    print("%% seeds x episodes per column:", file=sys.stderr)
    for key, head, _steps, _n in cols:
        print("%%   %-28s %d seeds, %d episodes" % (key, *meta[key]),
              file=sys.stderr)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "steps_to_milestone_pct_rows.tex").write_text(
            tex, encoding="utf-8")
        print("wrote %s/steps_to_milestone_pct_rows.tex" % args.out,
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
