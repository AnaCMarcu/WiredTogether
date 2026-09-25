#!/usr/bin/env python3
"""make_final_table_latex.py — tab:final_comparison in the paper's own format.

make_final_table_extended.py emits a markdown table and a flat table_rows.tex.
The paper's table is a two-panel side-by-side float whose formatting has to be
COMPUTED, not typed:

  bold     best within each block (a baseline and the couplings read against it)
  dagger   seed count below the six-seed target
  ---      a configuration that has not been run yet

Hand-maintaining that across 22 rows is how the 3-seed MAPPO+Heb row ended up
bolded as best-in-family after its extra seeds moved it to last.

Steps-to-first-completion moved to its own float (tab:steps_first_completion),
so this table carries only the three aggregate columns.

Coupling rows are named for the update rule, not the run directory:
  +plast.          eligibility-trace (three-factor) rule + signed death LTD
  +plast. (inst.)  the instantaneous (reward-modulated) rule
The RL panel's couplings additionally carry weight-gated experience sharing
(rho=0.3); see make_final_table.RL_HEB_ARMS.

Usage:
  python analysis/make_final_table_latex.py --out paper_assets/final_ext
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paths import group  # noqa: F401  (also puts siblings on sys.path)

import make_final_table as mft
from make_final_table_extended import NEW_ROWS

SEED_TARGET = 6

# The four RL trace-rule arms landed after this table was first written. They
# live in their own run groups and reuse exp36/exp37 dir names across the two
# backbones, so they cannot be addressed by dir name alone.
SR3F_QWEN = group("social_replay_3f_qwen")
SR3F_GEMMA = group("social_replay_3f_gemma4")
EXTRA_ROWS = [
    ("IPPO+plast", "exp37_ippo_hebbian_replay_3f", SR3F_QWEN,
     "Qwen3.5-2B", "IPPO (LoRA)", "trace rule + replay"),
    ("MAPPO+plast", "exp36_mappo_hebbian_replay_3f", SR3F_QWEN,
     "Qwen3.5-2B", "MAPPO (shared critic)", "trace rule + replay"),
    ("Gemma IPPO+plast", "exp37_ippo_hebbian_replay_3f", SR3F_GEMMA,
     "Gemma-4-E4B", "IPPO (LoRA)", "trace rule + replay"),
    ("Gemma MAPPO+plast", "exp36_mappo_hebbian_replay_3f", SR3F_GEMMA,
     "Gemma-4-E4B", "MAPPO (shared critic)", "trace rule + replay"),
]

# Registry key None = the run does not exist yet; the row prints "---" and a
# TODO comment. `note` is appended to the row's LaTeX comment.
PENDING = {
    "(l)": "trace rule on Qwen3.5-2B IPPO pending (exp37, social_replay_3f_qwen)",
    "(o)": "trace rule on Qwen3.5-2B MAPPO pending (exp36, social_replay_3f_qwen)",
    "(r)": "trace rule on Gemma-E4B IPPO pending (exp37, social_replay_3f_gemma4)",
    "(u)": "trace rule on Gemma-E4B MAPPO pending (exp36, social_replay_3f_gemma4)",
}

# (panel, [block, ...]); block = [(tag, printed label, registry key), ...].
# One block = one bold group.
PANELS = [
    ("Zero-shot VLM agents", [
        [("(a)", r"Qwen3.5-2B",                 "LLM-2B"),
         ("(b)", r"\ \ +trace",                "LLM-2B+Heb2.0"),
         ("(c)", r"\ \ +inst.",       "LLM-2B+Heb")],
        [("(d)", r"Qwen3.5-9B",                 "LLM-9B"),
         ("(e)", r"\ \ +trace",                "LLM-9B+Heb2.0"),
         ("(f)", r"\ \ +inst.",       "LLM-9B+Heb")],
        [("(g)", r"Gemma-E4B",                  "Gemma-E4B"),
         ("(h)", r"\ \ +Central Orch.",         "Gemma-E4B+Central Orch."),
         ("(i)", r"\ \ +trace",                "Gemma-E4B+Heb2.0"),
         ("(j)", r"\ \ +inst.",       "Gemma-E4B+Heb")],
    ]),
    ("RL fine-tuned agents", [
        [("(k)", r"Qwen3.5-2B IPPO",            "IPPO"),
         ("(l)", r"\ \ +trace",                "IPPO+plast"),
         ("(m)", r"\ \ +inst.",       "IPPO+Heb")],
        [("(n)", r"Qwen3.5-2B MAPPO",           "MAPPO"),
         ("(o)", r"\ \ +trace",                "MAPPO+plast"),
         ("(p)", r"\ \ +inst.",       "MAPPO+Heb")],
        [("(q)", r"Gemma-E4B IPPO",             "Gemma IPPO"),
         ("(r)", r"\ \ +trace",                "Gemma IPPO+plast"),
         ("(s)", r"\ \ +inst.",       "Gemma IPPO+Heb+SR")],
        [("(t)", r"Gemma-E4B MAPPO",            "Gemma MAPPO"),
         ("(u)", r"\ \ +trace",                "Gemma MAPPO+plast"),
         ("(v)", r"\ \ +inst.",       "Gemma MAPPO+Heb+SR")],
    ]),
]

CAPTION = r"""\textbf{RQ1: Effect of relational plasticity on cooperative task
performance.} Values are mean~$\pm$~std over pooled episodes across completed
seeds (three episodes per seed). $\dagger$ marks conditions below the
six-seed target, whose additional seeds are in progress; $\ddagger$ marks a
single-seed run, reported for completeness only and excluded from the
analysis. \textbf{+trace} is the eligibility-trace rule; \textbf{+inst.}\ is
the single-timescale ablation, which also differs in its learning and decay
rates and carries no death-LTD term. In the RL panel both couplings
additionally use weight-gated experience sharing ($\rho=0.3$).
``\,---\,'' marks a configuration that has not yet been run. Steps to first
completion for the zero-shot conditions are reported in
\cref{tab:steps_to_milestone}."""


def agg_cols(r):
    """The three aggregate columns as (sort key, rendered string)."""
    return [(r["task"][0],     r"$%.0f$ \pmm{%.0f}" % r["task"]),
            (r["ms_pct"][0],   r"$%.1f$ \pmm{%.1f}" % r["ms_pct"]),
            (r["coop_pct"][0], r"$%.1f$ \pmm{%.1f}" % r["coop_pct"])]


NCOL = 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    mft.ROWS = list(mft.ROWS) + NEW_ROWS + EXTRA_ROWS
    rows, _ = mft.collect()
    by = {r["name"]: r for r in rows}

    cells, seeds, missing = {}, {}, []
    for _panel, blocks in PANELS:
        for blk in blocks:
            for tag, lbl, key in blk:
                if key is None:
                    continue
                r = by.get(key)
                if r is None:
                    missing.append("%s %s (key %r)" % (tag, lbl, key))
                    continue
                cells[tag] = agg_cols(r)
                seeds[tag] = r["n_runs"]

    if missing:
        print("  [warn] no runs for: " + "; ".join(missing), file=sys.stderr)

    bold = set()
    for _panel, blocks in PANELS:
        for blk in blocks:
            tags = [t for t, _l, _k in blk if t in cells]
            for c in range(NCOL):
                best = max(tags, key=lambda t: cells[t][c][0])
                bold.add((best, c))

    def fmt(tag, c):
        val, sd = cells[tag][c][1].split(r" \pmm")
        val = val.strip("$")
        if (tag, c) in bold:
            val = r"\mathbf{%s}" % val
        return r"$%s$ \pmm%s" % (val, sd)

    panels = []
    for pi, (panel, blocks) in enumerate(PANELS):
        L = [r"\begin{minipage}[t]{0.49\textwidth}",
             r"\centering",
             r"{\footnotesize\textbf{(%s) %s}}\\[2pt]" % ("ab"[pi], panel),
             r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{l c cc}",
             r"\toprule",
             r"\textbf{Condition}",
             r"& \makecell{\textbf{Task}\\\textbf{return} $\uparrow$}",
             r"& \makecell{\textbf{Milest.}\\\textbf{\%} $\uparrow$}",
             r"& \makecell{\textbf{Coop.}\\\textbf{\%} $\uparrow$} \\",
             r"\midrule"]
        for bi, blk in enumerate(blocks):
            if bi:
                L.append(r"\midrule")
            for tag, lbl, key in blk:
                if tag not in cells:
                    L.append("%s %s" % (tag, lbl))
                    L.append(r"& --- & --- & --- \\ % TODO: "
                             + PENDING.get(tag, "no runs"))
                    continue
                if seeds[tag] <= 1:
                    dag = r"$^{\ddagger}$"   # single seed: reported, not analysed
                elif seeds[tag] < SEED_TARGET:
                    dag = r"$^{\dagger}$"
                else:
                    dag = ""
                L.append("%s %s%s" % (tag, lbl, dag))
                L.append("& " + " & ".join(fmt(tag, c) for c in range(NCOL))
                         + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\end{minipage}"]
        panels.append("\n".join(L))

    rows_tex = ("\n\\hfill\n".join(panels))
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{%s}" % CAPTION,
        r"\label{tab:final_comparison}", "",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}", "",
        rows_tex, "",
        r"\end{table}"])

    print(tex)
    print("\n%% seeds per row: " + ", ".join(
        "%s=%d" % (t, seeds[t]) for _p, bs in PANELS for b in bs
        for t, _l, _k in b if t in seeds), file=sys.stderr)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "final_comparison_rows.tex").write_text(rows_tex + "\n",
                                                            encoding="utf-8")
        (args.out / "final_comparison.tex").write_text(tex + "\n",
                                                       encoding="utf-8")
        print("wrote %s" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
