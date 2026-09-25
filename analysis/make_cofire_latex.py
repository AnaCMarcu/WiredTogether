#!/usr/bin/env python3
r"""make_cofire_latex.py - the Experiment-2 co-firing table in LaTeX.

cofire_table.py already computes every number and can emit them as CSV; this
only formats them, so the two can never disagree. It shells out to that script
rather than importing it, leaving the working script untouched.

Arms with several cues span their aggregate cells with \multirow, so the
preamble needs \usepackage{multirow} (plus booktabs, adjustbox and bm, as in
tab:final_comparison).

Bold marks the best arm in each aggregate column. Mean W is deliberately NOT
bolded: more bond is not better, and in this suite the arm with the most bond
mass and the arm with the least both beat the ones in between.

Usage:
  python analysis/make_cofire_latex.py <runs-root> <assets-dir> [--out FILE]
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

CUE_LABEL = {"comm": "Comm", "obs": "Obs", "imit": "Imit"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_root")
    ap.add_argument("assets_dir")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="tab:cofire")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    here = Path(__file__).resolve().parent
    proc = subprocess.run(
        [sys.executable, str(here / "cofire_table.py"),
         args.runs_root, args.assets_dir, "--csv"],
        capture_output=True, text=True, encoding="utf-8", check=True)
    rows = list(csv.DictReader(io.StringIO(proc.stdout)))

    # group consecutive rows by condition, preserving the ARMS order
    arms, order = {}, []
    for r in rows:
        c = r["condition"]
        if c not in arms:
            arms[c] = []
            order.append(c)
        arms[c].append(r)

    def f(v, p=1):
        return "--" if v in ("", None) else "%.*f" % (p, float(v))

    best = {k: max(order, key=lambda c: float(arms[c][0][col]))
            for k, col in (("task", "task"), ("allms", "allms_pct"),
                           ("coop", "coop_pct"))}

    L = []
    for c in order:
        rs = arms[c]
        n = len(rs)
        a = rs[0]

        def agg(val, sd, is_best, p=1):
            v = f(val, p)
            if is_best:
                v = "\\mathbf{%s}" % v
            return "$%s$ \\pmm{%s}" % (v, f(sd, p))

        def span(s):
            return s if n == 1 else "\\multirow{%d}{*}{%s}" % (n, s)

        head = [span(c),
                span(agg(a["task"], a["task_sd"], c == best["task"], 0)),
                span(agg(a["allms_pct"], a["allms_pct_sd"],
                         c == best["allms"])),
                span(agg(a["coop_pct"], a["coop_pct_sd"], c == best["coop"])),
                span("$%s$" % f(a["meanW"], 2))]

        for i, r_ in enumerate(rs):
            if r_["cue"]:
                tail = [CUE_LABEL.get(r_["cue"], r_["cue"]),
                        "$%s$ \\pmm{%s}" % (f(r_["act_use_pct"]),
                                            f(r_["act_use_sd"])),
                        "$%s$ \\pmm{%s}" % (f(r_["dW_pct"]), f(r_["dW_sd"])),
                        ("$%s$ \\pmm{%s}" % (f(r_["rho"], 2),
                                             f(r_["rho_sd"], 2))
                         if r_["rho"] else "--")]
            else:
                tail = ["--"] * 4
            L.append(" & ".join((head if i == 0 else [""] * 5) + tail)
                     + r" \\")
        if c != order[-1]:
            L.append(r"\midrule")
    body = "\n".join(L)

    head = r"""\begin{table}[t]
\centering
\caption{\textbf{Experiment 2: which social signal the bond update sees.}
Mean~$\pm$~std over 3 seeds $\times$ 3 episodes. \emph{Comm (forced)} is the
pre-experiment mechanism: communication happens as part of ordinary
deliberation, with no act to choose, no per-act reward and no co-firing credit
mask; observation and imitation do not exist in that mode. \emph{Comm
(choice)} offers the same single channel through the Experiment-2 apparatus,
so the two arms differ only in that apparatus. Act use~\% is the share of
social acts of that cue; $\Delta W$~\% the share of total bond growth
attributed to it; $\rho$ the Spearman correlation over the 6 directed pairs
$\times$ 3 episodes between cue acts $i\!\to\!j$ in an episode and $W_{ij}$ at
its end. Bold marks the best arm per aggregate column; mean $W$ is not bolded,
as more bond is not better. \textbf{Finding:} making communication a chosen,
rewarded act costs $118$ task return and $9.1$ cooperative points relative to
leaving it forced, and every arm containing chosen communication falls below
the no-cue null. Imitation is used least, carries the least attributed bond
growth, and is crowded out entirely when communication is available.}
\label{__LABEL__}

\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.2}

\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l ccc c l ccc}
\toprule
& \makecell{\textbf{Task}\\\textbf{return} $\uparrow$}
& \makecell{\textbf{Milest.}\\\textbf{\%} $\uparrow$}
& \makecell{\textbf{Coop.}\\\textbf{\%} $\uparrow$}
& \makecell{\textbf{mean}\\$\bm{W}$}
& \textbf{Cue}
& \makecell{\textbf{Act}\\\textbf{use \%}}
& \makecell{$\bm{\Delta W}$\\\textbf{\%}}
& $\bm{\rho}$ \\
\midrule
"""
    head = head.replace("__LABEL__", args.label)
    tail = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    tex = head + body + "\n" + tail
    print(tex)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tex, encoding="utf-8")
        print("wrote %s" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
