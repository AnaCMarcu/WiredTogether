#!/usr/bin/env python3
"""Aggregate the qualitative per-seed CSVs into paper tables.

Reads the per-seed outputs of the qualitative pipeline
(``analysis_qualitative/out/tables/``) and emits, per condition, the
mean +/- std ACROSS SEEDS of each metric, matching ``make_results.py``:
population std (``statistics.pstdev``), missing cells skipped per metric.

Sources (all per-seed, one row per (condition, seed)):
  social_interpretability.csv  -> Q-A  (coupling: read / ask / faithful / obey)
  bonds_behavior_runs.csv      -> rho  (W vs message / joint-dig / proximity)
  beliefs.csv                  -> Q-C  (perception grounding, partner-loc, contagion)

Outputs (into --out, default paper_assets_medium_ext/):
  qual_aggregated.csv   long form: condition, metric, mean, std, n_seeds
  qual_table_rows.tex   two paste-ready table environments:
                          tab:social_interp   (Q-A + rho)
                          tab:belief_quality   (Q-C)

Usage:
  python analysis_qualitative/make_qual_tables.py
  python analysis_qualitative/make_qual_tables.py --tables DIR --out DIR
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ── metric registry: key in CSV -> (short name, decimals, signed) ──────────
# 'signed' prints a leading +/- (used for correlations that can be negative).
Q_A = [
    ("bond_ref_coverage", "coverage", 2, False),
    ("ask_rate",          "ask",      2, False),
    ("ask_msg_adoption",  "adopt",    2, False),
    ("faith_mae",         "faith_mae", 3, False),
    ("faith_within_005",  "faith_w05", 2, False),
    ("follow_window",     "follow",   2, False),
    ("follow_baseline",   "follow_base", 2, False),
    ("argmax_consistency", "argmax",  2, False),
    ("respond_follow",    "respond",  2, False),
]
RHO = [
    ("rho_messages",   "rho_msg",  2, True),
    ("rho_joint_dig",  "rho_dig",  2, True),
    ("rho_proximity",  "rho_prox", 2, True),
]
Q_C = [
    ("perception_grounding_rate", "ground",   2, False),
    ("partner_loc_accuracy",      "ploc",     2, False),
    ("contagion_rate",            "contag",   2, False),
    ("belief_responsiveness",     "bresp",    2, False),
]

# condition display order
ORDER = ["LLM-2B", "LLM-9B", "LLM-2B+Heb", "LLM-9B+Heb",
         "IPPO", "MAPPO", "IPPO+Heb", "MAPPO+Heb",
         "No-bonds", "Allied-pair", "Allied-all"]


def _num(x):
    """float(x) or None for blank / nan / None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict], specs, exclude: frozenset = frozenset()) -> dict:
    """{label: {short: (mean, std, n)}} over seeds, skipping missing/quarantined."""
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        if str(r.get("quarantined", "")).strip().lower() == "true":
            continue
        if str(r.get("run", "")).strip() in exclude:
            continue
        by_label.setdefault(r["label"], []).append(r)
    out: dict[str, dict] = {}
    for label, seeds in by_label.items():
        stats = {}
        for key, short, _dp, _sg in specs:
            vals = [v for v in (_num(r.get(key)) for r in seeds) if v is not None]
            if vals:
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                stats[short] = (statistics.mean(vals), std, len(vals))
        out[label] = stats
    return out


def merge(*dicts) -> dict:
    out: dict[str, dict] = {}
    for d in dicts:
        for label, stats in d.items():
            out.setdefault(label, {}).update(stats)
    return out


def fmt(stat, dp, signed) -> str:
    if stat is None:
        return "--"
    m, s, _n = stat
    mstr = f"{m:+.{dp}f}" if signed else f"{m:.{dp}f}"
    return f"${mstr}$ \\pmm{{{s:.{dp}f}}}"


def table(agg, rows, cols, label, caption) -> str:
    """cols = list of (short, dp, signed); rows = list of condition labels."""
    colspec = "l" + "c" * len(cols)
    head = " & ".join(["Condition"] + [c[0].replace("_", "\\_") for c in cols])
    lines = [
        "\\begin{table*}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
        head + " \\\\", "\\midrule",
    ]
    for cond in rows:
        st = agg.get(cond, {})
        cells = [fmt(st.get(short), dp, sg) for short, dp, sg in cols]
        lines.append(f"{cond} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=HERE / "out" / "tables",
                    help="dir with the per-seed qualitative CSVs")
    ap.add_argument("--out", type=Path, default=REPO / "paper_assets_medium_ext",
                    help="dir to write qual_aggregated.csv + qual_table_rows.tex")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="EXP/seed_N",
                    help="runs to leave out of the aggregation")
    args = ap.parse_args()
    exc = frozenset(args.exclude)

    social = aggregate(load(args.tables / "social_interpretability.csv"), Q_A, exc)
    rho = aggregate(load(args.tables / "bonds_behavior_runs.csv"), RHO, exc)
    beliefs = aggregate(load(args.tables / "beliefs.csv"), Q_C, exc)
    agg = merge(social, rho, beliefs)

    args.out.mkdir(parents=True, exist_ok=True)

    # transparency CSV (long form)
    all_short = [s for _k, s, _d, _g in (Q_A + RHO + Q_C)]
    with (args.out / "qual_aggregated.csv").open("w", newline="",
                                                 encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "mean", "std", "n_seeds"])
        for cond in ORDER:
            for short in all_short:
                st = agg.get(cond, {}).get(short)
                if st is not None:
                    w.writerow([cond, short, round(st[0], 4),
                                round(st[1], 4), st[2]])

    # Table 1: Q-A + rho (graph conditions)
    t1_rows = ["LLM-2B+Heb", "LLM-9B+Heb", "IPPO+Heb", "MAPPO+Heb",
               "No-bonds", "Allied-pair", "Allied-all"]
    t1_cols = [("coverage", 2, False), ("ask", 2, False),
               ("adopt", 2, False), ("faith_mae", 3, False),
               ("rho_msg", 2, True), ("rho_dig", 2, True),
               ("rho_prox", 2, True)]
    t1 = table(
        agg, t1_rows, t1_cols, "tab:social_interp",
        "\\textbf{Is the social graph used, and what does it track?} "
        "Mean~$\\pm$~std across seeds. \\emph{coverage}/\\emph{ask}: fraction "
        "of social-module deliberations that cite the bond row / ask whom to "
        "message. \\emph{adopt}: text similarity between the module's "
        "suggested message and the one actually sent. \\emph{faith\\_mae}: "
        "mean error of recited bond values. $\\rho$: mean per-seed Spearman "
        "of bond strength vs.\\ per-pair message / joint-dig / proximity "
        "counts. RL rows have no social module (Q-A `--'); No-bonds/"
        "Allied-all have constant $\\mathbf{W}$ (no $\\rho$), and their "
        "\\emph{faith\\_mae} is trivially zero.")

    # Table 2: Q-C (belief quality + transfer; scale contrast, all conditions)
    t2_rows = ["LLM-2B", "LLM-2B+Heb", "LLM-9B", "LLM-9B+Heb",
               "IPPO", "IPPO+Heb", "MAPPO", "MAPPO+Heb",
               "No-bonds", "Allied-pair", "Allied-all"]
    t2_cols = [("ground", 2, False), ("ploc", 2, False),
               ("bresp", 2, False), ("contag", 2, False)]
    t2 = table(
        agg, t2_rows, t2_cols, "tab:belief_quality",
        "\\textbf{Belief quality and transfer.} "
        "Mean~$\\pm$~std across seeds. \\emph{ground}: perception statements "
        "with no chamber-impossible object. \\emph{ploc}: partner-location "
        "claims matching the partner's true chamber. \\emph{bresp}: received "
        "messages after which the receiver's partner belief updates within "
        "two belief intervals sharing the message's content. \\emph{contag}: "
        "hallucinated perceptions followed within 5 steps by \\emph{the same "
        "agent's} reward-less dig (own bad belief, not transfer).")

    (args.out / "qual_table_rows.tex").write_text(
        "% Generated by analysis_qualitative/make_qual_tables.py\n"
        "% Requires \\usepackage{makecell} and \\pmm (see floats.tex).\n\n"
        + t1 + "\n" + t2, encoding="utf-8")

    print(f"wrote {args.out/'qual_aggregated.csv'}")
    print(f"wrote {args.out/'qual_table_rows.tex'}")
    # quick console preview of the headline cells
    for cond in ("LLM-2B", "LLM-9B", "LLM-2B+Heb", "LLM-9B+Heb",
                 "IPPO+Heb", "MAPPO+Heb"):
        s = agg.get(cond, {})
        def g(k):
            v = s.get(k)
            return "--" if v is None else f"{v[0]:.2f}"
        print(f"  {cond:12s} cover={g('coverage')} ask={g('ask')} "
              f"rho_msg={g('rho_msg')} rho_dig={g('rho_dig')} "
              f"ground={g('ground')} contag={g('contag')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
