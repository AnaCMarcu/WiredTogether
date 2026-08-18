#!/usr/bin/env python3
"""make_final_table.py — the cross-model comparison table.

Two tables, emitted as Markdown and as LaTeX rows:

  Table 1 (design)   what each experiment IS — base model, learning, social
                     coupling, seeds. The results table then only names the
                     experiment, so no condition needs explaining twice.
  Table 2 (results)  task performance + perception, one row per experiment.

Every milestone quantity is PERCENTAGE COMPLETION, never a count:
  * milestone completion  = non-comm milestones fired / 25, per episode
  * cooperative           = Ch2-Ch5 milestones fired / 17, per episode
Both are team unions within an episode, pooled over episodes and seeds —
the same convention as make_results.py, whose aggregate() this reuses so the
numbers cannot drift from the paper's.

Task return is the DECOMPOSED team return (task + comm), which excludes
hebbian_diffuse — otherwise the +Heb arms would be credited with reward that
the diffusion mechanism merely moved between agents.

Steps-to-milestone is the within-episode step of first completion, median over
completing episodes, with the number of completing episodes in parentheses.
Perception grounding and partner-location accuracy come from the qualitative
pipeline's beliefs table.

Usage:
  python make_final_table.py
  python make_final_table.py --out paper_assets_final_comparison
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from make_results import (COOP_TRACKS, MILESTONE_TRACK, aggregate, load_runs)

NONCOMM_MAX = sum(1 for v in MILESTONE_TRACK.values() if v != "communication")
COOP_MAX = sum(1 for v in MILESTONE_TRACK.values() if v in COOP_TRACKS)

MEDIUM = Path("runs_from_daic/medium_runs")
GEMMA = Path("runs_from_daic/new_exp_0_gemma")

# name, dir, runs-root, base model, learning, social coupling
ROWS = [
    ("LLM-2B",        "exp01_llm_2b",               MEDIUM,
     "Qwen3.5-2B",  "none (frozen LLM)", "—"),
    ("LLM-2B+Heb",    "exp07_llm_2b_social_prompt", MEDIUM,
     "Qwen3.5-2B",  "none (frozen LLM)", "Hebbian + prompt"),
    ("LLM-9B",        "exp02_llm_9b",               MEDIUM,
     "Qwen3.5-9B",  "none (frozen LLM)", "—"),
    ("LLM-9B+Heb",    "exp08_llm_9b_social_prompt", MEDIUM,
     "Qwen3.5-9B",  "none (frozen LLM)", "Hebbian + prompt"),
    ("Gemma-E4B",     "new_exp_0_gemma_base",       GEMMA,
     "Gemma-4-E4B", "none (frozen LLM)", "—"),
    ("Gemma-E4B+Heb", "new_exp_0_gemma_hebbian",    GEMMA,
     "Gemma-4-E4B", "none (frozen LLM)", "Hebbian + prompt"),
    ("IPPO",          "exp04_ippo",                 MEDIUM,
     "Qwen3.5-2B",  "IPPO (LoRA)",       "—"),
    ("IPPO+Heb",      "exp06_ippo_hebbian",         MEDIUM,
     "Qwen3.5-2B",  "IPPO (LoRA)",       "Hebbian graph"),
    ("MAPPO",         "exp03_mappo",                MEDIUM,
     "Qwen3.5-2B",  "MAPPO (shared critic)", "—"),
    ("MAPPO+Heb",     "exp05_mappo_hebbian",        MEDIUM,
     "Qwen3.5-2B",  "MAPPO (shared critic)", "Hebbian graph"),
]

# Design-table cells in the paper's tab:conditions vocabulary:
# name -> (Model, RL, Hebbian, SM). Kept separate from ROWS so the markdown
# stays readable while the LaTeX matches the paper's abbreviations.
DESIGN_TEX = {
    "LLM-2B":        ("2B",   "--",    "--",              "--"),
    "LLM-2B+Heb":    ("2B",   "--",    "learned",         "prompt"),
    "LLM-9B":        ("9B",   "--",    "--",              "--"),
    "LLM-9B+Heb":    ("9B",   "--",    "learned",         "prompt"),
    "Gemma-E4B":     ("E4B",  "--",    "--",              "--"),
    "Gemma-E4B+Heb": ("E4B",  "--",    "learned",         "prompt"),
    "IPPO":          ("2B",   "IPPO",  "--",              "--"),
    "IPPO+Heb":      ("2B",   "IPPO",  "learned + diff.", "--"),
    "MAPPO":         ("2B",   "MAPPO", "--",              "--"),
    "MAPPO+Heb":     ("2B",   "MAPPO", "learned + diff.", "--"),
}
# Row order groups for the \midrule breaks in both LaTeX tables.
TEX_GROUPS = [["LLM-2B", "LLM-2B+Heb", "LLM-9B", "LLM-9B+Heb",
               "Gemma-E4B", "Gemma-E4B+Heb"],
              ["IPPO", "IPPO+Heb", "MAPPO", "MAPPO+Heb"]]

# repo id -> column header for the steps-to-milestone block
STEP_COLS = [
    ("m8_anvil_A1",        "Anvil A"),
    ("m9_anvil_B1",        "Anvil B"),
    ("m17_switch_pressed", "Switch"),
    ("m21_first_mob_kill", "Ch4 mob"),
]

BELIEF_TABLES = [Path("analysis_qualitative/out/tables/beliefs.csv"),
                 Path("analysis_qualitative/out_gemma/tables/beliefs.csv")]


def belief_metrics() -> dict:
    """{label: {metric: (mean, sd, n)}} from the qualitative beliefs tables."""
    acc = defaultdict(lambda: defaultdict(list))
    for path in BELIEF_TABLES:
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r.get("quarantined", "False") == "True":
                continue
            for key in ("perception_grounding_rate", "partner_loc_accuracy"):
                try:
                    acc[r["label"]][key].append(float(r[key]))
                except (KeyError, TypeError, ValueError):
                    pass
    return {lab: {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
                  for k, v in m.items()} for lab, m in acc.items()}


def pct(mean_sd, denom):
    m, s = mean_sd
    return 100.0 * m / denom, 100.0 * s / denom


def collect():
    beliefs = belief_metrics()
    out = []
    for name, dirname, root, model, learning, social in ROWS:
        runs = load_runs(root, dirname)
        if not runs:
            print(f"[skip] {name}: no runs under {root / dirname}")
            continue
        a = aggregate(runs)
        b = beliefs.get(name, {})
        out.append({
            "name": name, "model": model, "learning": learning,
            "social": social, "n_runs": a["n_runs"], "n_eps": a["n_eps"],
            "task": a["task"],
            "ms_pct": pct(a["allms_nc"], NONCOMM_MAX),
            "coop_pct": pct(a["coop"], COOP_MAX),
            "steps": {mid: a["steps"].get(mid, (None, 0)) for mid, _ in STEP_COLS},
            "ground": b.get("perception_grounding_rate"),
            "partner": b.get("partner_loc_accuracy"),
            "decomposed": a["task_is_decomposed"],
        })
    return out


def f2(pair, prec=1):
    return f"{pair[0]:.{prec}f} ± {pair[1]:.{prec}f}"


def f_steps(v):
    med, n = v
    return "—" if med is None else f"{med:.0f} ({n})"


def f_belief(v):
    return "—" if not v else f"{v[0]:.3f}"


def markdown(rows) -> str:
    L = ["## Table 1 — Experiment design", "",
         "| Experiment | Base model | Learning | Social coupling | Seeds |",
         "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['model']} | {r['learning']} | "
                 f"{r['social']} | {r['n_runs']} |")
    L += ["", "All conditions: 3 agents, 3 episodes × 1000 steps, "
              "communication enabled.", "",
          "## Table 2 — Task performance and perception", "",
          "Milestone figures are percentage completion per episode "
          f"(team union; denominators {NONCOMM_MAX} non-comm, {COOP_MAX} "
          "cooperative Ch2–Ch5). Steps-to-milestone is the median "
          "within-episode step of first completion, with completing episodes "
          "in parentheses.", ""]
    head = ("| Experiment | Task return | Milestone % | Coop. % | "
            + " | ".join(h for _, h in STEP_COLS)
            + " | Grounding | Partner loc. |")
    L += [head, "|" + "---|" * (5 + len(STEP_COLS) + 1)]
    for r in rows:
        cells = [r["name"], f2(r["task"], 0), f2(r["ms_pct"]), f2(r["coop_pct"])]
        cells += [f_steps(r["steps"][mid]) for mid, _ in STEP_COLS]
        cells += [f_belief(r["ground"]), f_belief(r["partner"])]
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def latex(rows) -> str:
    L = ["% ── Table 1: experiment design ───────────────────────────────"]
    for r in rows:
        L.append(f"{r['name']} & {r['model']} & {r['learning']} & "
                 f"{r['social']} & {r['n_runs']} \\\\")
    L += ["", "% ── Table 2: task performance + perception ──────────────────",
          "% cols: return, milestone \\%, coop \\%, steps(A,B,switch,mob), "
          "grounding, partner-loc"]
    for r in rows:
        cells = [r["name"], f"{r['task'][0]:.0f} $\\pm$ {r['task'][1]:.0f}",
                 f"{r['ms_pct'][0]:.1f} $\\pm$ {r['ms_pct'][1]:.1f}",
                 f"{r['coop_pct'][0]:.1f} $\\pm$ {r['coop_pct'][1]:.1f}"]
        cells += [f_steps(r["steps"][mid]).replace("—", "--")
                  for mid, _ in STEP_COLS]
        cells += [f_belief(r["ground"]).replace("—", "--"),
                  f_belief(r["partner"]).replace("—", "--")]
        L.append(" & ".join(cells) + " \\\\")
    return "\n".join(L) + "\n"


def latex_tables(rows) -> str:
    """Two complete table* environments in the paper's booktabs style."""
    by_name = {r["name"]: r for r in rows}

    def groups():
        for g, names in enumerate(TEX_GROUPS):
            present = [by_name[n] for n in names if n in by_name]
            if present:
                yield g, present

    def steps_cell(v):
        med, n = v
        return "--" if med is None else f"${med:.0f}_{{{n}}}$"

    def belief_cell(v):
        return "--" if not v else f"${v[0]:.2f}$"

    L = [
        "% Generated by make_final_table.py — do not edit by hand.",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{\\textbf{Compared conditions (cross-model comparison).}",
        "Model: Qwen3.5 (2B/9B) or Gemma-4 (E4B); all agents receive the game",
        "frame (vision) and may communicate. SM = social-module coupling of",
        "the bond row into the action prompt. Seeds = completed seeds",
        "aggregated in \\autoref{tab:final_comparison}.}",
        "\\label{tab:final_conditions}",
        "\\small",
        "\\begin{tabular}{lllllc}",
        "\\toprule",
        "Condition & Model & RL & Hebbian & SM & Seeds \\\\",
    ]
    for g, present in groups():
        L.append("\\midrule")
        for r in present:
            model, rl, heb, sm = DESIGN_TEX[r["name"]]
            L.append(f"{r['name']:14s} & {model} & {rl} & {heb} & {sm} & "
                     f"{r['n_runs']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]

    L += [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{\\textbf{Task performance and perception across models and",
        "RL baselines.} Mean~$\\pm$~std over the pooled episodes of all",
        "completed seeds (3 episodes each). \\emph{Milestones} = per-episode",
        f"team completion as \\% of the {NONCOMM_MAX} non-communication",
        f"milestones; \\emph{{Coop.}} = \\% of the {COOP_MAX} Ch2--Ch5",
        "milestones (\\autoref{eq:coop_milestones}); \\emph{Task return} =",
        "team-summed undiffused environment reward per episode. Steps columns:",
        "median within-episode step of first completion with the number of",
        "completing episodes as subscript; ``--'' = never completed.",
        "Grounding = perception-grounding rate; Partner = partner-location",
        "accuracy (both from the qualitative pipeline, run-level means).}",
        "\\label{tab:final_comparison}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{tabular}{l c cc cccc cc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Milestone \\%} & "
        "\\multicolumn{4}{c}{Steps to first completion} & "
        "\\multicolumn{2}{c}{Perception} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}",
        "Condition & Task return $\\uparrow$ & All $\\uparrow$ & "
        "Coop.\\ $\\uparrow$ & M8 anvil & M9 anvil & M13 switch & "
        "M17 mob & Ground.\\ $\\uparrow$ & Partner $\\uparrow$ \\\\",
    ]
    for g, present in groups():
        L.append("\\midrule")
        for r in present:
            cells = [
                f"{r['name']:14s}",
                f"${r['task'][0]:.0f} \\pm {r['task'][1]:.0f}$",
                f"${r['ms_pct'][0]:.1f} \\pm {r['ms_pct'][1]:.1f}$",
                f"${r['coop_pct'][0]:.1f} \\pm {r['coop_pct'][1]:.1f}$",
            ]
            cells += [steps_cell(r["steps"][mid]) for mid, _ in STEP_COLS]
            cells += [belief_cell(r["ground"]), belief_cell(r["partner"])]
            L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper_assets_final_comparison"))
    args = ap.parse_args()

    rows = collect()
    md = markdown(rows)
    print(md)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final_comparison.md").write_text(md, encoding="utf-8")
    (args.out / "final_comparison_rows.tex").write_text(latex(rows), encoding="utf-8")
    (args.out / "final_comparison_tables.tex").write_text(
        latex_tables(rows), encoding="utf-8")
    print(f"wrote {args.out}/final_comparison.md, final_comparison_rows.tex "
          f"and final_comparison_tables.tex")

    undecomposed = [r["name"] for r in rows if not r["decomposed"]]
    if undecomposed:
        print(f"[WARN] task return fell back to logged totals (includes "
              f"hebbian_diffuse) for: {', '.join(undecomposed)}")


if __name__ == "__main__":
    main()