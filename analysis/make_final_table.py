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
  python analysis/make_final_table.py
  python analysis/make_final_table.py --out paper_assets/final_comparison
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from paths import ASSETS, QUALITATIVE, RUNS  # noqa: E402  (also puts siblings on sys.path)

from make_results import (  # noqa: F401 (COOP_TRACKS/MILESTONE_TRACK re-exported)

    COOP_MAX, COOP_TRACKS, MILESTONE_TRACK, NONCOMM_MAX, aggregate, load_runs)

# Denominators are imported, NOT recomputed here. This file used to derive
# NONCOMM_MAX as "every track except communication", which was 25 when the
# published table was generated. Commit e47f286 then added the observation /
# imitation act-reward tracks (10 entries) to MILESTONE_TRACK, and
# make_results.NONCOMM_MAX was updated to exclude all of SOCIAL_ACT_TRACKS —
# but the local recomputation here silently drifted to 35, which would have
# scaled every "Milest. %" by 25/35 and printed "35" in the caption on the
# next regeneration. Single source of truth: make_results.
assert NONCOMM_MAX == 25 and COOP_MAX == 17, (NONCOMM_MAX, COOP_MAX)

MEDIUM = RUNS / "medium_runs"
GEMMA = RUNS / "new_exp_0_gemma"
SOCIAL_REPLAY = RUNS / "social_replay_qwen"
ORCHESTRATOR = RUNS / "orchestrator"

# Which arms stand for "+Heb" in the RL rows of the paper table:
#   "replay"    — exp30/exp31: reward diffusion + weight-gated experience
#                 sharing (Eq. 7, rho=0.3), runs_from_daic/social_replay_qwen
#   "diffusion" — exp05/exp06: reward diffusion only (the original rows)
# Both use the same medium config (3 agents, 3 eps x 1000 steps, Qwen3.5-2B),
# so the baselines (exp03/exp04) are shared.
#
# Set to "replay" once the vision-ON exp30/exp31 Qwen runs have landed in
# runs_from_daic/social_replay_qwen. The first batch (2026-08-23/24) ran with
# LLM_VISION_MODE=text by launcher mistake — no frames, vision=False — while
# every medium baseline ran vision=True; it is kept aside as
# runs_from_daic/social_replay_qwen_textonly and must not feed this table.
RL_HEB_ARMS = "diffusion"
_RL_HEB = {
    "replay": {
        "IPPO+Heb":  ("exp31_ippo_hebbian_replay",  SOCIAL_REPLAY,
                      "Hebbian graph + replay", "learned + diff. + replay"),
        "MAPPO+Heb": ("exp30_mappo_hebbian_replay", SOCIAL_REPLAY,
                      "Hebbian graph + replay", "learned + diff. + replay"),
    },
    "diffusion": {
        "IPPO+Heb":  ("exp06_ippo_hebbian",  MEDIUM, "Hebbian graph", "learned + diff."),
        "MAPPO+Heb": ("exp05_mappo_hebbian", MEDIUM, "Hebbian graph", "learned + diff."),
    },
}[RL_HEB_ARMS]

# The qualitative pipeline labels runs by make_results.CONDITIONS, where the
# replay arms are "IPPO+Heb+SR" / "MAPPO+Heb+SR"; map table names to those.
QUAL_LABEL = ({"IPPO+Heb": "IPPO+Heb+SR", "MAPPO+Heb": "MAPPO+Heb+SR"}
              if RL_HEB_ARMS == "replay" else {})

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
    # Centralized baseline: VillagerAgent-style DAG task orchestration.
    # Same base model / config as the two rows above (6 seeds x 3 eps x 1000
    # steps, Gemma-4-E4B, vision) -- the only difference is the coupling, so
    # it reads directly against Gemma-E4B and Gemma-E4B+Heb.
    ("Gemma-E4B+Central Orch.", "new_exp_0_gemma_orch_villager_advisory",
     ORCHESTRATOR,
     "Gemma-4-E4B", "none (frozen LLM)", "central DAG orchestrator"),
    ("IPPO",          "exp04_ippo",                 MEDIUM,
     "Qwen3.5-2B",  "IPPO (LoRA)",       "—"),
    ("IPPO+Heb",      _RL_HEB["IPPO+Heb"][0],       _RL_HEB["IPPO+Heb"][1],
     "Qwen3.5-2B",  "IPPO (LoRA)",       _RL_HEB["IPPO+Heb"][2]),
    ("MAPPO",         "exp03_mappo",                MEDIUM,
     "Qwen3.5-2B",  "MAPPO (shared critic)", "—"),
    ("MAPPO+Heb",     _RL_HEB["MAPPO+Heb"][0],      _RL_HEB["MAPPO+Heb"][1],
     "Qwen3.5-2B",  "MAPPO (shared critic)", _RL_HEB["MAPPO+Heb"][2]),
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
    "Gemma-E4B+Central Orch.": ("E4B", "--",  "--",              "DAG orch."),
    "IPPO":          ("2B",   "IPPO",  "--",              "--"),
    "IPPO+Heb":      ("2B",   "IPPO",  _RL_HEB["IPPO+Heb"][3],  "--"),
    "MAPPO":         ("2B",   "MAPPO", "--",              "--"),
    "MAPPO+Heb":     ("2B",   "MAPPO", _RL_HEB["MAPPO+Heb"][3], "--"),
}
# Row order groups for the \midrule breaks in both LaTeX tables.
TEX_GROUPS = [["LLM-2B", "LLM-2B+Heb", "LLM-9B", "LLM-9B+Heb",
               "Gemma-E4B", "Gemma-E4B+Heb",
               "Gemma-E4B+Central Orch."],
              ["IPPO", "IPPO+Heb", "MAPPO", "MAPPO+Heb"]]

# repo id -> column header for the steps-to-milestone block
STEP_COLS = [
    ("m8_anvil_A1",        "Anvil A"),
    ("m9_anvil_B1",        "Anvil B"),
    ("m17_switch_pressed", "Switch"),
    ("m21_first_mob_kill", "Ch4 mob"),
]

BELIEF_TABLES = [QUALITATIVE / "out" / "tables" / "beliefs.csv",
                 QUALITATIVE / "out_gemma" / "tables" / "beliefs.csv",
                 QUALITATIVE / "out_villager" / "tables" / "beliefs.csv",
                 QUALITATIVE / "out_social_replay" / "tables" / "beliefs.csv"]


# Perception grounding is a property of the BASE MODEL, not of the coupling.
# Pooled over every condition built on each model, base model explains 97% of
# its variance (eta^2 = 0.969) and the spread across a model's own couplings
# is <= 0.06 (0.002 for Gemma-E4B, over three couplings). Reporting it once
# per model in tab:perception_by_model therefore says everything the
# per-condition column said, without repeating a near-constant 11 times.
# Partner-location accuracy is NOT model-determined (eta^2 = 0.55, spread up
# to 0.18 within Qwen3.5-2B), so it stays a per-condition column in
# tab:final_comparison.
PERCEPTION_MODELS = [
    ("Qwen3.5-2B", ["LLM-2B", "LLM-2B+Heb",
                    "IPPO", "IPPO+Heb", "MAPPO", "MAPPO+Heb"]),
    ("Qwen3.5-9B", ["LLM-9B", "LLM-9B+Heb"]),
    ("Gemma-E4B",  ["Gemma-E4B", "Gemma-E4B+Heb", "Gemma-E4B+Central Orch."]),
]


def perception_by_model(raw: dict) -> list:
    """[{model, n, ground, partner, ground_spread, partner_spread}].

    ``*_spread`` is the largest gap between any two of that model's own
    couplings. Not emitted as a column -- it is the evidence behind the
    caption's claim that grounding is model-determined (<= 0.06) while
    partner-location accuracy is not (up to 0.18), so the reader is told
    the asymmetry rather than shown two more columns.
    """
    out = []
    for model, labels in PERCEPTION_MODELS:
        rec = {"model": model, "n": 0}
        ok = True
        for key, short in (("perception_grounding_rate", "ground"),
                           ("partner_loc_accuracy", "partner")):
            vals, per_cond = [], []
            for lab in labels:
                v = raw.get(lab, {}).get(key, [])
                if v:
                    vals += v
                    per_cond.append(st.mean(v))
            if not vals:
                ok = False
                break
            rec["n"] = max(rec["n"], len(vals))
            rec[short] = (st.mean(vals),
                          st.stdev(vals) if len(vals) > 1 else 0.0)
            rec[short + "_spread"] = ((max(per_cond) - min(per_cond))
                                      if per_cond else 0.0)
        if ok:
            out.append(rec)
    return out


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
    summary = {lab: {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0,
                         len(v)) for k, v in m.items()}
               for lab, m in acc.items()}
    return summary, {lab: dict(m) for lab, m in acc.items()}


def pct(mean_sd, denom):
    m, s = mean_sd
    return 100.0 * m / denom, 100.0 * s / denom


def collect():
    beliefs, raw = belief_metrics()
    perception = perception_by_model(raw)
    out = []
    for name, dirname, root, model, learning, social in ROWS:
        runs = load_runs(root, dirname)
        if not runs:
            print(f"[skip] {name}: no runs under {root / dirname}")
            continue
        a = aggregate(runs)
        b = beliefs.get(QUAL_LABEL.get(name, name), {})
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
    return out, perception


def f2(pair, prec=1):
    return f"{pair[0]:.{prec}f} ± {pair[1]:.{prec}f}"


def f_steps(v):
    med, n = v
    return "—" if med is None else f"{med:.0f} ({n})"


def f_belief(v):
    return "—" if not v else f"{v[0]:.3f}"


def markdown(rows, perception) -> str:
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
            + " |")
    L += [head, "|" + "---|" * (4 + len(STEP_COLS))]
    for r in rows:
        cells = [r["name"], f2(r["task"], 0), f2(r["ms_pct"]), f2(r["coop_pct"])]
        cells += [f_steps(r["steps"][mid]) for mid, _ in STEP_COLS]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Table 3 - Perception by base model", "",
          "Pooled over every condition built on each model.", "",
          "| Base model | Grounding | Partner loc. |", "|---|---|---|"]
    for r in perception:
        L.append(f"| {r['model']} | {r['ground'][0]:.2f} ± "
                 f"{r['ground'][1]:.2f} | {r['partner'][0]:.2f} ± "
                 f"{r['partner'][1]:.2f} |")
    L += ["", "Within-model spread across couplings: "
          + "; ".join(f"{r['model']} grounding {r['ground_spread']:.3f}, "
                      f"partner {r['partner_spread']:.3f}"
                      for r in perception) + "."]
    return "\n".join(L) + "\n"


def latex(rows) -> str:
    L = ["% ── Table 1: experiment design ───────────────────────────────"]
    for r in rows:
        L.append(f"{r['name']} & {r['model']} & {r['learning']} & "
                 f"{r['social']} & {r['n_runs']} \\\\")
    L += ["", "% ── Table 2: task performance + perception ──────────────────",
          "% cols: return, milestone \\%, coop \\%, steps(A,B,switch,mob)"]
    for r in rows:
        cells = [r["name"], f"{r['task'][0]:.0f} $\\pm$ {r['task'][1]:.0f}",
                 f"{r['ms_pct'][0]:.1f} $\\pm$ {r['ms_pct'][1]:.1f}",
                 f"{r['coop_pct'][0]:.1f} $\\pm$ {r['coop_pct'][1]:.1f}"]
        cells += [f_steps(r["steps"][mid]).replace("—", "--")
                  for mid, _ in STEP_COLS]
        L.append(" & ".join(cells) + " \\\\")
    return "\n".join(L) + "\n"


def latex_tables(rows, perception) -> str:
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
        "Perception measures are properties of the base model rather than",
        "of the coupling and are reported once per model in",
        "\\autoref{tab:perception_by_model}.}",
        "\\label{tab:final_comparison}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{tabular}{l c cc cccc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Milestone \\%} & "
        "\\multicolumn{4}{c}{Steps to first completion} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}",
        "Condition & Task return $\\uparrow$ & All $\\uparrow$ & "
        "Coop.\\ $\\uparrow$ & M8 anvil & M9 anvil & M13 switch & "
        "M17 mob \\\\",
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
            L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]

    L += [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{Perception by base model.}",
        "\\emph{Ground.} = perception-grounding rate; \\emph{Partner} =",
        "partner-location accuracy. Mean~$\\pm$~std over all seeds of every",
        "condition built on that model (base, +Heb, the RL policies for",
        "Qwen3.5-2B, and the centralized orchestrator for Gemma-E4B); the",
        "per-condition breakdown is omitted because the base model, not the",
        "social coupling, determines these measures. Grounding is almost",
        "entirely fixed by the model: it explains 97\\% of the variance",
        "($\\eta^2 = 0.97$) and the gap between a model's own couplings",
        "never exceeds $0.06$ (only $0.002$ across the three Gemma-E4B",
        "conditions). Partner-location accuracy is more sensitive to the",
        "coupling ($\\eta^2 = 0.55$; up to $0.18$ within Qwen3.5-2B, where",
        "the RL policies lose partner tracking relative to the frozen LLM),",
        "so its standard deviation here is correspondingly wider.",
        "\\textbf{Finding:} perception scales with the capability of the",
        "base model and is essentially unaffected by the social coupling.}",
        "\\label{tab:perception_by_model}",
        "\\footnotesize",
        "\\begin{tabular}{l c c}",
        "\\toprule",
        "\\textbf{Base model} & \\textbf{Ground.}\\ $\\uparrow$ & "
        "\\textbf{Partner}\\ $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for r in perception:
        L.append(f"{r['model']:12s} & ${r['ground'][0]:.2f} \\pm "
                 f"{r['ground'][1]:.2f}$ & ${r['partner'][0]:.2f} \\pm "
                 f"{r['partner'][1]:.2f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ASSETS / "final_comparison")
    args = ap.parse_args()

    rows, perception = collect()
    md = markdown(rows, perception)
    print(md)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final_comparison.md").write_text(md, encoding="utf-8")
    (args.out / "final_comparison_rows.tex").write_text(latex(rows), encoding="utf-8")
    (args.out / "final_comparison_tables.tex").write_text(
        latex_tables(rows, perception), encoding="utf-8")
    print(f"wrote {args.out}/final_comparison.md, final_comparison_rows.tex "
          f"and final_comparison_tables.tex")

    undecomposed = [r["name"] for r in rows if not r["decomposed"]]
    if undecomposed:
        print(f"[WARN] task return fell back to logged totals (includes "
              f"hebbian_diffuse) for: {', '.join(undecomposed)}")


if __name__ == "__main__":
    main()