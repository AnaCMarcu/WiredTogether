"""Generate the transplant experiment main results table (LaTeX).

Mirrors the Experiment-2 cofire table conventions EXACTLY by importing the
same machinery from make_results.py rather than reimplementing it:

  * load_runs()                — episode slicing + entry-honesty filter
                                 (m20 needs m18, m24 needs m22, same-episode
                                 pre-filter team union; fake entry rewards
                                 are subtracted from the task stream);
  * episode_milestone_sets()   — team union per episode (post-filter);
  * episode_task_returns(include_comm=False)
                               — team physical reward per episode, message
                                 pay excluded;
  * NONCOMM_MAX (25) / COOP_MAX (17) denominators.

Aggregation: per-episode values pooled over ALL episodes of all seeds of a
condition (no per-seed averaging — every episode weighs equally), reported
as mean ± population SD (statistics.pstdev).

Partner-preference columns: per episode, each pair's preference is the mean
over its two agents of P(message target = partner) from that episode's
messages.jsonl; pooled the same way. Chance level at N=6 is 0.20.

Usage:
    PYTHONPATH=src python src/mindforge/tools/make_transplant_table.py
Writes paper_assets_transplant/TRANSPLANT_MAIN_TABLE.tex; Qwen rows appear
automatically when runs_from_daic/pair_bonding_qwen/expB_* exists, and as
placeholders otherwise.
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import make_results as mr  # noqa: E402  (paper-table machinery)
from mindforge.tools.analyze_wiring import (  # noqa: E402
    agent_index,
    load_message_matrix,
    seatmate_preference,
)

N_AGENTS = 6
PAIRS = [(0, 1), (2, 3), (4, 5)]

MODELS = [
    ("Gemma 4 E4B", Path("runs_from_daic/pair_bonding")),
    ("Qwen3.5-9B", Path("runs_from_daic/pair_bonding_qwen")),
]
ARMS = [("Transplant", "expB_merged_transplant"),
        ("Shuffled", "expB_merged_shuffled")]

OUT = Path("paper_assets_transplant/TRANSPLANT_MAIN_TABLE.tex")


def pooled(vals, nd):
    """mean ± population SD over pooled per-episode values, \\pmm-formatted."""
    if not vals:
        return "--"
    m, sd = st.fmean(vals), st.pstdev(vals)
    return f"${m:.{nd}f}$ \\pmm{{{sd:.{nd}f}}}"


def pair_stats(runs_root, dir_name):
    """Per-pair per-episode metrics pooled over seeds of one condition.

    For every pair (a,b) and episode: partner preference (mean of the two
    members), within-pair message count, co-earned milestones (both members
    contributed to the same honesty-filtered milestone event group; all-hands
    groups dropped), and end-of-episode within-pair bond W (mean of W[a][b],
    W[b][a]). Also pools the cross-pair W baseline over all non-partner
    ordered pairs.
    """
    runs = mr.load_runs(runs_root, dir_name)
    if not runs:
        return None
    out = {p: {"pref": [], "msgs": [], "co": [], "w": [], "prox": []}
           for p in PAIRS}
    cross_w = []
    for run in runs:
        run_dir = Path(run["_path"]).parent
        _, per_ep = load_message_matrix(run_dir, N_AGENTS)
        for _ep, mat in sorted(per_ep.items()):
            prefs = seatmate_preference(mat)
            for a, b in PAIRS:
                pv = [v for v in (prefs[a][0], prefs[b][0]) if v is not None]
                if pv:
                    out[(a, b)]["pref"].append(st.fmean(pv))
                out[(a, b)]["msgs"].append(mat[a][b] + mat[b][a])
        # Co-milestones per episode: contributors sharing an honesty-filtered
        # (milestone_id, step) event group; all-hands groups say nothing
        # about pairing and are dropped.
        groups = defaultdict(set)
        for ev in run.get("milestone_events", []):
            idx = agent_index(ev.get("contributor"))
            if idx is not None:
                e, _ = mr.ep_of_step(run["_ep_bounds"],
                                     int(ev.get("step", -1)))
                if e is not None:
                    groups[(ev.get("milestone_id"), ev.get("step"))] |= {
                        (e, idx)}
        co = defaultdict(int)   # (episode, pair) -> count
        for (_mid, _step), members in groups.items():
            eps = {e for e, _ in members}
            idxs = {i for _, i in members}
            if len(idxs) < 2 or len(idxs) >= N_AGENTS:
                continue
            for e in eps:
                for a, b in PAIRS:
                    if a in idxs and b in idxs:
                        co[(e, (a, b))] += 1
        n_eps = len(run["_ep_bounds"])
        for e in range(n_eps):
            for p in PAIRS:
                out[p]["co"].append(co.get((e, p), 0))
        # End-of-episode within-pair W, cross-pair baseline, and the pair's
        # proximity share (its slice of the pair_interaction proximity plane;
        # chance = 1/15 per pair at N=6). Read from episode summaries — the
        # coop_eval nesting bug zeroes these tensors in final_metrics.json.
        for f in sorted(run_dir.glob("episodes/*/summary.json")):
            cm = json.load(open(f)).get("cooperation_metrics") or {}
            hw = cm.get("hebbian_W")
            if hw:
                for a, b in PAIRS:
                    out[(a, b)]["w"].append((hw[a][b] + hw[b][a]) / 2.0)
                cross_w.extend(hw[i][j] for i in range(N_AGENTS)
                               for j in range(N_AGENTS)
                               if i != j and i // 2 != j // 2)
            prox = (cm.get("pair_interaction") or {}).get("proximity")
            if prox:
                total = sum(prox[i][j] for i in range(N_AGENTS)
                            for j in range(N_AGENTS) if i != j)
                if total:
                    for a, b in PAIRS:
                        out[(a, b)]["prox"].append(
                            100.0 * (prox[a][b] + prox[b][a]) / total)
    return {"pairs": out, "cross_w": cross_w, "n_runs": len(runs)}


def condition_stats(runs_root, dir_name):
    """Pool per-episode metrics over every completed seed of one condition."""
    runs = mr.load_runs(runs_root, dir_name)
    if not runs:
        return None
    vals = {"ret": [], "ms": [], "coop": [],
            "pref": {p: [] for p in PAIRS}}
    n_runs = 0
    for run in runs:
        n_runs += 1
        rets, _ = mr.episode_task_returns(run, include_comm=False)
        vals["ret"].extend(rets)
        for team in mr.episode_milestone_sets(run):
            noncomm = sum(1 for m in team
                          if mr.MILESTONE_TRACK.get(m) != "communication")
            vals["ms"].append(100.0 * noncomm / mr.NONCOMM_MAX)
            vals["coop"].append(100.0 * mr.coop_count(team) / mr.COOP_MAX)
        run_dir = Path(run["_path"]).parent
        _, per_ep = load_message_matrix(run_dir, N_AGENTS)
        for _ep, mat in sorted(per_ep.items()):
            prefs = seatmate_preference(mat)
            for a, b in PAIRS:
                pair_vals = [prefs[a][0], prefs[b][0]]
                pair_vals = [v for v in pair_vals if v is not None]
                if pair_vals:
                    vals["pref"][(a, b)].append(st.fmean(pair_vals))
    vals["n_runs"] = n_runs
    vals["n_eps"] = len(vals["ms"])
    return vals


def build():
    """One combined table: condition-level metrics (multirow over the three
    pair rows) + per-pair breakdown."""
    # Cross-pair W baseline per model, for the caption.
    cross_notes = []
    pair_cache = {}
    for model_name, root in MODELS:
        xs = []
        for arm_label, dir_name in ARMS:
            if (root / dir_name).exists():
                ps = pair_stats(root, dir_name)
                pair_cache[(model_name, arm_label)] = ps
                if ps:
                    xs.extend(ps["cross_w"])
        if xs:
            cross_notes.append(
                f"{model_name}: ${st.fmean(xs):.3f}$ \\pmm{{{st.pstdev(xs):.3f}}}")

    lines = []
    lines.append("% ─── Transplant experiment results table (combined) ──")
    lines.append("% Generated by src/mindforge/tools/make_transplant_table.py")
    lines.append("% — regenerate, do not hand-edit. Mirrors the Experiment-2")
    lines.append("% conventions by importing make_results.py machinery")
    lines.append("% (team union per episode, entry-honesty filter, 25/17")
    lines.append("% denominators, pooled episodes, population SD).")
    lines.append("% Requires booktabs + makecell + multirow (+ \\pmm from floats.tex).")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{\\textbf{Pair transplant: do former partners "
                 "still wire together?} Six Phase~A agents per run, arranged "
                 "as three pairs; \\emph{Transplant} pairs each agent with "
                 "its real Phase~A partner (pairs 1--2 co-fired in Phase~A; "
                 "pair 3 shares an equally long history without a co-fired "
                 "achievement — the familiarity control), \\emph{Shuffled} "
                 "pairs agents from different Phase~A runs, so every agent "
                 "holds renamed memories of a shared history that never "
                 "happened. Bonds and memory volume are identical between "
                 "arms; 3 seeds $\\times$ 3 episodes per condition; all "
                 "values mean~$\\pm$~population SD over pooled episodes. "
                 "Team metrics: \\emph{Task return} (team-summed physical "
                 "reward, message pay excluded), \\emph{Milest.} (\\% of the "
                 "25 non-communication milestones), \\emph{Coop.} (\\% of "
                 "the 17 cooperative Ch2--Ch5 milestones, both after the "
                 "entry-honesty filter). Per-pair metrics: \\emph{Partner "
                 "pref.} $=P(\\text{message target}=\\text{partner})$, mean "
                 "over the pair's members (chance $0.20$); \\emph{Msgs "
                 "within} (messages exchanged inside the pair per episode); "
                 "\\emph{Prox.\\ share} (the pair's share of all pairwise "
                 "co-presence events, agents within $4$ blocks; chance "
                 "$1/15\\approx6.7\\%$); \\emph{Bond $W$} (end-of-episode "
                 "within-pair weight, both directions averaged; initialized "
                 "at $0.265$ for every pair). Cross-pair bond baseline "
                 "(initialized $0.10$), pooled per model: "
                 + "; ".join(cross_notes) + ".}")
    lines.append("\\label{tab:transplant_main}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4.0pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("\\begin{tabular}{l ccc l cccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Condition}")
    lines.append("& \\makecell{\\textbf{Task}\\\\\\textbf{return} $\\uparrow$}")
    lines.append("& \\makecell{\\textbf{Milest.}\\\\\\textbf{(\\%)} $\\uparrow$}")
    lines.append("& \\makecell{\\textbf{Coop.}\\\\\\textbf{(\\%)} $\\uparrow$}")
    lines.append("& \\textbf{Pair}")
    lines.append("& \\makecell{\\textbf{Partner}\\\\\\textbf{pref.}}")
    lines.append("& \\makecell{\\textbf{Msgs}\\\\\\textbf{within}}")
    lines.append("& \\makecell{\\textbf{Prox.}\\\\\\textbf{share (\\%)}}")
    lines.append("& \\makecell{\\textbf{Bond}\\\\\\textbf{$W$}} \\\\")
    for model_name, root in MODELS:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{model_name}}}}}\\\\")
        for a_i, (arm_label, dir_name) in enumerate(ARMS):
            if a_i:
                lines.append("\\addlinespace")
            cond = condition_stats(root, dir_name) \
                if (root / dir_name).exists() else None
            ps = pair_cache.get((model_name, arm_label))
            if cond is None or ps is None:
                for k in range(len(PAIRS)):
                    head = (f"\\multirow{{3}}{{*}}{{{arm_label}}} & "
                            f"\\multirow{{3}}{{*}}{{--}} & "
                            f"\\multirow{{3}}{{*}}{{--}} & "
                            f"\\multirow{{3}}{{*}}{{--}}"
                            if k == 0 else " & & &")
                    lines.append(f"{head} & {PAIR_LABELS[arm_label][k]} & "
                                 f"-- & -- & -- & -- \\\\  % pending")
                continue
            note = (f"  % {cond['n_runs']} runs, {cond['n_eps']} episodes"
                    if cond["n_runs"] != 3 or cond["n_eps"] != 9 else "")
            for k, p in enumerate(PAIRS):
                d = ps["pairs"][p]
                head = (f"\\multirow{{3}}{{*}}{{{arm_label}}} & "
                        f"\\multirow{{3}}{{*}}{{{pooled(cond['ret'], 0)}}} & "
                        f"\\multirow{{3}}{{*}}{{{pooled(cond['ms'], 1)}}} & "
                        f"\\multirow{{3}}{{*}}{{{pooled(cond['coop'], 1)}}}"
                        if k == 0 else " & & &")
                lines.append(
                    f"{head} & {PAIR_LABELS[arm_label][k]} & "
                    f"{pooled(d['pref'], 2)} & {pooled(d['msgs'], 0)} & "
                    f"{pooled(d['prox'], 1)} & {pooled(d['w'], 3)} \\\\"
                    + (note if k == 0 else ""))
            print(f"  {model_name:<12} {arm_label:<11} "
                  f"runs={cond['n_runs']} eps={cond['n_eps']}",
                  file=sys.stderr)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


PAIR_OUT = Path("paper_assets_transplant/TRANSPLANT_PAIR_TABLE.tex")

PAIR_LABELS = {
    "Transplant": ["Pair 1 (co-fired)", "Pair 2 (co-fired)",
                   "Pair 3 (control)"],
    "Shuffled": ["Pair 1 (strangers)", "Pair 2 (strangers)",
                 "Pair 3 (strangers)"],
}


if __name__ == "__main__":
    tex = build()
    OUT.write_text(tex, encoding="utf-8")
    print(f"wrote {OUT} ({len(tex.splitlines())} lines)")
    # The separate pair table is superseded by the combined layout; remove a
    # stale copy so it cannot drift out of sync with the main table.
    if PAIR_OUT.exists():
        PAIR_OUT.unlink()
        print(f"removed superseded {PAIR_OUT}")
