#!/usr/bin/env python3
r"""make_agent_completion_tables.py - Tables 1-3 with per-agent completion.

Replaces the agent-UNION milestone columns (Milest. % / Coop. %) of
tab:final_comparison, tab:cofire_main and tab:transplant_main with two
per-agent completion columns and regenerates the three tables. The existing
table scripts are left untouched, so the published (union) numbers stay
reproducible from their own code path.

Metric - completion %, per episode, after make_results' entry-honesty filter:

    credited(track)   = sum over agents of |milestones of that agent in track|
    achievable(track) = sum over milestones m in track of cap(m)
                        cap(m) = 1 for m_door1_open (one unlocker per team)
                               = N otherwise (Lua once=true per agent)
    completion %      = 100 * credited / achievable

Tracks:
    Solo  = the 8 Chamber-1 milestones            achievable 7N+1  (22 at N=3)
    Coop. = the 17 Chamber-2..5 milestones        achievable 17N   (51 at N=3)
    All   = the 25 non-communication milestones   achievable 24N+1 (73 at N=3)
            (written to the CSV, not printed)
Transplant Phase B starts inside Chamber 3 (Ch2 force-teleport), so its Coop.
track is the 12 REACHABLE Ch3-5 milestones (m16_enter_cell can never fire):
achievable 12N = 72 at N=6. Solo is not reachable there and is not printed.

Aggregation (--unit):
    seed     (default) mean over a run's episodes, then mean +/- SAMPLE SD
             across seeds - the seed is the independent unit
    episode  mean +/- POPULATION SD over pooled episodes (the paper's old
             convention)
The transplant cells are single-seed, so that table is always per episode.

Task return is unchanged from each table: decomposed task + comm streams in
Table 1; task stream only (message pay excluded) in the co-firing and
transplant tables. The co-firing cue columns (act use, attributed dW, rho)
and mean W are recomputed with cofire_table.py's own functions.

Outputs (--out, default paper_assets/agent_completion/):
    final_comparison_agent.tex   cofire_main_agent.tex   transplant_main_agent.tex
    agent_completion_rows.csv    (every printed value + All %, per condition)
    agent_completion_per_seed.csv
    METRICS.md                   (the definitions above, for the paper text)

Usage:
    python analysis/make_agent_completion_tables.py
    python analysis/make_agent_completion_tables.py --unit episode
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

import paths  # noqa: F401  (sys.path: analysis/, repo root, src/)
from paths import ASSETS

import make_final_table as mft
import make_results as MR
from make_final_table_extended import NEW_ROWS
from make_final_table_latex import EXTRA_ROWS, SEED_TARGET
import cofire_table as CT
import make_transplant_tables as MTT

# ─── milestone tracks ──────────────────────────────────────────────────────
CH1 = frozenset(m for m, t in MR.MILESTONE_TRACK.items() if t == "ch1_solo")
COOP = frozenset(m for m, t in MR.MILESTONE_TRACK.items() if t in MR.COOP_TRACKS)
NONCOMM = frozenset(m for m, t in MR.MILESTONE_TRACK.items()
                    if t not in MR.SOCIAL_ACT_TRACKS)
PHASE_B_UNREACHABLE = frozenset({"m16_enter_cell"})
COOP_PHASE_B = frozenset(m for m in COOP
                         if MR.MILESTONE_TRACK[m] != "ch2_anvils") - PHASE_B_UNREACHABLE
SINGLE_CREDIT = frozenset({"m_door1_open"})
assert len(CH1) == 8 and len(COOP) == 17 and len(NONCOMM) == 25 and len(COOP_PHASE_B) == 12


def achievable(track: frozenset, n_agents: int) -> int:
    return sum(1 if m in SINGLE_CREDIT else n_agents for m in track)


def episode_completion(run: dict, tracks: dict) -> list[dict]:
    """Per episode: completion % for each named track + task returns."""
    per_agent = run.get("milestones_per_episode", [])
    n = int(run["config"]["num_agents"])
    n_eps = max((len(a) for a in per_agent), default=0)
    task_c, _ = MR.episode_task_returns(run, include_comm=True)
    task_nc, _ = MR.episode_task_returns(run, include_comm=False)
    graph = MR.end_of_episode_graph_stats(run)
    out = []
    for e in range(n_eps):
        sets = [set(a[e]) for a in per_agent if e < len(a)]
        rec = {"n_agents": n,
               "task": task_c[e] if e < len(task_c) else None,
               "task_nc": task_nc[e] if e < len(task_nc) else None,
               "W_mean": graph[e]["mean"] if e < len(graph) else None}
        for name, track in tracks.items():
            credited = sum(len(s & track) for s in sets)
            rec[name] = 100.0 * credited / achievable(track, n)
            rec[name + "_credits"] = credited
        out.append(rec)
    return out


# ─── aggregation ───────────────────────────────────────────────────────────
def aggregate(runs: list[dict], tracks: dict, unit: str) -> dict:
    """{metric: (mean, sd)} plus n_seeds / n_eps and per-seed rows."""
    per_seed, pooled = [], []
    for run in runs:
        eps = episode_completion(run, tracks)
        pooled.extend(eps)
        seed = Path(run["_path"]).parent.name
        row = {"seed": seed, "n_eps": len(eps)}
        for k in eps[0]:
            if k == "n_agents":
                continue
            vals = [e[k] for e in eps if e[k] is not None]
            row[k] = st.fmean(vals) if vals else None
        per_seed.append(row)
    if unit == "seed" and len(per_seed) >= 2:
        src = per_seed
        sd = lambda v: st.stdev(v) if len(v) >= 2 else 0.0  # noqa: E731
    else:
        src = pooled
        sd = st.pstdev
    out = {"n_seeds": len(runs), "n_eps": len(pooled), "per_seed": per_seed,
           "unit": "seed" if src is per_seed else "episode"}
    for k in pooled[0]:
        if k == "n_agents":
            continue
        vals = [r[k] for r in src if r.get(k) is not None]
        out[k] = (st.fmean(vals), sd(vals)) if vals else (math.nan, math.nan)
    return out


def cell(ms, nd, mark=None):
    m, s = ms
    if m != m:
        return "--"
    body = f"{m:.{nd}f}"
    if mark == "b":
        body = r"\mathbf{%s}" % body
    elif mark == "u":
        body = r"\underline{%s}" % body
    return r"$%s$ \pmm{%s}" % (body, f"{s:.{nd}f}")


def best_marks(vals: dict, nd: int, second: bool = False) -> dict:
    """Bold every row tied for the largest PRINTED value; optional underline."""
    means = {k: round(v[0], nd) for k, v in vals.items() if v[0] == v[0]}
    if len(means) < 2:
        return {}
    top = max(means.values())
    marks = {k: "b" for k, m in means.items() if m == top}
    if second:
        rest = [m for m in means.values() if m < top]
        if rest:
            snd = max(rest)
            marks.update({k: "u" for k, m in means.items() if m == snd})
    return marks


def unit_text(unit: str, n_txt: str) -> str:
    if unit == "seed":
        return ("Values are mean~$\\pm$~SD across seeds, each seed being the mean of its "
                "three episodes" + n_txt)
    return "Values are mean~$\\pm$~SD across episodes pooled over seeds" + n_txt


METRIC_TEXT = (
    r"\emph{Solo~\%} and \emph{Coop.~\%} count every (agent, milestone) completion "
    r"credited in an episode, over the completions achievable by the team: the eight "
    r"Chamber~1 milestones (Solo; $7N{+}1$ achievable, since Door~1 is unlocked by one "
    r"agent) and the seventeen Chamber~2--5 milestones (Coop.; $17N$ achievable). An "
    r"agent is credited with a milestone at most once per episode, and a milestone "
    r"completed by three agents counts three times, not once.")

# ─── Table 1 ───────────────────────────────────────────────────────────────
TABLE1 = [
    ("Zero-shot VLM agents", [
        [("(a)", r"Qwen3.5-2B", "LLM-2B"),
         ("(b)", r"\ \ +Social plasticity", "LLM-2B+Heb2.0")],
        [("(c)", r"Qwen3.5-9B", "LLM-9B"),
         ("(d)", r"\ \ +Social plasticity", "LLM-9B+Heb2.0")],
        [("(e)", r"Gemma-E4B", "Gemma-E4B"),
         ("(f)", r"\ \ +Central orch.", "Gemma-E4B+Central Orch."),
         ("(g)", r"\ \ +Social plasticity", "Gemma-E4B+Heb2.0")],
    ]),
    ("RL fine-tuned agents", [
        [("(h)", r"Qwen3.5-2B IPPO", "IPPO"),
         ("(i)", r"\ \ +Social plasticity", "IPPO+plast")],
        [("(j)", r"Qwen3.5-2B MAPPO", "MAPPO"),
         ("(k)", r"\ \ +Social plasticity", "MAPPO+plast")],
        [("(l)", r"Gemma-E4B IPPO", "Gemma IPPO"),
         ("(m)", r"\ \ +Social plasticity", "Gemma IPPO+plast")],
        [("(n)", r"Gemma-E4B MAPPO", "Gemma MAPPO"),
         ("(o)", r"\ \ +Social plasticity", "Gemma MAPPO+plast")],
    ]),
]
T1_TRACKS = {"solo": CH1, "coop": COOP, "all": NONCOMM}
T1_COLS = [("task", 0), ("solo", 1), ("coop", 1)]


def table1(unit: str, csv_rows: list, seed_rows: list) -> str:
    registry = {n: (d, r) for n, d, r, *_ in
                list(mft.ROWS) + list(NEW_ROWS) + list(EXTRA_ROWS)}
    agg = {}
    for _panel, blocks in TABLE1:
        for blk in blocks:
            for tag, lbl, key in blk:
                d, root = registry[key]
                runs = MR.load_runs(Path(root), d)
                if not runs:
                    print("  [skip] %s %s: no runs" % (tag, lbl), file=sys.stderr)
                    continue
                a = aggregate(runs, T1_TRACKS, unit)
                agg[tag] = a
                _record(csv_rows, seed_rows, "tab:final_comparison", tag, lbl, d, a,
                        ["task", "solo", "coop", "all"])
    n_seeds = sorted({a["n_seeds"] for a in agg.values()})
    n_txt = (" (%d seeds per condition, $\\dagger$ marks conditions below the six-seed target)"
             % n_seeds[0] if len(n_seeds) == 1 else
             " ($\\dagger$ marks conditions below the six-seed target)")
    panels = []
    for pi, (panel, blocks) in enumerate(TABLE1):
        L = [r"\begin{minipage}[t]{0.49\textwidth}", r"\centering",
             r"{\footnotesize\textbf{(%s) %s}}\\[2pt]" % ("ab"[pi], panel),
             r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{l c cc}", r"\toprule",
             r"\textbf{Condition}",
             r"& \makecell{\textbf{Task}\\\textbf{return} $\uparrow$}",
             r"& \makecell{\textbf{Solo}\\\textbf{\%} $\uparrow$}",
             r"& \makecell{\textbf{Coop.}\\\textbf{\%} $\uparrow$} \\",
             r"\midrule"]
        for bi, blk in enumerate(blocks):
            if bi:
                L.append(r"\midrule")
            tags = [t for t, _l, _k in blk if t in agg]
            marks = {c: best_marks({t: agg[t][c] for t in tags}, nd) for c, nd in T1_COLS}
            for tag, lbl, _key in blk:
                if tag not in agg:
                    L.append("%s %s & --- & --- & --- \\\\" % (tag, lbl))
                    continue
                n = agg[tag]["n_seeds"]
                dag = r"$^{\ddagger}$" if n <= 1 else (r"$^{\dagger}$" if n < SEED_TARGET else "")
                L.append("%s %s%s" % (tag, lbl, dag))
                L.append("& " + " & ".join(cell(agg[tag][c], nd, marks[c].get(tag))
                                           for c, nd in T1_COLS) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{minipage}"]
        panels.append("\n".join(L))
    caption = (r"\textbf{RQ1: Does social plasticity improve cooperative performance?} "
               + unit_text(unit, n_txt) + ". " + METRIC_TEXT
               + r" \emph{Task return} is the decomposed team return (task and "
               r"communication streams). Steps to first completion for zero-shot agents "
               r"are reported in \cref{tab:steps_to_milestone}.")
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption,
        r"\label{tab:final_comparison}", "", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.05}", "",
        "\n\\hfill\n".join(panels), "", r"\end{table}"]) + "\n"


# ─── Table 2: co-firing ────────────────────────────────────────────────────
COFIRE_ORDER = ["null (pr)", "prc", "pro", "pri", "prco", "prcoi"]
COFIRE_LABEL = {"null (pr)": r"$\mathrm None$", "prc": r"$\mathrm Comm$",
                "pro": r"$\mathrm Obs$", "pri": r"$\mathrm Imit$",
                "prco": r"$\mathrm Comm+Obs $", "prcoi": r"$\mathrm Comm+Obs+Imit $"}
CUE_LABEL = {"comm": "Comm", "obs": "Obs", "imit": "Imit"}


def table2(unit: str, csv_rows: list, seed_rows: list, group: str = "cofiring_bidi_3f") -> str:
    root = paths.group(group)
    arms = {cond: (exp, label, cues) for cond, exp, label, cues in CT.ARMS}
    rows = []
    for cond in COFIRE_ORDER:
        exp, label, cues = arms[cond]
        runs = MR.load_runs(root, exp)
        if not runs:
            print("  [skip] cofire %s: no runs" % exp, file=sys.stderr)
            continue
        a = aggregate(runs, {"solo": CH1, "coop": COOP, "all": NONCOMM}, unit)
        seeds = [Path(r["_path"]).parent for r in runs]
        per_cue = {}
        for cue in cues:
            stats = [CT.seed_cue_stats(s, [cue])[cue] for s in seeds]
            d = {k: CT.agg([x[k] for x in stats]) for k in ("dW", "rho")}
            if unit == "seed":
                d["act_use"] = CT.agg([st.fmean(x["ep_use"]) for x in stats if x["ep_use"]])
            else:
                d["act_use"] = CT.agg([v for x in stats for v in x["ep_use"]])
            per_cue[cue] = d
        rows.append({"cond": cond, "label": label, "agg": a, "cues": per_cue, "arm": exp})
        _record(csv_rows, seed_rows, "tab:cofire_main", cond, label, exp, a,
                ["task_nc", "solo", "coop", "all", "W_mean"])

    marks = {c: best_marks({r["cond"]: r["agg"][c] for r in rows}, nd)
             for c, nd in (("task_nc", 0), ("solo", 1), ("coop", 1), ("W_mean", 2))}
    cue_vals = {k: {} for k in ("act_use", "dW", "rho")}
    for r in rows:
        for cue, d in r["cues"].items():
            for k in cue_vals:
                m = d[k][0]
                if m is not None:
                    cue_vals[k][(r["cond"], cue)] = (100 * m if k != "rho" else m, 0.0)
    cue_marks = {"act_use": best_marks(cue_vals["act_use"], 1),
                 "dW": best_marks(cue_vals["dW"], 1),
                 "rho": best_marks(cue_vals["rho"], 2)}

    def cue_cell(d, k, key, nd):
        m, s, _n = d[k]
        if m is None:
            return "--"
        scale = 1.0 if k == "rho" else 100.0
        return cell((m * scale, (s or 0.0) * scale), nd, cue_marks[k].get(key))

    L = [r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{l ccc c l ccc}", r"\toprule",
         r"& \makecell{\textbf{Task}\\\textbf{return} $\uparrow$}",
         r"& \makecell{\textbf{Solo}\\\textbf{\%} $\uparrow$}",
         r"& \makecell{\textbf{Coop.}\\\textbf{\%} $\uparrow$}",
         r"& \makecell{\textbf{Mean}\\$\bm{W}$}",
         r"& \textbf{Cue}",
         r"& \makecell{\textbf{Act}\\\textbf{use \%}}",
         r"& \makecell{\textbf{Attributed}\\$\bm{\Delta W}$ \textbf{\%}}",
         r"& $\bm{\rho}$ \\", r"\midrule"]
    prev_group = None
    for r in rows:
        grp = 0 if r["cond"] == "null (pr)" else (1 if len(r["cues"]) == 1 else 2)
        if prev_group is not None and grp != prev_group:
            L.append(r"\midrule")
        prev_group = grp
        a, c = r["agg"], r["cond"]
        w_m = a["W_mean"][0]
        w_txt = "--" if w_m != w_m else (r"$\mathbf{%.2f}$" % w_m if marks["W_mean"].get(c) == "b"
                                          else "$%.2f$" % w_m)
        head = [COFIRE_LABEL[c], cell(a["task_nc"], 0, marks["task_nc"].get(c)),
                cell(a["solo"], 1, marks["solo"].get(c)), cell(a["coop"], 1, marks["coop"].get(c)),
                w_txt]
        if not r["cues"]:
            L.append(" & ".join(head + ["--", "--", "--", "--"]) + r" \\")
            continue
        k = len(r["cues"])
        wrap = (lambda s: r"\multirow{%d}{*}{%s}" % (k, s)) if k > 1 else (lambda s: s)
        first = True
        for cue, d in r["cues"].items():
            key = (c, cue)
            lead = [wrap(h) for h in head] if first else ["", "", "", "", ""]
            L.append(" & ".join(lead + [CUE_LABEL[cue], cue_cell(d, "act_use", key, 1),
                                        cue_cell(d, "dW", key, 1), cue_cell(d, "rho", key, 2)])
                     + r" \\")
            first = False
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    n_seeds = sorted({r["agg"]["n_seeds"] for r in rows})
    n_txt = " (%d seeds per condition)" % n_seeds[0] if len(n_seeds) == 1 else ""
    caption = (r"\textbf{RQ2: Which social signals produce strong, informative bonds?} "
               r"Rows vary the social cues used to define pairwise co-firing. ``None'' retains "
               r"only engaged co-location, while the remaining conditions add communication, "
               r"observation, imitation, or their combinations. " + unit_text(unit, n_txt) + ". "
               + METRIC_TEXT + r" \emph{Task return} is the team task reward with message pay "
               r"excluded (the mute conditions cannot earn it). We further report mean bond "
               r"strength $\overline{W}$ at episode end, social-act use, the fraction of "
               r"positive bond change attributed to each cue ($\Delta W$), and the Spearman "
               r"correlation $\rho$ between pairwise bond strength and use of the corresponding "
               r"social act. For multi-cue conditions, cue-specific statistics are reported on "
               r"separate rows.")
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption,
                      r"\label{tab:cofire_main}", "", r"\footnotesize",
                      r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.05}", "",
                      "\n".join(L), r"\end{table}"]) + "\n"


# ─── Table 3: transplant ───────────────────────────────────────────────────
def table3(csv_rows: list, seed_rows: list, group: str = "pair_bonding_3f") -> str:
    root = paths.group(group)
    suffix = "_3f" if group.endswith("_3f") else ""
    rows = []
    for tag, label, stem, mem, bond, _manifest in MTT.CONDITIONS:
        dirname = stem + suffix
        runs = MR.load_runs(root, dirname) if (root / dirname).is_dir() else []
        if not runs:
            rows.append({"tag": tag, "label": label, "mem": mem, "bond": bond, "agg": None})
            continue
        a = aggregate(runs, {"coop": COOP_PHASE_B}, "episode")   # single-seed cells
        cond = MTT.condition_data(root, dirname, None)             # partner preference
        pref = cond["pref"]
        a["pref"] = (st.fmean(pref), st.pstdev(pref)) if pref else (math.nan, math.nan)
        rows.append({"tag": tag, "label": label, "mem": mem, "bond": bond, "agg": a,
                     "arm": dirname})
        _record(csv_rows, seed_rows, "tab:transplant_main", "(%s)" % tag, label, dirname, a,
                ["task_nc", "coop", "pref"])
    have = [r for r in rows if r["agg"]]
    marks = {c: best_marks({r["tag"]: r["agg"][c] for r in have}, nd, second=True)
             for c, nd in (("task_nc", 0), ("coop", 1), ("pref", 2))}
    L = [r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{l cc c c c}", r"\toprule",
         r"& \multicolumn{2}{c}{Transplanted} & & & \\",
         r"\cmidrule(lr){2-3}",
         r"Condition & Memory & Bonds & Task return $\uparrow$ & Coop.\ \% $\uparrow$ & "
         r"Partner pref.\ $\uparrow$ \\", r"\midrule"]
    for r in rows:
        head = r"\textbf{(%s)} %s & %s & %s" % (r["tag"], r["label"], r["mem"], r["bond"])
        if not r["agg"]:
            L.append(head + r" & -- & -- & -- \\  % pending")
            continue
        a, t = r["agg"], r["tag"]
        L.append("%s & %s & %s & %s \\\\" % (
            head, cell(a["task_nc"], 0, marks["task_nc"].get(t)),
            cell(a["coop"], 1, marks["coop"].get(t)), cell(a["pref"], 2, marks["pref"].get(t))))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    n_seeds = sorted({r["agg"]["n_seeds"] for r in have})
    n_txt = ("%d seed" % n_seeds[0] + ("s" if n_seeds[0] != 1 else "")) if len(n_seeds) == 1 \
        else "%d--%d seeds" % (n_seeds[0], n_seeds[-1])
    caption = (r"\textbf{RQ3: partner transfer with controlled initial bonds} "
               r"(Gemma-E4B, $+$Social plasticity; " + n_txt + r" $\times$ 3 episodes, mean "
               r"$\pm$ SD over pooled episodes). Six Phase~A agents are seated as three pairs "
               r"and start "
               r"Phase~B in Chamber~3. \emph{Memory}: true shared Phase~A history, a fabricated "
               r"one (partners from different Phase~A runs), or none. \emph{Bonds}: the learned "
               r"Phase~A graph (within-pair $W=0.265$, cross-pair $0.10$) or a flat graph of the "
               r"same mean weight ($0.133$). \emph{Task return}: team physical reward, message "
               r"pay excluded. \emph{Coop.~\%}: every (agent, milestone) completion credited in "
               r"an episode over the $12N=72$ achievable, counting the twelve Chamber~3--5 "
               r"milestones reachable from a Chamber~3 start (cell entry cannot fire); a "
               r"milestone completed by three agents counts three times. \emph{Partner pref.}: "
               r"fraction of an agent's messages addressed to its seat partner (chance "
               r"$1/5=0.20$). Bold: best; underlined: second best. Per-pair breakdown in "
               r"\cref{tab:transplant_pairs}.")
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption,
                      r"\label{tab:transplant_main}", r"\footnotesize",
                      r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.05}",
                      "\n".join(L), r"\end{table}"]) + "\n"


# ─── Table 3 companion: per-pair breakdown ─────────────────────────────────
import re as _re

_AGENT_RE = _re.compile(r"agent_?(\d+)")


def _agent_idx(name):
    m = _AGENT_RE.fullmatch(str(name))
    return int(m.group(1)) if m else None


def _handoffs(run: dict) -> list:
    """[(episode, presser, freed)]: a switch press (M13) and the door it
    opened (M14) fire at the same step for two different agents."""
    by = {}
    for ev in run.get("milestone_events", []):
        by.setdefault((int(ev["step"]), ev["milestone_id"]), []).append(_agent_idx(ev["contributor"]))
    out = []
    for (step, mid), who in by.items():
        if mid != "m17_switch_pressed":
            continue
        e, _ = MR.ep_of_step(run["_ep_bounds"], step)
        for p in who:
            for f in by.get((step, "m18_door_opened"), []):
                out.append((e, p, f))
    return out


def phase_a_pairs(manifest: dict) -> list:
    """Seat pairs whose members came from the same Phase A run."""
    by_run = {}
    for seat, a in (manifest.get("agents") or {}).items():
        by_run.setdefault(a.get("source_run"), []).append(int(seat))
    return [tuple(sorted(v)) for v in by_run.values() if len(v) == 2]


def pair_completions(run: dict) -> dict:
    """{(a, b): [(completions, with_partner)] per episode}.

    completions  = non-communication milestone credits of the two members
    with_partner = credits produced jointly with the partner. In Phase B the
                   only joint mechanism is the Chamber-3 handoff: one member's
                   switch press (M13) opens the partner's door (M14), so both
                   credits count. Each switch fires once per episode, so at
                   most one such handoff (two credits) per pair and episode.
                   Kills are individual and never count.
    """
    bounds = run["_ep_bounds"]
    per_agent = run.get("milestones_per_episode", [])
    n_eps = max((len(a) for a in per_agent), default=len(bounds))
    handoffs = _handoffs(run)
    out = {}
    for a, b in MTT.PAIRS:
        rows = []
        for e in range(n_eps):
            compl = sum(len({m for m in per_agent[i][e] if m in NONCOMM})
                        for i in (a, b) if e < len(per_agent[i]))
            joint = 2 * sum(1 for (ee, p, f) in handoffs if ee == e and {p, f} == {a, b})
            rows.append((compl, joint))
        out[(a, b)] = rows
    return out


def table_pairs(csv_rows: list, group: str = "pair_bonding_3f") -> str:
    root = paths.group(group)
    suffix = "_3f" if group.endswith("_3f") else ""
    merged_root = paths.group("pair_bonding") / "merged"
    conds = []
    for tag, label, stem, _mem, _bond, manifest in MTT.CONDITIONS:
        dirname = stem + suffix
        runs = MR.load_runs(root, dirname) if (root / dirname).is_dir() else []
        labels = MTT.pair_labels(MTT.load_manifest(merged_root, manifest))
        if not runs:
            conds.append({"tag": tag, "label": label, "dir": dirname, "labels": labels,
                          "data": None})
            continue
        cond = MTT.condition_data(root, dirname, None)      # pref / msgs / W per pair
        pc = {p: [] for p in MTT.PAIRS}
        for run in runs:
            for p, rows in pair_completions(run).items():
                pc[p].extend(rows)
        data = {}
        for p in MTT.PAIRS:
            pd = cond["pairs"][p]
            compl = [c for c, _ in pc[p]]
            joint = [j for _, j in pc[p]]
            frac = [100.0 * j / c for c, j in pc[p] if c > 0]
            data[p] = {"pref": pd["pref"], "msgs": pd["msgs"], "w": pd["w"],
                       "compl": compl, "joint": joint, "frac": frac}
            csv_rows.append({"table": "tab:transplant_pairs", "row": "(%s)" % tag,
                             "label": label, "arm": dirname, "pair": "%d-%d" % p,
                             "unit": "episode", "n_seeds": cond["n_runs"], "n_eps": cond["n_eps"],
                             "pref_mean": st.fmean(pd["pref"]) if pd["pref"] else math.nan,
                             "msgs_mean": st.fmean(pd["msgs"]) if pd["msgs"] else math.nan,
                             "completions_mean": st.fmean(compl) if compl else math.nan,
                             "with_partner_mean": st.fmean(joint) if joint else math.nan,
                             "with_partner_pct": st.fmean(frac) if frac else math.nan,
                             "W_mean": st.fmean(pd["w"]) if pd["w"] else math.nan})
        entry = {"tag": tag, "label": label, "dir": dirname, "labels": labels,
                 "data": data, "n_runs": cond["n_runs"], "n_eps": cond["n_eps"],
                 "cross_w": cond["cross_w"]}
        if manifest == "shuffled":
            # The Phase A teammates sit elsewhere here; say in the caption how
            # often THEY handed off, next to the seat (stranger) pairs.
            pa = phase_a_pairs(MTT.load_manifest(merged_root, manifest))
            hs = [h for run in runs for h in _handoffs(run)]
            entry["shuffled_note"] = (
                r" In (%s) the Phase~A teammates sit at seats %s; across its %d episodes "
                r"they completed %d handoff%s together, against %d for the stranger seat "
                r"pairs." % (tag, ", ".join("(%d,%d)" % p for p in pa), cond["n_eps"],
                             sum(1 for _, p, f in hs if tuple(sorted((p, f))) in pa),
                             "" if sum(1 for _, p, f in hs if tuple(sorted((p, f))) in pa) == 1 else "s",
                             sum(1 for _, p, f in hs if tuple(sorted((p, f))) in MTT.PAIRS)))
        conds.append(entry)

    def ms_(vals, nd):
        return (st.fmean(vals), st.pstdev(vals)) if vals else (math.nan, math.nan)

    series = {k: {} for k in ("pref", "msgs", "compl", "joint", "frac", "w")}
    for c in conds:
        if c["data"]:
            for k, p in enumerate(MTT.PAIRS):
                for col in series:
                    series[col][(c["tag"], k)] = ms_(c["data"][p][col], 3)
    marks = {col: best_marks(v, {"pref": 2, "msgs": 0, "compl": 1, "joint": 1, "frac": 0,
                                 "w": 3}[col], second=True) for col, v in series.items()}
    cross = ["(%s) %s" % (c["tag"], cell(ms_(c["cross_w"], 3), 3))
             for c in conds if c["data"] and c["cross_w"]]
    n_seeds = sorted({c["n_runs"] for c in conds if c["data"]})
    n_txt = ("%d seed%s" % (n_seeds[0], "" if n_seeds[0] == 1 else "s")) if len(n_seeds) == 1 \
        else "%d--%d seeds" % (n_seeds[0], n_seeds[-1])

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{RQ3: partner transfer, per-pair breakdown} (companion to "
         r"\cref{tab:transplant_main}; Gemma-E4B, $+$Social plasticity; " + n_txt +
         r" $\times$ 3 episodes, mean $\pm$ SD over pooled episodes). Conditions as in "
         r"\cref{tab:transplant_main}. \emph{Partner pref.}: fraction of a member's messages "
         r"addressed to its partner, averaged over the pair's two members (chance $1/5=0.20$). "
         r"\emph{Msgs within}: messages exchanged between the two members per episode. "
         r"\emph{Bond $W$}: end-of-episode within-pair weight, both directions averaged "
         r"(initialised at $0.265$ with learned bonds, $0.133$ with flat bonds). Cross-pair $W$ "
         r"at episode end, pooled per condition: " + "; ".join(cross) +
         r". Bold: best; underline: second best.}",
         r"\label{tab:transplant_pairs}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.05}",
         r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{l l c c c}", r"\toprule",
         r"Condition & Pair & Partner pref. & Msgs within & Bond $W$ \\", r"\midrule"]
    first = True
    for c in conds:
        if not first:
            L.append(r"\addlinespace")
        first = False
        for k, p in enumerate(MTT.PAIRS):
            head = (r"\multirow{3}{*}{\textbf{(%s)} %s}" % (c["tag"], c["label"])) if k == 0 else ""
            pair = r"\textbf{(%s-%d)} %s" % (c["tag"], k + 1, c["labels"][k])
            if not c["data"]:
                L.append("%s & %s & -- & -- & -- & -- & -- & -- \\\\%s" % (
                    head, pair, "  %% pending: %s" % c["dir"] if k == 0 else ""))
                continue
            d = c["data"][p]
            key = (c["tag"], k)
            L.append("%s & %s & %s & %s & %s \\\\" % (
                head, pair,
                cell(ms_(d["pref"], 2), 2, marks["pref"].get(key)),
                cell(ms_(d["msgs"], 0), 0, marks["msgs"].get(key)),
                cell(ms_(d["w"], 3), 3, marks["w"].get(key))))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ─── Table 12: imposed topology across horizons ────────────────────────────
HORIZON_SEEDS = {"seed_42", "seed_123"}        # the only seeds run at both budgets
HORIZON_CONDITIONS = [("No-bonds", "exp11_llm_9b_allied_none"),
                      ("Allied-pair", "exp10_llm_9b_allied_pair"),
                      ("Allied-all", "exp09_llm_9b_allied_all")]


def table_horizon(csv_rows: list) -> str:
    budgets = [("$1{,}000$", paths.group("medium_runs")), ("$2{,}000$", paths.group("medium_2k"))]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{RQ3: imposed topology across horizons} (Qwen3.5-9B; fixed "
         r"\emph{No-bonds}, \emph{Allied-pair}, and \emph{Allied-all} graphs at 1,000- and "
         r"2,000-step horizons on the same two seeds, $42$ and $123$; $n{=}6$ episodes per "
         r"cell, mean $\pm$ population SD over pooled episodes). " + METRIC_TEXT +
         r" \emph{All} covers the 25 non-communication milestones ($24N{+}1$ achievable). "
         r"\emph{Task return}: team-summed environment reward including communication pay, "
         r"as in the RQ1 table these runs share.}",
         r"\label{tab:topology_horizon}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{tabular}{l l ccc c}", r"\toprule",
         r"& & \multicolumn{3}{c}{Milestone \%} & \\", r"\cmidrule(lr){3-5}",
         r"Condition & Steps & Solo $\uparrow$ & Coop.\ $\uparrow$ & All $\uparrow$ & "
         r"Task return $\uparrow$ \\", r"\midrule"]
    tracks = {"solo": CH1, "coop": COOP, "all": NONCOMM}
    for ci, (label, dirname) in enumerate(HORIZON_CONDITIONS):
        if ci:
            L.append(r"\addlinespace")
        for bi, (steps, root) in enumerate(budgets):
            runs = [r for r in MR.load_runs(root, dirname)
                    if Path(r["_path"]).parent.name in HORIZON_SEEDS]
            head = (r"\multirow{2}{*}{%s}" % label) if bi == 0 else ""
            if not runs:
                L.append("%s & %s & -- & -- & -- & -- \\\\" % (head, steps))
                continue
            a = aggregate(runs, tracks, "episode")
            L.append("%s & %s & %s & %s & %s & %s \\\\" % (
                head, steps, cell(a["solo"], 1), cell(a["coop"], 1), cell(a["all"], 1),
                cell(a["task"], 0)))
            csv_rows.append({"table": "tab:topology_horizon", "row": label,
                             "label": "%s @ %s steps" % (label, steps.strip("$").replace("{,}", "")),
                             "arm": "%s/%s" % (root.name, dirname), "unit": "episode",
                             "n_seeds": a["n_seeds"], "n_eps": a["n_eps"],
                             "task_mean": a["task"][0], "task_sd": a["task"][1],
                             "solo_mean": a["solo"][0], "solo_sd": a["solo"][1],
                             "coop_mean": a["coop"][0], "coop_sd": a["coop"][1],
                             "all_mean": a["all"][0], "all_sd": a["all"][1]})
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ─── bookkeeping ───────────────────────────────────────────────────────────
def _record(csv_rows, seed_rows, table, tag, label, arm, a, metrics):
    row = {"table": table, "row": tag, "label": label.replace("\\ \\ ", "").strip(),
           "arm": arm, "unit": a["unit"], "n_seeds": a["n_seeds"], "n_eps": a["n_eps"]}
    for m in metrics:
        if m in a and isinstance(a[m], tuple):
            row[m + "_mean"], row[m + "_sd"] = a[m]
    csv_rows.append(row)
    for s in a["per_seed"]:
        seed_rows.append({"table": table, "row": tag, "arm": arm, **s})


METRICS_MD = """# Completion metrics used by the agent-completion tables

Generated by analysis/make_agent_completion_tables.py.

**Completion % (per episode).** Every (agent, milestone) pair credited in the
episode counts once (Lua fires each milestone at most once per agent per
episode). The denominator is what the team could achieve: N per milestone,
except Door 1 unlocked, which goes to exactly one agent.

| Column | Milestones | Achievable (N agents) | N=3 | N=6 |
|---|---|---|---|---|
| Solo % | 8 Chamber-1 milestones (M1-M7 + Door 1) | 7N + 1 | 22 | 43 |
| Coop. % | 17 Chamber-2..5 milestones | 17N | 51 | 102 |
| All % (CSV only) | 25 non-communication milestones | 24N + 1 | 73 | 145 |
| Coop. % (transplant) | 12 reachable Chamber-3..5 milestones (cell entry cannot fire from a Chamber-3 start) | 12N | -- | 72 |

The communication, observation and imitation tracks are never counted.
The entry-honesty filter of make_results is applied first (Chamber-4 / Chamber-5
entry without the enabling door / arena-clear milestone in the same episode is
removed).

**Difference to the previous columns.** The published Milest. % / Coop. %
took the UNION over agents (a milestone counted once for the team however
many agents completed it) over 25 / 17. Union values are roughly 1.6-1.7x the
per-agent values because on average that many agents are credited per fired
milestone.

**Aggregation.** Seed unit: each run's three episodes are averaged, then mean
+- sample SD (ddof=1) across seeds. Episode unit: mean +- population SD over
pooled episodes. Table 3 is single-seed and always uses the episode unit.

**Task return.** Table 1: decomposed task + communication reward streams
(unchanged). Tables 2 and 3: task stream only, message pay excluded (unchanged).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=["seed", "episode"], default="seed")
    ap.add_argument("--out", type=Path, default=ASSETS / "agent_completion")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    csv_rows, seed_rows = [], []
    t1 = table1(args.unit, csv_rows, seed_rows)
    t2 = table2(args.unit, csv_rows, seed_rows)
    t3 = table3(csv_rows, seed_rows)
    t4 = table_pairs(csv_rows)
    t5 = table_horizon(csv_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final_comparison_agent.tex").write_text(t1, encoding="utf-8")
    (args.out / "cofire_main_agent.tex").write_text(t2, encoding="utf-8")
    (args.out / "transplant_main_agent.tex").write_text(t3, encoding="utf-8")
    (args.out / "transplant_pairs_agent.tex").write_text(t4, encoding="utf-8")
    (args.out / "topology_horizon_agent.tex").write_text(t5, encoding="utf-8")
    (args.out / "METRICS.md").write_text(METRICS_MD, encoding="utf-8")
    keys = []
    for r in csv_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (args.out / "agent_completion_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    keys = []
    for r in seed_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (args.out / "agent_completion_per_seed.csv").open("w", newline="",
                                                          encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(seed_rows)

    print(t1)
    print(t2)
    print(t3)
    print(t4)
    print(t5)
    print("wrote %s/ (unit=%s)" % (args.out, args.unit), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
