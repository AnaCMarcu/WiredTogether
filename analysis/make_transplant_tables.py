"""RQ3 partner-transfer tables, split into a short main-text table and a
per-pair appendix table.

The paper's Table 3 (``tab:transplant_main``) grew a per-pair block that made
the main-text table wide.  This script emits two files instead:

* ``transplant_main_<group>.tex``  — one row per condition: the two factor
  levels (memory, bonds), task return, milestone %, cooperative milestone %,
  and partner preference (raw and ring-conditioned).
* ``transplant_pairs_<group>.tex`` — the per-pair breakdown (partner
  preference, messages within the pair, proximity share, end-of-episode bond
  W) for the appendix, ``tab:transplant_pairs``.

Conditions are the five memory x bond cells of the transplant design; a cell
whose runs have not landed is written as a ``--`` row with a ``% pending``
comment (or dropped with ``--drop-pending``, for a paper build that should
not show empty rows).

    cell  condition           memory      bonds     arm directory
    (a)   Co-fired Partner    true        learned   expB_merged_transplant
    (b)   Re-paired Partner   fabricated  learned   expB_merged_shuffled
    (c)   Memory only         true        flat      expB_memory_only
    (d)   Bonds only          none        learned   expB_bond_only
    (e)   Neither             none        flat      expB_neither

Numbers follow ``src/mindforge/tools/make_transplant_table.py`` exactly (the
published Table 3 reproduces with ``--group pair_bonding --seeds 42 123 456``):
``make_results.load_runs`` slicing + entry-honesty filter, team-union
milestones per episode, 25/17 denominators, message pay excluded from task
return, per-episode values pooled over all seeds, mean +/- population SD.
The one addition is the ring-conditioned preference, P(target = seatmate |
target is a ring neighbour), chance 0.50 at any N (see
``analyze_wiring.ring_conditioned_preference``); raw seatmate preference
(chance 0.20 at N = 6) is kept beside it because the paper text cites it.

Bold = best value in a column, underline = second best, over the rows that
have data (main table: over conditions; pair table: over all pair rows).

Usage (from anywhere)::

    python analysis/make_transplant_tables.py                      # pair_bonding_3f, +trace
    python analysis/make_transplant_tables.py --group pair_bonding \\
        --seeds 42 123 456 --drop-pending                          # the published +inst. table
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import paths  # noqa: F401  (sys.path: analysis/, repo root, src/)
import make_results as mr  # noqa: E402
from mindforge.tools.analyze_wiring import (  # noqa: E402
    load_message_matrix,
    mean_preference,
    ring_conditioned_preference,
    seatmate_preference,
)

N_AGENTS = 6
PAIRS = [(0, 1), (2, 3), (4, 5)]

# (tag, label, arm directory stem, memory level, bond level, manifest name)
# The manifest names the seating of the transplanted memories; the fresh-agent
# cells have none, so their pairs are labelled by seat only.
CONDITIONS = [
    ("a", "Co-fired Partner", "expB_merged_transplant", "true", "learned",
     "transplant"),
    ("b", "Re-paired Partner", "expB_merged_shuffled", "fabricated", "learned",
     "shuffled"),
    ("c", "Memory only", "expB_memory_only", "true", "flat", "transplant"),
    ("d", "Bonds only", "expB_bond_only", "none", "learned", None),
    ("e", "Neither", "expB_neither", "none", "flat", None),
]

# Rule label per group, for the captions. Anything else is printed verbatim.
RULE_LABEL = {"pair_bonding": "+inst.", "pair_bonding_3f": "+trace"}


# ── formatting ────────────────────────────────────────────────────────────

def _fmt(vals, nd, mark=None):
    """mean +/- population SD over pooled per-episode values, ``\\pmm`` style.

    mark: None, "b" (bold) or "u" (underline) on the mean.
    """
    if not vals:
        return "--"
    m, sd = st.fmean(vals), st.pstdev(vals)
    body = f"{m:.{nd}f}"
    if mark == "b":
        body = f"\\mathbf{{{body}}}"
    elif mark == "u":
        body = f"\\underline{{{body}}}"
    return f"${body}$ \\pmm{{{sd:.{nd}f}}}"


def _marks(series):
    """{key: 'b' | 'u'} for the largest and second-largest mean in `series`
    ({key: list-of-values}); rows without data are ignored. With a single
    row nothing is marked (there is nothing to compare against)."""
    means = {k: st.fmean(v) for k, v in series.items() if v}
    if len(means) < 2:
        return {}
    order = sorted(means, key=means.get, reverse=True)
    return {order[0]: "b", order[1]: "u"}


# ── data ──────────────────────────────────────────────────────────────────

def pair_labels(manifest):
    """Pair label per seat pair from the merge manifest's provenance, or the
    seat-only label when the arm transplants no memories."""
    if not manifest:
        return [f"Pair {k + 1} (no history)" for k in range(len(PAIRS))]
    # Every same-source pair co-fired in Phase A in the rule's sense: its bond
    # grew through co-activity (joint digging, messaging) to the same ~0.27.
    # The manifest's `cofired` flag marks the stricter anvil-milestone
    # criterion (pairs 1-2 only) and is stated in the caption, not the label.
    out = []
    by_seats = {tuple(sp["seats"]): sp for sp in manifest.get("seat_pairs", [])}
    for k, p in enumerate(PAIRS):
        sp = by_seats.get(p, {})
        kind = "co-fired" if sp.get("same_source_run") else "strangers"
        out.append(f"Pair {k + 1} ({kind})")
    return out


def load_manifest(merged_root, name):
    if name is None:
        return {}
    f = merged_root / name / "merged_manifest.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def condition_data(runs_root, dir_name, seeds):
    """Pooled per-episode metrics for one condition, or None without runs."""
    runs = mr.load_runs(runs_root, dir_name)
    if seeds:
        keep = {f"seed_{s}" for s in seeds}
        runs = [r for r in runs if Path(r["_path"]).parent.name in keep]
    if not runs:
        return None
    cond = {"ret": [], "ms": [], "coop": [], "pref": [], "ring": [],
            "cross_w": [], "n_runs": len(runs), "seeds": []}
    pairs = {p: {"pref": [], "ring": [], "msgs": [], "prox": [], "w": []}
             for p in PAIRS}
    for run in runs:
        run_dir = Path(run["_path"]).parent
        cond["seeds"].append(run_dir.name)
        rets, _ = mr.episode_task_returns(run, include_comm=False)
        cond["ret"].extend(rets)
        for team in mr.episode_milestone_sets(run):
            noncomm = sum(1 for m in team
                          if mr.MILESTONE_TRACK.get(m) != "communication")
            cond["ms"].append(100.0 * noncomm / mr.NONCOMM_MAX)
            cond["coop"].append(100.0 * mr.coop_count(team) / mr.COOP_MAX)
        _, per_ep = load_message_matrix(run_dir, N_AGENTS)
        for _ep, mat in sorted(per_ep.items()):
            raw = seatmate_preference(mat)
            ring = ring_conditioned_preference(mat)
            v = mean_preference(raw)
            if v is not None:
                cond["pref"].append(v)
            v = mean_preference(ring)
            if v is not None:
                cond["ring"].append(v)
            for a, b in PAIRS:
                pv = [x for x in (raw[a][0], raw[b][0]) if x is not None]
                if pv:
                    pairs[(a, b)]["pref"].append(st.fmean(pv))
                rv = [x for x in (ring[a][0], ring[b][0]) if x is not None]
                if rv:
                    pairs[(a, b)]["ring"].append(st.fmean(rv))
                pairs[(a, b)]["msgs"].append(mat[a][b] + mat[b][a])
        # End-of-episode W and proximity share come from the episode
        # summaries: final_metrics.json zeroes these tensors (coop_eval
        # nesting bug), so the per-episode files are the only source.
        for f in sorted(run_dir.glob("episodes/*/summary.json")):
            cm = json.loads(f.read_text(encoding="utf-8")).get(
                "cooperation_metrics") or {}
            hw = cm.get("hebbian_W")
            if hw:
                for a, b in PAIRS:
                    pairs[(a, b)]["w"].append((hw[a][b] + hw[b][a]) / 2.0)
                cond["cross_w"].extend(
                    hw[i][j] for i in range(N_AGENTS) for j in range(N_AGENTS)
                    if i != j and i // 2 != j // 2)
            prox = (cm.get("pair_interaction") or {}).get("proximity")
            if prox:
                total = sum(prox[i][j] for i in range(N_AGENTS)
                            for j in range(N_AGENTS) if i != j)
                if total:
                    for a, b in PAIRS:
                        pairs[(a, b)]["prox"].append(
                            100.0 * (prox[a][b] + prox[b][a]) / total)
    cond["n_eps"] = len(cond["ms"])
    cond["pairs"] = pairs
    return cond


def collect(group, suffix, seeds, merged_root):
    runs_root = paths.group(group)
    out = []
    for tag, label, stem, mem, bond, manifest_name in CONDITIONS:
        dir_name = stem + suffix
        cond = (condition_data(runs_root, dir_name, seeds)
                if (runs_root / dir_name).is_dir() else None)
        out.append({
            "tag": tag, "label": label, "dir": dir_name,
            "memory": mem, "bond": bond,
            "pair_labels": pair_labels(load_manifest(merged_root,
                                                     manifest_name)),
            "data": cond,
        })
    return out


# ── tables ────────────────────────────────────────────────────────────────

def _n_note(cond):
    """Comment flagging a condition that is not the nominal 3 seeds x 3 eps."""
    if cond["n_runs"] == 3 and cond["n_eps"] == 9:
        return ""
    return (f"  % {cond['n_runs']} runs ({', '.join(cond['seeds'])}), "
            f"{cond['n_eps']} episodes")


def _preamble(script_note):
    return [
        "% Generated by analysis/make_transplant_tables.py -- regenerate, do",
        f"% not hand-edit. {script_note}",
        "% Requires booktabs, multirow, adjustbox and \\pmm.",
    ]


def main_table(conds, rule, drop_pending):
    rows = [c for c in conds if c["data"] or not drop_pending]
    series = {k: {c["tag"]: (c["data"] or {}).get(k, []) for c in rows}
              for k in ("ret", "ms", "coop", "pref", "ring")}
    marks = {k: _marks(v) for k, v in series.items()}
    n_rows = [c for c in rows if c["data"]]
    seeds = sorted({c["data"]["n_runs"] for c in n_rows})
    n_txt = ("no runs yet" if not seeds
             else f"{seeds[0]} seed{'s' if seeds[0] != 1 else ''}"
             if len(seeds) == 1
             else f"{min(seeds)}--{max(seeds)} seeds")

    L = _preamble("Main-text version: one row per memory x bond cell.")
    L += [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{RQ3: partner transfer with controlled initial "
        f"bonds}} (Gemma-E4B, {rule}; {n_txt} $\\times$ 3 episodes, "
        "mean $\\pm$ SD over pooled episodes). Six Phase~A agents are seated "
        "as three pairs and start Phase~B in Chamber~3. \\emph{Memory}: "
        "whether the seat pairs carry their true shared Phase~A history, a "
        "fabricated one (partners drawn from different Phase~A runs), or "
        "none (fresh agents). \\emph{Bonds}: the learned Phase~A graph "
        "(within-pair $W=0.265$, cross-pair $0.10$) or a flat graph of the "
        "same mean weight ($0.133$). \\emph{Task return}: team physical "
        "reward, message pay excluded. \\emph{Milestone \\%}: share of the "
        "25 non-communication milestones (All) and of the 17 cooperative "
        "Ch2--5 milestones (Coop.). \\emph{Partner pref.}: fraction of an "
        "agent's messages addressed to its seat partner. \\emph{Raw} counts "
        "every message (chance $1/5=0.20$). \\emph{Ring} removes task "
        "geometry: in Chamber~3 each agent's switch opens the door of the "
        "next agent around the ring, so every agent has two task-linked "
        "neighbours, one of which is its seat partner; Ring is the fraction "
        "of the messages sent to those two neighbours that went to the "
        "partner (chance $0.50$). Per-pair breakdown in "
        "\\cref{tab:transplant_pairs}.}",
        "\\label{tab:transplant_main}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{l cc c cc cc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Transplanted} & & "
        "\\multicolumn{2}{c}{Milestone \\%} & "
        "\\multicolumn{2}{c}{Partner pref.} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        "Condition & Memory & Bonds & Task return $\\uparrow$ & "
        "All $\\uparrow$ & Coop.\\ $\\uparrow$ & Raw $\\uparrow$ & "
        "Ring $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for c in rows:
        head = (f"\\textbf{{({c['tag']})}} {c['label']} & {c['memory']} & "
                f"{c['bond']}")
        d = c["data"]
        if d is None:
            L.append(f"{head} & -- & -- & -- & -- & -- \\\\  % pending: "
                     f"{c['dir']}")
            continue
        t = c["tag"]
        L.append(
            f"{head} & {_fmt(d['ret'], 0, marks['ret'].get(t))} & "
            f"{_fmt(d['ms'], 1, marks['ms'].get(t))} & "
            f"{_fmt(d['coop'], 1, marks['coop'].get(t))} & "
            f"{_fmt(d['pref'], 2, marks['pref'].get(t))} & "
            f"{_fmt(d['ring'], 2, marks['ring'].get(t))} \\\\{_n_note(d)}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
          "\\end{table}"]
    return "\n".join(L) + "\n"


def pair_table(conds, rule, drop_pending):
    rows = [c for c in conds if c["data"] or not drop_pending]
    # bold / underline over every pair row of the table, per column
    series = {k: {} for k in ("pref", "ring", "msgs", "prox", "w")}
    for c in rows:
        if not c["data"]:
            continue
        for k, p in enumerate(PAIRS):
            for col in series:
                series[col][(c["tag"], k)] = c["data"]["pairs"][p][col]
    marks = {k: _marks(v) for k, v in series.items()}
    cross = [f"({c['tag']}) {_fmt(c['data']['cross_w'], 3)}"
             for c in rows if c["data"] and c["data"]["cross_w"]]

    L = _preamble("Appendix version: per-pair breakdown of "
                  "tab:transplant_main.")
    L += [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\textbf{RQ3: partner transfer, per-pair breakdown} "
        f"(companion to \\cref{{tab:transplant_main}}; Gemma-E4B, {rule}; "
        "mean $\\pm$ SD over pooled episodes). "
        "All three pairs of the co-fired conditions formed their bond "
        "through Phase~A co-activity (joint digging and messaging); "
        "pairs 1--2 also broke an anvil together. "
        "\\emph{Partner pref.}: fraction of a member's messages addressed "
        "to its partner, averaged over the pair's two members. \\emph{Raw} "
        "counts every message (chance $1/5=0.20$). \\emph{Ring} counts "
        "only messages sent to the agent's two task-linked neighbours "
        "(in Chamber~3 each agent's switch opens the door of the next agent "
        "around the ring; the seat partner is one of the two), so it asks "
        "which of two equally task-relevant teammates the agent chose "
        "(chance $0.50$). \\emph{Msgs within}: messages exchanged between "
        "the two members per episode. \\emph{Prox.\\ share}: a proximity "
        "event is one step on which two agents stand within 4 blocks of "
        "each other; the column is the share of all such events in the "
        "episode that involve this pair (15 pairs at $N=6$, so chance is "
        "$1/15\\approx6.7\\%$). \\emph{Bond $W$}: end-of-episode "
        "within-pair weight, both directions averaged (initialised at "
        "$0.265$ with learned bonds, $0.133$ with flat bonds). Cross-pair "
        "$W$ at episode end, pooled per condition: " + "; ".join(cross)
        + ".}",
        "\\label{tab:transplant_pairs}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{l l cc c c c}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{Partner pref.} & & & \\\\",
        "\\cmidrule(lr){3-4}",
        "Condition & Pair & Raw & Ring & Msgs within & "
        "Prox.\\ share (\\%) & Bond $W$ \\\\",
        "\\midrule",
    ]
    first = True
    for c in rows:
        if not first:
            L.append("\\addlinespace")
        first = False
        t = c["tag"]
        d = c["data"]
        for k, p in enumerate(PAIRS):
            head = (f"\\multirow{{3}}{{*}}{{\\textbf{{({t})}} {c['label']}}}"
                    if k == 0 else "")
            pair = f"\\textbf{{({t}-{k + 1})}} {c['pair_labels'][k]}"
            if d is None:
                L.append(f"{head} & {pair} & -- & -- & -- & -- & -- \\\\"
                         + (f"  % pending: {c['dir']}" if k == 0 else ""))
                continue
            pd = d["pairs"][p]
            key = (t, k)
            L.append(
                f"{head} & {pair} & "
                f"{_fmt(pd['pref'], 2, marks['pref'].get(key))} & "
                f"{_fmt(pd['ring'], 2, marks['ring'].get(key))} & "
                f"{_fmt(pd['msgs'], 0, marks['msgs'].get(key))} & "
                f"{_fmt(pd['prox'], 1, marks['prox'].get(key))} & "
                f"{_fmt(pd['w'], 3, marks['w'].get(key))} \\\\"
                + (_n_note(d) if k == 0 else ""))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
          "\\end{table}"]
    return "\n".join(L) + "\n"


# ── cli ───────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", default="pair_bonding_3f",
                    help="run group (pair_bonding_3f = +trace, "
                         "pair_bonding = the published +inst. runs)")
    ap.add_argument("--suffix", default=None,
                    help="arm-directory suffix; default '_3f' when the group "
                         "name ends in _3f, else ''")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="restrict to these seeds (default: every seed on "
                         "disk)")
    ap.add_argument("--merged-group", default="pair_bonding",
                    help="group whose merged/ manifests name the seating "
                         "(the _3f cells transplant the reward_modulated "
                         "Phase-A merge)")
    ap.add_argument("--drop-pending", action="store_true",
                    help="omit conditions with no runs instead of writing "
                         "-- rows")
    ap.add_argument("--out-dir", default=str(paths.ASSETS / "transplant"))
    args = ap.parse_args(argv)

    suffix = args.suffix
    if suffix is None:
        suffix = "_3f" if args.group.endswith("_3f") else ""
    merged_root = paths.group(args.merged_group) / "merged"
    rule = RULE_LABEL.get(args.group, args.group)

    conds = collect(args.group, suffix, args.seeds, merged_root)
    for c in conds:
        d = c["data"]
        print(f"  ({c['tag']}) {c['dir']:<32} "
              + (f"runs={d['n_runs']} ({', '.join(d['seeds'])}) "
                 f"eps={d['n_eps']}" if d else "no runs"),
              file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_f = out_dir / f"transplant_main_{args.group}.tex"
    pair_f = out_dir / f"transplant_pairs_{args.group}.tex"
    main_f.write_text(main_table(conds, rule, args.drop_pending),
                      encoding="utf-8")
    pair_f.write_text(pair_table(conds, rule, args.drop_pending),
                      encoding="utf-8")
    print(f"wrote {main_f}\nwrote {pair_f}")


if __name__ == "__main__":
    main()
