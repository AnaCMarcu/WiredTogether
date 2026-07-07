#!/usr/bin/env python3
"""make_results.py — fill the Wired Together results tables and figures.

Reads   runs/final/<condition>/seed_<N>/final_metrics.json   (schema from
src/mindforge/agent_modules/craftium_metric.py) and writes:

  <out>/table_rows.tex            rows for tab:main_results,
                                  tab:steps_to_milestone,
                                  tab:topology_ablation, tab:graph_stats
  <out>/summary.csv               every number, machine-readable
  <out>/milestone_progression.pdf fig:milestone_progression
  <out>/milestone_timeline.pdf    fig:milestone_timeline
  <out>/bond_evolution.pdf        fig:bond_evolution (first run with W
                                  snapshots, per --bond-run otherwise)

Usage:
  python make_results.py --runs-root runs/final --out paper_assets
  python make_results.py --runs-root runs/final --out paper_assets \
      --bond-run runs/final/exp05_mappo_hebbian/seed_42

Conventions (matching the paper):
  * Coop. milestones  = DISTINCT team milestones in Ch2–Ch5 per episode
                        (union over agents; Ch1 + comm track excluded).
  * Task return       = team-summed (task + comm_base + comm_milestone)
                        from reward_history_decomposed — EXCLUDES
                        hebbian_diffuse. Falls back to per_episode_returns
                        (with a warning) when decomposition is absent.
  * Steps-to-milestone = within-episode step of FIRST completion; median
                        over completing episodes + n completing.
  * Aggregation       = pooled over all episodes of all seeds (n = 15 for
                        3 seeds × 5 eps). Per-seed means also in the CSV.

Edit CONDITIONS below to match your experiment directory names.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Condition registry: label -> (dir name, group) ────────────────────
# Order here = row order in the LaTeX tables. Edit dir names to match
# runs/final/. Groups: main (tab:main_results + steps table),
# topo (tab:topology_ablation), and 'heb' flags rows for tab:graph_stats.
CONDITIONS = [
    # label          dir name                      group   hebbian?
    ("LLM-2B",       "exp01_llm_2b",               "main", False),
    ("LLM-9B",       "exp02_llm_9b",               "main", False),
    ("LLM-2B+Heb",   "exp07_llm_2b_social_prompt", "main", True),
    ("LLM-9B+Heb",   "exp08_llm_9b_social_prompt", "main", True),
    ("IPPO",         "exp04_ippo",                 "main", False),
    ("MAPPO",        "exp03_mappo",                "main", False),
    ("IPPO+Heb",     "exp06_ippo_hebbian",         "main", True),
    ("MAPPO+Heb",    "exp05_mappo_hebbian",        "main", True),
    ("No-bonds",     "exp11_llm_9b_allied_none",   "topo", False),
    ("Allied-pair",  "exp10_llm_9b_allied_pair",   "topo", False),
    ("Allied-all",   "exp09_llm_9b_allied_all",    "topo", False),
]
TOPO_REFERENCE = "LLM-9B"   # reference row repeated in the topology table
EXCLUDED_AGENT = "agent_2"  # the excluded agent in Allied-pair

# ─── Milestone schema (mirrors craftium_metric.py — keep in sync) ──────
MILESTONE_TRACK = {
    "m1_move_5": "ch1_solo", "m2_dig_3_any": "ch1_solo",
    "m3_pickup_3": "ch1_solo", "m4_dig_5_wood": "ch1_solo",
    "m5_kill_1_animal": "ch1_solo", "m6_kill_2_animals": "ch1_solo",
    "m7_dig_3_stone": "ch1_solo", "m_door1_open": "ch1_solo",
    "m8_anvil_A1": "ch2_anvils", "m9_anvil_B1": "ch2_anvils",
    "m14_sword_equipped": "ch2_anvils", "m15_chestplate_equipped": "ch2_anvils",
    "m16_enter_cell": "ch3_switches", "m17_switch_pressed": "ch3_switches",
    "m18_door_opened": "ch3_switches", "m19_all_in_communal": "ch3_switches",
    "m20_enter_ch4": "ch4_combat", "m21_first_mob_kill": "ch4_combat",
    "m22_all_mobs_killed": "ch4_combat", "m23_all_alive_ch4": "ch4_combat",
    "m24_enter_ch5": "ch5_boss", "m25_first_boss_dmg": "ch5_boss",
    "m26_boss_half_hp": "ch5_boss", "m27_boss_defeated": "ch5_boss",
    "m28_all_alive_bonus": "ch5_boss",
    "m_comm_ch1": "communication", "m_comm_ch2": "communication",
    "m_comm_ch3": "communication", "m_comm_ch4": "communication",
    "m_comm_ch5": "communication",
}
TRACK_ORDER = ["ch1_solo", "ch2_anvils", "ch3_switches",
               "ch4_combat", "ch5_boss", "communication"]
MILESTONE_ORDER = [m for t in TRACK_ORDER
                   for m in [k for k, v in MILESTONE_TRACK.items() if v == t]]
COOP_TRACKS = {"ch2_anvils", "ch3_switches", "ch4_combat", "ch5_boss"}
COOP_MAX = sum(1 for v in MILESTONE_TRACK.values() if v in COOP_TRACKS)

# Internal repo id -> (paper label, display name) for tab:steps_to_milestone.
# Internal ids keep the legacy numbering (m14..m28); the paper schedule
# renumbered compactly (M8..M24), so this map is the single source of truth
# for how a row is labelled. Keep in sync with tab:milestone_schedule.
PAPER_LABEL = {
    "m1_move_5":               ("M1",        "Move $>$5 from spawn"),
    "m2_dig_3_any":            ("M2",        "Dig 3 any"),
    "m3_pickup_3":             ("M3",        "Pick up 3 items"),
    "m4_dig_5_wood":           ("M4",        "Dig 5 wood"),
    "m5_kill_1_animal":        ("M5",        "Kill 1 animal"),
    "m6_kill_2_animals":       ("M6",        "Kill 2 animals"),
    "m7_dig_3_stone":          ("M7",        "Dig 3 stone"),
    "m_door1_open":            ("M\\_door1", "Door 1 unlocked"),
    "m8_anvil_A1":             ("M8",        "First anvil break"),
    "m9_anvil_B1":             ("M9",        "Second anvil break"),
    "m14_sword_equipped":      ("M10",       "Sword equipped"),
    "m15_chestplate_equipped": ("M11",       "Chestplate equipped"),
    "m16_enter_cell":          ("M12",       "Enter isolation cell"),
    "m17_switch_pressed":      ("M13",       "Press switch"),
    "m18_door_opened":         ("M14",       "Door opened"),
    "m19_all_in_communal":     ("M15",       "All in communal room"),
    "m20_enter_ch4":           ("M16",       "Enter Chamber 4"),
    "m21_first_mob_kill":      ("M17",       "First mob kill"),
    "m22_all_mobs_killed":     ("M18",       "All mobs killed"),
    "m23_all_alive_ch4":       ("M19",       "All survive (damage)"),
    "m24_enter_ch5":           ("M20",       "Enter Chamber 5"),
    "m25_first_boss_dmg":      ("M21",       "First boss damage"),
    "m26_boss_half_hp":        ("M22",       "Boss at 50\\% HP"),
    "m27_boss_defeated":       ("M23",       "Boss defeated"),
    "m28_all_alive_bonus":     ("M24",       "All alive at kill"),
    "m_comm_ch1":              ("M\\_comm\\_ch1", "Valid messaging, Ch1"),
    "m_comm_ch2":              ("M\\_comm\\_ch2", "Valid messaging, Ch2"),
    "m_comm_ch3":              ("M\\_comm\\_ch3", "Valid messaging, Ch3"),
    "m_comm_ch4":              ("M\\_comm\\_ch4", "Valid messaging, Ch4"),
    "m_comm_ch5":              ("M\\_comm\\_ch5", "Valid messaging, Ch5"),
}
# Track section titles for the grouped header rows in tab:steps_to_milestone.
TRACK_TITLE = {
    "ch1_solo":     "Chamber 1 --- solo skill acquisition",
    "ch2_anvils":   "Chamber 2 --- cooperative resource acquisition",
    "ch3_switches": "Chamber 3 --- communication under partial obs.",
    "ch4_combat":   "Chamber 4 --- team combat",
    "ch5_boss":     "Chamber 5 --- cooperative boss fight",
    "communication": "Communication track",
}
# Rows of tab:steps_to_milestone, in order (incl. the comm track, which
# has its own row group at the bottom of the table and the timeline fig).
STEPS_MILESTONES = list(MILESTONE_ORDER)

# repo id of the four key milestones (paper numbering in comments)
KEY_MILESTONES = [
    ("M8",  "m8_anvil_A1"),        # first anvil break   (paper M8)
    ("M15", "m19_all_in_communal"),# full regroup         (paper M15)
    ("M18", "m22_all_mobs_killed"),# arena clear          (paper M18)
    ("M23", "m27_boss_defeated"),  # boss defeated        (paper M23)
]
BOSS_MILESTONE = "m27_boss_defeated"
TRACK_TO_CHAMBER_NUM = {"ch1_solo": 1, "ch2_anvils": 2, "ch3_switches": 3,
                        "ch4_combat": 4, "ch5_boss": 5}
WARM_START_W = 0.1  # sparsity threshold = init weight

PALETTE = plt.get_cmap("tab10")


# ─── Loading ────────────────────────────────────────────────────────────
def load_runs(runs_root: Path, dir_name: str) -> list[dict]:
    """All seed runs for one condition, each augmented with episode slicing."""
    runs = []
    for f in sorted((runs_root / dir_name).glob("seed_*/final_metrics.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] unreadable {f}: {e}", file=sys.stderr)
            continue
        d["_path"] = str(f)
        lens = [int(x) for x in d.get("episode_lengths", [])]
        bounds, c = [], 0
        for L in lens:
            bounds.append((c, c + L))  # [start, end) in global steps
            c += L
        d["_ep_bounds"] = bounds
        runs.append(d)
    return runs


def ep_of_step(bounds, step):
    """Episode index + within-episode step for a global step; None if outside."""
    for e, (s0, s1) in enumerate(bounds):
        if s0 <= step < s1:
            return e, step - s0
    return None, None


# ─── Per-run extraction ─────────────────────────────────────────────────
def episode_milestone_sets(run) -> list[set]:
    """Distinct team milestone set per episode (union over agents)."""
    per_agent = run.get("milestones_per_episode", [])
    n_eps = max((len(a) for a in per_agent), default=0)
    out = []
    for e in range(n_eps):
        s = set()
        for a in per_agent:
            if e < len(a):
                s.update(a[e])
        out.append(s)
    return out


def coop_count(ms_set) -> int:
    return sum(1 for m in ms_set if MILESTONE_TRACK.get(m) in COOP_TRACKS)


def episode_task_returns(run) -> tuple[list[float], bool]:
    """Team task return per episode (excl. hebbian_diffuse).
    Returns (values, used_decomposition)."""
    dec = run.get("reward_history_decomposed") or []
    bounds = run["_ep_bounds"]
    if bounds and dec and any(dec):
        sums = [0.0] * len(bounds)
        for agent_recs in dec:
            for rec in agent_recs:
                e, _ = ep_of_step(bounds, int(rec.get("t", -1)))
                if e is not None:
                    sums[e] += (float(rec.get("task", 0.0))
                                + float(rec.get("comm_base", 0.0))
                                + float(rec.get("comm_milestone", 0.0)))
        return sums, True
    # Fallback: logged returns (include diffuse in Hebbian runs!)
    per_ep = run.get("per_episode_returns", [])
    n_eps = max((len(a) for a in per_ep), default=0)
    return [sum(a[e] for a in per_ep if e < len(a)) for e in range(n_eps)], False


def episode_first_fire(run) -> dict:
    """{(episode, milestone_id): within-episode step of first fire}."""
    out = {}
    for ev in run.get("milestone_events", []):
        mid = ev.get("milestone_id") or ev.get("milestone")
        e, ws = ep_of_step(run["_ep_bounds"], int(ev.get("step", -1)))
        if mid is None or e is None:
            continue
        key = (e, mid)
        if key not in out or ws < out[key]:
            out[key] = ws
    return out


def furthest_chamber(ms_set) -> int:
    chambers = [TRACK_TO_CHAMBER_NUM[MILESTONE_TRACK[m]]
                for m in ms_set
                if MILESTONE_TRACK.get(m) in TRACK_TO_CHAMBER_NUM]
    return max(chambers, default=0)


def end_of_episode_graph_stats(run) -> list[dict]:
    """Last graph snapshot within each episode → mean/max/sparsity/asym."""
    snaps = run.get("graph_snapshots") or []
    bounds = run["_ep_bounds"]
    if not snaps or not bounds:
        return []
    last_in_ep = {}
    for s in snaps:
        e, _ = ep_of_step(bounds, int(s.get("step", -1)))
        if e is not None:
            last_in_ep[e] = s  # snaps are chronological; keep overwriting
    out = []
    for e in sorted(last_in_ep):
        s = last_in_ep[e]
        W = s.get("W")
        rec = {"mean": float(s.get("mean_bond_strength", 0.0)),
               "sparsity": float(s.get("sparsity", 0.0)),
               "max": math.nan, "asym": math.nan}
        if W:
            n = len(W)
            off = [W[i][j] for i in range(n) for j in range(n) if i != j]
            rec["max"] = max(off) if off else math.nan
            rec["sparsity"] = (sum(1 for w in off if w < WARM_START_W)
                               / len(off)) if off else math.nan
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rec["asym"] = (statistics.fmean(
                abs(W[i][j] - W[j][i]) for i, j in pairs)
                if pairs else math.nan)
        out.append(rec)
    return out


# ─── Aggregation across runs of a condition ─────────────────────────────
def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan, math.nan
    return (statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) >= 2 else 0.0)


def aggregate(runs) -> dict:
    """Pool all episodes of all seeds; return every paper number."""
    coop, allms, task, furthest, boss = [], [], [], [], 0
    n_eps_total = 0
    used_decomp = True
    steps = {m: [] for m in STEPS_MILESTONES}
    graph = defaultdict(list)
    per_seed_coop = []

    for run in runs:
        sets_ = episode_milestone_sets(run)
        n_eps_total += len(sets_)
        seed_coop = []
        for s in sets_:
            c = coop_count(s)
            coop.append(c); seed_coop.append(c)
            allms.append(len(s))
            furthest.append(furthest_chamber(s))
            if BOSS_MILESTONE in s:
                boss += 1
        if seed_coop:
            per_seed_coop.append(statistics.fmean(seed_coop))

        t, used = episode_task_returns(run)
        task.extend(t); used_decomp &= used

        ff = episode_first_fire(run)
        for e in range(len(sets_)):
            for mid in STEPS_MILESTONES:
                if (e, mid) in ff:
                    steps[mid].append(ff[(e, mid)])

        for rec in end_of_episode_graph_stats(run):
            for k, v in rec.items():
                graph[k].append(v)

    # Credit Gini / exclusion share. The logged milestone_credit_total mixes
    # two key styles for the same agent ('agent_0' from Python, 'agent0'
    # from Lua contributors), so the stored credit_gini is computed over 2N
    # phantom agents and is wrong — merge the keys and recompute here.
    gini, excl = [], []
    for run in runs:
        cm = run.get("coop_metrics") or {}
        merged = defaultdict(float)
        for k, v in (cm.get("milestone_credit_total") or {}).items():
            m = re.fullmatch(r"agent_?(\d+)", k)
            if m:
                merged[f"agent_{m.group(1)}"] += float(v)
        vals = sorted(merged.values())
        s = sum(vals)
        if s > 0:
            n = len(vals)
            gini.append(sum(abs(a - b) for a in vals for b in vals)
                        / (2 * n * s))
            excl.append(merged.get(EXCLUDED_AGENT, 0.0) / s)

    return {
        "n_runs": len(runs), "n_eps": n_eps_total,
        "coop": mean_std(coop), "coop_per_seed": mean_std(per_seed_coop),
        "allms": mean_std(allms), "task": mean_std(task),
        "task_is_decomposed": used_decomp,
        "furthest_mode": (statistics.mode(furthest) if furthest else None),
        "boss": boss,
        "steps": {mid: (statistics.median(v) if v else None, len(v))
                  for mid, v in steps.items()},
        "gini": mean_std(gini), "excl": mean_std(excl),
        "graph": {k: mean_std(v) for k, v in graph.items()},
    }


# ─── LaTeX emission ─────────────────────────────────────────────────────
def fmt(ms, prec=1):
    m, s = ms
    if math.isnan(m):
        return "--"
    return f"${m:.{prec}f} \\pm {s:.{prec}f}$"


def emit_tables(stats: dict, out: Path):
    L = []
    L.append("% ── tab:main_results rows ─────────────────────────────")
    for label, _, grp, _ in CONDITIONS:
        if grp != "main" or label not in stats:
            continue
        a = stats[label]
        L.append(
            f"{label:<12} & {fmt(a['coop'])} & {fmt(a['allms'])} & "
            f"{fmt(a['task'], 0)} \\\\")
    L.append("")
    L.append("% ── tab:steps_to_milestone rows (milestone × condition) ──")
    L.append("% Paste between \\midrule and \\bottomrule, replacing the manual")
    L.append("% milestone rows. Cell = $median_{n}$; '--' = never completed.")
    main_labels = [lab for lab, _, g, _ in CONDITIONS if g == "main"]
    ncol = len(main_labels) + 2  # ID + Milestone + one column per condition
    cur_track = None
    for mid in STEPS_MILESTONES:
        track = MILESTONE_TRACK[mid]
        if track != cur_track:
            if cur_track is not None:
                L.append("\\addlinespace")
            L.append(f"\\multicolumn{{{ncol}}}{{l}}"
                     f"{{\\emph{{{TRACK_TITLE[track]}}}}}\\\\")
            cur_track = track
        plabel, name = PAPER_LABEL[mid]
        cells = []
        for lab in main_labels:
            med, n = stats.get(lab, {}).get("steps", {}).get(mid, (None, 0))
            cells.append(f"${med:.0f}_{{{n}}}$" if med is not None else "--")
        L.append(f"{plabel:<9} & {name:<22} & " + " & ".join(cells) + " \\\\")
    L.append("")
    L.append("% ── tab:topology_ablation rows ────────────────────────")
    topo_rows = [(TOPO_REFERENCE + " (ref.)", TOPO_REFERENCE)] + \
                [(lab, lab) for lab, _, g, _ in CONDITIONS if g == "topo"]
    for disp, key in topo_rows:
        if key not in stats:
            continue
        a = stats[key]
        L.append(f"{disp:<16} & {fmt(a['coop'])} & {fmt(a['task'], 0)} & "
                 f"{fmt(a['gini'], 2)} & {fmt(a['excl'], 2)} \\\\")
    L.append("")
    L.append("% ── tab:graph_stats rows ──────────────────────────────")
    for label, _, grp, heb in CONDITIONS:
        if not heb or label not in stats:
            continue
        g = stats[label]["graph"]
        if not g:
            continue
        L.append(f"{label:<12} & {fmt(g.get('mean', (math.nan, 0)), 2)} & "
                 f"{fmt(g.get('max', (math.nan, 0)), 2)} & "
                 f"{fmt(g.get('sparsity', (math.nan, 0)), 2)} & "
                 f"{fmt(g.get('asym', (math.nan, 0)), 3)} \\\\")
    (out / "table_rows.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def emit_csv(stats: dict, out: Path):
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_runs", "n_eps",
                    "coop_mean", "coop_std",
                    "coop_seedmean", "coop_seedstd",
                    "allms_mean", "allms_std", "task_mean", "task_std",
                    "task_is_decomposed", "furthest_mode", "boss_defeats",
                    *(f"{pid}_{x}" for pid, _ in KEY_MILESTONES
                      for x in ("median", "n")),
                    "gini_mean", "gini_std", "excl_mean", "excl_std",
                    "bond_mean", "bond_max", "sparsity", "asymmetry"])
        for label, _, _, _ in CONDITIONS:
            if label not in stats:
                continue
            a = stats[label]
            row = [label, a["n_runs"], a["n_eps"],
                   *a["coop"], *a["coop_per_seed"], *a["allms"], *a["task"],
                   a["task_is_decomposed"], a["furthest_mode"], a["boss"]]
            for _, mid in KEY_MILESTONES:
                med, n = a["steps"][mid]
                row += [med, n]
            row += [*a["gini"], *a["excl"],
                    a["graph"].get("mean", (math.nan,))[0],
                    a["graph"].get("max", (math.nan,))[0],
                    a["graph"].get("sparsity", (math.nan,))[0],
                    a["graph"].get("asym", (math.nan,))[0]]
            w.writerow(row)


# ─── Figures ────────────────────────────────────────────────────────────
def fig_progression(all_runs: dict, out: Path, max_steps: int):
    """Cumulative distinct Ch2–5 team milestones vs within-episode step."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    plotted = False
    for ci, (label, _, grp, _) in enumerate(CONDITIONS):
        if grp != "main" or label not in all_runs:
            continue
        curves = []
        for run in all_runs[label]:
            ff = episode_first_fire(run)
            n_eps = len(run["_ep_bounds"])
            for e in range(n_eps):
                fires = sorted(ws for (ee, mid), ws in ff.items()
                               if ee == e and
                               MILESTONE_TRACK.get(mid) in COOP_TRACKS)
                curve = np.zeros(max_steps)
                for k, ws in enumerate(fires, start=1):
                    if ws < max_steps:
                        curve[ws:] = k
                curves.append(curve)
        if not curves:
            continue
        arr = np.vstack(curves)
        m, s = arr.mean(0), arr.std(0)
        x = np.arange(max_steps)
        c = PALETTE(ci % 10)
        ax.plot(x, m, color=c, lw=1.8, label=label)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
        plotted = True
    if not plotted:
        plt.close(fig); return
    for b in range(1, 5):
        ax.axvline(b * max_steps // 5, color="#999", ls="--", lw=0.7)
    ax.set_xlabel("Within-episode step")
    ax.set_ylabel("Distinct team milestones (Ch2–Ch5)")
    ax.set_xlim(0, max_steps); ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "milestone_progression.pdf")
    plt.close(fig)


def fig_timeline(all_runs: dict, out: Path):
    """Raster: rows = milestones (chamber-grouped), x = median first step."""
    rows = list(MILESTONE_ORDER)
    y_of = {m: i for i, m in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(rows) + 1.5))
    # chamber bands (+ comm track)
    band_cols = ["#f4f0ff", "#fff4e6", "#e8f8ff", "#ffeeee", "#fff0c2",
                 "#eef7ee"]
    lo = 0
    for ti, t in enumerate(TRACK_ORDER):
        n = sum(1 for m in rows if MILESTONE_TRACK[m] == t)
        ax.axhspan(lo - 0.5, lo + n - 0.5, color=band_cols[ti], zorder=0)
        lo += n
    mains = [(ci, lab) for ci, (lab, _, g, _) in enumerate(CONDITIONS)
             if g == "main" and lab in all_runs]
    plotted = False
    for k, (ci, label) in enumerate(mains):
        jitter = (k - (len(mains) - 1) / 2) * (0.8 / max(len(mains), 1))
        for run_set in [all_runs[label]]:
            per_mid = defaultdict(list)
            for run in run_set:
                for (e, mid), ws in episode_first_fire(run).items():
                    per_mid[mid].append(ws)
            for mid, v in per_mid.items():
                if mid not in y_of:
                    continue
                y = y_of[mid] + jitter
                ax.plot([min(v), max(v)], [y, y], color=PALETTE(ci % 10),
                        lw=1.0, alpha=0.5, zorder=2)
                ax.scatter(statistics.median(v), y, s=22,
                           color=PALETTE(ci % 10), edgecolors="black",
                           linewidths=0.3, zorder=3,
                           label=label if mid == next(
                               m for m in rows if m in per_mid) else None)
                plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([PAPER_LABEL[m][0].replace("\\_", "_") for m in rows],
                       fontsize=7)
    ax.set_xlabel("Within-episode step of first completion")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "milestone_timeline.pdf")
    plt.close(fig)


def fig_bonds(run: dict, out: Path, max_steps: int = 1000):
    """Per-pair W_ij vs W_ji over one run, asymmetry shaded."""
    snaps = run.get("graph_snapshots") or []
    snaps = [s for s in snaps if s.get("W")]
    if not snaps:
        print("  [warn] bond figure skipped: no W in snapshots",
              file=sys.stderr)
        return
    N = len(snaps[0]["W"])
    steps = [s["step"] for s in snaps]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    fig, axes = plt.subplots(len(pairs), 1,
                             figsize=(7, 1.9 * len(pairs)),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (i, j) in zip(axes, pairs):
        wij = [s["W"][i][j] for s in snaps]
        wji = [s["W"][j][i] for s in snaps]
        ax.plot(steps, wij, color="#2c7fb8", lw=1.8,
                label=f"$W_{{{i}{j}}}$")
        ax.plot(steps, wji, color="#d95f0e", lw=1.8,
                label=f"$W_{{{j}{i}}}$")
        ax.fill_between(steps, np.minimum(wij, wji), np.maximum(wij, wji),
                        color="#888", alpha=0.18)
        # dotted: within-episode chamber timer boundaries (20% of the
        # episode budget each); dashed: episode boundaries
        chamber_len = max_steps // 5
        for s0, s1 in run["_ep_bounds"]:
            for k in range(1, 5):
                b = s0 + k * chamber_len
                if b < s1:
                    ax.axvline(b, color="#bbb", ls=":", lw=0.5)
        for s0, _ in run["_ep_bounds"][1:]:
            ax.axvline(s0, color="#666", ls="--", lw=0.9)
        ax.set_ylim(0, 1); ax.set_ylabel("Bond")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
        ax.set_title(f"pair (agent {i}, agent {j})", fontsize=9)
    axes[-1].set_xlabel("Environment step")
    fig.savefig(out / "bond_evolution.pdf")
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=Path("runs/final"))
    p.add_argument("--out", type=Path, default=Path("paper_assets"))
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--bond-run", type=Path, default=None,
                   help="run dir for fig:bond_evolution (default: first "
                        "Hebbian run that stored W snapshots)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_runs, stats = {}, {}
    for label, dirname, _, _ in CONDITIONS:
        runs = load_runs(args.runs_root, dirname)
        if not runs:
            print(f"[skip] {label}: no runs under "
                  f"{args.runs_root / dirname}")
            continue
        all_runs[label] = runs
        stats[label] = aggregate(runs)
        a = stats[label]
        flag = "" if a["task_is_decomposed"] else \
            "  [WARN: task return from logged totals — includes " \
            "hebbian_diffuse!]"
        print(f"[ok]   {label}: {a['n_runs']} seeds, {a['n_eps']} episodes, "
              f"coop={a['coop'][0]:.2f}±{a['coop'][1]:.2f}{flag}")

    if not stats:
        sys.exit("No runs found — check --runs-root and CONDITIONS.")

    emit_tables(stats, args.out)
    emit_csv(stats, args.out)
    fig_progression(all_runs, args.out, args.max_steps)
    fig_timeline(all_runs, args.out)

    bond_run = None
    if args.bond_run:
        # accept a seed dir containing final_metrics.json
        f = args.bond_run / "final_metrics.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            lens = [int(x) for x in d.get("episode_lengths", [])]
            bounds, c = [], 0
            for L in lens:
                bounds.append((c, c + L)); c += L
            d["_ep_bounds"] = bounds
            bond_run = d
    if bond_run is None:
        for label, _, _, heb in CONDITIONS:
            if heb and label in all_runs:
                for r in all_runs[label]:
                    if any(s.get("W") for s in
                           (r.get("graph_snapshots") or [])):
                        bond_run = r
                        break
            if bond_run:
                break
    if bond_run:
        fig_bonds(bond_run, args.out, args.max_steps)

    print(f"\nWrote: {args.out}/table_rows.tex, summary.csv, "
          f"milestone_progression.pdf, milestone_timeline.pdf"
          + (", bond_evolution.pdf" if bond_run else ""))


if __name__ == "__main__":
    main()
