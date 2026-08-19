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
    # Experiment 2 (co-firing without enforced communication). Group
    # "cofire" feeds ONLY the cofire_* emitters — the main/topo/graph
    # tables and summary.csv are untouched, so regenerating the
    # Experiment-1 assets from old runs roots stays byte-identical
    # (these dirs simply don't exist there).
    ("prc",          "exp20_cofire_prc",           "cofire", True),
    ("pro",          "exp21_cofire_pro",           "cofire", True),
    ("pri",          "exp22_cofire_pri",           "cofire", True),
    ("prco",         "exp29_cofire_prco",          "cofire", True),
    ("prcoi",        "exp23_cofire_prcoi",         "cofire", True),
    ("anchor",       "exp27_cofire_anchor",        "cofire", True),
    ("null (pr)",    "exp28_cofire_null",          "cofire", True),
    # Gemma 4 E4B arms (runs/new_exp_0_gemma). Group "gemma" for the same
    # reason as "cofire" above: no emitter selects it, so the Experiment-1
    # assets stay byte-identical. Present so the qualitative pipeline's
    # registry (which reads CONDITIONS) can discover these runs.
    ("Gemma-E4B",     "new_exp_0_gemma_base",      "gemma", False),
    ("Gemma-E4B+Heb", "new_exp_0_gemma_hebbian",   "gemma", True),
    # Agent-count scaling suite (runs/agent_scaling, submit_agent_scaling.sh).
    # Group "scaling": no table emitter selects it (same pattern as "cofire"
    # and "gemma"), present so the qualitative registry and
    # make_scaling_fig.py can discover the runs.
    ("Scale-N2",      "scale_gemma_base_n2",       "scaling", False),
    ("Scale-N3",      "scale_gemma_base_n3",       "scaling", False),
    ("Scale-N4",      "scale_gemma_base_n4",       "scaling", False),
    ("Scale-N5",      "scale_gemma_base_n5",       "scaling", False),
    ("Scale-N6",      "scale_gemma_base_n6",       "scaling", False),
    ("Scale-N9",      "scale_gemma_base_n9",       "scaling", False),
    ("Scale-N2+Heb",  "scale_gemma_hebbian_n2",    "scaling", True),
    ("Scale-N3+Heb",  "scale_gemma_hebbian_n3",    "scaling", True),
    ("Scale-N4+Heb",  "scale_gemma_hebbian_n4",    "scaling", True),
    ("Scale-N5+Heb",  "scale_gemma_hebbian_n5",    "scaling", True),
    ("Scale-N6+Heb",  "scale_gemma_hebbian_n6",    "scaling", True),
    ("Scale-N9+Heb",  "scale_gemma_hebbian_n9",    "scaling", True),
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
    # Act-reward symmetry suite: per-chamber observation/imitation
    # milestones (paid like the comm track; --social-act-rewards runs only).
    "m_obs_ch1": "observation", "m_obs_ch2": "observation",
    "m_obs_ch3": "observation", "m_obs_ch4": "observation",
    "m_obs_ch5": "observation",
    "m_imit_ch1": "imitation", "m_imit_ch2": "imitation",
    "m_imit_ch3": "imitation", "m_imit_ch4": "imitation",
    "m_imit_ch5": "imitation",
}
# Tracks that reward SOCIAL ACTS rather than physical task progress —
# excluded from the comparable milestone counts (mute/unflagged arms cannot
# earn them).
SOCIAL_ACT_TRACKS = {"communication", "observation", "imitation"}
TRACK_ORDER = ["ch1_solo", "ch2_anvils", "ch3_switches",
               "ch4_combat", "ch5_boss", "communication"]
MILESTONE_ORDER = [m for t in TRACK_ORDER
                   for m in [k for k, v in MILESTONE_TRACK.items() if v == t]]
COOP_TRACKS = {"ch2_anvils", "ch3_switches", "ch4_combat", "ch5_boss"}
COOP_MAX = sum(1 for v in MILESTONE_TRACK.values() if v in COOP_TRACKS)
# Comm-independent cooperative tracks (Experiment 2): chambers whose
# milestones do not structurally require communication. The mute cofire
# arms (pro / pri / null) cannot solve Ch3's comm-gated door by
# construction, so their task metric leans on this subset.
COOP_CI_TRACKS = {"ch2_anvils", "ch4_combat"}
# Denominator for the cofire table's milestone-completion percentage:
# every milestone except the communication track (= 25).
NONCOMM_MAX = sum(1 for v in MILESTONE_TRACK.values()
                  if v not in SOCIAL_ACT_TRACKS)
COOP_CI_MAX = sum(1 for v in MILESTONE_TRACK.values() if v in COOP_CI_TRACKS)

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


# ─── Entry-milestone honesty filter ─────────────────────────────────────
# m20_enter_ch4 / m24_enter_ch5 are position-triggered in Lua (mobs.lua
# globalstep detectors) and leak through two race conditions: (a) the
# episode-reset handler clears the door*_force_teleported suppression flags
# and milestone state while agents are still physically standing in Ch4/Ch5
# from the previous episode's rescue teleport (init.lua reset), and (b) in
# some runs the Lua reset modchannel message is lost, desyncing the whole
# timer/suppression state machine (lua_step >> episode length). A GENUINE
# entry requires the enabling door milestone in the SAME episode; events
# without it are teleport/reset artifacts and are removed from every
# downstream number: the event list (steps table + both milestone figures),
# the per-episode team milestone sets (coop / all-milestone counts), and the
# episode task return (their +30/+50 per-contributor rewards, which were
# drained into the task stream).
ENTRY_REQUIRES = {
    "m20_enter_ch4": "m18_door_opened",       # Ch3 puzzle door must be open
    "m24_enter_ch5": "m22_all_mobs_killed",   # Ch4 arena must be cleared
}


def _apply_entry_honesty_filter(run: dict) -> None:
    """Strip physically-impossible chamber-entry milestones from a run, in place."""
    bounds = run["_ep_bounds"]
    per_agent = run.get("milestones_per_episode", [])
    n_eps = max((len(a) for a in per_agent), default=len(bounds))

    # Team milestone set per episode (union over agents) — used to decide
    # whether the enabling milestone fired in that episode.
    team = [set() for _ in range(n_eps)]
    for a in per_agent:
        for e, ms in enumerate(a):
            team[e].update(ms)

    def dishonest(e: int, mid: str) -> bool:
        req = ENTRY_REQUIRES.get(mid)
        if req is None:
            return False
        return e is None or e >= n_eps or req not in team[e]

    # 1. Filter the event list (feeds steps-to-milestone + both figures) and
    #    accumulate the per-episode task-return adjustment (one event row per
    #    contributor, each carrying the full per-contributor reward).
    adjust = defaultdict(float)
    kept = []
    for ev in run.get("milestone_events", []):
        mid = ev.get("milestone_id") or ev.get("milestone")
        if mid in ENTRY_REQUIRES:
            e, _ = ep_of_step(bounds, int(ev.get("step", -1)))
            if dishonest(e, mid):
                if e is not None:
                    adjust[e] += float(ev.get("reward", 0.0))
                continue
        kept.append(ev)
    run["milestone_events"] = kept
    run["_task_adjust"] = dict(adjust)

    # 2. Filter the per-agent per-episode milestone sets (feed coop /
    #    all-milestone counts and furthest-chamber).
    for a in per_agent:
        for e in range(len(a)):
            a[e] = [m for m in a[e] if not dishonest(e, m)]


# ─── Loading ────────────────────────────────────────────────────────────
def load_runs(runs_root: Path, dir_name: str,
              exclude: frozenset = frozenset()) -> list[dict]:
    """All seed runs for one condition, each augmented with episode slicing."""
    runs = []
    for f in sorted((runs_root / dir_name).glob("seed_*/final_metrics.json")):
        if f"{dir_name}/{f.parent.name}" in exclude:
            print(f"  [excluded] {dir_name}/{f.parent.name}", file=sys.stderr)
            continue
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
        _apply_entry_honesty_filter(d)
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


def episode_task_returns(run, include_comm: bool = True) -> tuple[list[float], bool]:
    """Team task return per episode (excl. hebbian_diffuse).
    Returns (values, used_decomposition).

    ``include_comm=False`` (Experiment 2 cofire table) drops the comm_base +
    comm_milestone streams — message PAY, which mute arms structurally cannot
    earn — leaving only physical task progress. Default True = historical.
    """
    adjust = run.get("_task_adjust", {})  # honesty filter: fake entry rewards
    dec = run.get("reward_history_decomposed") or []
    bounds = run["_ep_bounds"]
    if bounds and dec and any(dec):
        sums = [0.0] * len(bounds)
        for agent_recs in dec:
            for rec in agent_recs:
                e, _ = ep_of_step(bounds, int(rec.get("t", -1)))
                if e is not None:
                    sums[e] += float(rec.get("task", 0.0))
                    if include_comm:
                        sums[e] += (float(rec.get("comm_base", 0.0))
                                    + float(rec.get("comm_milestone", 0.0)))
        return [s - adjust.get(e, 0.0) for e, s in enumerate(sums)], True
    # Fallback: logged returns (include diffuse in Hebbian runs!)
    per_ep = run.get("per_episode_returns", [])
    n_eps = max((len(a) for a in per_ep), default=0)
    return [sum(a[e] for a in per_ep if e < len(a)) - adjust.get(e, 0.0)
            for e in range(n_eps)], False


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
    coop_ci = []
    allms_nc = []
    task_nc = []
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
            coop_ci.append(sum(1 for m in s
                               if MILESTONE_TRACK.get(m) in COOP_CI_TRACKS))
            # Comm-track-free milestone count (Experiment 2): mute arms
            # structurally cannot fire m_comm_* (4 messages/chamber), so the
            # cofire table compares arms on this instead of raw allms.
            allms_nc.append(sum(1 for m in s
                                if MILESTONE_TRACK.get(m) not in SOCIAL_ACT_TRACKS))
            allms.append(len(s))
            furthest.append(furthest_chamber(s))
            if BOSS_MILESTONE in s:
                boss += 1
        if seed_coop:
            per_seed_coop.append(statistics.fmean(seed_coop))

        t, used = episode_task_returns(run)
        task.extend(t); used_decomp &= used
        t_nc, _ = episode_task_returns(run, include_comm=False)
        task_nc.extend(t_nc)

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
        "coop_ci": mean_std(coop_ci),
        "allms_nc": mean_std(allms_nc),
        "task_nc": mean_std(task_nc),
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


# ─── Experiment 2 (cofire) emitters ─────────────────────────────────────
# Separate files (cofire_summary.csv + cofire_table_rows.tex) so the
# Experiment-1 artifacts above never change shape.

_COFIRE_CHANNELS = ("spat", "comm", "obs", "imit")
# Guided imitation (2026-08-07): adoption steps are not separate acts, so
# the act mix is exactly the four fresh choices.
_COFIRE_ACTS = ("communicate", "observe", "imitate", "none")


def _run_cofire_stats(run: dict) -> dict | None:
    """Per-run act-mix / attribution / imitation shares from
    final_metrics['social_act_metrics'] (None for legacy runs, e.g. anchor)."""
    sam = run.get("social_act_metrics") or {}
    if not sam:
        return None
    acts = sam.get("act_counts") or {}
    total_acts = sum(acts.values())
    attr = sam.get("cofire_attribution") or {}
    attr_total = float(attr.get("total") or 0.0)

    def _chan_sum(ch):
        m = attr.get(ch)
        return float(sum(sum(row) for row in m)) if m else 0.0

    out = {}
    for a in _COFIRE_ACTS:
        out[f"act_{a}"] = (acts.get(a, 0) / total_acts) if total_acts else math.nan
    for ch in _COFIRE_CHANNELS:
        out[f"dW_{ch}"] = (_chan_sum(ch) / attr_total) if attr_total > 0 else math.nan
    # Guided imitation (2026-08-07): adoption rate = re-enacted steps over
    # delivered sequence elements; payoff = reward/step while adopting vs not.
    delivered = sam.get("imitation_delivered_actions", 0)
    adopted = sam.get("imitation_adopted_steps", 0)
    out["imit_adoption_rate"] = (adopted / delivered) if delivered else math.nan
    nrs = sam.get("nonadopted_steps", 0)
    out["adopted_reward_mean"] = (sam.get("adopted_reward_sum", 0.0) / adopted
                                  if adopted else math.nan)
    out["nonadopted_reward_mean"] = (sam.get("nonadopted_reward_sum", 0.0) / nrs
                                     if nrs else math.nan)
    out["cofire_pair_events"] = sam.get("cofire_pair_events", 0)
    return out


def emit_cofire(stats: dict, all_runs: dict, out: Path):
    """cofire_summary.csv + cofire_table_rows.tex for the cofire group."""
    labels = [lab for lab, _, g, _ in CONDITIONS
              if g == "cofire" and lab in stats]
    if not labels:
        return False

    per_run_keys = ([f"act_{a}" for a in _COFIRE_ACTS]
                    + [f"dW_{ch}" for ch in _COFIRE_CHANNELS]
                    + ["imit_adoption_rate", "adopted_reward_mean",
                       "nonadopted_reward_mean", "cofire_pair_events"])
    pooled = {}
    for lab in labels:
        rows = [r for r in (_run_cofire_stats(run) for run in all_runs[lab])
                if r is not None]
        pooled[lab] = {k: mean_std([r[k] for r in rows]) for k in per_run_keys} \
            if rows else {k: (math.nan, math.nan) for k in per_run_keys}

    with open(out / "cofire_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_runs", "n_eps",
                    "coop_pct_mean", "coop_pct_std",
                    "coop_ci_pct_mean", "coop_ci_pct_std",
                    "allms_nocomm_pct_mean", "allms_nocomm_pct_std",
                    "task_nocomm_mean", "task_nocomm_std",
                    "bond_mean", "bond_max", "sparsity", "asymmetry",
                    *(f"{k}_mean" for k in per_run_keys)])
        for lab in labels:
            a, p = stats[lab], pooled[lab]
            w.writerow([lab, a["n_runs"], a["n_eps"],
                        *(v * 100.0 / COOP_MAX for v in a["coop"]),
                        *(v * 100.0 / COOP_CI_MAX for v in a["coop_ci"]),
                        *(v * 100.0 / NONCOMM_MAX for v in a["allms_nc"]),
                        *a["task_nc"],
                        a["graph"].get("mean", (math.nan,))[0],
                        a["graph"].get("max", (math.nan,))[0],
                        a["graph"].get("sparsity", (math.nan,))[0],
                        a["graph"].get("asym", (math.nan,))[0],
                        *(p[k][0] for k in per_run_keys)])

    L = ["% ── tab:cofire rows (Experiment 2) ────────────────────────",
         "% label & coop%(of 17) & coop_ci%(of 8) & task_nocomm & bond mean & sparsity"
         " & asym & dW share comm/obs/imit \\\\"]
    for lab in labels:
        a, p = stats[lab], pooled[lab]
        g = a["graph"]
        share = "/".join(
            ("--" if math.isnan(p[f"dW_{ch}"][0])
             else f"{p[f'dW_{ch}'][0]:.2f}")
            for ch in ("comm", "obs", "imit"))
        _coop_pct = tuple(v * 100.0 / COOP_MAX for v in a["coop"])
        _ci_pct = tuple(v * 100.0 / COOP_CI_MAX for v in a["coop_ci"])
        L.append(f"{lab:<10} & {fmt(_coop_pct, 0)} & {fmt(_ci_pct, 0)} & "
                 f"{fmt(a['task_nc'], 0)} & "
                 f"{fmt(g.get('mean', (math.nan, 0)), 2)} & "
                 f"{fmt(g.get('sparsity', (math.nan, 0)), 2)} & "
                 f"{fmt(g.get('asym', (math.nan, 0)), 3)} & {share} \\\\")
    (out / "cofire_table_rows.tex").write_text("\n".join(L) + "\n",
                                               encoding="utf-8")
    return True


# ─── Figures ────────────────────────────────────────────────────────────
def fig_progression(all_runs: dict, out: Path, max_steps: int):
    """Cumulative distinct Ch2–5 team milestones vs within-episode step,
    one panel per baseline / +Hebbian backbone pair (shared y-scale)."""
    PAIRS = [("LLM-2B", "LLM-2B+Heb"), ("LLM-9B", "LLM-9B+Heb"),
             ("IPPO", "IPPO+Heb"), ("MAPPO", "MAPPO+Heb")]
    pairs = [(b, h) for b, h in PAIRS if b in all_runs or h in all_runs]
    if not pairs:
        return
    BASE_C, HEB_C = "#2c7fb8", "#d95f0e"  # CVD-safe blue/orange, matches fig:milestone_timeline

    def mean_std_curve(label):
        curves = []
        for run in all_runs.get(label, []):
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
            return None, None
        arr = np.vstack(curves)
        return arr.mean(0), arr.std(0)

    fig, axes = plt.subplots(1, len(pairs), sharey=True,
                             figsize=(2.15 * len(pairs) + 1.0, 2.7))
    axes = np.atleast_1d(axes)
    x = np.arange(max_steps)
    y_top = 0.0

    for ax, (base, heb) in zip(axes, pairs):
        for cond, color, ls in ((base, BASE_C, "-"), (heb, HEB_C, "--")):
            m, s = mean_std_curve(cond)
            if m is None:
                continue
            ax.plot(x, m, color=color, lw=1.8, ls=ls)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.14, lw=0)
            y_top = max(y_top, float((m + s).max()))
        for b in range(1, 5):
            ax.axvline(b * max_steps // 5, color="#c9c9c9", ls=":", lw=0.7)
        ax.set_xlim(0, max_steps)
        ax.set_xticks(range(0, max_steps + 1, max_steps // 5))
        ax.tick_params(labelsize=7)
        ax.set_title(base, fontsize=9)
        ax.grid(alpha=0.2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylim(0, max(1.0, y_top * 1.05))
    axes[0].set_ylabel("Distinct team milestones\n(Ch2–Ch5)", fontsize=8)
    fig.supxlabel("Within-episode step", fontsize=9)
    handles = [
        plt.Line2D([], [], color=BASE_C, lw=1.8, ls="-", label="baseline"),
        plt.Line2D([], [], color=HEB_C, lw=1.8, ls="--", label="+Hebbian"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=2,
               frameon=True, framealpha=0.95, columnspacing=1.2,
               handletextpad=0.5, bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    fig.savefig(out / "milestone_progression.pdf")
    plt.close(fig)


def fig_timeline(all_runs: dict, out: Path, max_steps: int = 1000):
    """Paired dot plot: one panel per baseline / +Hebbian backbone pair.

    Rows = milestones (chamber-grouped, labelled with paper ID + name);
    marker = median within-episode step of first completion (circle =
    baseline, diamond = +Hebbian), thin bar = min--max over completing
    episodes. Milestones never completed by ANY plotted condition are
    dropped, except the load-bearing M15/M18/M23 which stay as
    deliberately-empty rows.
    """
    PAIRS = [("LLM-2B", "LLM-2B+Heb"), ("LLM-9B", "LLM-9B+Heb"),
             ("IPPO", "IPPO+Heb"), ("MAPPO", "MAPPO+Heb")]
    pairs = [(b, h) for b, h in PAIRS if b in all_runs or h in all_runs]
    if not pairs:
        return
    BASE_C, HEB_C = "#2c7fb8", "#d95f0e"  # CVD-safe blue/orange, matches fig:bond_evolution

    def pooled_first_fires(label):
        per_mid = defaultdict(list)
        for run in all_runs.get(label, []):
            for (_e, mid), ws in episode_first_fire(run).items():
                per_mid[mid].append(ws)
        return per_mid

    fires = {lab: pooled_first_fires(lab)
             for pair in pairs for lab in pair}
    ever = set().union(*(set(f) for f in fires.values()))
    KEEP_EMPTY = {"m19_all_in_communal", "m22_all_mobs_killed",
                  "m27_boss_defeated"}
    rows = [m for m in MILESTONE_ORDER if m in ever or m in KEEP_EMPTY]
    y_of = {m: i for i, m in enumerate(rows)}

    def row_label(mid):
        pid, name = PAPER_LABEL[mid]
        pid = pid.replace("\\_", "_")
        name = name.replace("\\%", "%").replace("\\_", "_")
        return f"{pid}  {name}"

    # Chamber band extents over the pruned row list.
    TRACK_SHORT = {"ch1_solo": "Ch1", "ch2_anvils": "Ch2",
                   "ch3_switches": "Ch3", "ch4_combat": "Ch4",
                   "ch5_boss": "Ch5", "communication": "Comm"}
    band_edges, lo = [], 0
    for t in TRACK_ORDER:
        n = sum(1 for m in rows if MILESTONE_TRACK[m] == t)
        if n:
            band_edges.append((t, lo, lo + n))
            lo += n

    fig, axes = plt.subplots(
        1, len(pairs), sharey=True,
        figsize=(1.75 * len(pairs) + 2.5, 0.21 * len(rows) + 1.15))
    axes = np.atleast_1d(axes)

    for ax, (base, heb) in zip(axes, pairs):
        for bi, (_t, r0, r1) in enumerate(band_edges):
            if bi % 2 == 0:
                ax.axhspan(r0 - 0.5, r1 - 0.5, color="#f2f2f2", zorder=0)
        for cond, color, marker, dy in ((base, BASE_C, "o", -0.18),
                                        (heb, HEB_C, "D", 0.18)):
            for mid, v in fires.get(cond, {}).items():
                if mid not in y_of:
                    continue
                y = y_of[mid] + dy
                if len(v) > 1:
                    ax.plot([min(v), max(v)], [y, y], color=color,
                            lw=1.1, alpha=0.45, zorder=2,
                            solid_capstyle="butt")
                ax.scatter(statistics.median(v), y, s=26, marker=marker,
                           color=color, edgecolors="white",
                           linewidths=0.5, zorder=3)
        for b in range(max_steps // 5, max_steps, max_steps // 5):
            ax.axvline(b, color="#c9c9c9", ls=":", lw=0.6, zorder=1)
        ax.set_xlim(0, max_steps)
        ax.set_xticks(range(0, max_steps + 1, max_steps // 5))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(base, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    axes[0].invert_yaxis()  # Ch1 at top, communication track at bottom
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([row_label(m) for m in rows], fontsize=7)
    # Chamber tags along the right edge of the last panel.
    for t, r0, r1 in band_edges:
        axes[-1].text(1.03, (r0 + r1 - 1) / 2, TRACK_SHORT[t],
                      transform=axes[-1].get_yaxis_transform(),
                      fontsize=7, color="#666666", va="center", rotation=90)
    fig.supxlabel("Within-episode step of first completion", fontsize=9)
    handles = [
        plt.Line2D([], [], color=BASE_C, marker="o", ls="", ms=6,
                   label="baseline"),
        plt.Line2D([], [], color=HEB_C, marker="D", ls="", ms=5.5,
                   label="+Hebbian"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=2,
               frameon=True, framealpha=0.95, columnspacing=1.2,
               handletextpad=0.4, bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
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
    p.add_argument("--exclude", nargs="*", default=[], metavar="EXP/seed_N",
                   help="runs to leave out of the aggregation, e.g. "
                        "exp04_ippo/seed_1011")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    exclude = frozenset(args.exclude)

    all_runs, stats = {}, {}
    for label, dirname, _, _ in CONDITIONS:
        runs = load_runs(args.runs_root, dirname, exclude)
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
    wrote_cofire = emit_cofire(stats, all_runs, args.out)
    fig_progression(all_runs, args.out, args.max_steps)
    fig_timeline(all_runs, args.out, args.max_steps)

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
          + (", bond_evolution.pdf" if bond_run else "")
          + (", cofire_summary.csv, cofire_table_rows.tex"
             if wrote_cofire else ""))


if __name__ == "__main__":
    main()
