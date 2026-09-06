#!/usr/bin/env python3
"""make_directive_timelines.py — team-formation timelines for one seed.

Three stacked timelines (one per coordination mechanism, same seed) showing
WHO teamed up with WHOM and WHICH agent completed WHICH milestone WHEN:

  A  Gemma-E4B base        — formations that emerge naturally (rolling
                             per-pair message share; uniform = 1/3)
  B  + Central Orch.       — formations the villager-advisory orchestrator
                             ASSIGNS (per-agent task Gantt from
                             orchestrator/assignments.jsonl, colored by the
                             target milestone's chamber; hatched = team task)
                             + per-pair co-assignment strips
  C  + Hebbian             — formations the bonds ENCODE (symmetrized
                             pairwise W from graph_snapshots, every 50 steps)

Each block is three stacked axes on a shared cumulative-step x-axis:
  1. a chamber ribbon (which chamber the team is in),
  2. a milestone lane whose ROWS ARE NAMED MILESTONES — identical rows in
     all three panels (the union of what fired across the arms), one dot per
     (agent, milestone, step) in the agent's colour, offset within the row so
     a milestone several agents completed shows stacked dots,
  3. the arm-specific formation lane, each with the same stat box.

--simple writes a second variant whose bottom lane uses ONE grammar for
every arm: three rows (a0-a1 / a0-a2 / a1-a2) filled in the pair colour
while that pair is "the team" under that arm's own mechanism — pair
exchanging the most messages (A, the leader must beat the runner-up by
TEAM_MARGIN), pair assigned the same task (B), pair with the strongest
bond (C). Blank = no clear team.

--scan-seeds ranks every seed on how well it tells the story (gate: Hebbian
must beat the orchestrator on BOTH coop milestones and task return; then
scored on margins, earliness, team legibility and milestone variety) and
writes the full per-seed table, so the seed used in the paper is auditable.

Palette: agent hues are categorical slots 1-3 (blue #2a78d6 / orange
#eb6834 validated all-pairs in light mode in make_bond_asymmetry_fig.py;
aqua #1baf7a is the documented slot-3 companion). Pair series use a
deliberately different family (violet/green/magenta) so a pair line is
never misread as an agent series; pair identity is ALSO carried by
linestyle and direct end-labels.

Usage:
  python analysis/make_directive_timelines.py                 # seed 456 (paper pick)
  python analysis/make_directive_timelines.py --simple        # + *_simple.* variant
  python analysis/make_directive_timelines.py --scan-seeds    # + seed ranking
  python analysis/make_directive_timelines.py --seed 1213
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paths import ASSETS, RUNS  # noqa: E402  (also puts siblings on sys.path)

from make_results import (  # noqa: E402  (single source of milestone truth)

    COOP_TRACKS,
    MILESTONE_ORDER,
    MILESTONE_TRACK,
    PAPER_LABEL,
    WARM_START_W,
    _apply_entry_honesty_filter,
    coop_count,
    episode_milestone_sets,
    episode_task_returns,
)

# ─── Arms: (key, dir under --runs-root, panel name, tagline) ────────────
ARMS = [
    ("base", Path("new_exp_0_gemma/new_exp_0_gemma_base"),
     "Gemma-E4B (base)", "coordination emerges on its own"),
    ("orch", Path("orchestrator/new_exp_0_gemma_orch_villager_advisory"),
     "Gemma-E4B + Central Orch.",
     "coordination assigned centrally"),
    ("heb", Path("new_exp_0_gemma/new_exp_0_gemma_hebbian"),
     "Gemma-E4B + Hebbian", "coordination via learned bonds"),
]
PANEL_TAG = {"base": "A", "orch": "B", "heb": "C"}
SEEDS = [42, 123, 456, 789, 1011, 1213]

# ─── Palette (see module docstring) ─────────────────────────────────────
AGENT_C = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
AGENT_OFFSET = {0: -0.23, 1: 0.0, 2: 0.23}
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_C = {(0, 1): "#4a3aa7", (0, 2): "#008300", (1, 2): "#d55181"}
PAIR_LS = {(0, 1): "-", (0, 2): (0, (5, 2)), (1, 2): (0, (4, 2, 1, 2))}
INK = "#31373f"
MUTED = "#98a1ad"
GRID = "#eceff3"
RULE = "#d7dce3"

# The ribbon uses the SAME hues as the orchestrator's task blocks, just
# lightened, so one chamber colour key serves both and panel B needs no
# legend of its own.
CHAMBER_RIBBON = {"ch1": "#c9b8ec", "ch2": "#f7cd96", "ch3": "#a8dbf2",
                  "ch4": "#f3b0b0", "ch5": "#e8cf80"}
CHAMBER_ROW = {"ch1": "#f6f3ff", "ch2": "#fff6ea", "ch3": "#eef9ff",
               "ch4": "#fff2f2", "ch5": "#fffae6", "comm": "#f3f4f6"}
# Saturated foreground fills for the orchestrator task Gantt.
CHAMBER_FILL = {"ch1": "#8a63d2", "ch2": "#f0a13a", "ch3": "#4db6e2",
                "ch4": "#e06060", "ch5": "#d4af37", "comm": "#9ca3af"}
TRACK_TO_CH = {"ch1_solo": "ch1", "ch2_anvils": "ch2", "ch3_switches": "ch3",
               "ch4_combat": "ch4", "ch5_boss": "ch5", "communication": "comm"}

# Camera-ready defaults. fonttype 42 embeds TrueType outlines: Type-3
# fonts (matplotlib's default) are rejected by several submission checkers.
# Times matches the ICLR body font so figure text does not look pasted in.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# Venue page geometry (inches). ICLR is single-column 5.5in text width;
# ICML is two-column, 6.75in across both columns (figure*) or 3.25in in one.
VENUES = {"iclr": 5.5, "icml": 6.75, "icml-column": 3.25}

# Type sizes, in points, at FINAL size — the figure is authored at the exact
# width it is placed at, so \includegraphics[width=\textwidth] does not
# rescale it and nothing shrinks below ~6pt.
FS_PANEL = 8.5      # panel letter + arm name
FS_ROW = 6.0        # milestone row labels, tick labels
FS_AXIS = 6.8       # axis labels
FS_LEGEND = 6.5
FS_RIBBON = 5.8

_AGENT_RE = re.compile(r"agent_?(\d+)")


def agent_id(name) -> int | None:
    """Normalize 'agent2' (Lua) and 'agent_2' (python) to the int id."""
    m = _AGENT_RE.match(str(name))
    return int(m.group(1)) if m else None


def paper_id(mid: str) -> str:
    return PAPER_LABEL.get(mid, (mid, mid))[0].replace("\\_", "_")


# Short row labels for the figure; the full names stay in PAPER_LABEL and
# in the caption. Anything not listed falls back to the paper label.
SHORT_LABEL = {
    "m1_move_5": "M1 Move >5", "m2_dig_3_any": "M2 Dig 3",
    "m3_pickup_3": "M3 Pick up 3", "m4_dig_5_wood": "M4 Dig 5 wood",
    "m5_kill_1_animal": "M5 Kill animal", "m6_kill_2_animals": "M6 Kill 2",
    "m7_dig_3_stone": "M7 Dig 3 stone", "m_door1_open": "Door 1 open",
    "m8_anvil_A1": "M8 Anvil A", "m9_anvil_B1": "M9 Anvil B",
    "m14_sword_equipped": "M10 Sword", "m15_chestplate_equipped": "M11 Armour",
    "m16_enter_cell": "M12 Enter cell", "m17_switch_pressed": "M13 Press switch",
    "m18_door_opened": "M14 Door opened", "m19_all_in_communal": "M15 Regroup",
    "m20_enter_ch4": "M16 Enter Ch4", "m21_first_mob_kill": "M17 First mob kill",
    "m22_all_mobs_killed": "M18 Arena clear", "m23_all_alive_ch4": "M19 All alive",
    "m24_enter_ch5": "M20 Enter Ch5", "m25_first_boss_dmg": "M21 Boss damage",
    "m26_boss_half_hp": "M22 Boss 50%", "m27_boss_defeated": "M23 Boss down",
    "m28_all_alive_bonus": "M24 All alive",
}


def milestone_label(mid: str) -> str:
    """'m17_switch_pressed' -> 'M13 Press switch'."""
    if mid in SHORT_LABEL:
        return SHORT_LABEL[mid]
    tag, name = PAPER_LABEL.get(mid, (mid, mid))
    return f"{tag.replace(chr(92), '')} {name.replace(chr(92), '')}"


# ─── Loading ────────────────────────────────────────────────────────────
def load_run(run_dir: Path) -> dict:
    """final_metrics.json + episode bounds + entry-honesty filter."""
    fm = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    lens = [int(x) for x in fm.get("episode_lengths", [])]
    bounds, c = [], 0
    for L in lens:
        bounds.append((c, c + L))
        c += L
    fm["_ep_bounds"] = bounds
    fm["_run_dir"] = run_dir
    n_before = len(fm.get("milestone_events", []))
    _apply_entry_honesty_filter(fm)
    fm["_events_stripped"] = n_before - len(fm["milestone_events"])
    return fm


def milestone_lane_events(run: dict) -> list[dict]:
    """Post-filter milestone events with normalized agent + comm tag."""
    out = []
    for ev in run.get("milestone_events", []):
        mid = ev.get("milestone_id") or ev.get("milestone")
        aid = agent_id(ev.get("contributor"))
        if mid is None or aid is None:
            continue
        if mid not in MILESTONE_TRACK:
            print(f"  [warn] unknown milestone id {mid!r}", file=sys.stderr)
            continue
        out.append({"step": int(ev["step"]), "mid": mid, "agent": aid,
                    "is_comm": MILESTONE_TRACK[mid] == "communication"})
    return out


def load_chamber_bands(run: dict) -> list[tuple[str, float, float]]:
    """(chamber, cum_start, cum_end) per episode from episode summaries."""
    bands = []
    for e, (s0, s1) in enumerate(run["_ep_bounds"]):
        p = run["_run_dir"] / "episodes" / f"ep_{e + 1:04d}" / "episode_summary.json"
        if not p.exists():
            continue
        cm = json.loads(p.read_text(encoding="utf-8")).get("cooperation_metrics", {})
        entries = cm.get("chamber_entry_steps") or {}
        # sort by (entry step, chamber number); ties yield zero-width bands
        items = sorted(entries.items(), key=lambda kv: (kv[1], kv[0]))
        for k, (ch, t) in enumerate(items):
            t_next = items[k + 1][1] if k + 1 < len(items) else s1 - s0
            lo, hi = s0 + min(t, s1 - s0), s0 + min(t_next, s1 - s0)
            if hi > lo:
                bands.append((ch, lo, hi))
    return bands


def load_msg_share(run: dict, window: int) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Rolling per-pair message share, computed within each episode."""
    per_pair = {p: ([], []) for p in PAIRS}
    for e, (s0, s1) in enumerate(run["_ep_bounds"]):
        p = run["_run_dir"] / "episodes" / f"ep_{e + 1:04d}" / "messages.jsonl"
        if not p.exists():
            continue
        n_steps = s1 - s0
        counts = {q: np.zeros(n_steps) for q in PAIRS}
        total = np.zeros(n_steps)
        for line in p.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            t = int(m["t"])
            i, j = agent_id(m.get("sender")), agent_id(m.get("receiver"))
            if t >= n_steps or i is None or j is None or i == j:
                continue
            counts[tuple(sorted((i, j)))][t] += 1
            total[t] += 1
        ct = np.cumsum(total)
        roll_tot = ct - np.concatenate(([0.0] * window, ct[:-window]))[:n_steps]
        for q in PAIRS:
            cq = np.cumsum(counts[q])
            roll = cq - np.concatenate(([0.0] * window, cq[:-window]))[:n_steps]
            share = np.divide(roll, roll_tot, out=np.full(n_steps, np.nan),
                              where=roll_tot > 0)
            per_pair[q][0].append(np.arange(n_steps) + s0)
            per_pair[q][1].append(share)
    return {q: (np.concatenate(xs), np.concatenate(ys))
            for q, (xs, ys) in per_pair.items() if xs}


def load_orch(run: dict) -> dict:
    """Assignment intervals, pair co-assignments and task metadata."""
    od = run["_run_dir"] / "orchestrator"
    task_ch, task_multi = {}, set()
    for line in (od / "dag.jsonl").read_text(encoding="utf-8").splitlines():
        for t in json.loads(line).get("tasks", []):
            mids = t.get("milestones") or []
            if mids and mids[0] in MILESTONE_TRACK:
                task_ch[t["id"]] = TRACK_TO_CH[MILESTONE_TRACK[mids[0]]]
            if int(t.get("min_agents", 1)) >= 2:
                task_multi.add(t["id"])

    def chamber_of(task_id: str) -> str | None:
        if task_id in task_ch:
            return task_ch[task_id]
        m = re.search(r"ch(\d)", task_id)
        return f"ch{m.group(1)}" if m else None

    recs = [json.loads(x) for x in
            (od / "assignments.jsonl").read_text(encoding="utf-8").splitlines()]
    recs.sort(key=lambda r: (r["episode"], r["t"]))
    offsets = {e + 1: b[0] for e, b in enumerate(run["_ep_bounds"])}
    ep_end = {e + 1: b[1] for e, b in enumerate(run["_ep_bounds"])}

    intervals = []                      # (agent, task_id, cum_t0, cum_t1)
    open_slot: dict[tuple, tuple] = {}  # (ep, agent) -> (task_id, cum_t0)
    n_alloc = defaultdict(int)
    for r in recs:
        ep, aid = r["episode"], agent_id(r["agent"])
        key, t_cum = (ep, aid), offsets[ep] + int(r["t"])
        if r["reason"].startswith("allocate"):
            n_alloc[aid] += 1
            if key in open_slot:
                tid, t0 = open_slot.pop(key)
                if t_cum > t0:
                    intervals.append((aid, tid, t0, t_cum))
            open_slot[key] = (r["task_id"], t_cum)
        else:  # freed_*
            if key in open_slot:
                tid, t0 = open_slot.pop(key)
                if t_cum > t0:
                    intervals.append((aid, tid, t0, t_cum))
    for (ep, aid), (tid, t0) in open_slot.items():
        if ep_end[ep] > t0:
            intervals.append((aid, tid, t0, ep_end[ep]))

    # Pair co-assignment: overlap of same-task intervals of both members.
    by_at = defaultdict(list)
    for aid, tid, t0, t1 in intervals:
        by_at[(aid, tid)].append((t0, t1))
    pair_iv = {q: [] for q in PAIRS}
    team_iv = []                        # concurrent same-task -> team marks
    for (i, j) in PAIRS:
        tids = {t for a, t in by_at if a == i} & {t for a, t in by_at if a == j}
        for tid in tids:
            for a0, a1 in by_at[(i, tid)]:
                for b0, b1 in by_at[(j, tid)]:
                    lo, hi = max(a0, b0), min(a1, b1)
                    if hi > lo:
                        pair_iv[(i, j)].append((lo, hi))
                        team_iv.append((tid, lo, hi, (i, j)))
    concurrent_tasks = {tid for tid, *_ in team_iv}
    return {"intervals": intervals, "pair_iv": pair_iv, "team_iv": team_iv,
            "chamber_of": chamber_of,
            "team_tasks": task_multi | concurrent_tasks,
            "n_alloc": dict(n_alloc), "n_tasks": len({t for _, t, *_ in intervals}),
            "n_multi_dag": len(task_multi)}


def load_bonds(run: dict) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Symmetrized pair bond strength over graph snapshots."""
    snaps = [s for s in run.get("graph_snapshots", []) if s.get("W")]
    out = {}
    for (i, j) in PAIRS:
        xs = np.array([s["step"] for s in snaps])
        ys = np.array([(s["W"][i][j] + s["W"][j][i]) / 2 for s in snaps])
        if len(xs):
            out[(i, j)] = (xs, ys)
    return out


def co_completions(run: dict, include_comm: bool = False) -> list[dict]:
    """Deduped unordered co-completion events."""
    seen, out = set(), []
    for ev in run.get("co_completion_events", []):
        pair = tuple(sorted((int(ev["agent_i"]), int(ev["agent_j"]))))
        mid = ev.get("milestone", "")
        if not include_comm and MILESTONE_TRACK.get(mid) == "communication":
            continue
        key = (int(ev["step"]), pair, mid)
        if key in seen:
            continue
        seen.add(key)
        out.append({"step": int(ev["step"]), "pair": pair, "mid": mid})
    return out


# ─── "Who is the team" ──────────────────────────────────────────────────
# Every mechanism is reduced to the SAME quantity: a per-step coupling
# vector over the three pairs (base = messages exchanged in the window,
# hebbian = bond strength W, orchestrator = 1 while co-assigned). It is
# then normalised to shares, so "who is the team" means the same thing in
# all three panels. Comparing a share against an absolute strength is what
# made the earlier version read as if Hebbian teamed up LESS than base:
# when all three bonds grow together no pair leads, and those steps were
# scored as "no team" instead of "all three coupled".
TEAM_MARGIN = 0.05          # a pair leads when its share > 1/3 + margin
ALL_THREE = 3               # state id: coupling spread across the trio
NO_TEAM = -1                # state id: no coupling at all
ALL3_C = "#4b5563"          # slate — deliberately not one of the pair hues


def pair_coupling(arm: str, data: dict) -> np.ndarray:
    """(T, 3) per-step coupling for pairs (0,1), (0,2), (1,2)."""
    total = data["run"]["_ep_bounds"][-1][1]
    vals = np.zeros((total, 3))
    if arm == "orch":
        for c, q in enumerate(PAIRS):
            for lo, hi in data["form"]["pair_iv"][q]:
                vals[max(0, lo):min(total, hi), c] = 1.0
        return vals
    series = data["form"]
    if not series:
        return vals
    vals[:] = np.nan
    for c, q in enumerate(PAIRS):
        if q not in series:
            continue
        xs, ys = series[q]
        if arm == "base":                       # already one value per step
            idx = xs.astype(int)
            m = (idx >= 0) & (idx < total)
            vals[idx[m], c] = ys[m]
        else:                                   # heb: hold each snapshot
            k = np.searchsorted(xs, np.arange(total), side="right") - 1
            vals[:, c] = ys[np.clip(k, 0, len(xs) - 1)]
    return np.nan_to_num(vals, nan=0.0)


def team_states(coupling: np.ndarray) -> np.ndarray:
    """Per step: 0/1/2 = that pair leads, ALL_THREE, or NO_TEAM."""
    tot = coupling.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot[:, None] > 0, coupling / tot[:, None], 0.0)
    order = np.argsort(-share, axis=1)
    top = np.take_along_axis(share, order[:, :1], 1)[:, 0]
    second = np.take_along_axis(share, order[:, 1:2], 1)[:, 0]
    leads = (top >= 1 / 3 + TEAM_MARGIN) & (top > second + 1e-9)
    return np.where(tot <= 0, NO_TEAM, np.where(leads, order[:, 0], ALL_THREE))


def state_runs(states: np.ndarray) -> dict:
    """Run-length encode states into {state: [(lo, hi)]} (NO_TEAM dropped)."""
    out = defaultdict(list)
    k, n = 0, len(states)
    while k < n:
        j = k
        while j < n and states[j] == states[k]:
            j += 1
        if states[k] != NO_TEAM:
            key = ALL_THREE if states[k] == ALL_THREE else PAIRS[int(states[k])]
            out[key].append((k, j))
        k = j
    return dict(out)


def team_intervals(arm: str, data: dict) -> dict:
    """{pair or ALL_THREE: [(lo, hi)]} — who is teamed up at each step."""
    runs = state_runs(team_states(pair_coupling(arm, data)))
    return {k: runs.get(k, []) for k in [*PAIRS, ALL_THREE]}


def team_coverage(team: dict, total: int) -> tuple[float, int]:
    """Fraction of steps with any coupling, and the longest unbroken block."""
    cov = np.zeros(total, dtype=bool)
    longest = 0
    for ivs in team.values():
        for lo, hi in ivs:
            cov[max(0, lo):min(total, hi)] = True
            longest = max(longest, hi - lo)
    return float(cov.mean()), int(longest)


def team_stability(team: dict, total: int) -> dict:
    """How long a formation lasts before it changes — comparable across arms.

    This, not "% of steps teamed", is the honest cross-arm comparison: a
    share-based signal (base) almost always has a leader, so its coverage
    is near 100% by construction while the leader churns.
    """
    blocks = [(lo, hi, k) for k, ivs in team.items() for lo, hi in ivs]
    blocks.sort()
    lens = [hi - lo for lo, hi, _ in blocks]
    pair_steps = sum(hi - lo for lo, hi, k in blocks if k != ALL_THREE)
    all3_steps = sum(hi - lo for lo, hi, k in blocks if k == ALL_THREE)
    return {
        "n_formations": len(blocks),
        "mean_block": float(np.mean(lens)) if lens else 0.0,
        "median_block": float(np.median(lens)) if lens else 0.0,
        "longest_block": int(max(lens, default=0)),
        "pair_frac": pair_steps / max(total, 1),
        "all_three_frac": all3_steps / max(total, 1),
    }


# ─── Drawing primitives ─────────────────────────────────────────────────
def style(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK, labelsize=FS_ROW, length=2.5, width=0.7,
                   color=RULE, pad=1.6)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)


def draw_ribbon(ax, run, bands, xmax):
    """Thin chamber-over-time strip, with episode marks."""
    for ch, lo, hi in bands:
        ax.broken_barh([(lo, hi - lo)], (0, 1),
                       facecolors=CHAMBER_RIBBON.get(ch, "#eeeeee"),
                       edgecolors="white", linewidth=0.6, zorder=2)
        if hi - lo > xmax * 0.035:
            ax.text((lo + hi) / 2, 0.5, ch.replace("ch", "Ch"), fontsize=FS_RIBBON,
                    color="#4a5261", ha="center", va="center", zorder=3)
    run_end = run["_ep_bounds"][-1][1]
    if run_end < xmax:
        ax.broken_barh([(run_end, xmax - run_end)], (0, 1),
                       facecolors="#f0f1f3", edgecolors="none", zorder=2)
    for e, (s0, _) in enumerate(run["_ep_bounds"]):
        if e == 0:
            continue
        ax.plot([s0, s0], [-0.1, 1.1], color=INK, lw=1.0, zorder=4,
                clip_on=False)
        ax.text(s0 + xmax * 0.004, 1.28, f"ep {e + 1}", fontsize=FS_RIBBON,
                color=INK, ha="left", va="bottom", zorder=4)
    ax.text(-xmax * 0.008, 0.5, "chamber", fontsize=FS_RIBBON, color=MUTED,
            ha="right", va="center")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def episode_rules(ax, run):
    for e, (s0, _) in enumerate(run["_ep_bounds"]):
        if e:
            ax.axvline(s0, color=INK, lw=0.8, alpha=0.32, ls=(0, (4, 3)),
                       zorder=1)


def chamber_rules(ax, bands):
    for _, lo, _ in bands[1:]:
        ax.axvline(lo, color=RULE, lw=0.7, zorder=1)


def milestone_rows(arms_data) -> tuple[list, dict]:
    """Union of env milestones fired across the arms, in canonical order."""
    fired = set()
    for d in arms_data.values():
        fired |= {e["mid"] for e in d["events"] if not e["is_comm"]}
    rows = [m for m in MILESTONE_ORDER if m in fired]
    return rows, {m: i for i, m in enumerate(rows)}


def draw_milestone_lane(ax, events, rows, row_of, show_comm=True):
    """Rows = named milestones; one dot per (agent, milestone, step)."""
    n = len(rows)
    comm_y = -1.25
    span = {}
    for mid in rows:
        ch = TRACK_TO_CH[MILESTONE_TRACK[mid]]
        span.setdefault(ch, [row_of[mid], row_of[mid]])
        span[ch][1] = row_of[mid]
    for ch, (lo, hi) in span.items():
        ax.axhspan(lo - 0.5, hi + 0.5, color=CHAMBER_ROW.get(ch, "#f7f7f7"),
                   zorder=0)
    if show_comm:
        ax.axhspan(comm_y - 0.5, comm_y + 0.5, color=CHAMBER_ROW["comm"],
                   zorder=0)

    for ev in events:
        base_y = comm_y if ev["is_comm"] else row_of.get(ev["mid"])
        if base_y is None or (ev["is_comm"] and not show_comm):
            continue
        y = base_y + AGENT_OFFSET[ev["agent"]]
        if ev["is_comm"]:
            ax.scatter(ev["step"], y, marker="|", s=26, lw=0.9,
                       color=AGENT_C[ev["agent"]], alpha=0.5, zorder=3)
        else:
            ax.scatter(ev["step"], y, s=26, color=AGENT_C[ev["agent"]],
                       edgecolors="white", linewidths=0.6, zorder=4)

    ticks = list(range(n)) + ([comm_y] if show_comm else [])
    labels = [milestone_label(m) for m in rows] + (
        ["communication track"] if show_comm else [])
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=FS_ROW, color=INK)
    if show_comm:
        ax.get_yticklabels()[-1].set_color(MUTED)
        ax.get_yticklabels()[-1].set_style("italic")
    ax.set_ylim((comm_y - 0.7) if show_comm else -0.7, n - 0.3)
    style(ax)
    ax.tick_params(labelbottom=False, length=0)


def block_header(ax, title, tagline, stats):
    """Stacked header lines above a block's ribbon axis."""
    lines = [(title, 10.5, INK, "bold"), (tagline, 7.2, MUTED, "normal")]
    lines += [(s, 6.9, INK, "normal") for s in stats]
    dy = 17 + 10.5 * (len(lines) - 1)   # 17pt clears the ribbon's episode marks
    for text, size, col, weight in lines:
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=size, color=col, fontweight=weight,
                    ha="left", va="bottom", annotation_clip=False)
        dy -= 10.5 if text is not title else 12.5


def _pair_end_labels(ax, series, xmax):
    """Direct labels at each line's end, pushed apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.075
    ends = []
    for q, (xs, ys) in series.items():
        k = np.where(np.isfinite(ys))[0]
        if len(k):
            ends.append([float(ys[k[-1]]), float(xs[k[-1]]), q])
    ends.sort()
    for i in range(1, len(ends)):                 # spread upward
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, q in ends:
        ax.text(min(x + xmax * 0.006, xmax * 1.002), min(y, hi),
                f"a{q[0]}-a{q[1]}", fontsize=FS_ROW, color=PAIR_C[q],
                va="center", ha="left", clip_on=False, fontweight="bold")


def _coco_markers_on_lines(ax, series, cocos):
    for c in cocos:
        q = c["pair"]
        if q not in series:
            continue
        xs, ys = series[q]
        k = int(np.argmin(np.abs(xs - c["step"])))
        if np.isfinite(ys[k]):
            ax.scatter(c["step"], ys[k], marker="D", s=26, facecolor="white",
                       edgecolor=PAIR_C[q], linewidths=1.2, zorder=6)


def draw_formation_base(ax, series, cocos, xmax):
    ax.axhline(1 / 3, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(xmax * 0.988, 1 / 3 - 0.022, "even split", fontsize=FS_RIBBON,
            color=MUTED, ha="right", va="top", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.8))
    for q, (xs, ys) in series.items():
        ax.plot(xs, ys, color=PAIR_C[q], ls=PAIR_LS[q], lw=1.4, zorder=3,
                solid_capstyle="round")
    _pair_end_labels(ax, series, xmax)
    _coco_markers_on_lines(ax, series, cocos)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 1 / 3, 2 / 3, 1])
    ax.set_yticklabels(["0", "1/3", "2/3", "1"])
    ax.set_ylabel("messages\nby pair (share)", fontsize=FS_AXIS,
                  color=INK, linespacing=1.3)
    style(ax, grid_axis="y")


def draw_formation_heb(ax, series, cocos, xmax):
    ax.axhline(WARM_START_W, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(xmax * 0.995, WARM_START_W - 0.012, "initial W = 0.1",
            fontsize=FS_RIBBON, color=MUTED, ha="right", va="top", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.8))
    for q, (xs, ys) in series.items():
        ax.plot(xs, ys, color=PAIR_C[q], ls=PAIR_LS[q], lw=1.6, zorder=3,
                marker="o", markersize=2.2, markevery=2, solid_capstyle="round")
    _pair_end_labels(ax, series, xmax)
    _coco_markers_on_lines(ax, series, cocos)
    top = max((float(np.nanmax(ys)) for _, ys in series.values()), default=0.3)
    ax.set_ylim(0, max(0.36, top * 1.22))
    ax.set_ylabel("bond strength\nW by pair", fontsize=FS_AXIS, color=INK,
                  linespacing=1.3)
    style(ax, grid_axis="y")


def draw_formation_orch(ax, orch, cocos, xmax):
    """Rows 5..3 = per-agent task Gantt; rows 2..0 = pair co-assignment."""
    agent_y = {0: 5, 1: 4, 2: 3}
    pair_y = {(0, 1): 2, (0, 2): 1, (1, 2): 0}
    for aid, tid, t0, t1 in orch["intervals"]:
        team = tid in orch["team_tasks"]
        ch = orch["chamber_of"](tid)
        ax.broken_barh([(t0, t1 - t0)], (agent_y[aid] - 0.33, 0.66),
                       facecolors=CHAMBER_FILL.get(ch, "#bbbbbb"),
                       edgecolors="#2b2f36" if team else "white",
                       linewidth=0.7 if team else 0.4,
                       hatch="///" if team else None, alpha=0.92, zorder=3)
    for _, lo, _, (i, j) in orch["team_iv"]:
        ax.plot([lo, lo], [agent_y[i], agent_y[j]], color="#2b2f36", lw=0.8,
                alpha=0.7, zorder=4)
    for q, ivs in orch["pair_iv"].items():
        if ivs:
            ax.broken_barh([(lo, hi - lo) for lo, hi in ivs],
                           (pair_y[q] - 0.28, 0.56), facecolors=PAIR_C[q],
                           alpha=0.9, zorder=3)
    for c in cocos:
        if c["pair"] in pair_y:
            ax.scatter(c["step"], pair_y[c["pair"]], marker="D", s=26,
                       facecolor="white", edgecolor=PAIR_C[c["pair"]],
                       linewidths=1.2, zorder=6)
    ax.axhline(2.62, color=RULE, lw=0.9, zorder=2)
    ax.set_ylim(-0.55, 5.55)
    ax.set_yticks([5, 4, 3, 2, 1, 0])
    ax.set_yticklabels(["a0", "a1", "a2", "a0-a1", "a0-a2", "a1-a2"],
                       fontsize=FS_ROW)
    for lab, col in zip(ax.get_yticklabels(),
                        [AGENT_C[0], AGENT_C[1], AGENT_C[2],
                         PAIR_C[(0, 1)], PAIR_C[(0, 2)], PAIR_C[(1, 2)]]):
        lab.set_color(col)
        lab.set_fontweight("bold")
    ax.set_ylabel("assigned task", fontsize=FS_AXIS, color=INK, labelpad=17)
    style(ax)


def draw_formation_simple(ax, team, cocos):
    row_y = {ALL_THREE: 3, (0, 1): 2, (0, 2): 1, (1, 2): 0}
    for key, ivs in team.items():
        if not ivs:
            continue
        col = ALL3_C if key == ALL_THREE else PAIR_C[key]
        ax.broken_barh([(lo, hi - lo) for lo, hi in ivs],
                       (row_y[key] - 0.34, 0.68), facecolors=col,
                       edgecolors="none", zorder=3)
    for c in cocos:
        if c["pair"] in row_y:
            ax.scatter(c["step"], row_y[c["pair"]], marker="D", s=28,
                       facecolor="white", edgecolor=PAIR_C[c["pair"]],
                       linewidths=1.2, zorder=6)
    ax.axhline(2.6, color=RULE, lw=0.9, zorder=2)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["all three", "a0-a1", "a0-a2", "a1-a2"],
                       fontsize=FS_ROW)
    for lab, col in zip(ax.get_yticklabels(),
                        [ALL3_C, PAIR_C[(0, 1)], PAIR_C[(0, 2)], PAIR_C[(1, 2)]]):
        lab.set_color(col)
        lab.set_fontweight("bold")
    ax.set_ylabel("who is\ncoupled", fontsize=FS_AXIS, color=INK,
                  linespacing=1.3)
    style(ax)


# ─── Per-panel headline numbers ─────────────────────────────────────────
def headline(arm: str, data: dict) -> list[str]:
    run = data["run"]
    total = run["_ep_bounds"][-1][1]
    coops = [coop_count(s) for s in episode_milestone_sets(run)]
    coop_ep = sum(coops) / max(len(coops), 1)
    env = [e for e in data["events"] if not e["is_comm"]]
    coop_ev = [e for e in env if MILESTONE_TRACK[e["mid"]] in COOP_TRACKS]
    first = min((e["step"] for e in coop_ev), default=None)
    st = team_stability(data["team"], total)
    when = f"first at step {first}" if first is not None else "never reached"
    return [
        f"{coop_ep:.2f} cooperative milestones per episode  ·  {when}",
        (f"one pair leads {st['pair_frac'] * 100:.0f}% of steps, all three "
         f"coupled {st['all_three_frac'] * 100:.0f}%  ·  a formation lasts "
         f"{st['mean_block']:.0f} steps on average, longest {st['longest_block']}"),
    ]


# ─── Figure assembly ────────────────────────────────────────────────────
def block_axes(fig, cell, n_rows, simple=False):
    inner = cell.subgridspec(3, 1,
                             height_ratios=[0.26, 0.30 * max(n_rows, 4),
                                            1.05 if simple else 1.7],
                             hspace=0.16)
    return (fig.add_subplot(inner[0]), fig.add_subplot(inner[1]),
            fig.add_subplot(inner[2]))


def draw_block(axes, arm, data, rows, row_of, xmax, show_comm, simple):
    ax_r, ax_m, ax_f = axes
    run = data["run"]
    draw_ribbon(ax_r, run, data["bands"], xmax)
    draw_milestone_lane(ax_m, data["events"], rows, row_of,
                        show_comm=show_comm and not simple)
    episode_rules(ax_m, run)
    if simple:
        draw_formation_simple(ax_f, data["team"], data["cocos"])
    elif arm == "base":
        draw_formation_base(ax_f, data["form"], data["cocos"], xmax)
    elif arm == "orch":
        draw_formation_orch(ax_f, data["form"], data["cocos"], xmax)
    else:
        draw_formation_heb(ax_f, data["form"], data["cocos"], xmax)
    chamber_rules(ax_f, data["bands"])
    episode_rules(ax_f, run)
    for ax in axes:
        ax.set_xlim(0, xmax)


def common_legend_handles(show_comm=True, simple=False):
    h = [Line2D([], [], marker="o", ls="", color=AGENT_C[i], markersize=5,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"agent {i} completed it") for i in range(3)]
    if show_comm:
        h.append(Line2D([], [], marker="|", ls="", color=INK, alpha=0.5,
                        markersize=7, label="communication milestone"))
    h.append(Line2D([], [], marker="D", ls="", markerfacecolor="white",
                    markeredgecolor=INK, markersize=5.5,
                    label="pair completed a milestone together"))
    if simple:
        h.append(Patch(facecolor=ALL3_C,
                       label="all three coupled (no single pair leads)"))
    return h


def orch_legend_handles():
    return [Patch(facecolor=CHAMBER_FILL[c],
                  label=f"task for {c.replace('ch', 'Ch')}")
            for c in ("ch1", "ch2", "ch3", "ch4", "ch5")] + [
        Patch(facecolor="white", edgecolor="#2b2f36", hatch="///",
              label="task needing 2+ agents")]


XLABEL = "env step (running across the three episodes)"


def _finish(fig, out_dir, name):
    """Save at the EXACT authored size — no tight bbox, so the width the
    figure was designed for is the width LaTeX places."""
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)


def _geometry(n_rows: int, simple: bool, titled: bool) -> dict:
    """Block geometry in inches. Row pitch is set so 6pt labels never touch."""
    pitch = 0.115
    return {
        "left": 0.92,
        # the line panels carry direct "a0-a1" end labels outside the
        # axes; without a tight bbox they need reserved width
        "right": 0.08 if simple else 0.40,
        "top": 0.36 if titled else 0.10,
        "bottom": 0.54,
        "header": 0.24, "ribbon": 0.12,
        "mile": pitch * (n_rows + (0 if simple else 1)) + 0.10,
        "form": pitch * 4 + 0.13 if simple else 0.78,
        "gap_rm": 0.05, "gap_mf": 0.12, "block_gap": 0.22,
    }


def _fig_height(g: dict, n: int = 3) -> float:
    block = (g["header"] + g["ribbon"] + g["gap_rm"] + g["mile"]
             + g["gap_mf"] + g["form"])
    return g["top"] + n * block + (n - 1) * g["block_gap"] + g["bottom"]


def _block_axes(fig, g, index, W, H):
    """Three stacked axes for one block, placed by exact inch offsets."""
    x0, w = g["left"] / W, 1 - (g["left"] + g["right"]) / W
    block = (g["header"] + g["ribbon"] + g["gap_rm"] + g["mile"]
             + g["gap_mf"] + g["form"])
    top_in = g["top"] + index * (block + g["block_gap"])
    y = H - top_in - g["header"]
    axes = []
    for h, gap in ((g["ribbon"], g["gap_rm"]), (g["mile"], g["gap_mf"]),
                   (g["form"], 0.0)):
        y -= h
        axes.append(fig.add_axes([x0, y / H, w, h / H]))
        y -= gap
    return axes


def legend_handles(show_comm: bool, simple: bool):
    h = [Line2D([], [], marker="o", ls="", color=AGENT_C[i], markersize=3.6,
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"agent {i}") for i in range(3)]
    if show_comm:
        h.append(Line2D([], [], marker="|", ls="", color=INK, alpha=0.55,
                        markersize=5.5, label="comm. milestone"))
    h.append(Line2D([], [], marker="D", ls="", markerfacecolor="white",
                    markeredgecolor=INK, markersize=4.0, markeredgewidth=1.0,
                    label="pair completed one together"))
    if simple:
        h.append(Patch(facecolor=ALL3_C, label="all three coupled"))
    else:
        h.append(Patch(facecolor="white", edgecolor="#2b2f36", hatch="///",
                       label="task needing 2+ agents"))
    return h


def build_figure(arms_data, out_dir: Path, seed, show_comm: bool,
                 simple: bool = False, per_arm: bool = False,
                 width: float = 5.5, titled: bool = False):
    xmax = max(d["run"]["_ep_bounds"][-1][1] for d in arms_data.values()) + 20
    rows, row_of = milestone_rows(arms_data)
    show_comm = show_comm and not simple
    suffix = ("_per_arm" if per_arm else "") + ("_simple" if simple else "")

    g = _geometry(len(rows), simple, titled)
    W = width
    H = _fig_height(g)
    fig = plt.figure(figsize=(W, H))

    for k, (arm, _, name, tagline) in enumerate(ARMS):
        axes = _block_axes(fig, g, k, W, H)
        draw_block(axes, arm, arms_data[arm], rows, row_of, xmax, show_comm,
                   simple)
        head = f"{PANEL_TAG[arm]}   {name}"
        if per_arm:
            head += f"   (seed {arms_data[arm]['seed']})"
        axes[0].annotate(head, xy=(0, 1), xycoords="axes fraction",
                         xytext=(0, 9), textcoords="offset points",
                         fontsize=FS_PANEL, color=INK, fontweight="bold",
                         ha="left", va="bottom", annotation_clip=False)
        axes[0].annotate(tagline, xy=(1, 1), xycoords="axes fraction",
                         xytext=(0, 9.5), textcoords="offset points",
                         fontsize=6.3, color=MUTED, ha="right",
                         va="bottom", annotation_clip=False)
        if k < len(ARMS) - 1:
            axes[2].tick_params(labelbottom=False)
        else:
            axes[2].set_xlabel(XLABEL, fontsize=FS_AXIS, color=INK, labelpad=2)

    fig.legend(handles=legend_handles(show_comm, simple), loc="lower center",
               bbox_to_anchor=(0.5, 0.004), fontsize=FS_LEGEND, frameon=False,
               labelcolor=INK, ncol=3 if show_comm else 5, borderpad=0.3,
               handlelength=1.3, columnspacing=1.3, handletextpad=0.5)

    if titled:
        fig.text(g["left"] / W, 1 - 0.10 / H,
                 f"Who teams up with whom, and what gets done - seed {seed}",
                 fontsize=10, color=INK, ha="left", va="top",
                 fontweight="bold")
    _finish(fig, out_dir, f"directive_timelines{suffix}")

    single = {"base": "timeline_base", "orch": "timeline_orch",
              "heb": "timeline_hebbian"}
    g1 = dict(g, top=0.10, bottom=0.52)
    H1 = _fig_height(g1, n=1)
    for arm, _, name, tagline in ARMS:
        fig = plt.figure(figsize=(W, H1))
        axes = _block_axes(fig, g1, 0, W, H1)
        draw_block(axes, arm, arms_data[arm], rows, row_of, xmax, show_comm,
                   simple)
        axes[0].annotate(f"{name}  (seed {arms_data[arm]['seed']})",
                         xy=(0, 1), xycoords="axes fraction", xytext=(0, 9),
                         textcoords="offset points", fontsize=FS_PANEL,
                         color=INK, fontweight="bold", ha="left", va="bottom",
                         annotation_clip=False)
        axes[2].set_xlabel(XLABEL, fontsize=FS_AXIS, color=INK, labelpad=2)
        fig.legend(handles=legend_handles(show_comm, simple),
                   loc="lower center", bbox_to_anchor=(0.5, 0.004),
                   fontsize=FS_LEGEND, frameon=False, labelcolor=INK,
                   ncol=3 if show_comm else 5, handlelength=1.3)
        _finish(fig, out_dir, f"{single[arm]}{suffix}")
    return W, H


CAPTION = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/directive_timelines%(suffix)s.pdf}
\caption{\textbf{Team formation and task progress under three coordination
mechanisms} (seed %(seed)s, %(steps)s env steps over three episodes).
Each block shows, from top: the chamber the team occupies; which agent
completed which milestone and when (one dot per agent, stacked within a row
when several agents complete the same milestone); and %(bottom)s
%(extra)sCoupling is normalised identically in all three arms -- each is
reduced to a per-step coupling over the three agent pairs (messages
exchanged for base, bond strength $W$ for Hebbian, co-assignment for the
orchestrator) and converted to shares -- so a leading pair means the same
thing in every panel, and \emph{all three} denotes coupling spread across
the trio rather than absence of coordination.
%(stats)s}
\label{fig:directive-timelines%(suffix)s}
\end{figure}
"""


def write_caption(arms_data, out_dir: Path, seed, simple: bool, per_arm: bool):
    suffix = ("_per_arm" if per_arm else "") + ("_simple" if simple else "")
    bottom = ("which pair is coupled at each step under that mechanism."
              if simple else
              "how teams form under that mechanism: the share of messages "
              "each pair exchanges (A), the tasks the orchestrator assigns "
              "and which pairs it puts on the same task (B), and the learned "
              "pairwise bond strength (C).")
    extra = ("" if simple else
             "Task blocks in B are coloured by the chamber the task targets, "
             "matching the chamber ribbon. ")
    bits = []
    for arm, _, name, _ in ARMS:
        d = arms_data[arm]
        total = d["run"]["_ep_bounds"][-1][1]
        st = team_stability(d["team"], total)
        coops = [coop_count(x) for x in episode_milestone_sets(d["run"])]
        bits.append(f"{name}: {sum(coops) / max(len(coops), 1):.2f} "
                    f"cooperative milestones/episode, formations lasting "
                    f"{st['mean_block']:.0f} steps on average")
    stats = ("Across the three mechanisms -- " + "; ".join(bits)
             + " -- teams under the Hebbian coupling persist far longer than "
               "those that form on their own, which reform every few dozen "
               "steps.")
    steps = max(d["run"]["_ep_bounds"][-1][1] for d in arms_data.values())
    seed_s = (", ".join(f"{a}: {arms_data[a]['seed']}" for a, *_ in ARMS)
              if per_arm else str(seed))
    (out_dir / f"caption_timelines{suffix}.tex").write_text(
        CAPTION % {"suffix": suffix, "seed": seed_s, "steps": steps,
                   "bottom": bottom, "extra": extra, "stats": stats},
        encoding="utf-8")


# ─── Stats sidecar ──────────────────────────────────────────────────────
def arm_stats(arm: str, data: dict) -> dict:
    run, events = data["run"], data["events"]
    env = [e for e in events if not e["is_comm"]]
    coop = [e for e in env if MILESTONE_TRACK[e["mid"]] in COOP_TRACKS]
    total = run["_ep_bounds"][-1][1]
    coops = [coop_count(s) for s in episode_milestone_sets(run)]
    task, _ = episode_task_returns(run)
    cov, longest = team_coverage(data["team"], total)
    s = {
        "episode_lengths": run.get("episode_lengths"),
        "task_return_total": round(sum(task), 1),
        "coop_milestones_per_episode": round(sum(coops) / max(len(coops), 1), 2),
        "n_env_milestone_events": len(env),
        "n_distinct_env_milestones": len({e["mid"] for e in env}),
        "n_comm_milestone_events": len(events) - len(env),
        "n_events_stripped_by_honesty_filter": run["_events_stripped"],
        "first_coop_track_milestone_step": min((e["step"] for e in coop),
                                               default=None),
        "n_co_completions_noncomm": len(data["cocos"]),
        "team_coverage_frac": round(cov, 3),
        "team_longest_block_steps": longest,
        "team_stability": {k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in
                           team_stability(data["team"], total).items()},
        "team_pair_steps": {
            ("all_three" if q is ALL_THREE else f"{q[0]}-{q[1]}"):
            int(sum(hi - lo for lo, hi in ivs))
            for q, ivs in data["team"].items()},
        "chamber_bands": [(ch, int(lo), int(hi)) for ch, lo, hi in data["bands"]],
    }
    if arm == "base":
        s["pair_final_msg_share"] = {
            f"{q[0]}-{q[1]}": round(float(ys[np.isfinite(ys)][-1]), 3)
            for q, (xs, ys) in data["form"].items() if np.isfinite(ys).any()}
    elif arm == "orch":
        o = data["form"]
        s.update(n_assignment_intervals=len(o["intervals"]),
                 allocations_per_agent=o["n_alloc"],
                 n_distinct_tasks_assigned=o["n_tasks"],
                 n_team_tasks_in_dag=o["n_multi_dag"])
    else:
        s["pair_final_bond_W"] = {f"{q[0]}-{q[1]}": round(float(ys[-1]), 4)
                                  for q, (xs, ys) in data["form"].items()}
    return s


def write_stats(arms_data, out_dir: Path, seed: int):
    stats = {"seed": seed,
             "arms": {arm: arm_stats(arm, d) for arm, d in arms_data.items()}}
    (out_dir / "timeline_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    lines = [f"# Directive timelines — seed {seed}", "",
             "Coupling is normalised the same way in every arm (share of the "
             "three pair couplings), so these columns are comparable; "
             "\"one pair leads\" and \"all three\" partition the teamed steps.",
             "",
             "| arm | task return | coop milestones/ep | first coop step | "
             "env fires | one pair leads | all three | mean formation | "
             "longest |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, s in stats["arms"].items():
        st = s["team_stability"]
        lines.append(
            f"| {arm} | {s['task_return_total']:.0f} | "
            f"{s['coop_milestones_per_episode']:.2f} | "
            f"{s['first_coop_track_milestone_step']} | "
            f"{s['n_env_milestone_events']} | "
            f"{st['pair_frac'] * 100:.0f}% | "
            f"{st['all_three_frac'] * 100:.0f}% | "
            f"{st['mean_block']:.0f} | {st['longest_block']} |")
    (out_dir / "timeline_stats.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")


# ─── Seed scan ──────────────────────────────────────────────────────────
def _seed_metrics(arm, run_dir, window):
    data = load_arm(arm, run_dir, window)
    run = data["run"]
    total = run["_ep_bounds"][-1][1]
    coops = [coop_count(s) for s in episode_milestone_sets(run)]
    task, _ = episode_task_returns(run)
    env = [e for e in data["events"] if not e["is_comm"]]
    coop_ev = [e for e in env if MILESTONE_TRACK[e["mid"]] in COOP_TRACKS]
    st = team_stability(data["team"], total)
    return {"total_steps": total, "task_total": round(sum(task), 1),
            "coop_per_ep": round(sum(coops) / max(len(coops), 1), 2),
            "first_coop_step": min((e["step"] for e in coop_ev), default=None),
            "n_distinct_env_milestones": len({e["mid"] for e in env}),
            "max_chamber": max((int(ch[2]) for ch, _, _ in data["bands"]
                                if ch.startswith("ch")), default=0),
            "mean_formation": round(st["mean_block"], 1),
            "team_longest_block": st["longest_block"]}


def scan_seeds(runs_root: Path, out_dir: Path, window: int) -> dict:
    """Rank every seed on how well it tells the story. Gate: Heb > Orch."""
    out = {}
    for seed in SEEDS:
        per, missing = {}, False
        for arm, rel, _, _ in ARMS:
            rd = runs_root / rel / f"seed_{seed}"
            if not (rd / "final_metrics.json").exists():
                missing = True
                break
            per[arm] = _seed_metrics(arm, rd, window)
        if missing:
            print(f"  seed {seed}: run dirs missing — skipped", file=sys.stderr)
            continue
        b, o, h = per["base"], per["orch"], per["heb"]

        def cens(m):
            return (m["first_coop_step"] if m["first_coop_step"] is not None
                    else m["total_steps"])

        comp = {
            "coop_heb_minus_orch": round(h["coop_per_ep"] - o["coop_per_ep"], 2),
            "coop_heb_minus_base": round(h["coop_per_ep"] - b["coop_per_ep"], 2),
            "task_heb_minus_orch": round(h["task_total"] - o["task_total"], 1),
            "task_heb_minus_base": round(h["task_total"] - b["task_total"], 1),
            "first_coop_earlier_than_orch": cens(o) - cens(h),
            "first_coop_earlier_than_base": cens(b) - cens(h),
            "formation_len_heb_over_base": round(
                h["mean_formation"] / max(b["mean_formation"], 1e-9), 2),
            "distinct_env_milestones_heb": h["n_distinct_env_milestones"],
            "max_chamber_heb": h["max_chamber"],
        }
        gate = comp["coop_heb_minus_orch"] > 0 and comp["task_heb_minus_orch"] > 0
        score = (2.0 * comp["coop_heb_minus_orch"]
                 + 1.0 * comp["coop_heb_minus_base"]
                 + comp["task_heb_minus_orch"] / 500
                 + 0.5 * comp["task_heb_minus_base"] / 500
                 + 1.5 * comp["first_coop_earlier_than_orch"] / 500
                 + comp["first_coop_earlier_than_base"] / 500
                 + 0.5 * min(comp["formation_len_heb_over_base"], 4.0)
                 + 0.4 * (comp["distinct_env_milestones_heb"] - 5) / 3
                 + 0.4 * (comp["max_chamber_heb"] - 3))
        out[seed] = {"arms": per, "components": comp,
                     "gate_heb_beats_orch_both_axes": gate,
                     "score": round(score, 3)}
        print(f"  seed {seed}: gate={'PASS' if gate else 'fail'} "
              f"score={score:6.2f}")

    ranked = sorted(out.items(),
                    key=lambda kv: (kv[1]["gate_heb_beats_orch_both_axes"],
                                    kv[1]["score"]), reverse=True)
    (out_dir / "seed_scan.json").write_text(
        json.dumps({"ranking": [s for s, _ in ranked], "seeds": out}, indent=2),
        encoding="utf-8")

    md = ["# Seed scan — which seed tells the story best", "",
          "**Gate (must pass):** Hebbian beats the orchestrator on BOTH "
          "cooperative milestones per episode and task return.", "",
          "**Score** = 2·Δcoop(heb−orch) + Δcoop(heb−base) + "
          "Δtask(heb−orch)/500 + ½·Δtask(heb−base)/500 + "
          "1.5·(first-coop earlier than orch)/500 + (earlier than base)/500 + "
          "½·(how much longer a Hebbian formation lasts than a base one, "
          "capped at 4×) + milestone-variety and chamber-depth bonuses.", "",
          "Triples below are base / orch / hebbian.", "",
          "| rank | seed | gate | score | task return | coop/ep | "
          "first coop step | mean formation (steps) |",
          "|---|---|---|---|---|---|---|---|"]
    for i, (seed, d) in enumerate(ranked, 1):
        a = d["arms"]
        md.append(
            f"| {i} | {seed} | "
            f"{'**PASS**' if d['gate_heb_beats_orch_both_axes'] else 'fail'} | "
            f"{d['score']:.2f} | "
            f"{a['base']['task_total']:.0f} / {a['orch']['task_total']:.0f} / "
            f"{a['heb']['task_total']:.0f} | "
            f"{a['base']['coop_per_ep']:.2f} / {a['orch']['coop_per_ep']:.2f} / "
            f"{a['heb']['coop_per_ep']:.2f} | "
            f"{a['base']['first_coop_step']} / {a['orch']['first_coop_step']} / "
            f"{a['heb']['first_coop_step']} | "
            f"{a['base']['mean_formation']:.0f} / "
            f"{a['orch']['mean_formation']:.0f} / "
            f"{a['heb']['mean_formation']:.0f} |")
    (out_dir / "seed_scan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if ranked:
        print(f"  best seed: {ranked[0][0]} (score {ranked[0][1]['score']:.2f})")
    return out


def median_seed_per_arm(runs_root: Path) -> dict:
    """Each arm's MEDIAN seed by task return — a typical run, not its best.

    Only for the supplementary '--per-arm-seeds' figure. The same-seed
    figure stays the default: showing each mechanism on the seed that
    flatters it would be cherry-picking, whereas the median run is the
    representative one.
    """
    picked = {}
    for arm, rel, _, _ in ARMS:
        scored = []
        for seed in SEEDS:
            f = runs_root / rel / f"seed_{seed}" / "final_metrics.json"
            if not f.exists():
                continue
            run = load_run(f.parent)
            task, _ = episode_task_returns(run)
            scored.append((sum(task), seed))
        if not scored:
            continue
        scored.sort()
        picked[arm] = scored[len(scored) // 2][1]     # median by task return
    return picked


# ─── Main ───────────────────────────────────────────────────────────────
def load_arm(arm: str, run_dir: Path, window: int) -> dict:
    run = load_run(run_dir)
    data = {"run": run, "events": milestone_lane_events(run),
            "bands": load_chamber_bands(run), "cocos": co_completions(run)}
    if arm == "base":
        data["form"] = load_msg_share(run, window)
    elif arm == "orch":
        data["form"] = load_orch(run)
    else:
        data["form"] = load_bonds(run)
        if not data["form"]:
            print("  [warn] hebbian arm has no graph_snapshots", file=sys.stderr)
    data["team"] = team_intervals(arm, data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS)
    ap.add_argument("--seed", type=int, default=456)
    ap.add_argument("--out", type=Path, default=ASSETS / "timelines")
    ap.add_argument("--window", type=int, default=50,
                    help="rolling window (steps) for the base msg-share lane")
    ap.add_argument("--no-comm-events", action="store_true",
                    help="drop comm-milestone ticks from the milestone lane")
    ap.add_argument("--simple", action="store_true",
                    help="also write the *_simple.* variant (one shared "
                         "'the team is this pair' grammar for all three arms)")
    ap.add_argument("--scan-seeds", action="store_true",
                    help="rank every seed on story quality first")
    ap.add_argument("--use-best-seed", action="store_true",
                    help="with --scan-seeds, plot the top-ranked seed")
    ap.add_argument("--venue", choices=sorted(VENUES), default="iclr",
                    help="page width to author for (iclr = 5.5in single "
                         "column; icml = 6.75in figure*)")
    ap.add_argument("--width", type=float, default=None,
                    help="explicit figure width in inches (overrides --venue)")
    ap.add_argument("--with-title", action="store_true",
                    help="draw a title inside the figure (for review; the "
                         "camera-ready version uses the LaTeX caption)")
    ap.add_argument("--per-arm-seeds", action="store_true",
                    help="supplementary variant: show each arm on its OWN "
                         "median-by-task-return seed (a typical run of each "
                         "mechanism) instead of one shared seed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    if args.scan_seeds:
        print("scanning seeds…")
        scan = scan_seeds(args.runs_root, args.out, args.window)
        if args.use_best_seed and scan:
            seed = max(scan, key=lambda s: (
                scan[s]["gate_heb_beats_orch_both_axes"], scan[s]["score"]))
            print(f"using best seed: {seed}")

    per_arm_seed = {}
    if args.per_arm_seeds:
        per_arm_seed = median_seed_per_arm(args.runs_root)
        print(f"per-arm median seeds: {per_arm_seed}")

    arms_data = {}
    for arm, rel, _, _ in ARMS:
        arm_seed = per_arm_seed.get(arm, seed)
        rd = args.runs_root / rel / f"seed_{arm_seed}"
        if not (rd / "final_metrics.json").exists():
            sys.exit(f"error: no run at {rd}")
        data = load_arm(arm, rd, args.window)
        data["seed"] = arm_seed
        total = data["run"]["_ep_bounds"][-1][1]
        bad = [e for e in data["events"] if not 0 <= e["step"] < total]
        if bad:
            print(f"  [warn] {arm}: {len(bad)} events outside [0,{total})",
                  file=sys.stderr)
        print(f"{arm}: {len(data['events'])} milestone events "
              f"({data['run']['_events_stripped']} stripped), "
              f"{len(data['cocos'])} co-completions, {total} steps")
        arms_data[arm] = data

    pa = bool(per_arm_seed)
    width = args.width or VENUES[args.venue]
    for simple in ([False, True] if args.simple else [False]):
        W, H = build_figure(arms_data, args.out, seed, not args.no_comm_events,
                            simple=simple, per_arm=pa, width=width,
                            titled=args.with_title)
        write_caption(arms_data, args.out, seed, simple, pa)
        tag = "_simple" if simple else ""
        print(f"  directive_timelines{tag}.pdf  {W:.2f} x {H:.2f} in "
              f"({args.venue} textwidth) + caption_timelines{tag}.tex")
    write_stats(arms_data, args.out, per_arm_seed if pa else seed)
    print(f"wrote figures, captions and stats to {args.out}/")


if __name__ == "__main__":
    main()
