#!/usr/bin/env python3
r"""make_counterfactual_n6.py - six-agent runs in the counterfactual_full grammar.

The paper's counterfactual_full_orch / counterfactual_full_plast figures are
``counterfactual_{a,b}`` from make_counterfactual_story.py: the WIDE layout of
make_final_figures - a mutual-message strip, one mechanism lane with chamber
labels and a cumulative step axis, shaded focus windows with corner
connectors, and three bordered storyboard panels of verbatim chat lines
interleaved with frame pairs. Every loader in that family is written for
three agents (PAIRS = 3 pairs, range(3) rows, a (T,3,3) bond array), so this
script keeps the renderer (mff.build_figure / mff.panel) and swaps in
six-agent versions of the pieces that assume N = 3. Nothing in the working
scripts is edited; the replacements are monkeypatched onto ``mff`` before
build_figure runs, exactly as make_counterfactual_story.py does for its lane.

THREE FIGURES

  counterfactual_full_orch_n6
      Central orchestrator, 6 agents, seed 42 (runs/agent_scaling_orch).
      Lane: one tenure row per agent on the team-size ramp. Panels:
        1  ep1 Ch4  six single-agent kill tasks; a0 asks a5 to confirm the
                    target so it can join, a5 kills alone at t=682
        2  ep2 Ch4  a0 calls focus fire from t=605 and kills at t=626 while
                    a5 is still "moving forward to join" in another room
        3  ep3 Ch2  the run's only joint milestone (anvil A, t=391, credited
                    to a1 + a2) - the DAG had a0+a1 and a3+a4 on that anvil
                    and a2 unassigned since t=316; a1 and a2 never exchanged
                    a message. Four freed_success rows follow, for a0 a1 a3
                    a4.

  counterfactual_full_transplant_s123 / _s42
      RQ3 pair transplant (runs/pair_bonding/expB_merged_transplant): six
      agents seeded with three bonded pairs a0-a1, a2-a3, a4-a5 at W = 0.27
      and started in Ch3. Lane: the symmetrised bond of the three transplanted
      pairs in colour, the strongest cross pairs in grey. The two seeds are
      the two orders the same thing happens in:
        s123  the pairs drift apart in ep1 (0.27 -> 0.19 / 0.12 / 0.08,
              a0-a2 up to 0.27) and get back together in ep2 (-> 0.29 /
              0.27 / 0.29), then drift again in ep3.
        s42   the pairs hold through ep1 (up to 0.33), dissolve in ep2 into
              a0-a4 / a3-a5 / a1-a2 (0.31 / 0.29 / 0.27, the transplanted
              pairs at 0.04 / 0.09 / 0.11), and re-form late in ep3 (0.27 /
              0.28 / 0.28).
      Panels follow that arc per seed, with the chamber events that go with
      it: the Ch3 switch/door pairs, which are the joint milestones of these
      runs, and the Ch4 kills.

BOND SOURCE: the recorded 50-step graph_snapshots (6 x 6, interpolated),
i.e. lane "bonds_snap". These runs use the reward-modulated rule, so the
three-factor replay does not apply.

WHAT "JOINT" MEANS HERE: two or more contributors on the same lua_step in a
cooperative track (mtt.events' rule), applied over all 15 pairs. A switch
press and the door it opens fire together and are credited to two agents,
which is why the Ch3 events are the joint marks of the transplant runs.

Usage:  python analysis/make_counterfactual_n6.py [orch|s123|s42]
Out:    paper_assets/timelines/counterfactual/counterfactual_full_{orch_n6,
        transplant_s123,transplant_s42}.{pdf,png,svg}
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from paths import ASSETS, group  # noqa: E402  (also puts siblings on sys.path)

import make_final_figures as mff  # noqa: E402   (the WIDE renderer)
from make_directive_timelines import load_run  # noqa: E402
from make_results import MILESTONE_TRACK  # noqa: E402
from make_team_tenure import COOP, aid  # noqa: E402

N = 6
OUT = ASSETS / "timelines" / "counterfactual"
ORCH_RUN = group("agent_scaling_orch") / "scale_gemma_orch_villager_n6" / "seed_42"
TRANSPLANT = group("pair_bonding") / "expB_merged_transplant"
TRANSPLANTED = [(0, 1), (2, 3), (4, 5)]

# ── six-agent design tokens, spliced into the family ────────────────────
AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73",
           3: "#CC79A7", 4: "#E69F00", 5: "#56B4E9"}
CROSS = "#8b95a1"
PAIR_C = defaultdict(lambda: CROSS)
PAIR_C.update({(0, 1): "#5B4EA4", (2, 3): "#2E8B6E", (4, 5): "#B2652C",
               (0, 5): "#B2652C", (0, 2): "#7a5c3e", (0, 4): "#c0862f",
               (3, 5): "#a0567a", (1, 2): "#4f7f7a"})
TEAM_RAMP = ["#cfdbe9", "#a9bfdc", "#7b9dc3", "#4f7bb0", "#2a63a0", "#0F4D92"]
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]

mff.AGENT_C = AGENT_C
mff.PAIR_C = PAIR_C
# panel titles say what happened, not just where: ~60 characters need the
# family's 11.5 pt title one and a half notches smaller to stay on one line
mff.FS["panel_title"] = 9.2

STRIP_PAIRS: list = []            # set per figure before build_figure runs


def pair(i, j):
    return (i, j) if i < j else (j, i)


# ── data: N-agnostic replacements for the family's loaders ──────────────
def chamber_spans(arm, run, min_len=12):
    """Majority chamber across agents per step, from step_log.csv.

    Same rule as mff.chamber_spans_from_steps, minus the N = 3 array it
    reads through, plus the same-label merge make_counterfactual_story adds
    (a visit shorter than min_len is absorbed into the preceding span, which
    otherwise prints "Ch2   Ch2"). Never merges across an episode boundary.
    """
    ch_int = {"ch1": 1, "ch2": 2, "ch3": 3, "ch4": 4, "ch5": 5}
    bounds = run["_ep_bounds"]
    T = bounds[-1][1]
    maj = np.zeros(T, dtype=int)
    for e, (s0, s1) in enumerate(bounds):
        per = defaultdict(list)
        p = run["_run_dir"] / "episodes" / f"ep_{e + 1:04d}" / "step_log.csv"
        with p.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                c = ch_int.get(r.get("chamber") or "", 0)
                if c:
                    per[int(r["step"])].append(c)
        for t in range(s1 - s0):
            k = s0 + t
            maj[k] = (int(np.median(per[t])) if per.get(t)
                      else (maj[k - 1] if k > s0 else 0))
    starts = {b for b, _ in bounds}
    spans, t0 = [], 0
    for t in range(1, T + 1):
        if t == T or maj[t] != maj[t0] or t in starts:
            if maj[t0] > 0 and t - t0 >= min_len:
                spans.append((f"Ch{maj[t0]}", t0, t))
            elif spans and maj[t0] > 0 and t0 not in starts:
                spans[-1] = (spans[-1][0], spans[-1][1], t)
            t0 = t
    out = []
    for lab, lo, hi in spans:
        if out and out[-1][0] == lab and lo <= out[-1][2] and lo not in starts:
            out[-1] = (lab, out[-1][1], hi)
        else:
            out.append((lab, lo, hi))
    return out


def load_bond_series(arm):
    """(T, N, N) directed bonds, the 50-step snapshots interpolated."""
    if arm["lane"] != "bonds_snap":
        return None
    fm = json.loads((arm["run"] / "final_metrics.json").read_text(encoding="utf-8"))
    snaps = sorted((s["step"], np.array(s["W"], dtype=float))
                   for s in fm.get("graph_snapshots", []) if s.get("W"))
    T = max(snaps[-1][0] + 1, sum(int(x) for x in fm["episode_lengths"]))
    W = np.zeros((T, N, N))
    ss = [s for s, _ in snaps]
    for i in range(N):
        for j in range(N):
            W[:, i, j] = np.interp(np.arange(T), ss, [m[i, j] for _, m in snaps])
    return W


# Milestones one event can credit to two agents: an anvil break, a switch
# press with the door it opens, a kill or boss hit. Per-agent milestones that
# happen to fire on the same lua_step - m14_sword_equipped fires six times at
# ep3 t=391 of the N=6 run, one per agent, and m24_enter_ch5 fires for four
# agents at once in the transplant runs - are NOT joint and are excluded here,
# where mtt.events (written for N=3, where the overlap is rarer) groups by
# lua_step alone.
SHAREABLE = {"m8_anvil_A1", "m9_anvil_B1", "m17_switch_pressed",
             "m18_door_opened", "m21_first_mob_kill", "m22_all_mobs_killed",
             "m25_first_boss_dmg", "m26_boss_half_hp", "m27_boss_defeated"}


def events(fm):
    """joints [(step, pair)] over all 15 pairs, kills [(step, killer)]."""
    by, step, kills = defaultdict(set), {}, []
    for e in fm["milestone_events"]:
        mid = e["milestone_id"]
        if MILESTONE_TRACK.get(mid) not in COOP or mid not in SHAREABLE:
            continue
        by[e["lua_step"]].add(aid(e["contributor"]))
        step[e["lua_step"]] = min(step.get(e["lua_step"], 10 ** 9), e["step"])
        if mid == "m21_first_mob_kill":
            kills.append((e["step"], aid(e["contributor"])))
    joints = []
    for k, ags in by.items():
        ags = sorted(ags)
        for x in range(len(ags)):
            for y in range(x + 1, len(ags)):
                joints.append((step[k], (ags[x], ags[y])))
    return sorted(joints), sorted(kills)


def assignment_grid(run):
    """(T, N) task id or None on the cumulative clock, from the ledger."""
    T = run["_ep_bounds"][-1][1]
    offs = {e + 1: b[0] for e, b in enumerate(run["_ep_bounds"])}
    ev = defaultdict(list)
    for line in (run["_run_dir"] / "orchestrator" / "assignments.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ev[offs[r["episode"]] + r["t"]].append((aid(r["agent"]), r["task_id"], r["reason"]))
    task = np.empty((T, N), dtype=object)
    live = {}
    for t in range(T):
        for a, tid, rs in ev.get(t, []):        # file order: frees, then allocates
            live[a] = tid if rs.startswith("allocate") else None
        for a in range(N):
            task[t, a] = live.get(a)
    return task


# ── the three family pieces that hard-code N = 3 ────────────────────────
def _chamber_rules(ax, spans):
    """Dotted rule at each labelled chamber start (story-script convention)."""
    if not spans:
        return
    total = spans[-1][2] - spans[0][1]
    last = -1e9
    for _lab, lo, hi in spans:
        if hi - lo <= 40 or lo - last < total * 0.03 or lo <= 0:
            continue
        ax.axvline(lo, color="#b3bcc7", lw=0.7, ls=(0, (1.2, 2.4)), zorder=0)
        last = lo


def mutual_strip(ax, messages, spans, main=False):
    """One row per pair in STRIP_PAIRS; a tick where both directions fire."""
    _chamber_rules(ax, spans)
    per_step = {}
    for t, snd, rcv in messages:
        per_step.setdefault(t, set()).add((snd, rcv))
    pairs = list(STRIP_PAIRS)
    y_of = {q: len(pairs) - 1 - i for i, q in enumerate(pairs)}
    for t, dirs in per_step.items():
        for i, j in pairs:
            if (i, j) in dirs and (j, i) in dirs:
                y = y_of[(i, j)]
                ax.vlines(t, y + 0.14, y + 0.86, color=PAIR_C[(i, j)],
                          lw=0.6 if main else 0.5, alpha=0.85, zorder=2)
    ax.set_yticks([y_of[q] + 0.5 for q in pairs])
    ax.set_yticklabels([f"a{i}↔a{j}" for i, j in pairs], fontsize=mff.FS["raster"])
    for tick, q in zip(ax.get_yticklabels(), pairs):
        tick.set_color(PAIR_C[q])
    ax.set_ylim(-0.4 if main else 0, len(pairs) + (0.6 if main else 0))
    ax.set_ylabel("mutual\nmessages", fontsize=mff.FS["axis"], color=mff.INK)
    ax.tick_params(length=0)
    return y_of


def lane_bonds(ax, arm, Wr, spans, xlim):
    """Transplanted pairs in colour, the named cross pairs in grey."""
    _chamber_rules(ax, spans)
    steps = np.arange(len(Wr))

    def wbar(q):
        return (Wr[:, q[0], q[1]] + Wr[:, q[1], q[0]]) / 2

    handles = []
    for q in arm.get("cross", []):
        h, = ax.plot(steps, wbar(q), color=CROSS, lw=1.1, ls=(0, (3, 2)),
                     alpha=0.9, zorder=2,
                     label="$\\bar{W}$(a%d–a%d)" % q)
        handles.append(h)
    for q in arm["pairs"]:
        h, = ax.plot(steps, wbar(q), color=PAIR_C[q], lw=2.0,
                     label="$\\bar{W}$(a%d–a%d)" % q,
                     solid_capstyle="round", zorder=3)
        handles.insert(len(handles) - len(arm.get("cross", [])), h)
    for t, q in arm["joints"]:
        t = min(t, len(Wr) - 1)
        ax.scatter(t, wbar(q)[t], marker="o", s=58, facecolor="white",
                   edgecolor=PAIR_C[q] if q in arm["pairs"] else CROSS,
                   linewidths=1.8, zorder=6)
    for t, killer in arm["kills"]:
        t = min(t, len(Wr) - 1)
        ys = [wbar(q)[t] for q in arm["pairs"] + list(arm.get("cross", []))
              if killer in q]
        ax.scatter(t, max(ys) if ys else 0.1, marker="*", s=160,
                   facecolor=AGENT_C[killer], edgecolor="white",
                   linewidths=0.6, zorder=6)
    ymax = float(np.nanmax(Wr))
    mff.chamber_labels(ax, spans, 0.005, xlim)
    ax.grid(axis="y", color="#eef1f4", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("bond strength", fontsize=mff.FS["axis"], color=mff.INK)
    ax.tick_params(labelsize=mff.FS["tick"])
    leg1 = ax.legend(handles=handles, fontsize=8.5, ncol=len(handles),
                     frameon=False, loc="upper left", borderaxespad=0.2,
                     columnspacing=1.2, handlelength=1.8)
    ax.add_artist(leg1)
    ax.legend(handles=mff._event_handles(), fontsize=8, ncol=2, frameon=False,
              loc="upper right", borderaxespad=0.2, handletextpad=0.4)
    ax.set_ylim(0.0, ymax * 1.30)


def lane_orch(ax, arm, run, spans, xlim):
    """One tenure row per agent; a bar is the task held, shaded by how many
    agents the orchestrator put on it. Events on the credited agent's row."""
    _chamber_rules(ax, spans)
    task = assignment_grid(run)
    T = task.shape[0]
    rows = {a: N - 1 - a for a in range(N)}
    bh, gap = 0.30, 2.0
    for a in range(N):
        keys = []
        for t in range(T):
            tid = task[t, a]
            keys.append(None if tid is None else
                        (tid, sum(1 for b in range(N) if b != a and task[t, b] == tid)))
        t0 = 0
        for t in range(1, T + 1):
            if t == T or keys[t] != keys[t0]:
                if keys[t0] is not None:
                    g = gap if t - t0 > 4 * gap else 0
                    ax.broken_barh([(t0 + g, t - t0 - 2 * g)],
                                   (rows[a] - bh, 2 * bh),
                                   facecolors=TEAM_RAMP[keys[t0][1]],
                                   linewidth=0, zorder=3)
                t0 = t
    for t, q in arm["joints"]:
        for a in q:
            ax.scatter(t, rows[a], marker="o", s=58, facecolor="white",
                       edgecolor=PAIR_C[q], linewidths=1.5, zorder=8,
                       path_effects=HALO)
    for t, killer in arm["kills"]:
        ax.scatter(t, rows[killer], marker="*", s=160,
                   facecolor=AGENT_C[killer], edgecolor="white",
                   linewidths=0.6, zorder=8, path_effects=HALO)
    mff.chamber_labels(ax, spans, -0.98, xlim)
    ax.set_yticks([rows[a] for a in range(N)])
    ax.set_yticklabels([f"a{a}" for a in range(N)], fontsize=mff.FS["raster"] + 1)
    for tick, a in zip(ax.get_yticklabels(), range(N)):
        tick.set_color(AGENT_C[a])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-1.08, N - 1 + 1.55)
    ax.set_ylabel("orchestrator\nassignments", fontsize=mff.FS["axis"],
                  color=mff.INK, labelpad=4)
    ax.spines["left"].set_visible(False)
    team_key = ax.legend(
        handles=[Patch(facecolor=TEAM_RAMP[k], label=lab) for k, lab in
                 ((0, "alone"), (1, "paired"), (2, "three"), (5, "all six"))],
        fontsize=8.5, ncol=4, loc="upper left", frameon=False,
        borderaxespad=0.3, handlelength=1.2, handleheight=0.8,
        handletextpad=0.5, columnspacing=1.4)
    ax.add_artist(team_key)
    ax.legend(handles=mff._event_handles(), fontsize=8, ncol=2, frameon=False,
              loc="upper right", borderaxespad=0.2, handletextpad=0.4)


mff.chamber_spans_from_steps = chamber_spans
mff.load_bond_series = load_bond_series
mff.mutual_strip = mutual_strip
mff.lane_bonds = lane_bonds
mff.lane_coassign = lane_orch
mff.shade_chambers = _chamber_rules


# ── the three stories ───────────────────────────────────────────────────
def _ex(key, ep, off, lo, hi, agents, title, frames):
    return dict(key=key, x=(off[ep] + lo, off[ep] + hi), ep=ep, agents=agents,
                dpairs=[], title=title, frames=frames, outcome="")


def _offsets(run_dir):
    run = load_run(run_dir)
    return {e + 1: b[0] for e, b in enumerate(run["_ep_bounds"])}


def arm_orch():
    off = _offsets(ORCH_RUN)
    return dict(
        name="counterfactual_full_orch_n6", ph=0.50,
        run=ORCH_RUN, exp="scale_gemma_orch_villager_n6", seed=42,
        lane="coassign", strip=[(0, 1), (0, 5), (2, 3)],
        examples=[
            _ex("solo_kill", 1, off, 655, 686, (5, 0),
                "Ch4 split into six solo tasks: a5 kills alone, a0 never joins",
                [("M", 681, 5, 0, "Target locked on zombie, ready to attack."),
                 ("F", (5, 680, "a5: zombie centred"),
                  (0, 682, "a0: the red door, agent1 beside it")),
                 ("M", 682, 0, 5, "Agent 5, confirm zombie location so I can join the attack."),
                 ("M", 682, 5, 0, "I see the zombie centered, starting attack now."),
                 ("F", (5, 684, "a5: the zombie flashes red under its hits, a0 a2 named"),
                  (0, 688, "a0: a2 a3 a5 in the lit room, no zombie")),
                 ("M", 683, 0, 5, "Agent 5, if you have the zombie centered, let me know when you start attacking so I can move closer to assist.")]),
            _ex("focus_fire", 2, off, 605, 630, (0, 5),
                "a0 calls focus fire and kills alone; a5 stays \"moving forward\"",
                [("M", 605, 0, 5, "Zombie centered. Ready to attack. Let's focus fire."),
                 ("M", 618, 5, 0, "Agent 0, I'm moving forward to join the attack; let me know when you have a target centered."),
                 ("F", (0, 623, "a0: zombie in reach, a5's nametag in view"),
                  (5, 618, "a5: a0 a2 a3 far down the lit corridor")),
                 ("M", 619, 0, 5, "I have a centered zombie target, starting attack now. Focus fire!"),
                 ("M", 624, 5, 0, "Moving forward to join focus fire. Let me know if you need backup on a specific zombie."),
                 ("F", (0, 626, "a0: the kill lands (t=626), a1 a2 a4 named"),
                  (5, 626, "a5: a wall")),
                 ("M", 626, 0, 5, "I'm attacking the central zombie boss. Let me know if you need backup on a specific target.")]),
            _ex("anvil", 3, off, 375, 395, (1, 2),
                "Anvil A breaks under a1 + a2, a pair the DAG never assigned",
                [("M", 386, 2, 5, "Anvil A is visible and centered in view. I am moving forward to align for the double dig."),
                 ("F", (1, 384, "a1: a chicken first, anvil A behind it"),
                  (2, 388, "a2: anvil A filling the view, a4 beside")),
                 ("M", 390, 1, 0, "Anvil A is visible now. I'm moving in to coordinate the strike with you."),
                 ("M", 391, 1, 0, "Anvil visible and centered. Ready to coordinate a hit on Anvil A with agent_0."),
                 ("F", (1, 393, "a1: the slab where anvil A stood (broken t=391)"),
                  (2, 393, "a2: diamond sword equipped, a0 a4 a5 around")),
                 ("M", 391, 2, 5, "Anvil A is centered! I'm digging now; please coordinate with me!")]),
        ],
        figures=[dict(name="counterfactual_full_orch_n6", xlim=None, keys=None, delib=False)])


def arm_s123():
    run_dir = TRANSPLANT / "seed_123"
    off = _offsets(run_dir)
    return dict(
        name="counterfactual_full_transplant_s123", ph=0.50,
        run=run_dir, exp="expB_merged_transplant", seed=123,
        lane="bonds_snap", pairs=TRANSPLANTED, cross=[(0, 2)],
        strip=[(0, 1), (2, 3), (4, 5), (0, 2)],
        examples=[
            _ex("drift", 1, off, 880, 905, (0, 2),
                "ep1: a0 drifts to a2 while its transplanted bond to a1 decays",
                [("M", 880, 2, 0, "Keep sweeping the flanks, I'm moving forward to advance the search."),
                 ("F", (0, 890, "a0: a2 a3 a4 and the zombies ahead"),
                  (2, 890, "a2: a0 and a1 in view, same room")),
                 ("M", 885, 0, 2, "I'm turning right to sweep the right flank; let me know if you spot Door 4 from your angle."),
                 ("M", 890, 0, 2, "I'm turning left to face the mobs; keep scanning the flanks for the door."),
                 ("F", (0, 898, "a0: zombie point-blank"),
                  (2, 902, "a2: a0 a1 a3 still in view")),
                 ("M", 905, 0, 2, "See red door on right, turning to check it. Let me know if you see any door changes.")]),
            _ex("switch", 2, off, 99, 108, (5, 4),
                "ep2: a5 presses the switch for a4 and the pair's bond recovers",
                [("M", 102, 5, 4, "Switch centered, digging now."),
                 ("F", (5, 102, "a5: the blue switch cube"),
                  (4, 104, "a4: a0 a1 a2 across the cells")),
                 ("M", 104, 4, 5, "Can you confirm the switch was pressed? I'm turning to check the south wall."),
                 ("M", 105, 5, 4, "Agent 4, are you seeing the switch? I'm moving forward to get closer to the south wall."),
                 ("F", (5, 104, "a5 pressing, hearts up"),
                  (4, 107, "a4: the pink door and the blue cube")),
                 ("M", 106, 4, 5, "Agent 5, I see the switch area. Let me align to help monitor the door status.")]),
            _ex("retreat", 2, off, 335, 351, (3, 2),
                "ep2: a3 and a2 coordinate a retreat in Ch4, bond back at 0.27",
                [("M", 335, 3, 2, "Keep scanning S wall for cube!"),
                 ("F", (3, 340, "a3: a0 a1 and three zombies at the door"),
                  (2, 337, "a2: a3 a4 and the zombies, red door")),
                 ("M", 345, 3, 2, "Need to pull back from combat zone immediately."),
                 ("M", 348, 3, 2, "Retreating due to low health; focusing on clearing mobs."),
                 ("F", (3, 348, "a3 backs off: a4 a5 a2 and the red door"),
                  (2, 338, "a2: zombies point-blank")),
                 ("M", 349, 2, 3, "I'm moving FWD, scanning the corridor for the zombies or the door.")]),
        ],
        figures=[dict(name="counterfactual_full_transplant_s123", xlim=None, keys=None, delib=False)])


def arm_s42():
    run_dir = TRANSPLANT / "seed_42"
    off = _offsets(run_dir)
    return dict(
        name="counterfactual_full_transplant_s42", ph=0.50,
        run=run_dir, exp="expB_merged_transplant", seed=42,
        lane="bonds_snap", pairs=TRANSPLANTED, cross=[(0, 4), (3, 5), (0, 5)],
        strip=[(0, 1), (2, 3), (4, 5), (0, 4), (0, 5)],
        examples=[
            _ex("door", 1, off, 109, 125, (4, 5),
                "ep1: a4 opens a5's door, the transplanted pair working as seeded",
                [("M", 109, 4, 5, "Switch is on south wall. Waiting for signal."),
                 ("M", 111, 5, 4, "I see agent4 and agent3 nearby. Let me know if you spot the blue switch on the south wall."),
                 ("F", (4, 116, "a4: the blue switch cube"),
                  (5, 114, "a5: a4 and a0 in view")),
                 ("M", 116, 5, 4, "I see the blue cube slightly to the right; I'm turning to face it now."),
                 ("M", 118, 4, 5, "Still sealed in cell, need switch press."),
                 ("F", (4, 122, "a4 presses (t=122)"),
                  (5, 125, "a5: its door open, corridor lit"))]),
            _ex("swap", 2, off, 265, 284, (5, 0),
                "ep2: a5 presses for a0 instead; a0–a5 climbs to 0.33 as a4–a5 falls",
                [("M", 266, 5, 0, "I'm moving forward to help scan the south wall area too."),
                 ("F", (5, 272, "a5: the red (locked) door"),
                  (0, 266, "a0: a1 a2 a5 across the cells")),
                 ("M", 270, 5, 0, "Keep scanning the south wall for the blue cube!"),
                 ("M", 277, 5, 0, "Cube is centered. Digging now."),
                 ("F", (5, 278, "a5 presses the cube (t=277); a0's door opens"),
                  (0, 282, "a0: through its open door")),
                 ("M", 279, 5, 0, "Cube centered, digging now. Watch for door change.")]),
            _ex("combat", 3, off, 386, 393, (3, 2),
                "ep3: a3 kills with a2 moving in, the original pair re-formed",
                [("M", 388, 3, 2, "I see a zombie centered, starting combat now. Focus on the door!"),
                 ("F", (3, 389, "a3: zombie centred, a4 a1 a5 beside"),
                  (2, 386, "a2: on the heels of two teammates")),
                 ("M", 389, 2, 3, "Moving to support combat near the door."),
                 ("M", 390, 2, 3, "I'm moving towards the zombie Agent 3 saw. Focus on the door!"),
                 ("F", (3, 392, "a3: the hit flashes red, a2 now named beside"),
                  (2, 390, "a2: a5 and the flashing zombie")),
                 ("M", 393, 3, 2, "I'm attacking the centered zombie. Let's focus on taking it down quickly to open the door.")]),
        ],
        figures=[dict(name="counterfactual_full_transplant_s42", xlim=None, keys=None, delib=False)])


def arm_plast_n6():
    """Relational plasticity at six agents: the three-factor scaling run
    behind counterfactual_compact_n6_a, over all three episodes.

    No anvil or switch fires in this run, so the vignettes are its kills
    and one hand-over: (1) ep1 a1 kills with a5 3.5 blocks away while
    narrating every swing to a3, twelve blocks off at a door; (2) ep2 the
    three-agent fight of the compact figure (a2 388/391, a0 394, a5 kills
    396; a2 reports to a4); (3) ep3 a0 swings at 340-341, a1 closes to 3.8
    blocks saying it will take the zombie a0 is fighting, and swings at
    349-352 while a0 pushes on to the red door. No m21 fires in ep3.
    Lane: the pairs that lead at some point (a0-a5, a2-a4, a1-a3 in ep2;
    a0-a1 in ep3) in colour, a4-a5 (ep1 and ep3 leader) grey.
    """
    PAIR_C.update({(2, 4): "#B8860B", (1, 3): "#a0567a"})
    run_dir = group("agent_scaling_3f") / "scale_gemma_hebbian_n6" / "seed_456"
    off = _offsets(run_dir)
    return dict(
        name="counterfactual_full_plast_n6", ph=0.50,
        run=run_dir, exp="scale_gemma_hebbian_n6", seed=456,
        lane="bonds_snap", pairs=[(0, 5), (2, 4), (1, 3), (0, 1)], cross=[(4, 5)],
        strip=[(0, 5), (2, 4), (1, 3), (0, 1), (4, 5)],
        examples=[
            _ex("kill1", 1, off, 384, 393, (1, 3),
                "ep1: a1 kills with a5 beside it, narrating each swing to a3, a door away",
                [("F", (1, 387, "a1: the whole team in the room"),
                  (3, 384, "a3: its last look at the team")),
                 ("M", 389, 1, 3, "Zombie centered, starting attack now."),
                 ("M", 389, 3, 1, "Still moving forward, scanning corridor for changes."),
                 ("M", 391, 1, 3, "Zombie centered, attacking now."),
                 ("F", (1, 391, "a1: the kill (t=391), a0 a3 a4 named"),
                  (3, 391, "a3: the red door, twelve blocks away")),
                 ("M", 391, 3, 1, "Still moving forward, keeping eyes on the walls for texture changes.")]),
            _ex("kill2", 2, off, 386, 398, (5, 0),
                "ep2: a2, a0 and a5 swing at the same zombies in turn; a5 kills",
                [("M", 388, 2, 4, "Zombie is centered, I'm attacking it now; keep an eye on the door state."),
                 ("F", (5, 394, "a5: zombies at arm's length, a1 a2 a4 named"),
                  (0, 394, "a0 swings: zombie ahead")),
                 ("M", 394, 0, 5, "Zombies are centered now. I'm engaging the closest one to start the kill sequence."),
                 ("M", 396, 5, 0, "Agent0, I see a zombie centered now. I'm engaging it to help clear the mobs for the door."),
                 ("F", (5, 398, "a5: its hits land, a2 beside"),
                  (0, 396, "a0: a1 a4 and the zombies")),
                 ("M", 398, 0, 5, "I see the zombie, I'll join the attack sequence on the right.")]),
            _ex("handover3", 3, off, 339, 353, (0, 1),
                "ep3: a1 takes over the zombie a0 was hitting while a0 pushes to the door",
                [("M", 341, 0, 1, "I'm attacking the zombie directly in front of me, expect HP drop soon."),
                 ("F", (0, 340, "a0 swings: zombie right, the team around"),
                  (1, 344, "a1 arrives: a2 a0 a3 a4 ahead")),
                 ("M", 343, 1, 0, "Moving forward to engage the zombies. I'll focus on one Agent 0 is fighting."),
                 ("M", 350, 1, 0, "I'm on the center zombie now. I'll focus fire to help clear this quickly."),
                 ("F", (0, 350, "a0: the red door"),
                  (1, 352, "a1: its hit lands")),
                 ("M", 353, 0, 1, "Still blocked here, check if you see the door from your side.")]),
        ],
        figures=[dict(name="counterfactual_full_plast_n6", xlim=None, keys=None, delib=False)])


ARMS = {"orch": arm_orch, "plast": arm_plast_n6, "s123": arm_s123, "s42": arm_s42}


def main(which: str | None = None) -> int:
    global STRIP_PAIRS
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, make in ARMS.items():
        if which and which != tag:
            continue
        arm = make()
        print(f"== {arm['name']} ==")
        for ex in arm["examples"]:                 # column purity, as in mff.main
            aL, aR = ex["agents"]
            for row in ex["frames"]:
                if row[0] != "F":
                    continue
                for slot, want in ((row[1], aL), (row[2], aR)):
                    if slot is not None and slot[0] != want:
                        raise AssertionError(f"{arm['name']}/{ex['key']}: frame "
                                             f"a{slot[0]}@{slot[1]} sits in the a{want} column")
        run = load_run(arm["run"])
        arm["joints"], arm["kills"] = events(run)
        messages = mff.load_messages(run)
        spans = mff.chamber_spans_from_steps(arm, run)
        Wr = mff.load_bond_series(arm)
        wanted = [(s[0], ex["ep"], s[1]) for ex in arm["examples"]
                  for r in ex["frames"] if r[0] == "F" for s in (r[1], r[2]) if s]
        frames = mff.grab_frames(arm, wanted)
        missing = len(set(wanted)) - len(frames)
        print(f"  frames {len(frames)}/{len(set(wanted))}"
              + (f"  [{missing} missing - pull this seed's gifs/ to fill]" if missing else "")
              + f", joints {len(arm['joints'])}, kills {len(arm['kills'])}, "
              f"chamber spans {len(spans)}")
        STRIP_PAIRS = list(arm["strip"])
        with plt.rc_context(mff.RC):
            for spec in arm["figures"]:
                mff.build_figure(arm["name"], arm, spec, run, Wr, messages, spans, frames, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
