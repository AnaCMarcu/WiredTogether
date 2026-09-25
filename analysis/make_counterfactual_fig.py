#!/usr/bin/env python3
r"""make_counterfactual_fig.py - same map, same episode: plan fails, bonds work.

Two columns over ONE (seed, episode, cooperative milestone) cell picked by
make_counterfactual_scan.py. Left: the central-orchestrator team, which was
assigned to that milestone and did not complete it. Right: the Hebbian team
on the identical map, which completed it with no instruction. Three rows:

  mechanism   orchestrator: per-agent task Gantt inside the window, tasks
              that target the milestone in colour (x = slot timed out);
              Hebbian: the three pair bonds W from the RECORDED 50-step
              graph snapshots (the per-step three-factor replay is not used:
              it omits the death-LTD term and runs 0.01-0.04 above the
              recorded W), with the fire marked on the contributor's pair.
  frames      four first-person views per column, captioned.
  chat        the exchange among the agents that matter, verbatim.

Curated shots and lines for the cells in CASES; any other cell falls back to
automatic picks (contributor around the fire; assigned agents mid-slot;
keyword-salient messages), which are serviceable but not paper-ready.

--appendix draws the full comparison instead: one row per HEB_ONLY and
ORCH_ONLY cell from the scan's cells.csv, mechanism lanes only over the
whole episode, plus a LaTeX table of the cells. That is the honest version:
it shows the cells that go the other way too.

Usage:
  python analysis/make_counterfactual_fig.py --heb heb2 --seed 42 --ep 1
  python analysis/make_counterfactual_fig.py --heb heb2 --seed 456 --ep 1
  python analysis/make_counterfactual_fig.py --heb heb2 --appendix
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from paths import ASSETS  # noqa: E402  (also puts siblings on sys.path)

import make_directive_timelines as mdt  # noqa: E402
from make_counterfactual_scan import (  # noqa: E402
    HEB_ARMS, ORCH, fires_by_ep, orch_plan)
from make_story_timelines_multi import grab_frames  # noqa: E402

AGENT_C = mdt.AGENT_C
PAIR_C = mdt.PAIR_C
INK, MUTED, RULE, GRID = "#31373f", "#98a1ad", "#c3c9d0", "#eceff3"
TARGET_C, OTHER_C = "#d95f0e", "#d5dae1"
CHAMBER_FILL = mdt.CHAMBER_FILL
PAIRS = mdt.PAIRS


def W(t, n=58):
    return "\n".join(textwrap.wrap(t, n))


# ─── curated cells ──────────────────────────────────────────────────────
# frames: (agent, within-episode step, caption); chat: (step, agent, text)
CASES = {
    ("heb2", 42, 1, "m21_first_mob_kill"): dict(
        window=(596, 802),
        orch_frames=[
            (1, 668, "a1 sees a0 standing beside a zombie;\nnobody attacks - slot 1 times out"),
            (2, 690, "a2 pressed against a0's back:\nthe trio bunches up"),
            (0, 700, "a0 faces a wall: 'still dark,\nmoving forward to find the fight'"),
            (1, 715, "a1 sees a0 in the lit room again;\nslot 2 has timed out too")],
        orch_chat=[
            (688, 1, "I'm moving forward to join the fight, let me know if you need me to target a specific zombie."),
            (694, 1, "I'm turning right to try and spot the zombies Agent 0 is targeting. Can you confirm which zombie you mean?"),
            (700, 2, "Zombies are here, I'm engaging the nearest one. Let's focus fire."),
            (704, 1, "Still dark, moving forward to find the fight."),
            (709, 1, "I'm moving forward to locate the zombies. Keep pressure on the center one."),
            (710, 2, "Still moving forward, looking for zombies.")],
        heb_frames=[
            (1, 707, "a1 spots the zombies ahead:\n'Engaging now'"),
            (0, 711, "a0 keeps pushing the passage\nand calling the flanks"),
            (1, 711, "a1 centres the zombie:\n'Engaging combat now'"),
            (1, 715, "kill lands (t=715): first\nmob kill of the episode")],
        heb_chat=[
            (704, 1, "I'll move forward now to cover the flanks for you."),
            (706, 1, "Turning left now to check the flank while you watch the right."),
            (707, 1, "Zombies spotted up ahead. Engaging now."),
            (708, 0, "Keep watching flanks for zombies while I advance into the passage."),
            (711, 1, "Zombie centered. Engaging combat now."),
            (715, 1, "Zombies are here; let's focus on clearing them for the door.")],
        orch_stats="3 agents assigned to Ch4 combat tasks for 420 agent-steps; 7 slot timeouts; 0 kills",
        heb_stats="a1 kills at t=715; W(a0-a1) 0.45 -> 0.49 across the kill; 29 a0<->a1 messages in the run-up",
    ),
    ("heb2", 456, 1, "m21_first_mob_kill"): dict(
        window=(596, 802),
        orch_frames=[
            (0, 645, "a0 - the one agent assigned\n'kill zombie' - has zombies in view"),
            (1, 645, "a1, unassigned, sees the\nsame zombies and red door"),
            (2, 656, "a2 has a zombie at arm's\nlength; no attack follows"),
            (0, 660, "a0's kill task times out;\na0 is facing a wall")],
        orch_chat=[
            (642, 1, "Moving forward to assist combat. I'm ready to engage when you target one."),
            (645, 2, "I'm turning left to scan the room for zombies, agent_0. Let me know if you see anything on your side."),
            (647, 0, "Still scanning for zombies near the door, Agent 1. Let me know if you spot one first."),
            (652, 2, "I see a zombie to the right; I'll turn to engage."),
            (655, 2, "I see mobs ahead. I'm moving closer to check them out."),
            (656, 0, "Zombies are centered now. I'm gearing up to attack.")],
        heb_frames=[
            (1, 643, "a1 sees a2 across the lit\nroom and moves to assist"),
            (2, 646, "a2 centres the zombie:\n'engaging now'"),
            (2, 650, "kill lands (t=650):\nfirst mob kill"),
            (1, 669, "a1's own kill 19 steps\nlater, zombie at arm's length")],
        heb_chat=[
            (638, 1, "Zombies are here, I need to find one to engage. Look out for them."),
            (640, 2, "Zombies in Ch4? I'm moving towards the communal area to assist."),
            (643, 2, "I see a zombie here, I will engage it now."),
            (644, 1, "I'm moving forward to assist Agent2 with the zombie."),
            (646, 2, "Zombie is centered, engaging now. Keep an eye on it."),
            (650, 2, "Zombie centered, attacking now.")],
        orch_stats="a0 alone on 'kill zombie' t=600-660 (timed out); then all three on entry/advance tasks; 0 kills",
        heb_stats="a2 kills at t=650, a1 at t=669; W(a1-a2) 0.20 -> 0.25; the a1<->a2 exchange is the only pair talking",
    ),
}


# ─── data ───────────────────────────────────────────────────────────────
def load_cell(heb: str, seed: int, ep: int, mid: str) -> dict:
    heb_root, heb_title = HEB_ARMS[heb]
    o_dir, h_dir = ORCH / f"seed_{seed}", heb_root / f"seed_{seed}"
    o_run, h_run = mdt.load_run(o_dir), mdt.load_run(h_dir)
    o_lo, o_hi = o_run["_ep_bounds"][ep - 1]
    h_lo, h_hi = h_run["_ep_bounds"][ep - 1]
    plan, targets = orch_plan(o_run)
    orch = mdt.load_orch(o_run)
    intervals = [(a, tid, t0 - o_lo, t1 - o_lo) for a, tid, t0, t1 in orch["intervals"]
                 if o_lo <= t0 < o_hi]
    tgt_tasks = {tid for tid, ms in targets[ep].items() if mid in ms}
    ends = defaultdict(list)
    for line in (o_dir / "orchestrator/assignments.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["episode"] == ep and r["reason"].startswith("freed") and r["task_id"] in tgt_tasks:
            ends[r["reason"]].append((int(r["t"]), mdt.agent_id(r["agent"])))

    def bands(run, lo, hi):
        return [(c, max(a, lo) - lo, min(b, hi) - lo)
                for c, a, b in mdt.load_chamber_bands(run) if b > lo and a < hi]

    o_f, h_f = fires_by_ep(o_run), fires_by_ep(h_run)
    bonds = {q: (xs - h_lo, ys) for q, (xs, ys) in mdt.load_bonds(h_run).items()}
    return dict(
        heb=heb, heb_title=heb_title, seed=seed, ep=ep, mid=mid,
        label=mdt.milestone_label(mid), pid=mdt.paper_id(mid),
        o_dir=o_dir, h_dir=h_dir, o_len=o_hi - o_lo, h_len=h_hi - h_lo,
        o_lo=o_lo, h_lo=h_lo,
        intervals=intervals, tgt_tasks=tgt_tasks, ends=ends,
        o_bands=bands(o_run, o_lo, o_hi), h_bands=bands(h_run, h_lo, h_hi),
        o_fire=o_f.get((ep, mid)), h_fire=h_f.get((ep, mid)),
        h_fires_all=[(k[1], v) for k, v in h_f.items() if k[0] == ep],
        o_fires_all=[(k[1], v) for k, v in o_f.items() if k[0] == ep],
        bonds=bonds, plan=plan[ep].get(mid),
        o_exp=o_dir.parent.name, h_exp=h_dir.parent.name,
    )


def chat_log(run_dir: Path, lo: int):
    log = json.loads((run_dir / "communication_log.json").read_text(encoding="utf-8"))
    return [(t - lo, mdt.agent_id(s), mdt.agent_id(r), x) for t, s, x, r in log]


KEYWORDS = ("zombie", "engag", "attack", "fight", "kill", "switch", "door", "press", "anvil")


def auto_chat(msgs, agents, t0, t1, n=6):
    """Keyword-salient messages among `agents` in [t0, t1], spread in time."""
    cand = [(t, s, x) for t, s, r, x in msgs
            if t0 <= t <= t1 and s in agents and (len(agents) < 2 or r in agents)
            and any(k in x.lower() for k in KEYWORDS)]
    if len(cand) <= n:
        return cand
    idx = np.linspace(0, len(cand) - 1, n).round().astype(int)
    return [cand[i] for i in idx]


def auto_frames(cell, side):
    if side == "heb" and cell["h_fire"]:
        t, a = cell["h_fire"]["t"], cell["h_fire"]["agents"][0]
        return [(a, t - 6, f"a{a}, 6 steps before"), (a, t - 3, f"a{a}, 3 steps before"),
                (a, t, f"a{a}: fires {cell['pid']} (t={t})"), (a, t + 3, f"a{a}, 3 steps after")]
    tg = [iv for iv in cell["intervals"] if iv[1] in cell["tgt_tasks"]]
    if tg:
        a, tid, t0, t1 = max(tg, key=lambda iv: iv[3] - iv[2])
        mid_t = int((t0 + t1) / 2)
        return [(x, mid_t, f"a{x} mid-slot on {tid}"[:40]) for x in (0, 1, 2)] + \
               [(a, int(t1) - 1, f"a{a} at slot end (t={int(t1)})")]
    w0, w1 = cell["window"]
    return [(x, int((w0 + w1) / 2), f"a{x}") for x in (0, 1, 2)]


# ─── lanes ──────────────────────────────────────────────────────────────
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK, labelsize=7.5, length=2.5, width=0.7, color=RULE, pad=1.6)


def _chambers(ax, bands, x0, x1):
    for ch, lo, hi in bands:
        if hi > x0 and lo < x1:
            ax.axvspan(max(lo, x0), min(hi, x1), color=CHAMBER_FILL.get(ch, "#ccc"),
                       alpha=0.10, lw=0, zorder=0)
            cx = (max(lo, x0) + min(hi, x1)) / 2
            if min(hi, x1) - max(lo, x0) > (x1 - x0) * 0.08:
                ax.text(cx, 1.0, ch.upper(), transform=ax.get_xaxis_transform(),
                        fontsize=6.5, color=MUTED, ha="center", va="bottom")


def draw_orch_lane(ax, cell, x0, x1, compact=False):
    _chambers(ax, cell["o_bands"], x0, x1)
    for a, tid, t0, t1 in cell["intervals"]:
        if t1 < x0 or t0 > x1:
            continue
        tgt = tid in cell["tgt_tasks"]
        ax.broken_barh([(max(t0, x0), min(t1, x1) - max(t0, x0))], (a - 0.34, 0.68),
                       facecolors=TARGET_C if tgt else OTHER_C,
                       edgecolors="white", linewidth=0.4, zorder=2,
                       hatch="////" if tgt else None, alpha=0.95 if tgt else 1.0)
        if tgt and not compact and (t1 - t0) > (x1 - x0) * 0.09:
            ax.text((max(t0, x0) + min(t1, x1)) / 2, a, tid.replace("_", " ")[:26],
                    fontsize=5.6, color="white", ha="center", va="center", zorder=3,
                    fontweight="bold")
    for reason, lst in cell["ends"].items():
        for t, a in lst:
            if x0 <= t <= x1:
                ax.scatter(t, a, marker="x", s=38 if not compact else 22, color=INK,
                           linewidths=1.3, zorder=5)
    for mid, f in cell["o_fires_all"]:
        if x0 <= f["t"] <= x1:
            for a in f["agents"]:
                ax.scatter(f["t"], a, marker="D", s=40 if not compact else 22,
                           facecolor="white", edgecolor=AGENT_C[a], linewidths=1.3, zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.7, 2.7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["a0", "a1", "a2"], fontsize=7.5)
    for tick, a in zip(ax.get_yticklabels(), (0, 1, 2)):
        tick.set_color(AGENT_C[a])
    ax.invert_yaxis()
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def draw_heb_lane(ax, cell, x0, x1, compact=False):
    _chambers(ax, cell["h_bands"], x0, x1)
    ymax = 0.0
    for q in PAIRS:
        if q not in cell["bonds"]:
            continue
        xs, ys = cell["bonds"][q]
        m = (xs >= x0 - 60) & (xs <= x1 + 60)
        ax.plot(xs[m], ys[m], "-o", color=PAIR_C[q], lw=1.4, ms=3 if not compact else 2,
                mfc="white", mew=1.0, zorder=3, label=f"a{q[0]}-a{q[1]}")
        ymax = max(ymax, float(ys[m].max()) if m.any() else 0)
    for mid, f in cell["h_fires_all"]:
        if not (x0 <= f["t"] <= x1):
            continue
        ax.axvline(f["t"], color=INK, lw=0.7, ls=(0, (2, 2)), zorder=2)
        for a in f["agents"]:
            q = max([p for p in PAIRS if a in p and p in cell["bonds"]],
                    key=lambda p: float(np.interp(f["t"], *cell["bonds"][p])), default=None)
            y = float(np.interp(f["t"], *cell["bonds"][q])) if q else ymax * 0.5
            ax.scatter(f["t"], y, marker="X", s=70 if not compact else 34, color=AGENT_C[a],
                       edgecolor="white", linewidths=0.8, zorder=6)
            if not compact:
                ax.annotate(f"a{a} {mdt.paper_id(mid)}", (f["t"], y), xytext=(4, 8),
                            textcoords="offset points", fontsize=6.5, color=AGENT_C[a],
                            fontweight="bold")
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, max(0.75, ymax * 1.25))
    _style(ax)
    ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if not compact:
        ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper left",
                  handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)


# ─── frames & chat ──────────────────────────────────────────────────────
def draw_frames(fig, frames, shots, ep, x0, w, y_top, fig_wh):
    """2 x 2 grid of 16:9 frames, title above, caption wrapped to the frame
    width below. The axes box is sized to the image aspect up front (and
    imshow told to fill it), so a missing frame renders in exactly the same
    place as a real one instead of a taller empty box. Returns the height
    used, in figure fraction."""
    FW, FH = fig_wh
    ncol, gap = 2, 0.014
    fw = (w - gap) / ncol
    fh = fw * FW / FH * 9 / 16
    title_h, cap_h = 0.014, 0.056
    chars = max(18, int(fw * FW * 13.5))      # ~13.5 chars / inch at 6.3 pt
    y = y_top
    for row in (shots[i:i + ncol] for i in range(0, len(shots), ncol)):
        y -= title_h
        for k, (a, t, cap) in enumerate(row):
            ax = fig.add_axes([x0 + k * (fw + gap), y - fh, fw, fh])
            ax.axis("off")
            fr = frames.get((a, ep, t))
            if fr is not None:
                ax.imshow(fr, aspect="auto")
            else:
                ax.set_facecolor(GRID)
                ax.text(0.5, 0.5, "frame unavailable", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6.5, color=MUTED)
            ax.text(0.0, 1.04, f"a{a} view - t={t}", transform=ax.transAxes,
                    fontsize=6.8, color=AGENT_C[a], va="bottom", fontweight="bold")
            ax.text(0.0, -0.05, "\n".join(textwrap.wrap(cap.replace("\n", " "), chars)),
                    transform=ax.transAxes, fontsize=6.3, color=INK, va="top",
                    linespacing=1.15)
        y -= fh + cap_h
    return y_top - y


def draw_chat(fig, chat, x0, w, y_top, h_avail):
    ax = fig.add_axes([x0, y_top - h_avail, w, h_avail])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig_h_in = fig.get_size_inches()[1]
    ax_h_in = h_avail * fig_h_in
    wrapped = [(t, a, W(x, 62)) for t, a, x in chat]
    heights = [0.16 + 0.125 * (txt.count("\n") + 1) for _, _, txt in wrapped]   # inches
    total = sum(heights) + 0.07 * len(heights)
    scale = min(1.0, (ax_h_in - 0.05) / total) if total > 0 else 1.0
    y = 0.995
    for (t, a, txt), h_in in zip(wrapped, heights):
        h = h_in * scale / ax_h_in
        ax.add_patch(FancyBboxPatch((0.01, y - h), 0.98, h - 0.004,
                                    boxstyle="round,pad=0.004", facecolor=AGENT_C[a] + "1c",
                                    edgecolor=AGENT_C[a], lw=0.8))
        ax.text(0.025, y - 0.012, f"agent {a} - t={t}", fontsize=6.4, color=AGENT_C[a],
                va="top", fontweight="bold")
        ax.text(0.025, y - 0.012 - 0.115 * scale / ax_h_in, txt, fontsize=6.9, color=INK,
                va="top", linespacing=1.22)
        y -= h + 0.06 * scale / ax_h_in


# ─── headline figure ────────────────────────────────────────────────────
def build(cell: dict, out_dir: Path, curated: dict | None):
    x0, x1 = (curated or {}).get("window") or (
        max(0, (cell["h_fire"]["t"] if cell["h_fire"] else 0) - 120),
        min(cell["h_len"], (cell["h_fire"]["t"] if cell["h_fire"] else cell["h_len"]) + 40))
    cell["window"] = (x0, x1)
    o_msgs = chat_log(cell["o_dir"], cell["o_lo"])
    h_msgs = chat_log(cell["h_dir"], cell["h_lo"])
    if curated:
        o_shots, h_shots = curated["orch_frames"], curated["heb_frames"]
        o_chat, h_chat = curated["orch_chat"], curated["heb_chat"]
        o_stats, h_stats = curated["orch_stats"], curated["heb_stats"]
    else:
        o_shots, h_shots = auto_frames(cell, "orch"), auto_frames(cell, "heb")
        o_agents = {a for a, tid, *_ in cell["intervals"] if tid in cell["tgt_tasks"]} or {0, 1, 2}
        h_agents = set(cell["h_fire"]["agents"]) if cell["h_fire"] else {0, 1, 2}
        if len(h_agents) == 1:
            a = next(iter(h_agents))
            q = max([p for p in PAIRS if a in p], key=lambda p: float(np.interp(
                cell["h_fire"]["t"], *cell["bonds"][p])))
            h_agents = set(q)
        o_chat = auto_chat(o_msgs, o_agents, x0, x1)
        h_chat = auto_chat(h_msgs, h_agents, x0, (cell["h_fire"]["t"] if cell["h_fire"] else x1))
        p = cell["plan"]
        o_stats = (f"{len(p['agents'])} agents on tasks targeting {cell['pid']} for "
                   f"{p['agent_steps']} agent-steps; ends: "
                   + ", ".join(f"{k.replace('freed_', '')} x{v}" for k, v in p["ends"].items())
                   + "; not completed") if p else "no task targeting this milestone; not completed"
        h_stats = (f"a{''.join(map(str, cell['h_fire']['agents']))} fires {cell['pid']} at t="
                   f"{cell['h_fire']['t']}" if cell["h_fire"] else "not completed")

    ep = cell["ep"]
    o_frames = grab_frames({"run": cell["o_dir"], "seed": cell["seed"], "exp": cell["o_exp"]},
                           [(a, ep, t) for a, t, _ in o_shots])
    h_frames = grab_frames({"run": cell["h_dir"], "seed": cell["seed"], "exp": cell["h_exp"]},
                           [(a, ep, t) for a, t, _ in h_shots])

    FW, FH = 7.4, 9.2
    fig = plt.figure(figsize=(FW, FH))
    colx = {"orch": 0.075, "heb": 0.555}
    colw = 0.42
    # headers
    heads = {"orch": ("Central orchestrator", "assigned to it - did not complete"),
             "heb": (cell["heb_title"].split(" (")[0].replace("Gemma-E4B + ", "+ "),
                     "no instruction - completed")}
    for side, (t1, t2) in heads.items():
        fig.text(colx[side], 0.975, t1, fontsize=10.5, color=INK, fontweight="bold", va="top")
        fig.text(colx[side], 0.955, t2, fontsize=8.2, color=TARGET_C if side == "orch" else "#1b7f4a",
                 va="top", style="italic")
    fig.text(0.5, 0.993, f"seed {cell['seed']}, episode {ep}: {cell['label']} ({cell['pid']})  -  "
             f"t = {x0}-{x1} of {cell['h_len']}", fontsize=8.5, color=MUTED, ha="center", va="top")

    # row 1: mechanism lanes
    lane_h, lane_top = 0.155, 0.93
    ax_o = fig.add_axes([colx["orch"], lane_top - lane_h, colw, lane_h])
    draw_orch_lane(ax_o, cell, x0, x1)
    ax_o.set_ylabel("assigned task", fontsize=7.5, color=INK)
    ax_o.set_xlabel("environment step (within episode)", fontsize=7.5, color=INK)
    ax_h = fig.add_axes([colx["heb"], lane_top - lane_h, colw, lane_h])
    draw_heb_lane(ax_h, cell, x0, x1)
    ax_h.set_ylabel("bond strength W", fontsize=7.5, color=INK)
    ax_h.set_xlabel("environment step (within episode)", fontsize=7.5, color=INK)
    ax_o.legend(handles=[Patch(facecolor=TARGET_C, hatch="////", label=f"task targets {cell['pid']}"),
                         Patch(facecolor=OTHER_C, label="other task"),
                         plt.Line2D([], [], marker="x", color=INK, lw=0, ms=5, label="slot timed out")],
                fontsize=6.3, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02),
                handlelength=1.4, columnspacing=0.9, borderaxespad=0.1)

    # row 2: frames (2 x 2 per column); the x-label of the lanes needs ~0.045
    fr_top = lane_top - lane_h - 0.055
    used = draw_frames(fig, o_frames, o_shots, ep, colx["orch"], colw, fr_top, (FW, FH))
    draw_frames(fig, h_frames, h_shots, ep, colx["heb"], colw, fr_top, (FW, FH))

    # row 3: chat, down to the stats line
    ch_top = fr_top - used - 0.012
    ch_h = ch_top - 0.088
    draw_chat(fig, o_chat, colx["orch"], colw, ch_top, ch_h)
    draw_chat(fig, h_chat, colx["heb"], colw, ch_top, ch_h)

    # footer stats
    for side, txt in (("orch", o_stats), ("heb", h_stats)):
        fig.text(colx[side], 0.072, W(txt, 78), fontsize=6.9, color=INK, va="top",
                 linespacing=1.25, fontweight="bold")
    fig.text(0.5, 0.008, "Diamonds: milestone fires (agent colour). Orchestrator: x = 60-step task "
             "slot freed by timeout. Hebbian: W from the recorded 50-step graph snapshots "
             "(symmetrised); X = fire on the contributor's strongest pair.",
             fontsize=6.2, color=MUTED, ha="center", va="bottom")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"counterfactual_s{cell['seed']}_ep{ep}_{cell['pid']}"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    cap = (f"Seed {cell['seed']}, episode {ep}, {cell['label']} ({cell['pid']}), steps {x0}-{x1}. "
           f"Left: central orchestrator - {o_stats}. Right: {cell['heb_title']} - {h_stats}. "
           f"Same map and episode; the orchestrator team held tasks targeting the milestone "
           f"and did not complete it, the Hebbian team completed it without instruction.")
    stem.with_suffix(".caption.txt").write_text(cap + "\n", encoding="utf-8")
    missing = [k for k in [(a, ep, t) for a, t, _ in h_shots] if k not in h_frames]
    print(f"wrote {stem}.png/.pdf" + (f"  [{len(missing)} Hebbian frames unavailable]" if missing else ""))


# ─── appendix grid ──────────────────────────────────────────────────────
def build_appendix(heb: str, out_dir: Path, classes=("HEB_ONLY", "ORCH_ONLY")):
    rows = list(csv.DictReader(open(out_dir / "cells.csv", encoding="utf-8")))
    rows = [r for r in rows if r["class"] in classes]
    order = {"m21_first_mob_kill": 0, "m17_switch_pressed": 1, "m18_door_opened": 2}
    rows.sort(key=lambda r: (r["class"] != "HEB_ONLY", order.get(r["mid"], 9),
                             int(r["seed"]), int(r["ep"])))
    # switch and door fire on the same step; show the pair once per (seed, ep, class)
    seen, kept = set(), []
    for r in rows:
        key = (r["seed"], r["ep"], r["class"], "ch3" if r["mid"] in
               ("m17_switch_pressed", "m18_door_opened") else r["mid"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    n = len(kept)
    if not n:
        print("no cells to draw")
        return
    row_h, top, bot = 0.62, 0.55, 0.45
    FH = top + bot + n * row_h
    fig = plt.figure(figsize=(7.4, FH))
    heb_title = HEB_ARMS[heb][1]
    fig.text(0.5, 1 - 0.12 / FH, f"Every cell where exactly one arm fired - orchestrator vs {heb_title}",
             fontsize=9.5, color=INK, ha="center", va="top", fontweight="bold")
    fig.text(0.29, 1 - 0.34 / FH, "central orchestrator - assigned tasks", fontsize=8, color=INK, ha="center")
    fig.text(0.745, 1 - 0.34 / FH, "Hebbian - pair bonds W", fontsize=8, color=INK, ha="center")
    for i, r in enumerate(kept):
        cell = load_cell(heb, int(r["seed"]), int(r["ep"]), r["mid"])
        y0 = 1 - (top + (i + 1) * row_h - 0.10) / FH
        h = (row_h - 0.16) / FH
        tag = "H" if r["class"] == "HEB_ONLY" else "O"
        fig.text(0.012, y0 + h / 2, f"seed {r['seed']}\nep {r['ep']}\n{cell['pid']}  [{tag}]",
                 fontsize=6.6, color=INK, va="center", linespacing=1.2)
        # one x-range per row: the two runs of a seed can differ in episode
        # length (a Ch5 death ends an episode early), and a shared axis is
        # what makes "at the same time" readable across the two columns.
        xmax = max(cell["o_len"], cell["h_len"])
        ax_o = fig.add_axes([0.10, y0, 0.39, h])
        draw_orch_lane(ax_o, cell, 0, xmax, compact=True)
        ax_h = fig.add_axes([0.555, y0, 0.39, h])
        draw_heb_lane(ax_h, cell, 0, xmax, compact=True)
        for ax in (ax_o, ax_h):
            ax.tick_params(labelsize=6)
            if i != n - 1:
                ax.set_xticklabels([])
        ax_o.set_yticklabels(["a0", "a1", "a2"], fontsize=6)
    fig.text(0.5, 0.06 / FH, "[H] Hebbian only, [O] orchestrator only. Orange hatched = task targeting the cell's "
             "milestone, x = slot timed out, diamond = fire; X = Hebbian fire on the contributor's pair.",
             fontsize=6.2, color=MUTED, ha="center", va="bottom")
    stem = out_dir / "counterfactual_appendix_grid"
    fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    # LaTeX table over the same cells
    L = [r"\begin{tabular}{l c l c c c c c}", r"\toprule",
         r"seed & ep & milestone & who fired & orch.\ agent-steps & orch.\ ends & Heb.\ fire $t$ & $W$ at fire \\",
         r"\midrule"]
    for r in kept:
        who = "Hebbian" if r["class"] == "HEB_ONLY" else "orchestrator"
        ends = r["orch_ends"].replace("freed_", "").replace(";", ", ").replace(":", r"$\times$") or "--"
        ft = r["heb_fired_t"] or "--"
        wv = ("%.2f" % float(r["heb_W"])) if r["heb_W"] not in ("", "nan") and r["heb_W"] == r["heb_W"] else "--"
        L.append(f"{r['seed']} & {r['ep']} & {r['milestone']} & {who} & {r['orch_agent_steps']} & "
                 f"{ends} & {ft} & {wv} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "counterfactual_appendix_table.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {stem}.png/.pdf and counterfactual_appendix_table.tex ({n} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heb", default="heb2", choices=sorted(HEB_ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--mid", default="m21_first_mob_kill")
    ap.add_argument("--appendix", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    out = args.out or (ASSETS / "counterfactual" / args.heb)
    if args.appendix:
        build_appendix(args.heb, out)
        return 0
    cell = load_cell(args.heb, args.seed, args.ep, args.mid)
    build(cell, out, CASES.get((args.heb, args.seed, args.ep, args.mid)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
