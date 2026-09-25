#!/usr/bin/env python3
r"""make_rq3_dynamics_fig.py - RQ3 partner re-selection dynamics, full transfer vs bonds only.

ICLR text-width figure (5.5 in) with four panels:

  (A) schematic of the Chamber-3 dependency ring. Six agents seated in the
      three Phase-A pairs; agent i's switch opens agent (i+1)'s door. For one
      example agent it marks its OLD PARTNER (its Phase-A teammate, solid) and
      its TASK-LINKED NEIGHBOUR (the other ring neighbour, from a different
      Phase-A pair, dashed) - the two line styles reused in panel C.
  (B) share of messages sent to the old partner, 50-step bins (chance 0.20).
  (C) mean bond W to the old partner (solid) and to the task-linked
      neighbour (dashed), averaged over the six agents, snapshots every 50 steps.
  (D) switch presses (M13), one marker per press.

Episodes are rescaled to [0, 1] so the two conditions align (lengths differ
by < 2%); the shaded band is the Chamber-3 phase. One seed (pair_bonding_3f,
seed 42), as in tab:transplant_main.

Usage:
    python analysis/make_rq3_dynamics_fig.py [--out paper_assets/agent_completion/figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

import paths  # noqa: F401
from paths import ASSETS, REPO, group
import make_results as MR

ROOT = group("pair_bonding_3f")
FULL, BONDS = "#e8622a", "#2a78d6"
CONDS = [("(a) Full transfer (memory + bonds)", "expB_merged_transplant_3f", FULL),
         ("(c) Bonds only", "expB_bond_only_3f", BONDS)]
SEAT = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
OTHER = {i: ({(i - 1) % 6, (i + 1) % 6} - {SEAT[i]}).pop() for i in range(6)}
BIN = 50
DASH = (0, (3.2, 1.6))
INK, MUTED, FAINT, GRID, BAND = "#1a1a19", "#55554e", "#9a9990", "#e6e5df", "#f2f0e9"


def ai(name):
    return int(str(name).split("_")[-1].replace("agent", ""))


def load(d):
    run = MR.load_runs(ROOT, d)[0]
    rd = Path(run["_path"]).parent
    out = {"share": [], "w_old": [], "w_other": [], "press": [], "ch3_end": []}
    snaps = sorted(run["graph_snapshots"], key=lambda g: g["step"])
    for e, (s0, s1) in enumerate(run["_ep_bounds"]):
        L = s1 - s0
        msgs = [json.loads(l) for l in open(rd / f"episodes/ep_{e + 1:04d}/messages.jsonl", encoding="utf-8")]
        for b in range(0, L, BIN):
            m = [x for x in msgs if b <= x["t"] < b + BIN]
            if m:
                share = sum(1 for x in m if ai(x["receiver"]) == SEAT[ai(x["sender"])]) / len(m)
                out["share"].append((e + (b + BIN / 2) / L, share))
        for g in snaps:
            if s0 <= g["step"] < s1:
                W = g["W"]
                x = e + (g["step"] - s0) / L
                out["w_old"].append((x, st.fmean(W[i][SEAT[i]] for i in range(6))))
                out["w_other"].append((x, st.fmean(W[i][OTHER[i]] for i in range(6))))
        rows = list(csv.DictReader(open(rd / f"episodes/ep_{e + 1:04d}/step_log.csv", encoding="utf-8")))
        ch4 = [int(r["step"]) for r in rows if r["chamber"] in ("ch4", "ch5")]
        out["ch3_end"].append(e + (min(ch4) if ch4 else 332) / L)
        for ev in run["milestone_events"]:
            s = int(ev["step"])
            if ev["milestone_id"] == "m17_switch_pressed" and s0 <= s < s1:
                out["press"].append(e + (s - s0) / L)
    return out


def draw_ring(ax):
    """Chamber-3 dependency ring with the two relationships of agent a2."""
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ang = {i: math.pi / 2 - i * 2 * math.pi / 6 + math.pi / 6 for i in range(6)}
    pos = {i: (math.cos(ang[i]), math.sin(ang[i])) for i in range(6)}
    for a, b in ((0, 1), (2, 3), (4, 5)):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color=BAND, lw=15, solid_capstyle="round", zorder=0)
    for i in range(6):
        ax.add_patch(FancyArrowPatch(pos[i], pos[(i + 1) % 6], connectionstyle="arc3,rad=-0.28",
                                     arrowstyle="-|>", mutation_scale=6, lw=0.7, color=FAINT,
                                     shrinkA=7, shrinkB=7, zorder=1))
    focus = 2
    ax.plot(*zip(pos[focus], pos[SEAT[focus]]), color=INK, lw=1.3, zorder=2)
    ax.plot(*zip(pos[focus], pos[OTHER[focus]]), color=INK, lw=1.3, ls=DASH, zorder=2)
    for i, (x, y) in pos.items():
        hl = i == focus
        ax.scatter([x], [y], s=150, color=INK if hl else "white", edgecolor=INK, lw=0.9, zorder=3)
        ax.text(x, y, f"a{i}", ha="center", va="center", fontsize=6,
                color="white" if hl else INK, zorder=4)
    ax.text(0, 0, "Chamber 3\nswitch chain", ha="center", va="center", fontsize=5.8, color=MUTED)


def draw_key(ax):
    ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], color=BAND, lw=7, label="Phase-A pair"),
        Line2D([], [], color=FAINT, lw=0.8, marker=">", ms=3,
               label="switch $i$ opens\ndoor of agent $i{+}1$"),
        Line2D([], [], color=INK, lw=1.3, label="old partner:\nPhase-A teammate"),
        Line2D([], [], color=INK, lw=1.3, ls=DASH,
               label="task-linked neighbour:\nother ring neighbour,\nfrom a different pair")],
        loc="upper left", bbox_to_anchor=(0.02, 1.0), frameon=False, fontsize=6,
        handlelength=1.7, labelspacing=0.7, borderaxespad=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ASSETS / "agent_completion" / "figures")
    args = ap.parse_args()
    data = [(lab, load(d), c) for lab, d, c in CONDS]
    ch3_end = data[0][1]["ch3_end"]

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "ytick.major.width": 0.5, "ytick.major.size": 2.5})
    fig = plt.figure(figsize=(5.5, 3.25))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 3.2], height_ratios=[1.1, 1.1, 0.5],
                          wspace=0.24, hspace=0.16, left=0.015, right=0.985, top=0.905, bottom=0.075)
    gl = gs[:, 0].subgridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.02)
    axr = fig.add_subplot(gl[0])
    axk = fig.add_subplot(gl[1])
    draw_key(axk)
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    draw_ring(axr)

    for ax in (ax1, ax2, ax3):
        for e in range(3):
            ax.axvspan(e, ch3_end[e], color=BAND, lw=0, zorder=0)
            if e:
                ax.axvline(e, color=FAINT, lw=0.6, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)

    # (B) old-partner message share
    for lab, d, c in data:
        ax1.plot(*zip(*d["share"]), color=c, lw=1.3, zorder=3)
    ax1.axhline(0.2, color=MUTED, ls=":", lw=0.7, zorder=2)
    ax1.text(2.995, 0.215, "chance", ha="right", va="bottom", fontsize=6, color=MUTED)
    ax1.set_ylim(-0.04, 1.06)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_ylabel("Msgs to\nold partner")

    # (C) bonds
    for lab, d, c in data:
        ax2.plot(*zip(*d["w_old"]), color=c, lw=1.3, zorder=3)
        ax2.plot(*zip(*d["w_other"]), color=c, lw=1.1, ls=DASH, zorder=3)
    ax2.set_ylabel("Bond $W$")
    ax2.set_ylim(0.08, 0.66)
    ax2.set_yticks([0.2, 0.4, 0.6])
    ax2.legend(handles=[Line2D([], [], color=MUTED, lw=1.3, label="old partner"),
                        Line2D([], [], color=MUTED, lw=1.1, ls=DASH, label="task-linked neighbour")],
               loc="lower right", frameon=False, fontsize=6, ncol=2, handlelength=2.2,
               columnspacing=1.2, borderaxespad=0.1)

    # (D) switch presses
    for row, (lab, d, c) in enumerate(data):
        y = 1 - row
        ax3.scatter(d["press"], [y] * len(d["press"]), marker="v", s=22, color=c, zorder=3,
                    edgecolor="white", linewidth=0.5)
    ax3.set_ylim(-0.7, 1.7)
    ax3.set_yticks([1, 0])
    ax3.set_yticklabels(["(a)", "(c)"])
    ax3.tick_params(axis="y", length=0)
    ax3.set_ylabel("Switch\npress")
    ax3.set_xlim(0, 3)
    ax3.set_xticks([0.5, 1.5, 2.5])
    ax3.set_xticklabels(["Episode 1", "Episode 2", "Episode 3"], fontsize=7)
    ax3.text(ch3_end[0] / 2, 1.62, "Chamber 3", ha="center", va="top", fontsize=6, color=MUTED)

    fig.align_ylabels([ax1, ax2, ax3])
    fig.legend(handles=[Line2D([], [], color=c, lw=1.6, label=lab) for lab, _, c in data],
               loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=2, frameon=False,
               fontsize=7, handlelength=1.8, columnspacing=1.6)
    x_left = ax1.get_position().x0 - 0.112
    for ax, letter in ((ax1, "B"), (ax2, "C"), (ax3, "D")):
        fig.text(x_left, ax.get_position().y1, letter, fontsize=8.5, fontweight="bold",
                 va="top", ha="left", color=INK)
    fig.text(0.015, ax1.get_position().y1, "A", fontsize=8.5, fontweight="bold",
             va="top", ha="left", color=INK)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "rq3_dynamics"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    print("wrote %s (+ .pdf)" % stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
