"""Planned teams vs. realised teams: orchestrator vs Gemma-4B + Hebbian 2.0.

Both mechanisms are asked the same question -- "which pair is a team right
now?" -- and the answer is laid under the pairs that actually get credited
together on a cooperative-track milestone.

  orchestrator : a pair is a team while both agents hold the same task id,
                 replayed per step from orchestrator/assignments.jsonl.
  Hebbian 2.0  : a pair is a team in proportion to its bond, from the
                 validated per-step replay of the three-factor rule.

Rows are labelled by pair and by agent, so nothing has to be decoded from
colour. The bottom row carries the accountability chain: how much joint work
the orchestrator planned, how much was ever staffed, how much finished, and
-- for both mechanisms -- whether the joint work that did happen was the work
the mechanism pointed at.

Run:  python analysis/make_team_comparison.py [--wide]
Out:  paper_assets/timelines/comparison/team_formation[_wide].{pdf,png,svg}
"""
import json
import re
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paths import ASSETS, RUNS            # noqa: F401 (adds siblings to path)
from make_results import MILESTONE_TRACK
from replay_hebbian_terms import load_inputs
from prototype_three_factor_rule import replay_three_factor

COOP = {"ch2_anvils", "ch3_switches", "ch4_combat", "ch5_boss"}
PAIRS = [(0, 1), (0, 2), (1, 2)]
AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
TEAM_C, BUSY_C, TRIO_C = "#4a5568", "#d9dee5", "#a7b0bb"
INK, MUTED, GRID = "#31373f", "#8b95a1", "#e6e9ee"
OK_C, NO_C = "#1b7f5a", "#b03a3a"

ORCH = RUNS / "orchestrator/new_exp_0_gemma_orch_villager_advisory"
G3F = RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f"
OUT = ASSETS / "timelines" / "comparison"
ORCH_SEED, HEB_SEED = "seed_789", "seed_123"


def aid(name):
    return int(re.search(r"(\d+)", name).group(1))


# -- data ---------------------------------------------------------------
def episode_offsets(fm):
    offs, c = [], 0
    for L in fm.get("episode_lengths") or []:
        offs.append(c)
        c += L
    return offs, c


def coop_fires(run_dir):
    """[(step, agents credited)] merged per lua_step, + per-agent credits."""
    fm = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    by, step, contrib = defaultdict(set), {}, Counter()
    for e in fm["milestone_events"]:
        if MILESTONE_TRACK.get(e["milestone_id"]) not in COOP:
            continue
        by[e["lua_step"]].add(aid(e["contributor"]))
        step[e["lua_step"]] = min(step.get(e["lua_step"], 10 ** 9), e["step"])
        contrib[aid(e["contributor"])] += 1
    return fm, sorted((step[k], tuple(sorted(v))) for k, v in by.items()), contrib


def orch_signals(run_dir, fm):
    """Per pair: same-task mask and both-busy-elsewhere mask, per step."""
    offs, T = episode_offsets(fm)
    ev = defaultdict(list)
    for line in open(run_dir / "orchestrator/assignments.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ev[r["t"] + offs[r["episode"] - 1]].append(
            (aid(r["agent"]), r["task_id"], r["reason"]))
    task = np.empty((T, 3), dtype=object)
    live = {}
    for t in range(T):
        for a, tid, reason in ev.get(t, []):
            live[a] = tid if reason.startswith("allocate") else None
        for a in range(3):
            task[t, a] = live.get(a)
    same = {p: np.array([task[t, p[0]] is not None
                         and task[t, p[0]] == task[t, p[1]] for t in range(T)])
            for p in PAIRS}
    trio = np.array([task[t, 0] is not None and task[t, 0] == task[t, 1]
                     == task[t, 2] for t in range(T)])
    excl = {p: same[p] & ~trio for p in PAIRS}
    busy = {p: np.array([task[t, p[0]] is not None and task[t, p[1]] is not None
                         and task[t, p[0]] != task[t, p[1]] for t in range(T)])
            for p in PAIRS}
    return same, busy, excl, trio, T


def heb_signals(run_dir):
    W = replay_three_factor(load_inputs(run_dir))["W"]
    bond = {p: (W[:, p[0], p[1]] + W[:, p[1], p[0]]) / 2 for p in PAIRS}
    return bond, len(W)


def leading_pair(bond, t, eps=0.02):
    v = sorted(((bond[p][t], p) for p in PAIRS), reverse=True)
    return v[0][1] if v[0][0] - v[1][0] > eps else None


def collect():
    d = {}

    hits = tot = proposed = staffed = finished = 0
    contrib = Counter()
    seeds = sorted(ORCH.glob("seed_*"))
    for sd in seeds:
        fm, fires, c = coop_fires(sd)
        same, busy, excl, trio, T = orch_signals(sd, fm)
        contrib += c
        prop, staff, done = set(), set(), set()
        for line in open(sd / "orchestrator/dag.jsonl", encoding="utf-8"):
            r = json.loads(line)
            for tk in r.get("tasks", []):
                if tk.get("min_agents", 1) < 2:
                    continue
                k = (r.get("episode"), tk["id"])
                prop.add(k)
                if len(tk.get("assigned") or []) >= 2:
                    staff.add(k)
                if tk.get("status") == "success":
                    done.add(k)
        proposed += len(prop)
        staffed += len(staff)
        finished += len(done)
        for st, ags in fires:
            if len(ags) < 2:
                continue
            tot += 1
            s = min(st, T - 1)
            hits += any(same[p][s] for p in PAIRS if set(p) <= set(ags))
        if sd.name == ORCH_SEED:
            d["orch_show"] = (same, busy, excl, trio, fires, T)
    d["orch"] = dict(hits=hits, tot=tot, contrib=contrib, seeds=len(seeds),
                     proposed=proposed, staffed=staffed, finished=finished,
                     chance=0.40)

    hits = tot = 0
    contrib = Counter()
    seeds = sorted(G3F.glob("seed_*"))
    for sd in seeds:
        fm, fires, c = coop_fires(sd)
        bond, T = heb_signals(sd)
        contrib += c
        for st, ags in fires:
            if len(ags) < 2:
                continue
            tot += 1
            lead = leading_pair(bond, min(st, T - 1))
            hits += lead is not None and set(lead) <= set(ags)
        if sd.name == HEB_SEED:
            d["heb_show"] = (bond, fires, T)
    d["heb"] = dict(hits=hits, tot=tot, contrib=contrib, seeds=len(seeds),
                    chance=1 / 3)
    return d


# -- drawing ------------------------------------------------------------
def blocks(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    return [(idx[a], idx[b] - idx[a] + 1)
            for a, b in zip(np.r_[0, brk + 1], np.r_[brk, idx.size - 1])]


def pair_rows(ax, fires, T, fs, orch=None, bond=None):
    """Three self-labelled pair rows: team signal + joint completions."""
    ys = {p: 2 - i for i, p in enumerate(PAIRS)}
    if bond is not None:
        mean_bond = sum(bond.values()) / len(bond)
        peak = max(np.abs(b - mean_bond).max() for b in bond.values())
    else:
        mean_bond, peak = None, 1.0
    for p, y in ys.items():
        ax.axhline(y - 0.28, color=GRID, lw=0.7, zorder=0)
        if orch is not None:
            same, busy, excl, trio = orch
            for mask, col, hgt in ((busy[p], BUSY_C, 0.16),
                                   (same[p] & trio, TRIO_C, 0.34),
                                   (excl[p], TEAM_C, 0.62)):
                for x0, w in blocks(mask):
                    ax.add_patch(plt.Rectangle((x0, y - 0.28), w, hgt,
                                               facecolor=col, lw=0, zorder=1))
        else:
            v = (bond[p] - mean_bond) / max(peak, 1e-9)
            base = y - 0.02
            ax.axhline(base, color="#c9ced6", lw=0.6, zorder=1)
            ax.fill_between(np.arange(T), base,
                            base + np.clip(v, 0, None) * 0.30,
                            color=TEAM_C, alpha=0.85, lw=0, zorder=2)
            ax.fill_between(np.arange(T), base,
                            base + np.clip(v, None, 0) * 0.30,
                            color=BUSY_C, alpha=0.95, lw=0, zorder=2)
    for st, ags in fires:
        if len(ags) < 2:
            continue
        for p in PAIRS:
            if not set(p) <= set(ags):
                continue
            s = min(st, T - 1)
            named = bool(orch[0][p][s]) if orch is not None else \
                (leading_pair(bond, s) == p)
            ax.scatter(st, ys[p] + 0.50, marker="D", s=30, facecolor="white",
                       edgecolor=OK_C if named else NO_C, lw=1.4, zorder=5,
                       clip_on=False)
    ax.set_yticks([y - 0.05 for y in ys.values()])
    ax.set_yticklabels(["a%d-a%d" % p for p in PAIRS], fontsize=fs["tick"])
    ax.set_ylim(-0.45, 2.85)
    ax.set_xlim(0, T)
    ax.tick_params(length=0, labelsize=fs["tick"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def agent_rows(ax, fires, T, fs):
    for a in range(3):
        ax.axhline(2 - a, color=GRID, lw=0.7, zorder=0)
    for st, ags in fires:
        joint = len(ags) >= 2
        if joint:
            ax.plot([st, st], [2 - max(ags), 2 - min(ags)], color=INK,
                    lw=0.9, alpha=0.5, zorder=2)
        for a in ags:
            ax.scatter(st, 2 - a, s=30 if joint else 13, zorder=3,
                       facecolor=AGENT_C[a], edgecolor="white", lw=0.5)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["a0", "a1", "a2"], fontsize=fs["tick"])
    for tick, a in zip(ax.get_yticklabels(), range(3)):
        tick.set_color(AGENT_C[a])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0, T)
    ax.tick_params(length=0, labelsize=fs["tick"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def build(wide=False):
    d = collect()
    o, h = d["orch"], d["heb"]
    FIG_W = 11.0 if wide else 5.5
    fs = dict(axis=8.2 if wide else 6.3, tick=8.0 if wide else 6.0,
              title=11.0 if wide else 7.8, body=8.8 if wide else 6.4,
              small=7.6 if wide else 5.7)
    H = 5.30 if wide else 4.30
    rc = {"font.family": "serif",
          "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
          "axes.edgecolor": "#9aa3ad", "axes.linewidth": 0.6,
          "svg.fonttype": "none", "pdf.fonttype": 42,
          "mathtext.fontset": "stix"}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(FIG_W, H))
        gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 0.60, 0.92],
                              hspace=0.55, wspace=0.24,
                              left=0.105, right=0.985, top=0.885, bottom=0.115)

        same, busy, excl, trio, ofires, oT = d["orch_show"]
        bond, hfires, hT = d["heb_show"]
        specs = [(0, "Central orchestrator",
                  "who the orchestrator put on the same task",
                  ORCH_SEED, ofires, oT, dict(orch=(same, busy, excl, trio))),
                 (1, "Gemma-4B + Hebbian 2.0",
                  "bond above (dark) or below (light) the pair average",
                  HEB_SEED, hfires, hT, dict(bond=bond))]
        for k, title, sub, seed, fires, T, kw in specs:
            axP = fig.add_subplot(gs[0, k])
            axA = fig.add_subplot(gs[1, k], sharex=axP)
            pair_rows(axP, fires, T, fs, **kw)
            agent_rows(axA, fires, T, fs)
            axP.set_title(title, fontsize=fs["title"], color=INK,
                          fontweight="bold", pad=17)
            axP.text(0.5, 1.055, sub + "   (" + seed.replace("_", " ") + ")",
                     transform=axP.transAxes, fontsize=fs["small"],
                     color=MUTED, ha="center", va="bottom")
            axP.set_ylabel("team", fontsize=fs["axis"], color=INK)
            axA.set_ylabel("credits", fontsize=fs["axis"], color=INK)
            axA.set_xlabel("environment step (cumulative)",
                           fontsize=fs["axis"], labelpad=1.5)
            axA.tick_params(labelsize=fs["tick"], length=2.5)
            plt.setp(axP.get_xticklabels(), visible=False)

        axL = fig.add_subplot(gs[2, 0])
        labels = ["planned", "ever staffed\nwith 2+ agents", "ever completed"]
        vals = [o["proposed"], o["staffed"], o["finished"]]
        axL.barh([2, 1, 0], vals, color=[TEAM_C, "#9aa3ad", NO_C], height=0.5,
                 zorder=3)
        for y, v in zip([2, 1, 0], vals):
            axL.text(v + o["proposed"] * 0.025, y, str(v), va="center",
                     fontsize=fs["body"], color=INK, fontweight="bold")
        axL.set_yticks([2, 1, 0])
        axL.set_yticklabels(labels, fontsize=fs["small"], linespacing=1.1)
        axL.set_xlim(0, o["proposed"] * 1.2)
        axL.set_ylim(-1.0, 2.6)
        axL.set_xticks([])
        axL.tick_params(length=0)
        axL.set_title("Multi-agent tasks the orchestrator planned  (6 seeds)",
                      fontsize=fs["axis"], color=INK, pad=5, loc="left")
        for s in ("top", "right", "bottom", "left"):
            axL.spines[s].set_visible(False)
        axL.text(0, -0.92, "yet %d joint completions happened anyway "
                 "(%.1f per seed, vs %.1f for Hebbian 2.0)"
                 % (o["tot"], o["tot"] / o["seeds"], h["tot"] / h["seeds"]),
                 fontsize=fs["small"], color=MUTED, va="center")

        axR = fig.add_subplot(gs[2, 1])
        vals = [o["hits"] / max(o["tot"], 1), h["hits"] / max(h["tot"], 1)]
        bars = axR.bar(["orchestrator", "Hebbian 2.0"], vals,
                       color=[MUTED, TEAM_C], width=0.42, zorder=3)
        for k, (b, m) in enumerate(zip(bars, (o, h))):
            axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                     "%d/%d" % (m["hits"], m["tot"]), ha="center",
                     fontsize=fs["body"], color=INK, fontweight="bold")
            axR.plot([k - 0.30, k + 0.30], [m["chance"], m["chance"]],
                     color=NO_C, lw=1.1, ls=(0, (2.5, 2)), zorder=4)
            axR.text(k + 0.33, m["chance"], "chance", fontsize=fs["small"],
                     color=NO_C, va="center", ha="left")
        axR.set_xlim(-0.6, 1.75)
        axR.set_ylim(0, 1.10)
        axR.set_yticks([0, 0.5, 1.0])
        axR.set_yticklabels(["0", "50%", "100%"], fontsize=fs["tick"])
        axR.set_title("When a joint completion happened,\n"
                      "was that pair the mechanism's team?",
                      fontsize=fs["axis"], color=INK, pad=5, loc="left")
        axR.tick_params(labelsize=fs["tick"])
        axR.grid(axis="y", color="#eef1f4", lw=0.6)
        axR.set_axisbelow(True)
        for s in ("top", "right"):
            axR.spines[s].set_visible(False)

        handles = [Patch(facecolor=TEAM_C, label="this pair only"),
                   Patch(facecolor=TRIO_C, label="all three on one task"),
                   Patch(facecolor=BUSY_C, label="both busy, different tasks"),
                   plt.Line2D([], [], marker="D", ls="none", mfc="white",
                              mec=OK_C, mew=1.4, ms=6,
                              label="joint completion by the named team"),
                   plt.Line2D([], [], marker="D", ls="none", mfc="white",
                              mec=NO_C, mew=1.4, ms=6,
                              label="joint completion the team missed")]
        fig.legend(handles=handles, loc="lower center", ncol=5,
                   fontsize=fs["small"], frameon=False,
                   bbox_to_anchor=(0.5, -0.004), handlelength=1.3,
                   columnspacing=1.4, handletextpad=0.5)

        OUT.mkdir(parents=True, exist_ok=True)
        stem = "team_formation_wide" if wide else "team_formation"
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / (stem + "." + ext),
                        dpi=300 if ext == "png" else None, facecolor="white")
        plt.close(fig)

    print("  orchestrator: planned %d multi-agent tasks, %d ever staffed, %d "
          "completed; %d joint completions, %d matched a co-assignment"
          % (o["proposed"], o["staffed"], o["finished"], o["tot"], o["hits"]))
    print("  Hebbian 2.0 : %d joint completions, %d matched the leading bond"
          % (h["tot"], h["hits"]))
    print("  wrote " + str(OUT) + "/" + stem + ".(pdf|png|svg)")


if __name__ == "__main__":
    build(wide="--wide" in sys.argv)
