r"""Central orchestrator vs Hebbian bonds, in the wide_gemma_hebbian3f grammar.

Two mechanism lanes on one seed, sharing an exact step axis. Neither carries a
panel title -- both are meant to be labelled from the caption.

  lane a : central orchestrator. One row per agent; a bar marks the task it
           holds, on one blue ramp by team size: pale = alone, mid = paired
           with one other agent, dark = all three on one task. Every visible
           seam is a reassignment. Events sit on the credited agent's row.
  lane b : Hebbian 2.0. The symmetrised bond weight
           Wbar_ij = (W_i->j + W_j->i)/2 per pair over time.

Shared with the make_final_figures family: chamber shading and labels, agent /
pair colours, the Wbar(a0-a1) legend notation, and the event glyphs on both
lanes -- open circle in the pair colour for a joint cooperative milestone, filled
star in the killer's colour for a mob kill. Everything else follows the
figures4papers look: plain black left/bottom axes, no grid, no hairlines,
frameless legends inside the axes, and the event key left to the caption.

Note the two lanes are different *runs* at the same seed (orchestrator vs
Hebbian arm), so the step axis is shared exactly but each lane keeps its own
chamber spans -- they agree through episode 1 and drift afterwards.

Sizing: the canvas is the ICLR text width (5.5 in), so every figure renders
1:1 at ``width=\linewidth`` and the point sizes in FS_PRINT are the printed
ones. The split lanes use fixed margins (no tight bbox) so their plot boxes
stay column-aligned when both are placed at the same width.

Run:  python analysis/make_team_tenure.py [--wide | --split] [--seed seed_456]
Out:  paper_assets/timelines/comparison/
        team_tenure[_wide].{pdf,png,svg}   both lanes stacked
        team_tenure_{a,b}.{pdf,png,svg}    one lane each, for \subcaption
"""
import json
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import patheffects as pe
from matplotlib.ticker import MultipleLocator

from paths import ASSETS, RUNS            # noqa: F401 (adds siblings to path)
from make_final_figures import (PAIR_C, AGENT_C, INK, MUTED, RC,
                                shade_chambers, chamber_spans_from_steps)
from make_directive_timelines import load_run
from make_results import MILESTONE_TRACK
from replay_hebbian_terms import load_inputs
from prototype_three_factor_rule import replay_three_factor

COOP = {"ch2_anvils", "ch3_switches", "ch4_combat", "ch5_boss"}
PAIRS = [(0, 1), (0, 2), (1, 2)]
TIMEOUT = 60                              # orchestrator node_timeout_steps

# ── design tokens (plot layer only; nothing here touches the data) ──────
# team size on one blue ramp, pre-mixed over white so the bands do not bleed
# through: 0 mates = independent, 1 = paired, 2 = full team
TEAM_C = {0: "#cfdbe9", 1: "#7b9dc3", 2: "#0F4D92"}
# lane (a) draws its events on top of the dark full-team blocks, where a
# white halo keeps them legible; it is invisible over the pale ones
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]

ORCH = RUNS / "orchestrator/new_exp_0_gemma_orch_villager_advisory"
G3F = RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f"
OUT = ASSETS / "timelines" / "comparison"
DASH = "–"


def aid(name):
    return int(re.search(r"(\d+)", name).group(1))


def events(fm):
    """joints [(step, pair)], kills [(step, killer)] from milestone_events."""
    by, step, kills = defaultdict(set), {}, []
    for e in fm["milestone_events"]:
        mid = e["milestone_id"]
        if MILESTONE_TRACK.get(mid) not in COOP:
            continue
        by[e["lua_step"]].add(aid(e["contributor"]))
        step[e["lua_step"]] = min(step.get(e["lua_step"], 10 ** 9), e["step"])
        if mid == "m21_first_mob_kill":
            kills.append((e["step"], aid(e["contributor"])))
    joints = []
    for k, ags in by.items():
        if len(ags) >= 2:
            for q in PAIRS:
                if set(q) <= ags:
                    joints.append((step[k], q))
    return sorted(joints), sorted(kills)


def orch_lane_data(run):
    """Per-step task per agent + free reasons, from the assignment ledger."""
    T = run["_ep_bounds"][-1][1]
    offs = {e + 1: b[0] for e, b in enumerate(run["_ep_bounds"])}
    ev, reason_at = defaultdict(list), {}
    for line in open(run["_run_dir"] / "orchestrator/assignments.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        t = offs[r["episode"]] + r["t"]
        ev[t].append((aid(r["agent"]), r["task_id"], r["reason"]))
        if not r["reason"].startswith("allocate"):
            reason_at[(aid(r["agent"]), t)] = r["reason"].replace("freed_", "")
    task = np.empty((T, 3), dtype=object)
    live = {}
    for t in range(T):
        for a, tid, rs in ev.get(t, []):
            live[a] = tid if rs.startswith("allocate") else None
        for a in range(3):
            task[t, a] = live.get(a)
    trio = np.array([task[t, 0] is not None and task[t, 0] == task[t, 1]
                     == task[t, 2] for t in range(T)])
    same = {q: np.array([task[t, q[0]] is not None
                         and task[t, q[0]] == task[t, q[1]] for t in range(T)])
            for q in PAIRS}
    busy = {q: np.array([task[t, q[0]] is not None and task[t, q[1]] is not None
                         and task[t, q[0]] != task[t, q[1]] for t in range(T)])
            for q in PAIRS}
    excl = {q: same[q] & ~trio for q in PAIRS}
    # exact team instances: (agents, task_id, t0, t1) contiguous in time
    open_t, teams = {}, []
    for t in range(T + 1):
        now = {}
        if t < T:
            for tid in {x for x in task[t] if x is not None}:
                grp = tuple(a for a in range(3) if task[t, a] == tid)
                if len(grp) >= 2:
                    now[(grp, tid)] = True
        for key in list(open_t):
            if key not in now:
                teams.append((key[0], key[1], open_t.pop(key), t - 1))
        for key in now:
            open_t.setdefault(key, t)
    return dict(T=T, task=task, trio=trio, excl=excl, busy=busy,
                reason_at=reason_at, teams=sorted(teams, key=lambda x: x[2]))


def blocks(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    return [(int(idx[a]), int(idx[b])) for a, b in
            zip(np.r_[0, brk + 1], np.r_[brk, idx.size - 1])]


def end_reason(d, agents, last):
    rs = [d["reason_at"].get((a, last + 1)) for a in agents]
    rs = [r for r in rs if r]
    for pick in ("success", "timeout"):
        if pick in rs:
            return pick
    return rs[0] if rs else None


# ── lane furniture ─────────────────────────────────────────────────────
def _chamber_labels(ax, spans, y, xmax, fs, min_len=90):
    for lab, lo, hi in spans:
        if hi - lo < min_len:
            continue
        cx = (lo + hi) / 2
        if 0 < cx < xmax:
            ax.text(cx, y, lab, fontsize=fs["chamber"], color=MUTED,
                    ha="center", va="bottom", zorder=2)


def _finish_x(ax, xmax, fs):
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.set_xlabel("environment step (cumulative)", fontsize=fs["axis"],
                  labelpad=3)


# ── lane (a): central orchestrator ─────────────────────────────────────
def lane_orch(ax, run, spans, joints, kills, fs, xmax):
    """One row per agent; a bar is the task it holds, darker = bigger team."""
    d = orch_lane_data(run)
    task, T = d["task"], d["T"]
    shade_chambers(ax, spans)
    rows = {0: 2, 1: 1, 2: 0}
    bh, gap = fs["bar"], 2.0          # a hairline of white marks each seam
    for a in range(3):
        keys = []
        for t in range(T):
            tid = task[t, a]
            if tid is None:
                keys.append(None)
                continue
            mates = tuple(b for b in range(3) if b != a and task[t, b] == tid)
            keys.append((tid, mates))
        t0 = 0
        for t in range(1, T + 1):
            if t == T or keys[t] != keys[t0]:
                if keys[t0] is not None:
                    col = TEAM_C[len(keys[t0][1])]
                    g = gap if t - t0 > 4 * gap else 0
                    ax.broken_barh([(t0 + g, t - t0 - 2 * g)],
                                   (rows[a] - bh, 2 * bh), facecolors=col,
                                   linewidth=0, zorder=3)
                t0 = t
    n_to = n_ok = 0
    for agents, tid, lo, hi in d["teams"]:
        r = end_reason(d, agents, hi)
        n_to += r == "timeout"
        n_ok += r == "success"
    # events in the family grammar, on the row of each credited agent:
    # open circle in the pair colour = joint milestone, filled star in the
    # killer's colour = mob kill
    for t, q in joints:
        for a in q:
            ax.scatter(t, rows[a], marker="o", s=fs["joint"],
                       facecolor="white", edgecolor=PAIR_C[q],
                       linewidths=1.5, zorder=8, path_effects=HALO)
    for t, killer in kills:
        ax.scatter(t, rows[killer], marker="*", s=fs["star"],
                   facecolor=AGENT_C[killer], edgecolor="white",
                   linewidths=0.6, zorder=8, path_effects=HALO)
    _chamber_labels(ax, spans, -0.98, xmax, fs)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["a0", "a1", "a2"])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-1.08, 3.55)
    ax.set_ylabel("orchestrator\nassignments", fontsize=fs["axis"],
                  labelpad=4)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(facecolor=TEAM_C[k], label=lab) for k, lab in
                       ((0, "independent"), (1, "paired"), (2, "full team"))],
              fontsize=fs["leg"], ncol=3, loc="upper left", frameon=False,
              borderaxespad=0.3, handlelength=1.2, handleheight=0.8,
              handletextpad=0.5, columnspacing=1.4)
    lens = [hi - lo + 1 for *_, lo, hi in d["teams"]]
    return len(lens), int(np.median(lens)) if lens else 0, n_to, n_ok


# ── lane (b): Hebbian bonds ────────────────────────────────────────────
def lane_bonds(ax, run_dir, spans, joints, kills, fs, xmax):
    """Family grammar: pair-coloured Wbar, open circle = joint milestone in
    the pair colour, filled star = mob kill in the killer's colour."""
    Wr = replay_three_factor(load_inputs(run_dir))["W"]
    shade_chambers(ax, spans)
    steps = np.arange(len(Wr))
    wbar = {q: (Wr[:, q[0], q[1]] + Wr[:, q[1], q[0]]) / 2 for q in PAIRS}
    curves = [ax.plot(steps, wbar[q], color=PAIR_C[q], lw=fs["line"],
                      label="$\\bar{W}$(a%d%sa%d)" % (q[0], DASH, q[1]),
                      solid_capstyle="round", zorder=4)[0] for q in PAIRS]
    for t, q in joints:
        ax.scatter(t, wbar[q][min(t, len(Wr) - 1)], marker="o", s=fs["joint"],
                   facecolor="white", edgecolor=PAIR_C[q], linewidths=1.5,
                   zorder=6)
    for t, killer in kills:
        ys = [wbar[q][min(t, len(Wr) - 1)] for q in PAIRS if killer in q]
        ax.scatter(t, max(ys), marker="*", s=fs["star"],
                   facecolor=AGENT_C[killer], edgecolor="white",
                   linewidths=0.6, zorder=6)
    ymax = float(np.nanmax(Wr))
    ax.set_yticks(np.arange(0.2, 0.81, 0.2))
    ax.set_ylim(0.0, ymax * 1.28)
    ax.set_ylabel("bond strength", fontsize=fs["axis"], labelpad=4)
    _chamber_labels(ax, spans, 0.02, xmax, fs)
    ax.legend(handles=curves, fontsize=fs["leg"], ncol=3, loc="upper left",
              frameon=False, borderaxespad=0.3, handlelength=1.6,
              handletextpad=0.5, columnspacing=1.4)
    return wbar


# ── figure assembly ────────────────────────────────────────────────────
# Printed point sizes: the canvas is the ICLR text width, so every figure here
# renders 1:1 at \linewidth and these are the sizes the reader actually sees.
FS_PRINT = dict(axis=8.5, tick=8.0, leg=7.8, chamber=6.8,
                joint=32, star=80, line=1.3, bar=0.30)
FIG_W = 5.5
LEFT, RIGHT = 0.120, 0.995


def _inputs(seed):
    orun = load_run(ORCH / seed)
    hrun = load_run(G3F / seed)
    return dict(
        orun=orun, hrun=hrun,
        oj_ok=events(orun), hj_hk=events(hrun),
        ospans=chamber_spans_from_steps({"run": ORCH / seed}, orun),
        hspans=chamber_spans_from_steps({"run": G3F / seed}, hrun),
        xmax=max(orun["_ep_bounds"][-1][1], hrun["_ep_bounds"][-1][1]),
        seed=seed)


def _rc(fs):
    """Family fonts on plain black axes, in the figures4papers manner."""
    rc = dict(RC)
    rc.update({"mathtext.fontset": "custom",
               "mathtext.rm": "Times New Roman",
               "mathtext.it": "Times New Roman:italic",
               "mathtext.bf": "Times New Roman:bold",
               "font.size": fs["tick"], "text.color": INK,
               "axes.edgecolor": INK, "axes.labelcolor": INK,
               "axes.linewidth": 0.8, "axes.spines.top": False,
               "axes.spines.right": False,
               "xtick.color": INK, "ytick.color": INK,
               "xtick.labelsize": fs["tick"], "ytick.labelsize": fs["tick"],
               "xtick.major.size": 3, "xtick.major.width": 0.8,
               "ytick.major.size": 3, "ytick.major.width": 0.8,
               "xtick.direction": "out", "ytick.direction": "out",
               "legend.frameon": False})
    return rc


def _save(fig, stem, tight):
    OUT.mkdir(parents=True, exist_ok=True)
    extra = dict(bbox_inches="tight", pad_inches=0.012) if tight else {}
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / (stem + "." + ext), facecolor="white",
                    dpi=600 if ext == "png" else None, **extra)
    plt.close(fig)
    return stem


def _report(stem, seed, stats, wbar, oj, ok, hj, hk):
    n_teams, med, n_to, n_ok = stats
    lead = max(PAIRS, key=lambda q: float(np.mean(wbar[q])))
    share = np.mean([max(PAIRS, key=lambda q: wbar[q][t]) == lead
                     for t in range(len(wbar[lead]))])
    print("  orchestrator %s: %d teams, median %d steps, %d timeout / %d "
          "success | joints %s kills %s" % (seed, n_teams, med, n_to, n_ok,
                                            oj, ok))
    print("  hebbian      %s: joints %s kills %s" % (seed, hj, hk))
    print("  caption nums: %d teams, median %d steps, %d/%d dissolved at the "
          "%d-step timeout, %d at success; a%d-a%d strongest for %.0f%%"
          % (n_teams, med, n_to, n_teams, TIMEOUT, n_ok, lead[0], lead[1],
             100 * share))
    print("  wrote " + str(OUT) + "/" + stem + ".(pdf|png|svg)")


def build(wide=False, seed="seed_456"):
    """Both lanes stacked in one titleless figure."""
    d = _inputs(seed)
    fs = dict(FS_PRINT)
    H = 2.40 if wide else 2.90
    with plt.rc_context(_rc(fs)):
        fig = plt.figure(figsize=(FIG_W, H))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.44, 0.56], hspace=0.22,
                              left=LEFT, right=RIGHT, top=0.985,
                              bottom=0.175 if wide else 0.150)
        axO = fig.add_subplot(gs[0])
        axH = fig.add_subplot(gs[1], sharex=axO)
        stats = lane_orch(axO, d["orun"], d["ospans"], *d["oj_ok"], fs,
                          d["xmax"])
        wbar = lane_bonds(axH, G3F / seed, d["hspans"], *d["hj_hk"], fs,
                          d["xmax"])
        axO.set_xlim(0, d["xmax"])
        axO.tick_params(axis="x", length=0, labelbottom=False)
        axO.spines["bottom"].set_visible(False)
        _finish_x(axH, d["xmax"], fs)
        stem = _save(fig, "team_tenure_wide" if wide else "team_tenure", True)
    _report(stem, seed, stats, wbar, *d["oj_ok"], *d["hj_hk"])


def build_split(seed="seed_456"):
    """The same two lanes as standalone figures, one per subcaption.

    Fixed margins and no tight bbox, so the two files share an identical plot
    box and stay column-aligned when both are placed at the same width.
    """
    d = _inputs(seed)
    fs = dict(FS_PRINT)
    with plt.rc_context(_rc(fs)):
        fig = plt.figure(figsize=(FIG_W, 1.42))
        axO = fig.add_axes([LEFT, 0.275, RIGHT - LEFT, 0.705])
        stats = lane_orch(axO, d["orun"], d["ospans"], *d["oj_ok"], fs,
                          d["xmax"])
        _finish_x(axO, d["xmax"], fs)
        _save(fig, "team_tenure_a", False)

        fig = plt.figure(figsize=(FIG_W, 1.68))
        axH = fig.add_axes([LEFT, 0.235, RIGHT - LEFT, 0.745])
        wbar = lane_bonds(axH, G3F / seed, d["hspans"], *d["hj_hk"], fs,
                          d["xmax"])
        _finish_x(axH, d["xmax"], fs)
        _save(fig, "team_tenure_b", False)
    _report("team_tenure_{a,b}", seed, stats, wbar, *d["oj_ok"], *d["hj_hk"])


if __name__ == "__main__":
    sd = "seed_456"
    if "--seed" in sys.argv:
        sd = sys.argv[sys.argv.index("--seed") + 1]
    if "--split" in sys.argv:
        build_split(seed=sd)
    else:
        build(wide="--wide" in sys.argv, seed=sd)
