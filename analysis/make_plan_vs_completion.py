"""When does cooperation actually start? Orchestrator vs Hebbian 2.0.

Two stacked timelines over the same three seeds, so the x-axis is directly
comparable.

  top    : the central orchestrator. Pale bars mark the steps at which an
           agent was holding a task that targets a cooperative milestone --
           what the plan asked for. Dots are cooperative milestone
           completions, one per credited agent.
  bottom : Gemma-4B + Hebbian 2.0. No plan, so no bars; same dots.

A triangle marks the first cooperative completion of each run. The bonds
reach cooperation earlier and with far less spread across seeds, while the
orchestrator's planned cooperative work runs the whole episode without
closing.

Run:  python analysis/make_plan_vs_completion.py [--wide]
Out:  paper_assets/timelines/comparison/plan_vs_completion[_wide].{pdf,png,svg}
"""
import json
import re
import statistics as stats
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paths import ASSETS, RUNS            # noqa: F401 (adds siblings to path)
from make_results import MILESTONE_TRACK

COOP = {"ch2_anvils", "ch3_switches", "ch4_combat", "ch5_boss"}
AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
PLAN_C = "#d5dae1"
INK, MUTED, GRID = "#31373f", "#8b95a1", "#eceff3"
FIRST_C = "#b03a3a"

ORCH = RUNS / "orchestrator/new_exp_0_gemma_orch_villager_advisory"
G3F = RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f"
OUT = ASSETS / "timelines" / "comparison"
SEEDS = ["seed_42", "seed_123", "seed_456"]


def aid(name):
    return int(re.search(r"(\d+)", name).group(1))


def episode_offsets(fm):
    offs, c = [], 0
    for L in fm.get("episode_lengths") or []:
        offs.append(c)
        c += L
    return offs, c


def coop_events(run_dir):
    """[(step, [agents])] cooperative-track fires, merged per lua_step."""
    fm = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    by, step = defaultdict(set), {}
    for e in fm["milestone_events"]:
        if MILESTONE_TRACK.get(e["milestone_id"]) not in COOP:
            continue
        by[e["lua_step"]].add(aid(e["contributor"]))
        step[e["lua_step"]] = min(step.get(e["lua_step"], 10 ** 9), e["step"])
    return fm, sorted((step[k], sorted(v)) for k, v in by.items())


def planned_mask(run_dir, fm):
    """Steps at which an agent held a task targeting a coop milestone."""
    offs, T = episode_offsets(fm)
    coop_tasks = set()
    for line in open(run_dir / "orchestrator/dag.jsonl", encoding="utf-8"):
        for t in json.loads(line).get("tasks", []):
            if any(MILESTONE_TRACK.get(m) in COOP
                   for m in (t.get("milestones") or [])):
                coop_tasks.add(t["id"])
    ev = defaultdict(list)
    for line in open(run_dir / "orchestrator/assignments.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ev[r["t"] + offs[r["episode"] - 1]].append(
            (aid(r["agent"]), r["task_id"], r["reason"]))
    live, mask = {}, np.zeros(T, dtype=bool)
    for t in range(T):
        for a, tid, reason in ev.get(t, []):
            live[a] = tid if reason.startswith("allocate") else None
        mask[t] = any(tid in coop_tasks for tid in live.values() if tid)
    return mask, T, offs


def blocks(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    return [(idx[a], idx[b] - idx[a] + 1)
            for a, b in zip(np.r_[0, brk + 1], np.r_[brk, idx.size - 1])]


def collect():
    rows = {"orch": [], "heb": []}
    for s in SEEDS:
        fm, ev = coop_events(ORCH / s)
        mask, T, offs = planned_mask(ORCH / s, fm)
        rows["orch"].append(dict(seed=s, ev=ev, mask=mask, T=T, offs=offs))
        fm, ev = coop_events(G3F / s)
        offs, T = episode_offsets(fm)
        rows["heb"].append(dict(seed=s, ev=ev, mask=None, T=T, offs=offs))
    return rows


def draw(ax, rows, fs, xmax, show_plan):  # xmax scales label offset
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        for o in r["offs"][1:]:
            ax.plot([o, o], [y - 0.34, y + 0.34], color="#d3d8de", lw=0.7,
                    zorder=1)
        if show_plan and r["mask"] is not None:
            for x0, w in blocks(r["mask"]):
                ax.add_patch(plt.Rectangle((x0, y - 0.26), w, 0.52,
                                           facecolor=PLAN_C, lw=0, zorder=1))
        for st, ags in r["ev"]:
            dys = np.linspace(-0.16, 0.16, len(ags)) if len(ags) > 1 else [0.0]
            for a, dy in zip(ags, dys):
                ax.scatter(st, y + dy, s=fs["dot"], facecolor=AGENT_C[a],
                           edgecolor="white", lw=0.4, zorder=4)
        if r["ev"]:
            first = r["ev"][0][0]
            ax.scatter(first, y + 0.44, marker="v", s=fs["dot"] * 1.5,
                       color=FIRST_C, zorder=5, clip_on=False)
            ax.text(first + xmax * 0.008, y + 0.46, str(first),
                    fontsize=fs["small"], color=FIRST_C, ha="left",
                    va="center", clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["seed"].replace("seed_", "seed ")
                        for r in reversed(rows)], fontsize=fs["tick"])
    ax.set_ylim(-0.6, len(rows) - 0.15)
    ax.set_xlim(0, xmax)
    ax.tick_params(labelsize=fs["tick"], length=2.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def build(wide=False):
    rows = collect()
    xmax = max(r["T"] for arm in rows.values() for r in arm)
    firsts = {k: [r["ev"][0][0] for r in v if r["ev"]] for k, v in rows.items()}

    FIG_W = 11.0 if wide else 5.5
    fs = dict(axis=8.4 if wide else 6.4, tick=8.2 if wide else 6.2,
              title=11.0 if wide else 8.0, small=7.8 if wide else 5.9,
              dot=26 if wide else 15)
    H = 3.35 if wide else 2.65
    rc = {"font.family": "serif",
          "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
          "axes.edgecolor": "#9aa3ad", "axes.linewidth": 0.6,
          "svg.fonttype": "none", "pdf.fonttype": 42,
          "mathtext.fontset": "stix"}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(FIG_W, H))
        gs = fig.add_gridspec(2, 1, hspace=0.62, left=0.135, right=0.985,
                              top=0.845, bottom=0.20)
        axO = fig.add_subplot(gs[0])
        axH = fig.add_subplot(gs[1], sharex=axO)

        draw(axO, rows["orch"], fs, xmax, show_plan=True)
        draw(axH, rows["heb"], fs, xmax, show_plan=False)

        axO.set_title("Central orchestrator", fontsize=fs["title"], color=INK,
                      fontweight="bold", pad=13, loc="left")
        axO.text(1.0, 1.10, "first cooperation at step %d on average "
                 "(%s)" % (stats.mean(firsts["orch"]),
                           ", ".join(str(f) for f in firsts["orch"])),
                 transform=axO.transAxes, fontsize=fs["small"], color=MUTED,
                 ha="right", va="bottom")
        axH.set_title("Gemma-4B + Hebbian 2.0", fontsize=fs["title"],
                      color=INK, fontweight="bold", pad=13, loc="left")
        axH.text(1.0, 1.10, "first cooperation at step %d on average "
                 "(%s)" % (stats.mean(firsts["heb"]),
                           ", ".join(str(f) for f in firsts["heb"])),
                 transform=axH.transAxes, fontsize=fs["small"], color=MUTED,
                 ha="right", va="bottom")
        axH.set_xlabel("environment step (cumulative)", fontsize=fs["axis"],
                       labelpad=1.5)
        plt.setp(axO.get_xticklabels(), visible=False)

        handles = [Patch(facecolor=PLAN_C,
                         label="an agent is on a task for a cooperative milestone")]
        handles += [plt.Line2D([], [], marker="o", ls="none", color=AGENT_C[a],
                               ms=4.5, label="a%d credited" % a)
                    for a in range(3)]
        handles += [plt.Line2D([], [], marker="v", ls="none", color=FIRST_C,
                               ms=5, label="first cooperation")]
        fig.legend(handles=handles, loc="lower center", ncol=5,
                   fontsize=fs["small"], frameon=False,
                   bbox_to_anchor=(0.5, -0.012), handlelength=1.2,
                   columnspacing=1.2, handletextpad=0.45)

        OUT.mkdir(parents=True, exist_ok=True)
        stem = "plan_vs_completion_wide" if wide else "plan_vs_completion"
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / (stem + "." + ext),
                        dpi=300 if ext == "png" else None, facecolor="white")
        plt.close(fig)

    for k, name in (("orch", "orchestrator"), ("heb", "Hebbian 2.0")):
        f = firsts[k]
        n = sum(len(r["ev"]) for r in rows[k])
        print("  %-12s first cooperation %s  mean %4.0f  spread %4.0f  |  "
              "%d cooperative completions"
              % (name, f, stats.mean(f), max(f) - min(f), n))
    print("  wrote " + str(OUT) + "/" + stem + ".(pdf|png|svg)")


if __name__ == "__main__":
    build(wide="--wide" in sys.argv)
