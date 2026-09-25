#!/usr/bin/env python3
r"""make_transplant_bonds.py - every bond of the six-agent transplant runs.

make_counterfactual_n6.py's transplant figures draw the three transplanted
pairs and one or three named cross pairs - enough to carry a storyboard, not
enough to see the whole graph. This figure draws ALL 15 symmetrised bonds
Wbar_ij = (W_ij + W_ji)/2 for every seed of a group, one row per seed, with
the recorded 6 x 6 bond matrix at the end of each episode beside the lines.

  lines     the three seeded pairs in the family's pair colours (whichever
            pairs start above 0.2 in the first snapshot - a0-a1, a2-a3,
            a4-a5 in both groups), the twelve cross pairs in thin grey; the
            cross pair that is strongest at the end of each episode is
            named at that end
  circles   joint milestones on the line of the pair credited (the Ch3
            switch/door pairs in these runs; grey when a cross pair)
  matrices  Wbar at the last snapshot of each episode, seeded cells outlined,
            so "the pairs came back" can be read as three cells on the
            diagonal blocks lighting up again

Bond source is the run's 50-step ``graph_snapshots`` (reward-modulated
rule, no replay). Chamber labels come from the majority chamber in
step_log.csv, as in make_counterfactual_n6.

Usage:  python analysis/make_transplant_bonds.py [--group transplant|shuffled]
                                                 [--seeds 42,123]
Out:    paper_assets/timelines/transplant/all_bonds_<group>.{pdf,png,svg}
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from paths import ASSETS, group  # noqa: E402  (also puts siblings on sys.path)

import make_final_figures as mff  # noqa: E402
from make_counterfactual_n6 import (  # noqa: E402
    AGENT_C, CROSS, PAIR_C, chamber_spans, events, load_bond_series,
)
from make_directive_timelines import load_run  # noqa: E402

N = 6
OUT = ASSETS / "timelines" / "transplant"
GROUPS = {"transplant": "expB_merged_transplant",
          "shuffled": "expB_merged_shuffled"}
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
FS = dict(axis=8.0, tick=7.0, seed=9.0, lab=6.6, chamber=7.0, hm=6.4,
          title=8.4)
VMAX = 0.36


def wbar(W, q):
    return (W[:, q[0], q[1]] + W[:, q[1], q[0]]) / 2


def seeded_pairs(W):
    return [q for q in PAIRS if wbar(W, q)[0] > 0.2]


def draw_row(fig, axL, axHs, run, W, seeded, joints, spans, seed):
    T = len(W)
    steps = np.arange(T)
    bounds = run["_ep_bounds"]
    xmax = bounds[-1][1]

    for q in PAIRS:
        if q in seeded:
            continue
        axL.plot(steps, wbar(W, q), color=CROSS, lw=0.8, alpha=0.75, zorder=2)
    for q in seeded:
        axL.plot(steps, wbar(W, q), color=PAIR_C[q], lw=1.9,
                 solid_capstyle="round", zorder=4,
                 label="$\\bar{W}$(a%d–a%d)" % q)
    # the strongest cross pair at each episode end, named at that end. Naming
    # it at its peak put the label on the "ep k" caption whenever the peak
    # was near the top of the axis, which is exactly when it matters.
    for e, (s0, s1) in enumerate(bounds):
        s1 = min(s1, T)
        best = max((q for q in PAIRS if q not in seeded),
                   key=lambda q: wbar(W, q)[s1 - 1])
        y_end = wbar(W, best)[s1 - 1]
        # skip weak ends, and episodes too short to hold a label without
        # sitting on the "ep k" caption (shuffled seed 456's ep2 is 62 steps)
        if y_end < 0.15 or s1 - s0 < 120:
            continue
        axL.annotate("a%d–a%d" % best, (s1 - 1, y_end), xytext=(-3, 2),
                     textcoords="offset points", fontsize=FS["lab"],
                     color="#5b6472", ha="right", va="bottom", zorder=6,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.6,
                               alpha=0.8))
    for t, q in joints:
        t = min(t, T - 1)
        axL.scatter(t, wbar(W, q)[t], marker="o", s=30, facecolor="white",
                    edgecolor=PAIR_C[q] if q in seeded else CROSS,
                    linewidths=1.3, zorder=7)
    for e, (o, _) in enumerate(bounds):
        if e:
            axL.axvline(o, color=mff.INK, lw=0.8, ls=(0, (4, 3)), alpha=0.35)
    for _lab, lo, hi in spans:
        if hi - lo > 40 and lo > 0:
            axL.axvline(lo, color="#b3bcc7", lw=0.6, ls=(0, (1.2, 2.4)), zorder=0)
    mff.FS["chamber"] = FS["chamber"]
    mff.chamber_labels(axL, spans, 0.005, (0, xmax))
    for e, (s0, s1) in enumerate(bounds):
        axL.text((s0 + min(s1, T)) / 2, VMAX * 0.985, f"ep {e + 1}",
                 fontsize=FS["lab"], color=mff.MUTED, ha="center", va="top")
    axL.set_xlim(-xmax * 0.005, xmax * 1.005)
    axL.set_ylim(0.0, VMAX)
    axL.set_yticks([0.0, 0.1, 0.2, 0.3])
    axL.grid(axis="y", color="#eef1f4", lw=0.8)
    axL.set_axisbelow(True)
    axL.tick_params(labelsize=FS["tick"])
    axL.set_ylabel(f"seed {seed}\nbond strength", fontsize=FS["axis"],
                   color=mff.INK)
    for s_ in ("top", "right"):
        axL.spines[s_].set_visible(False)

    # end-of-episode matrices
    for e, ((s0, s1), ax) in enumerate(zip(bounds, axHs)):
        k = min(s1, T) - 1
        M = (W[k] + W[k].T) / 2
        np.fill_diagonal(M, np.nan)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=VMAX, interpolation="nearest")
        for i, j in seeded:
            for a, b in ((i, j), (j, i)):
                ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False,
                                       edgecolor=PAIR_C[(i, j)], lw=1.1))
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels([f"a{a}" for a in range(N)], fontsize=FS["hm"] - 0.6)
        ax.set_yticklabels([f"a{a}" for a in range(N)], fontsize=FS["hm"] - 0.6)
        for tick, a in zip(ax.get_xticklabels(), range(N)):
            tick.set_color(AGENT_C[a])
        for tick, a in zip(ax.get_yticklabels(), range(N)):
            tick.set_color(AGENT_C[a])
        ax.tick_params(length=0, pad=1.2)
        ax.set_title(f"end of ep {e + 1}", fontsize=FS["hm"], color=mff.INK,
                     pad=2.5)
        for s_ in ax.spines.values():
            s_.set_color("#c3c9d0")
            s_.set_linewidth(0.6)


def report(seed, run, W, seeded):
    parts = []
    for e, (s0, s1) in enumerate(run["_ep_bounds"]):
        k = min(s1, len(W)) - 1
        vals = " ".join(f"{wbar(W, q)[k]:.2f}" for q in seeded)
        best = max((q for q in PAIRS if q not in seeded), key=lambda q: wbar(W, q)[k])
        parts.append(f"ep{e + 1} [{vals}] cross a{best[0]}-a{best[1]} "
                     f"{wbar(W, best)[k]:.2f}")
    print(f"  seed {seed:<4}  " + " | ".join(parts))


def build(group_name: str, seeds: list[int] | None, dpi: int) -> None:
    root = group("pair_bonding") / GROUPS[group_name]
    found = sorted(int(p.name.split("_")[1]) for p in root.glob("seed_*")
                   if (p / "final_metrics.json").exists())
    seeds = [s for s in (seeds or found) if s in found]
    n = len(seeds)
    row_h, top, bot = 1.62, 0.52, 0.42
    fig_h = top + n * row_h + bot
    print(f"{group_name}: seeds {seeds}")
    with plt.rc_context(mff.RC):
        fig = plt.figure(figsize=(7.2, fig_h))
        for r, seed in enumerate(seeds):
            run_dir = root / f"seed_{seed}"
            run = load_run(run_dir)
            arm = dict(run=run_dir, lane="bonds_snap")
            W = load_bond_series(arm)
            seeded = seeded_pairs(W)
            joints, _kills = events(run)
            spans = chamber_spans(arm, run)
            y1 = 1 - (top + r * row_h) / fig_h
            y0 = 1 - (top + (r + 1) * row_h - 0.30) / fig_h
            axL = fig.add_axes([0.085, y0, 0.585, y1 - y0])
            hm_w = 0.085
            axHs = [fig.add_axes([0.705 + k * (hm_w + 0.012),
                                  y0 + (y1 - y0 - hm_w * 7.2 / fig_h) / 2,
                                  hm_w, hm_w * 7.2 / fig_h]) for k in range(3)]
            draw_row(fig, axL, axHs, run, W, seeded, joints, spans, seed)
            if r == 0:
                # in the header strip, not on the axes: the first row's
                # lines start at 0.27 and the legend sat on top of them
                handles, labels = axL.get_legend_handles_labels()
                fig.legend(handles, labels, fontsize=FS["lab"] + 0.6, ncol=3,
                           frameon=False, loc="upper left",
                           bbox_to_anchor=(0.085, 1 - 0.30 / fig_h),
                           handlelength=1.6, columnspacing=1.2,
                           borderaxespad=0.0)
            if r < n - 1:
                axL.set_xticklabels([])
            else:
                axL.set_xlabel("environment step (cumulative)",
                               fontsize=FS["axis"], labelpad=2)
            report(seed, run, W, seeded)
        title = ("Pair transplant: all 15 bonds" if group_name == "transplant"
                 else "Shuffled transplant: all 15 bonds")
        fig.text(0.085, 1 - 0.13 / fig_h, title, fontsize=FS["title"],
                 color=mff.INK, fontweight="bold", va="top")
        fig.text(0.985, 1 - 0.13 / fig_h,
                 "seeded pairs in colour · cross pairs grey · "
                 "○ joint milestone · matrices: $\\bar{W}$ at episode end",
                 fontsize=FS["lab"], color=mff.MUTED, va="top", ha="right")
        OUT.mkdir(parents=True, exist_ok=True)
        stem = f"all_bonds_{group_name}"
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi if ext == "png" else None,
                        facecolor="white")
        plt.close(fig)
    print(f"wrote {OUT.name}/{stem}.(pdf|png|svg)   7.20 x {fig_h:.2f} in")


def main() -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="All 15 bonds of the transplant runs.")
    ap.add_argument("--group", choices=sorted(GROUPS), default="transplant")
    ap.add_argument("--seeds", default=None, help="comma-separated; default all")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    build(args.group, seeds, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
