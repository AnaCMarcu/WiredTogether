#!/usr/bin/env python3
r"""make_social_frontier_fig.py - the social-interval frontier, labelled legibly.

make_pareto_social_fig.py emits this figure among twenty others, but on a
LINEAR x-axis with alternating fixed label offsets. Social-module compute spans
4.8e13 to 2.2e16 FLOPs/episode - nearly three decades - so on a linear axis
five of the seven intervals pile into the leftmost 7% of the plot and their
labels collide. This redraws the one figure for the paper:

  * symlog x, which gives the three-decade spread the room it needs while
    still admitting the no-social-module arm at x=0 as an ordinary point;
  * labels placed by MEASURED bounding box, trying candidate offsets until one
    lands clear - a rule based on the gap in x alone is not enough, because
    the two peak points sit 0.15 dex apart but 1.3 points apart in y;
  * seed counts kept out of the plot; they belong in the caption.

Reads pareto_social.csv, which make_pareto_social_fig.py writes, so the two
figures can never disagree about the numbers.

Usage:
  python analysis/make_social_frontier_fig.py paper_assets/pareto_social_3f
  python analysis/make_social_frontier_fig.py <dir> --metric coop --drop 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = "#d95f0e"   # hebbian house colour (matches make_scaling_fig)
BASE = "#2c7fb8"     # baseline house colour
INK = "#666666"   # label grey, as in the original sweep figure

METRICS = {
    "milestone": ("total_pct_mean", "total_pct_sd", "Milestone completion [%]"),
    "coop": ("coop_pct_mean", "coop_pct_sd", "Cooperative milestones [%]"),
    "return": ("task_return_mean", "task_return_sd", "Task return"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("assets_dir", type=Path)
    ap.add_argument("--metric", default="milestone", choices=sorted(METRICS))
    ap.add_argument("--drop", type=int, action="append", default=[],
                    metavar="INTERVAL",
                    help="omit this interval from the figure (repeatable)")
    ap.add_argument("--err", action="store_true",
                    help="draw +-1 SD bars (large at n<=3; off by default)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    mean_k, sd_k, ylabel = METRICS[args.metric]
    rows = list(csv.DictReader(open(args.assets_dir / "pareto_social.csv")))

    pts = []
    for r in rows:
        iv = int(r["interval"]) if r["interval"] else 0
        if iv in args.drop:
            print("  dropped interval %d" % iv, file=sys.stderr)
            continue
        pts.append({"iv": iv,
                    "x": float(r["social_flops_mean"]),
                    "y": float(r[mean_k]),
                    "sd": float(r[sd_k]),
                    "n": int(r["n_seeds"])})
    pts.sort(key=lambda p: p["x"])
    if not pts:
        print("no points left to plot", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    # symlog, not log: the no-social-module arm sits at exactly x=0, and a
    # linear threshold just under the cheapest sweep point keeps it on the
    # same axis as the rest instead of demoting it to a reference line.
    xs_pos = [p["x"] for p in pts if p["x"] > 0]
    ax.set_xscale("symlog", linthresh=min(xs_pos) * 0.6, linscale=0.45)

    ax.plot([p["x"] for p in pts], [p["y"] for p in pts],
            "-", color=SERIES, lw=1.5, zorder=2)
    if args.err:
        ax.errorbar([p["x"] for p in pts], [p["y"] for p in pts],
                    yerr=[p["sd"] for p in pts], fmt="none", ecolor=SERIES,
                    elinewidth=1.0, capsize=3, alpha=0.5, zorder=2)
    for p in pts:
        is_base = p["iv"] == 0
        ax.plot(p["x"], p["y"], "s" if is_base else "o",
                ms=8 if is_base else 7.5,
                color=BASE if is_base else SERIES,
                mfc="none", mew=1.6, zorder=3)

    # Labels by measured bounding box: draw, measure, and keep the first
    # candidate offset that clears everything already placed.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = []
    CANDIDATES = [(0, 12, "center", "bottom"), (0, -14, "center", "top"),
                  (26, 6, "left", "bottom"), (-26, 6, "right", "bottom"),
                  (30, -10, "left", "top"), (-30, -10, "right", "top")]
    for p in pts:
        tag = r"$\Delta$=%d" % p["iv"]
        colour = BASE if p["iv"] == 0 else INK
        for dx, dy, ha, va in CANDIDATES:
            ann = ax.annotate(tag, (p["x"], p["y"]),
                              textcoords="offset points", xytext=(dx, dy),
                              ha=ha, va=va, fontsize=8.5, color=colour)
            fig.canvas.draw()
            bb = ann.get_window_extent(rend).expanded(1.06, 1.12)
            if not any(bb.overlaps(q) for q in placed):
                placed.append(bb)
                break
            ann.remove()
        else:                       # every candidate blocked - keep it anyway
            ann = ax.annotate(tag, (p["x"], p["y"]),
                              textcoords="offset points", xytext=(0, 12),
                              ha="center", va="bottom", fontsize=8.5,
                              color=colour)
            placed.append(ann.get_window_extent(rend))

    ax.set_xlabel("Compute  (FLOPs / episode, social module)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)

    ys = [p["y"] for p in pts]
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - 0.20 * (hi - lo), hi + 0.28 * (hi - lo))

    handles = [
        plt.Line2D([], [], color=SERIES, marker="o", ms=7.5, lw=1.5,
                   mfc="none", mew=1.6, label="+Hebbian"),
        plt.Line2D([], [], color=BASE, marker="s", ms=8, lw=0,
                   mfc="none", mew=1.6,
                   label=r"no Hebbian ($\Delta$ = 0)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right",
              handlelength=2.0)

    fig.tight_layout()
    out = args.out or (args.assets_dir / ("social_frontier_%s.png" % args.metric))
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote %s (+ .pdf)" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
