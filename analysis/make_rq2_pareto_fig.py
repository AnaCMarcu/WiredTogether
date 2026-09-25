#!/usr/bin/env python3
r"""make_rq2_pareto_fig.py - the RQ2 co-firing Pareto: bond mass vs cooperation.

RQ2 (tab:cofire_main) varies the explicit co-firing cues
K subset of {comm, obs, imit} and reports, per condition, the mean bond
strength the rule produces and the cooperation it buys. The table states the
finding in numbers; this plots it, with mean bond strength on x and
cooperative milestone completion on y, so the trade-off is visible as a
frontier: only the two conditions with the LEAST bond mass are non-dominated,
and every condition that reaches a high mean W does so while completing fewer
cooperative milestones. Bond mass is treated as a cost axis here purely to
make the domination structure explicit -- W is an outcome of the rule, not a
budget the agents spend, and the figure makes no claim that lowering W is
itself an objective.

Numbers come from ``cofire_summary.csv`` -- the same file make_cofire_latex.py
formats into the paper's table -- so the figure and the table can never
disagree. The only quantity the table does not carry is the dispersion of the
bond mean, which is recomputed here through make_results' own
``end_of_episode_graph_stats`` and drawn only if its mean reproduces the CSV's
``bond_mean``; a mismatch means the runs and the assets have drifted apart, and
the x bars are dropped rather than describing a different number than the one
plotted.

Layout follows make_pareto_perception_fig.py (5.0 x 3.5 in, default fonts,
grey 8.5 pt point labels, framed legend), so the three Pareto figures read as
one family. Condition identity is carried by marker shape AND colour, never
colour alone, and every point is labelled with the table's own condition name.

The rank correlation between the two axes goes to the console, never into the
figure. tab:cofire_main already carries a rho column meaning something else
entirely -- per-pair bond strength against per-pair use of that cue, over the
6 directed pairs x 3 episodes -- and a second rho in the same subsection would
be read as that one. The frontier already states the trend, and over six
designed conditions the correlation adds no evidence the picture lacks.

Usage (from the repo root):
    python analysis/make_rq2_pareto_fig.py
    python analysis/make_rq2_pareto_fig.py --metric return --with-anchor
    python analysis/make_rq2_pareto_fig.py \
        --copy-to "C:/Users/marcu/Downloads/WIRED_TOGETHER_revised/figures"
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from statistics import fmean, pstdev

from paths import ASSETS, group  # noqa: E402  (also puts siblings on sys.path)

from cofire_table import ARMS, spearman  # noqa: E402

# The paper's RQ2 table drops the Experiment-2 apparatus from the label of the
# communication-only arm (there is no forced-comm row to disambiguate it
# from), so the figure follows the table.
PAPER_LABEL = {"Comm (choice)": "Comm"}
ANCHOR = "anchor"

#: y-axis choices: (mean column, sd column, axis label, filename stem).
METRICS = {
    "coop": ("coop_pct_mean", "coop_pct_std",
             "Cooperative milestones [%]", "pareto_bond_coop"),
    "milestone": ("allms_nocomm_pct_mean", "allms_nocomm_pct_std",
                  "Milestone completion [%]", "pareto_bond_milestone"),
    "return": ("task_nocomm_mean", "task_nocomm_std",
               "Task return", "pareto_bond_return"),
}

# Three visual groups, distinguished by shape as well as hue so the figure
# survives greyscale and colour-vision deficiency. Hues are the repo's house
# pair (make_scaling_fig / make_social_frontier_fig) plus ink for the
# reference arm.
STYLE = {
    "none": {"color": "#444444", "marker": "s", "ms": 7.0, "mew": 1.6,
             "label": "None (no explicit cue)"},
    "quiet": {"color": "#2c7fb8", "marker": "o", "ms": 7.5, "mew": 1.6,
              "label": "Comm-free cue (obs, imit)"},
    "comm": {"color": "#d95f0e", "marker": "^", "ms": 8.0, "mew": 1.6,
             "label": "Cue set includes comm"},
    "forced": {"color": "#666666", "marker": "D", "ms": 6.5, "mew": 1.6,
               "label": "Comm (forced), reference"},
}
INK = "#555555"


def style_of(cond: str, cues: list[str]) -> str:
    """Which visual group an arm belongs to, from its cue set."""
    if cond == ANCHOR:
        return "forced"
    if not cues:
        return "none"
    return "comm" if "comm" in cues else "quiet"


def bond_sd(runs_root: Path, exp: str) -> tuple[float, float] | None:
    """(mean, population sd) of end-of-episode mean bond strength, or None.

    Pooled over seeds x episodes, exactly as make_results pools it for
    ``bond_mean`` -- the mean is returned so the caller can check it against
    the CSV before trusting the sd.
    """
    try:
        import make_results as M
    except ImportError as e:                       # pragma: no cover
        print("  [warn] make_results unimportable (%s); no x bars" % e,
              file=sys.stderr)
        return None
    runs = M.load_runs(runs_root, exp)
    vals = [r["mean"] for run in runs for r in M.end_of_episode_graph_stats(run)]
    if not vals:
        return None
    return fmean(vals), pstdev(vals)


def collect(assets: Path, runs_root: Path, metric: str, with_anchor: bool):
    """One dict per arm: label, style key, x/y means and sds."""
    mean_k, sd_k, _, _ = METRICS[metric]
    pooled = {r["condition"]: r
              for r in csv.DictReader(open(assets / "cofire_summary.csv"))}

    pts = []
    for cond, exp, label, cues in ARMS:
        if cond not in pooled:
            print("  [warn] %s missing from cofire_summary.csv" % cond,
                  file=sys.stderr)
            continue
        if cond == ANCHOR and not with_anchor:
            continue
        p = pooled[cond]
        x = float(p["bond_mean"])
        got = bond_sd(runs_root, exp) if runs_root.is_dir() else None
        if got is None:
            xsd = None
        elif abs(got[0] - x) > 1e-3:
            # The runs on disk no longer produce the mean the assets record;
            # an sd from those runs would not describe the plotted point.
            print("  [warn] %s bond mean %.4f (runs) != %.4f (csv); no x bar"
                  % (cond, got[0], x), file=sys.stderr)
            xsd = None
        else:
            xsd = got[1]
        pts.append({"cond": cond,
                    "label": PAPER_LABEL.get(label, label),
                    "style": style_of(cond, cues),
                    "x": x, "xsd": xsd,
                    "y": float(p[mean_k]), "ysd": float(p[sd_k])})
    return pts


def frontier(pts):
    """Non-dominated points under (maximise y, minimise x), sorted by x."""
    out = []
    for p in pts:
        if not any(q is not p and q["x"] <= p["x"] and q["y"] >= p["y"]
                   and (q["x"] < p["x"] or q["y"] > p["y"]) for q in pts):
            out.append(p)
    return sorted(out, key=lambda p: p["x"])


def staircase(front, x_right):
    """The frontier as a step line, extended right at the last y."""
    xs, ys = [], []
    for i, p in enumerate(front):
        if i:
            xs.append(p["x"])
            ys.append(front[i - 1]["y"])
        xs.append(p["x"])
        ys.append(p["y"])
    xs.append(x_right)
    ys.append(front[-1]["y"])
    return xs, ys


def place_labels(fig, ax, pts, obstacles):
    """Point labels placed by MEASURED bounding box.

    Markers, the frontier line and the legend are seeded as obstacles, so no
    label lands on top of them or on a neighbour's label; with Comm+Obs and
    Comm+Obs+Imit 0.006 apart in x, a fixed offset rule is not enough. The
    error bars are deliberately NOT obstacles: they span up to eight
    percentage points, and avoiding them pushes a label so far from its
    marker that it stops reading as that point's name.

    Must run after tight_layout -- the placement is measured in canvas
    coordinates, which tight_layout moves.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    placed = list(obstacles)
    CANDIDATES = [(0, 10, "center", "bottom"), (0, -12, "center", "top"),
                  (10, 6, "left", "bottom"), (-10, 6, "right", "bottom"),
                  (10, -8, "left", "top"), (-10, -8, "right", "top"),
                  (13, 0, "left", "center"), (-13, 0, "right", "center"),
                  (0, 22, "center", "bottom"), (0, -24, "center", "top"),
                  (20, 11, "left", "bottom"), (-20, 11, "right", "bottom"),
                  (20, -13, "left", "top"), (-20, -13, "right", "top"),
                  (0, 34, "center", "bottom"), (0, -36, "center", "top")]
    for p in pts:
        for dx, dy, ha, va in CANDIDATES:
            ann = ax.annotate(p["label"], (p["x"], p["y"]),
                              textcoords="offset points", xytext=(dx, dy),
                              ha=ha, va=va, fontsize=8.5, color=INK, zorder=4)
            fig.canvas.draw()
            bb = ann.get_window_extent(rend).expanded(1.04, 1.10)
            if not any(bb.overlaps(q) for q in placed):
                placed.append(bb)
                break
            ann.remove()
        else:                           # every candidate blocked - keep it
            ann = ax.annotate(p["label"], (p["x"], p["y"]),
                              textcoords="offset points", xytext=(0, 10),
                              ha="center", va="bottom", fontsize=8.5,
                              color=INK, zorder=4)
            print("  [warn] no clear spot for the %r label" % p["label"],
                  file=sys.stderr)
            placed.append(ann.get_window_extent(rend))


def draw(pts, out_png: Path, ylabel: str, show_frontier=True,
         shade=True, legend_loc="upper right"):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()             # the Pareto figures run on defaults
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ax.errorbar([p["x"] for p in pts], [p["y"] for p in pts],
                yerr=[p["ysd"] for p in pts],
                xerr=([p["xsd"] or 0.0 for p in pts]
                      if any(p["xsd"] is not None for p in pts) else None),
                fmt="none", ecolor="#aaaaaa", elinewidth=0.8, capsize=2.5,
                alpha=0.6, zorder=1)
    for key in ("none", "quiet", "comm", "forced"):
        grp = [p for p in pts if p["style"] == key]
        if not grp:
            continue
        st = STYLE[key]
        ax.plot([p["x"] for p in grp], [p["y"] for p in grp], ls="none",
                marker=st["marker"], ms=st["ms"], color=st["color"],
                mfc="none", mec=st["color"], mew=st["mew"], zorder=3)

    xs = [p["x"] + (p["xsd"] or 0.0) for p in pts]
    xs += [p["x"] - (p["xsd"] or 0.0) for p in pts]
    ys = [p["y"] + p["ysd"] for p in pts] + [p["y"] - p["ysd"] for p in pts]
    xpad = 0.12 * (max(xs) - min(xs))
    ypad = 0.10 * (max(ys) - min(ys))
    # Extra room on the right: the three comm arms sit within 0.04 of each
    # other, so one of their labels is always pushed out sideways and would
    # otherwise run into the spine.
    ax.set_xlim(min(xs) - xpad, max(xs) + 1.9 * xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + 1.7 * ypad)

    fx = fy = None
    if show_frontier:
        # The reference arm is an apparatus control, not a cue condition, so
        # it never defines the frontier of the cue sweep.
        front = frontier([p for p in pts if p["style"] != "forced"])
        fx, fy = staircase(front, ax.get_xlim()[1])
        y0 = ax.get_ylim()[0]
        ax.plot(fx, fy, ls="--", lw=1.2, color="#999999", zorder=2)
        if shade:
            # No in-plot tag on the shaded area: the caption names it, and a
            # word floating in the corner reads as a data label.
            ax.fill_between(fx, y0, fy, color="#000000", alpha=0.045, lw=0,
                            zorder=0)

    ax.set_xlabel(r"Mean bond strength  $\overline{W}$")
    ax.set_ylabel(ylabel)

    handles = [Line2D([], [], ls="none", color=STYLE[k]["color"],
                      marker=STYLE[k]["marker"], ms=STYLE[k]["ms"],
                      mfc="none", mew=STYLE[k]["mew"], label=STYLE[k]["label"])
               for k in ("none", "quiet", "comm", "forced")
               if any(p["style"] == k for p in pts)]
    leg = ax.legend(handles=handles, loc=legend_loc, fontsize=8.5,
                    framealpha=0.95)

    # tight_layout moves the axes, so everything the labels must dodge is
    # measured after it, not before.
    fig.tight_layout()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    obstacles = [leg.get_window_extent(rend).padded(3.0)]
    for p in pts:
        cx, cy = ax.transData.transform((p["x"], p["y"]))
        obstacles.append(Bbox.from_bounds(cx - 7, cy - 7, 14, 14))
    if fx is not None:                  # the frontier polyline, segment-wise
        seg = ax.transData.transform(list(zip(fx, fy)))
        for (x0, ya), (x1, yb) in zip(seg, seg[1:]):
            obstacles.append(Bbox.from_extents(min(x0, x1), min(ya, yb),
                                               max(x0, x1), max(ya, yb))
                             .padded(2.0))
    place_labels(fig, ax, pts, obstacles)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_pdf)                # LaTeX wants the PDF
    plt.close(fig)
    print("wrote %s (+ .pdf)" % out_png)
    return out_png, out_pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path,
                    default=ASSETS / "cofire_bidi_3f",
                    help="dir holding cofire_summary.csv (the table's source)")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="run group for the bond sd (default: cofiring_bidi_3f)")
    ap.add_argument("--metric", default="coop", choices=sorted(METRICS))
    ap.add_argument("--with-anchor", action="store_true",
                    help="also plot the forced-communication reference arm, "
                         "which the paper's RQ2 table omits")
    ap.add_argument("--no-frontier", action="store_true")
    ap.add_argument("--no-shade", action="store_true")
    ap.add_argument("--legend-loc", default="upper right")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--copy-to", type=Path, default=None,
                    help="also drop the PNG and PDF in this directory")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    runs_root = args.runs_root or group("cofiring_bidi_3f")
    pts = collect(args.assets, runs_root, args.metric, args.with_anchor)
    if not pts:
        print("no conditions found in %s" % args.assets, file=sys.stderr)
        return 1

    cue = [p for p in pts if p["style"] != "forced"]
    rho = spearman([p["x"] for p in cue], [p["y"] for p in cue])
    for p in sorted(pts, key=lambda q: q["x"]):
        print("  %-14s W %.3f%s   y %6.2f +-%.2f" % (
            p["label"], p["x"],
            "" if p["xsd"] is None else " +-%.3f" % p["xsd"],
            p["y"], p["ysd"]))
    front = frontier(cue)
    print("  frontier: %s" % ", ".join(p["label"] for p in front))
    print("  rho(W, y) over %d cue conditions (console only): %s"
          % (len(cue), "--" if rho is None else "%.3f" % rho))

    _, _, ylabel, stem = METRICS[args.metric]
    out = args.out or (args.assets / (stem + ".png"))
    png, pdf = draw(pts, out, ylabel,
                    show_frontier=not args.no_frontier,
                    shade=not args.no_shade, legend_loc=args.legend_loc)
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for f in (png, pdf):
            shutil.copy2(f, args.copy_to / f.name)
        print("copied to %s" % args.copy_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
