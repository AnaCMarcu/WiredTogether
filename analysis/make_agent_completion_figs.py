#!/usr/bin/env python3
r"""make_agent_completion_figs.py - the three Pareto figures on per-agent completion.

Regenerates, with the metric of make_agent_completion_tables.py (per-agent
completion, no agent union, split into Solo = Chamber 1 and Coop. = Chambers
2-5), the appendix figures that currently plot the union metric:

  1. social-module frequency      figures/social_frontier_milestone.pdf
     (Gemma-E4B, deliberation interval swept, x = social-module FLOPs/episode)
  2. agent capability Pareto      figures/pareto_perception.pdf, pareto_partner.pdf
     (five backbones, x = perception grounding rate / partner-location
     accuracy from the beliefs CSV)
  3. RQ2 co-firing Pareto         (x = mean bond strength W, y = completion;
     drawn by make_rq2_pareto_fig.draw, not currently included in the paper)

Every figure comes out twice, once per y metric (``*_solo`` and ``*_coop``),
as PNG + PDF, in ONE folder (default paper_assets/agent_completion/figures/)
next to a CSV of every plotted number and a README naming the files.

Labels are placed by MEASURED bounding box against every marker, every series
line segment and the legend (centre-above first, then centre-below, then the
diagonals), so no label sits on a line or on a neighbour.

--paper renders at the size the ICLR appendix prints them (each panel is
0.48\textwidth ~ 2.6 in), i.e. 3.5 x 2.6 in with 8 pt text, into <out>/paper/,
and --copy-to drops those PDFs into the paper's figures directory.

Metric: completion % = credited (agent, milestone) pairs / achievable pairs
per episode, after the entry-honesty filter. y = mean over a seed's
episodes, then mean across seeds; --err adds +-1 sample SD across seeds.
Both variants share the +-SD y range so the bare one cannot zoom into the
means. Interval 200 (one seed) is dropped from the social figure by default.

Usage:
    python analysis/make_agent_completion_figs.py --err
    python analysis/make_agent_completion_figs.py --paper --err \
        --copy-to path/to/paper/figures
    python analysis/make_agent_completion_figs.py --only social --drop -1
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

import paths  # noqa: F401  (sys.path: analysis/, repo root, src/)
from paths import ASSETS, RUNS

import make_results as MR
from make_agent_completion_tables import CH1, COOP, NONCOMM, aggregate
import make_pareto_social_fig as MPS
import make_rq2_pareto_fig as RQ2
from make_pareto_grid import load_beliefs, SHORT_LABEL
from make_pareto_fig import SIZES, exp_to_size_arm
from cofire_table import ARMS as COFIRE_ARMS

TRACKS = {"solo": CH1, "coop": COOP, "all": NONCOMM}
YLABEL = {"solo": "Solo milestone completion [%]",
          "coop": "Coop. milestone completion [%]",
          "all": "Milestone completion [%]"}     # all 25 task milestones
SERIES, BASE, INK = "#d95f0e", "#2c7fb8", "#666666"
FAMILY_OPEN = {"gemma": True, "qwen": False}   # marker fill carries the family

# (interval, root key, exp dir in the 3f views); None = no social module.
SOCIAL_POINTS = [
    (None, "anchors", "new_exp_0_gemma_base"),
    (2,    "sweep",   "new_exp_0_gemma_si2"),
    (8,    "anchors", "new_exp_0_gemma_hebbian"),   # -> si3f8 in the 3f view
    (20,   "sweep",   "new_exp_0_gemma_si20"),
    (50,   "sweep",   "new_exp_0_gemma_si50"),
    (100,  "sweep",   "new_exp_0_gemma_si100"),
    (200,  "sweep",   "new_exp_0_gemma_si200"),
    (500,  "sweep",   "new_exp_0_gemma_si500"),
]

# Rendering presets: screen (the repo's usual 5.8 x 3.9) and paper (print size).
STYLE = {
    "screen": dict(figsize=(5.8, 3.9), font=10, tick=9, ann=8.5, legend=9, ms=7.5, lw=1.5),
    "paper":  dict(figsize=(3.5, 2.6), font=8, tick=7, ann=7.5, legend=7, ms=6, lw=1.3),
}
_S = STYLE["screen"]


def save_both(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    print("wrote %s (+ .pdf)" % stem.with_suffix(".png"))


# ─── label placement ───────────────────────────────────────────────────────
CANDIDATES = [(0, 9, "center", "bottom"), (0, -10, "center", "top"),
              (0, 18, "center", "bottom"), (0, -19, "center", "top"),
              (9, 6, "left", "bottom"), (-9, 6, "right", "bottom"),
              (9, -7, "left", "top"), (-9, -7, "right", "top"),
              (12, 0, "left", "center"), (-12, 0, "right", "center"),
              (0, 28, "center", "bottom"), (0, -29, "center", "top")]


def line_obstacles(ax, xs, ys, pad=2.0):
    """Bounding boxes of every segment of a polyline, in display coords."""
    seg = ax.transData.transform(list(zip(xs, ys)))
    out = []
    for (x0, y0), (x1, y1) in zip(seg, seg[1:]):
        out.append(Bbox.from_extents(min(x0, x1), min(y0, y1),
                                     max(x0, x1), max(y0, y1)).padded(pad))
    return out


def marker_obstacles(ax, pts, half=6.0):
    out = []
    for x, y in pts:
        cx, cy = ax.transData.transform((x, y))
        out.append(Bbox.from_bounds(cx - half, cy - half, 2 * half, 2 * half))
    return out


def place_labels(fig, ax, items, obstacles, fontsize):
    """items: [(x, y, text, colour)]. Tries CANDIDATES in order and keeps the
    first offset whose measured box clears every obstacle, every earlier
    label and the tick labels, and stays inside the figure canvas.
    Must run after tight_layout (placement is measured on the canvas)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = list(obstacles)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        if t.get_text():
            placed.append(t.get_window_extent(rend).padded(1.0))
    canvas = fig.bbox.padded(-2.0)
    for x, y, text, colour in items:
        for dx, dy, ha, va in CANDIDATES:
            ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                              ha=ha, va=va, fontsize=fontsize, color=colour, zorder=5,
                              annotation_clip=False)
            fig.canvas.draw()
            bb = ann.get_window_extent(rend).expanded(1.06, 1.12)
            inside = (bb.x0 >= canvas.x0 and bb.x1 <= canvas.x1
                      and bb.y0 >= canvas.y0 and bb.y1 <= canvas.y1)
            if inside and not any(bb.overlaps(q) for q in placed):
                placed.append(bb)
                break
            ann.remove()
        else:
            ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 9),
                              ha="center", va="bottom", fontsize=fontsize, color=colour,
                              zorder=5)
            placed.append(ann.get_window_extent(rend))
            print("  [warn] no clear spot for label %r" % text, file=sys.stderr)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=_S["tick"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ─── 1. social-module frequency ────────────────────────────────────────────
def social_rows(anchors: Path, sweep: Path, flops_args, seed_matched: bool):
    roots = {"anchors": anchors, "sweep": sweep}
    loaded = []
    for iv, key, d in SOCIAL_POINTS:
        runs = MR.load_runs(roots[key], d)
        if runs:
            loaded.append((iv, d, runs))
        else:
            print("  [warn] %s: no runs" % d, file=sys.stderr)
    if seed_matched and loaded:
        common = set.intersection(*({Path(r["_path"]).parent.name for r in runs}
                                    for _, _, runs in loaded))
        loaded = [(iv, d, [r for r in runs if Path(r["_path"]).parent.name in common])
                  for iv, d, runs in loaded]
        print("  seed-matched to %s" % sorted(common), file=sys.stderr)
    rows = []
    for iv, d, runs in loaded:
        soc, kept = [], []
        for run in runs:
            run_dir = Path(run["_path"]).parent
            n_eps = len(run["episode_lengths"])
            f = MPS.social_flops_for_run(run_dir, flops_args)
            if f is None:
                print("  [warn] %s: no log.txt/llm_logs, skipped" % run_dir, file=sys.stderr)
                continue
            soc.append(f / n_eps)
            kept.append(run)
        if not kept:
            continue
        a = aggregate(kept, TRACKS, "seed")
        rows.append({"interval": iv, "exp": d, "n_seeds": len(kept),
                     "seeds": ";".join(sorted(Path(r["_path"]).parent.name for r in kept)),
                     "social_flops_mean": st.fmean(soc),
                     "social_flops_sd": st.stdev(soc) if len(soc) > 1 else 0.0,
                     **{"%s_%s" % (k, s): a[k][i] for k in TRACKS for i, s in
                        enumerate(("mean", "sd"))},
                     "task_mean": a["task"][0], "task_sd": a["task"][1]})
    return rows


def draw_social(rows, ykey: str, out_stem: Path, err: bool, drop: set):
    pts = sorted(({"iv": r["interval"] or 0, "x": r["social_flops_mean"],
                   "y": r["%s_mean" % ykey], "sd": r["%s_sd" % ykey], "n": r["n_seeds"]}
                  for r in rows if (r["interval"] or 0) not in drop),
                 key=lambda p: p["x"])
    fig, ax = plt.subplots(figsize=_S["figsize"])
    xs_pos = [p["x"] for p in pts if p["x"] > 0]
    # symlog: the no-module arm sits at exactly x = 0 on the same axis.
    ax.set_xscale("symlog", linthresh=min(xs_pos) * 0.6, linscale=0.45)
    xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
    ax.plot(xs, ys, "-", color=SERIES, lw=_S["lw"], zorder=2)
    if err:
        ax.errorbar(xs, ys, yerr=[p["sd"] for p in pts], fmt="none", ecolor=SERIES,
                    elinewidth=0.9, capsize=2.5, alpha=0.5, zorder=2)
    for p in pts:
        is_base = p["iv"] == 0
        ax.plot(p["x"], p["y"], "s" if is_base else "o",
                ms=_S["ms"] + (0.5 if is_base else 0), color=BASE if is_base else SERIES,
                mfc="none", mew=1.5, zorder=3)
    ax.set_xlabel("Compute  (FLOPs / episode, social module)", fontsize=_S["font"])
    ax.set_ylabel(YLABEL[ykey], fontsize=_S["font"])
    style_axes(ax)
    # Major ticks at 0 and the decades from 1e14: the symlog origin and the
    # 1e13 tick otherwise print on top of each other.
    ticks = [0.0] + [10.0 ** k for k in range(14, 17) if 10.0 ** k <= max(xs_pos) * 1.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + ["$10^{%d}$" % k for k in range(14, 14 + len(ticks) - 1)])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # Room to the left of x = 0 so the baseline marker and its label do not
    # sit on the y spine (symlog admits negative x).
    ax.set_xlim(left=-0.9 * min(xs_pos) * 0.6, right=max(xs_pos) * 2.2)
    env = [p["y"] + p["sd"] for p in pts] + [p["y"] - p["sd"] for p in pts]
    lo, hi = min(env), max(env)
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.30 * (hi - lo))
    # Legend in the (empty) upper right; what the point labels mean is said
    # in the caption, not in a sentence-long legend entry.
    handles = [Line2D([], [], color=SERIES, marker="o", ms=_S["ms"], lw=_S["lw"], mfc="none",
                      mew=1.5, label="+Social plasticity"),
               Line2D([], [], color=BASE, marker="s", ms=_S["ms"] + 0.5, lw=0, mfc="none",
                      mew=1.5, label="no social module")]
    leg = ax.legend(handles=handles, fontsize=_S["legend"], loc="upper right",
                    handlelength=1.8, framealpha=0.95)
    fig.tight_layout()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    obstacles = [leg.get_window_extent(rend).padded(3.0)]
    obstacles += marker_obstacles(ax, [(p["x"], p["y"]) for p in pts])
    obstacles += line_obstacles(ax, xs, ys)
    if err:   # keep labels off the bars too, they are the only vertical ink
        obstacles += line_obstacles(ax, [x for x in xs for _ in (0, 1)],
                                    [v for p in pts for v in (p["y"] - p["sd"], p["y"] + p["sd"])],
                                    pad=1.0)
    items = [(p["x"], p["y"], "no module" if p["iv"] == 0 else "%d" % p["iv"],
              BASE if p["iv"] == 0 else INK) for p in pts]
    place_labels(fig, ax, items, obstacles, _S["ann"])
    save_both(fig, out_stem)
    plt.close(fig)


# ─── 2. agent-capability (perception) Pareto ───────────────────────────────
def perception_rows(root: Path, beliefs_csv: Path, sizes, x_key: str):
    """{size: {family, x:(m,sd), base:{solo,coop,...}, hebbian:{...}}}; x is
    pooled over BOTH arms of a backbone (a backbone property, as in the
    original figure)."""
    beliefs = load_beliefs([beliefs_csv])
    rows, table = {}, []
    for size in sizes:
        row = {"family": SIZES[size]["family"]}
        gx_all, ok = [], True
        for arm in ("base", "hebbian"):
            cond = next((p.name for p in root.iterdir()
                         if exp_to_size_arm(p.name) == (size, arm)), None)
            runs = MR.load_runs(root, cond) if cond else []
            gx = beliefs.get((size, arm), {}).get(x_key) or []
            if not runs or not gx:
                ok = False
                break
            a = aggregate(runs, TRACKS, "seed")
            gx_all += gx
            row[arm] = {k: a[k] for k in TRACKS}
            row[arm]["n"] = a["n_seeds"]
            table.append({"x_axis": x_key, "size": size, "arm": arm, "n_seeds": a["n_seeds"],
                          "x_mean": st.fmean(gx), "x_sd": st.pstdev(gx) if len(gx) > 1 else 0.0,
                          **{"%s_%s" % (k, s): a[k][i] for k in TRACKS
                             for i, s in enumerate(("mean", "sd"))},
                          "task_mean": a["task"][0], "task_sd": a["task"][1]})
        if ok:
            row["x"] = (st.fmean(gx_all), st.pstdev(gx_all) if len(gx_all) > 1 else 0.0)
            rows[size] = row
        else:
            print("  (skipping %s: incomplete data)" % size, file=sys.stderr)
    return rows, table


ARM = {"base": dict(color=BASE, marker="s", label="Base"),
       "hebbian": dict(color=SERIES, marker="o", label="+Social plasticity")}


def draw_perception(rows, ykey: str, out_stem: Path, xlabel: str, err: bool, legend_loc: str):
    fig, ax = plt.subplots(figsize=_S["figsize"])
    env, lines, markers = [], [], []
    for arm, stl in ARM.items():
        pts = sorted(((r["x"][0], r[arm][ykey][0], r[arm][ykey][1], r["family"])
                      for r in rows.values()), key=lambda t: t[0])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=stl["color"], ls="-", lw=_S["lw"], zorder=2)
        lines.append((xs, ys))
        if err:
            ax.errorbar(xs, ys, yerr=[p[2] for p in pts], fmt="none", ecolor=stl["color"],
                        elinewidth=0.9, capsize=2.5, alpha=0.55, zorder=1)
            env += [p[1] + p[2] for p in pts] + [p[1] - p[2] for p in pts]
        for fam in dict.fromkeys(p[3] for p in pts):
            fp = [p for p in pts if p[3] == fam]
            ax.plot([p[0] for p in fp], [p[1] for p in fp], ls="none", marker=stl["marker"],
                    ms=_S["ms"], color=stl["color"],
                    mfc="none" if FAMILY_OPEN.get(fam, True) else stl["color"],
                    mec=stl["color"], mew=1.5, zorder=3)
        markers += list(zip(xs, ys))
        env += ys
    lo, hi = min(env), max(env)
    pad = max(0.02 * (abs(hi) or 1.0), 0.22 * (hi - lo))
    ax.set_ylim(lo - pad, hi + 1.6 * pad)
    ax.margins(x=0.10)
    ax.set_xlabel(xlabel, fontsize=_S["font"])
    ax.set_ylabel(YLABEL[ykey], fontsize=_S["font"])
    style_axes(ax)
    handles = [Line2D([], [], ls="none", color=s["color"], marker=s["marker"], ms=_S["ms"],
                      mfc="none", mew=1.5, label=s["label"]) for s in ARM.values()]
    leg = ax.legend(handles=handles, loc=legend_loc, fontsize=_S["legend"], framealpha=0.95)
    fig.tight_layout()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    obstacles = [leg.get_window_extent(rend).padded(3.0)] + marker_obstacles(ax, markers)
    for xs, ys in lines:
        obstacles += line_obstacles(ax, xs, ys)
    # One label per backbone, anchored at its higher arm.
    items = []
    for size, r in sorted(rows.items(), key=lambda kv: kv[1]["x"][0]):
        hi_arm = max(("base", "hebbian"), key=lambda a: r[a][ykey][0])
        items.append((r["x"][0], r[hi_arm][ykey][0], SHORT_LABEL.get(size, size), "#555555"))
    place_labels(fig, ax, items, obstacles, _S["ann"])
    save_both(fig, out_stem)
    plt.close(fig)


# ─── 3. RQ2 co-firing Pareto ───────────────────────────────────────────────
def cofire_points(root: Path, ykey: str):
    pts, table = [], []
    for cond, exp, label, cues in COFIRE_ARMS:
        if cond == RQ2.ANCHOR:
            continue
        runs = MR.load_runs(root, exp)
        if not runs:
            print("  [warn] cofire %s: no runs" % exp, file=sys.stderr)
            continue
        a = aggregate(runs, TRACKS, "seed")
        w = [r["mean"] for run in runs for r in MR.end_of_episode_graph_stats(run)]
        pts.append({"cond": cond, "label": RQ2.PAPER_LABEL.get(label, label),
                    "style": RQ2.style_of(cond, cues),
                    "x": st.fmean(w), "xsd": st.pstdev(w) if len(w) > 1 else 0.0,
                    "y": a[ykey][0], "ysd": a[ykey][1]})
        table.append({"cond": cond, "label": label, "arm": exp, "n_seeds": a["n_seeds"],
                      "W_mean": st.fmean(w), "W_sd": st.pstdev(w) if len(w) > 1 else 0.0,
                      **{"%s_%s" % (k, s): a[k][i] for k in TRACKS
                         for i, s in enumerate(("mean", "sd"))},
                      "task_nc_mean": a["task_nc"][0], "task_nc_sd": a["task_nc"][1]})
    return pts, table


def draw_cofire(pts, ykey: str, out_stem: Path):
    """make_rq2_pareto_fig.draw (its own measured label placement); it always
    draws +-1 SD bars and derives its padding from them, so one variant."""
    RQ2.draw(pts, out_stem.with_suffix(".png"), YLABEL[ykey],
             show_frontier=True, shade=True, legend_loc="upper right")


# ─── main ──────────────────────────────────────────────────────────────────
README = """# Pareto figures on per-agent completion

Generated by analysis/make_agent_completion_figs.py. Metric: completion % =
credited (agent, milestone) pairs / achievable pairs per episode (Solo = 8
Chamber-1 milestones over 7N+1; Coop. = 17 Chamber-2..5 milestones over 17N),
entry-honesty filter applied, no agent union. y = mean over each seed's
episodes, then mean across seeds; `_err` variants add +-1 sample SD across
seeds; both variants share the +-SD y range. Definitions in ../METRICS.md.
`paper/` holds the same figures at ICLR print size (3.5 x 2.6 in, 8 pt).

| File | Replaces | x axis | Inputs |
|---|---|---|---|
| social_frontier_{all,solo,coop}[_err] | figures/social_frontier_milestone.pdf | social-module FLOPs / episode (symlog); interval 200 (1 seed) dropped by default; `all` = all 25 task milestones over 24N+1 | compute/_social_3f_sweep + _social_3f_anchors (Gemma-E4B, interval sweep, 3f rule) |
| pareto_perception_{all,solo,coop}[_err] | figures/pareto_perception.pdf | perception grounding rate (strict) | compute/_pareto_3f_view + perception_3f/beliefs_3f.csv |
| pareto_partner_{all,solo,coop}[_err] | figures/pareto_partner.pdf | partner-location accuracy | same |
| rq2_pareto_{solo,coop} | (not in the paper) figures/pareto_bond_coop.pdf | mean bond strength W at episode end, +-1 SD bars | rq2_cofiring/cofiring_bidi_3f |

Numbers behind every point: social_points.csv, perception_points.csv,
cofire_points.csv (both metrics plus All % and task return).
"""


def main() -> int:
    global _S
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ASSETS / "agent_completion" / "figures")
    ap.add_argument("--only", choices=["social", "perception", "cofire"], default=None)
    ap.add_argument("--err", action="store_true", help="ALSO emit +-1 seed-SD variants")
    ap.add_argument("--paper", action="store_true",
                    help="print-size rendering (3.5 x 2.6 in, 8 pt) into <out>/paper/")
    ap.add_argument("--copy-to", type=Path, default=None,
                    help="also copy the PDFs into this directory (the paper's figures/)")
    ap.add_argument("--drop", type=int, action="append", default=None,
                    help="omit this deliberation interval from the social figure "
                         "(default: 200, a single seed; --drop -1 keeps every interval)")
    ap.add_argument("--seed-matched", action="store_true",
                    help="restrict the social sweep to seeds common to every arm")
    ap.add_argument("--beliefs", type=Path, default=ASSETS / "perception_3f" / "beliefs_3f.csv")
    ap.add_argument("--sizes", default="e2b,e4b,qwen2b,12b,qwen9b")
    ap.add_argument("--n-eff", type=float, default=4.5e9)
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    _S = STYLE["paper" if args.paper else "screen"]
    out = args.out / "paper" if args.paper else args.out
    out.mkdir(parents=True, exist_ok=True)
    errs = (False, True) if args.err else (False,)
    drop = {200} if args.drop is None else set(args.drop)
    written = []

    def emit(fn, stem, *a, **kw):
        fn(*a, stem, **kw)
        written.append(stem.with_suffix(".pdf"))

    def write_csv(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % (out / name))

    if args.only in (None, "social"):
        flops_args = SimpleNamespace(n_eff=args.n_eff, image_tokens=args.image_tokens,
                                     overhead_tokens=args.overhead_tokens,
                                     chars_per_token=args.chars_per_token)
        rows = social_rows(RUNS / "compute" / "_social_3f_anchors",
                           RUNS / "compute" / "_social_3f_sweep", flops_args, args.seed_matched)
        write_csv("social_points.csv", rows)
        for ykey in ("all", "solo", "coop"):
            for e in errs:
                stem = out / ("social_frontier_%s%s" % (ykey, "_err" if e else ""))
                draw_social(rows, ykey, stem, e, drop)
                written.append(stem.with_suffix(".pdf"))

    if args.only in (None, "perception"):
        root = RUNS / "compute" / "_pareto_3f_view"
        sizes = [s for s in args.sizes.split(",") if s]
        table_all = []
        for xk, xlabel, stem_name in (
                ("grounding_strict", "Perception grounding rate", "pareto_perception"),
                ("partner_loc", "Partner-location accuracy", "pareto_partner")):
            rows, table = perception_rows(root, args.beliefs, sizes, xk)
            table_all += table
            for ykey in ("all", "solo", "coop"):
                for e in errs:
                    stem = out / ("%s_%s%s" % (stem_name, ykey, "_err" if e else ""))
                    draw_perception(rows, ykey, stem, xlabel, e, legend_loc="upper left")
                    written.append(stem.with_suffix(".pdf"))
        write_csv("perception_points.csv", table_all)

    if args.only in (None, "cofire"):
        root = paths.group("cofiring_bidi_3f")
        for ykey in ("solo", "coop"):
            pts, table = cofire_points(root, ykey)
            if ykey == "solo":
                write_csv("cofire_points.csv", table)
            front = RQ2.frontier(pts)
            print("  RQ2 %s frontier: %s" % (ykey, ", ".join(p["label"] for p in front)))
            stem = out / ("rq2_pareto_%s" % ykey)
            draw_cofire(pts, ykey, stem)
            written.append(stem.with_suffix(".pdf"))

    (args.out / "README.md").write_text(README, encoding="utf-8")
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for pdf in written:
            if pdf.is_file():
                shutil.copy2(pdf, args.copy_to / pdf.name)
        print("copied %d PDFs to %s" % (len(written), args.copy_to))
    print("all figures in %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
