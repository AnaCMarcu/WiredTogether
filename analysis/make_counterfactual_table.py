#!/usr/bin/env python3
r"""make_counterfactual_table.py - the matched-window case studies, laid out
as a *teaser table* rather than as two stacked panels.

WHAT THIS IS
------------
``make_counterfactual_compact.py`` renders the same evidence as two separate
canvases (``counterfactual_compact_a/_b``) that the LaTeX source places side
by side. That works, but it leaves the reader to infer the correspondence
between the two columns: nothing on the page says that the frame strip on the
left and the frame strip on the right are the same kind of evidence.

This script renders ONE figure in the layout that multi-agent Minecraft papers
use for their task teasers: bold column headers over a heavy rule, a two-level
row-label gutter down the left edge, alternating row bands, and a monospace
outcome row at the foot. Columns are conditions, rows are evidence types, and
the grid itself carries the "same measurement, different mechanism" claim.

Two figures come out of it:

  counterfactual_table    Relational plasticity vs Central orchestration,
                          both 3 agents, seed 42, episode 3, steps 580-720.
                          Same map, same episode, same milestone. This is the
                          restyled version of counterfactual_compact_{a,b}.

  orchestrator_n6_table   Central orchestration at 3 agents vs at 6 agents.
                          Horizon-matched (both 3 x 1000 steps, villager DAG,
                          advisory, cadence 8, node timeout 60). Each column
                          is that run's own first Ch4 combat window.

SCOPE OF THE CLAIMS
-------------------
Both figures are single-window case studies, and every column is labelled with
the step range it covers, because the window-level reading does NOT always
generalise to the run:

  * In the N=6 Ch4 window the orchestrator writes six single-agent subtasks.
    Run-wide it is not more solo-happy than N=3 (46% of its staffed tasks are
    multi-agent vs 31% at N=3). What DOES hold run-wide is that it writes far
    more tasks it never staffs at all (210/299 = 70% vs 85/166 = 51%).
  * ``freed_success`` counts agent-task pairs, not events. Of the 12 successes
    at N=6, six are the single t=682 kill closing six separate solo tasks and
    four are one anvil event closing four; only two are independent. Success
    counts are therefore not comparable across team sizes.

The N=6 column carries one uncontrolled difference from the N=3 column:
``--team-scaling`` (N-templated prompts) is on at N=6 and off at N=3. Ch4
difficulty IS matched - ``--ch4-mob-count 3`` pins the N=6 run to the three
zombies the N=3 run faces by default (Lua spawns ``NUM_AGENTS`` otherwise).

FRAMES
------
First-person frames are read from ``<run>/gifs/<exp>/seed_<S>_agent_<a>_ep<E>
.mp4``. ``pull_new.sh`` excludes ``gifs/`` by default, so a freshly pulled run
has none; those cells render as labelled placeholders and the script says so
on stdout. To fill them in, see the header of ``pull_media.sh``.

Usage:  python analysis/make_counterfactual_table.py [--only pair|n6]
Out:    paper_assets/timelines/counterfactual/{counterfactual_table,
        orchestrator_n6_table}.{pdf,png,svg}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from paths import ASSETS, group  # noqa: E402  (also puts siblings on sys.path)

import make_directive_timelines as mdt  # noqa: E402
import make_final_figures as mff  # noqa: E402

# -- runs ---------------------------------------------------------------
SEED = 42
HEB_RUN = group("pareto_social_3f") / "new_exp_0_gemma_si3f8" / f"seed_{SEED}"
HEB_EXP = "new_exp_0_gemma_si3f8"
ORCH_RUN = (group("orchestrator") / "new_exp_0_gemma_orch_villager_advisory"
            / f"seed_{SEED}")
ORCH_EXP = "new_exp_0_gemma_orch_villager_advisory"
N6_RUN = (group("agent_scaling_orch") / "scale_gemma_orch_villager_n6"
          / f"seed_{SEED}")
N6_EXP = "scale_gemma_orch_villager_n6"

OUT = ASSETS / "timelines" / "counterfactual"

# -- design -------------------------------------------------------------
# Humanist sans first (what the reference layout uses), then the portable
# fallbacks, so the figure still builds on a machine without the Windows
# fonts. Swap the head of this list to re-face the whole figure.
SANS = ["Calibri", "Segoe UI", "Arial", "DejaVu Sans"]
MONO = ["Consolas", "DejaVu Sans Mono"]

INK = "#2b3138"          # body text
HEAD_INK = "#12181f"     # column headers
MUTED = "#79838f"        # row-group labels, secondary notes
RULE_HEAVY = "#1F3864"   # the thick rule under the headers
RULE_HAIR = "#dde1e6"    # row separators
BAND = "#F4F5F7"         # alternating row band
PLACEHOLDER = "#eceef1"

AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73",
           3: "#CC79A7", 4: "#E69F00", 5: "#56B4E9"}
PAIR_C = mff.PAIR_C
# The light end has to stay clearly visible against white: a solo block is
# often only a dozen steps wide, and at the original #cfdbe9 the 665-683
# single-agent slot on each N=6 lane read as a gap in the bar rather than as
# a task.
TEAM_CMAP = LinearSegmentedColormap.from_list(
    "team", ["#aec6e0", "#6f95bf", "#0F4D92"])

FS = dict(head=11.0, group=7.4, label=7.4, body=7.6, chat=6.6, tag=5.6,
          mono=7.0, tick=5.8, axis=6.2, lane=5.8, note=6.0)

RC = {"svg.fonttype": "none", "pdf.fonttype": 42,
      "font.family": "sans-serif", "font.sans-serif": SANS,
      "mathtext.fontset": "dejavusans",
      "axes.linewidth": 0.5, "image.interpolation": "none",
      "image.resample": False,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5,
      "xtick.major.size": 1.8, "ytick.major.size": 1.8}

# -- inline rich text (**bold**, `mono`) --------------------------------
_STYLE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def _words(md: str):
    """Split markdown into words, each a list of styled runs.

    A *word* is a list of runs rather than a string so that punctuation
    stuck to a bold span ("**foo**,") stays in the same word and never
    picks up a space when the line is laid out.
    """
    segs = []
    for part in _STYLE_RE.split(md):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            segs.append((part[2:-2], True, False))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            segs.append((part[1:-1], False, True))
        else:
            segs.append((part, False, False))
    words, cur = [], []
    for txt, bold, mono in segs:
        for piece in re.split(r"(\s+)", txt):
            if not piece:
                continue
            if piece.isspace():
                if cur:
                    words.append(cur)
                    cur = []
            else:
                cur.append((piece, bold, mono))
    if cur:
        words.append(cur)
    return words


class Measurer:
    """Text widths in figure fractions, memoised.

    matplotlib has no inline rich text, so every styled run is positioned by
    hand; that needs a real width per run, which only the renderer can give.
    """

    def __init__(self, fig):
        self.fig = fig
        self.rend = fig.canvas.get_renderer()
        self.px = fig.get_figwidth() * fig.dpi
        self._c = {}

    def run(self, txt, size, bold=False, mono=False):
        key = (txt, size, bold, mono)
        if key not in self._c:
            t = self.fig.text(0, 0, txt, fontsize=size,
                              fontfamily=MONO if mono else SANS,
                              fontweight="bold" if bold else "normal")
            w = t.get_window_extent(renderer=self.rend).width
            t.remove()
            self._c[key] = w / self.px
        return self._c[key]

    def space(self, size, mono=False):
        # measured as a difference: a lone " " has no reliable advance
        return (self.run("a a", size, mono=mono)
                - self.run("aa", size, mono=mono))

    def word(self, w, size):
        return sum(self.run(t, size, b, m) for t, b, m in w)


def rich_text(fig, mz, x, y, md, width, size, color=INK, leading=1.42,
              mono_default=False):
    """Draw **bold** / `mono` markdown at (x, y) top-left, wrapped to width.

    ``mono_default`` sets the face for *unmarked* runs, so a chat line can be
    monospace throughout while still honouring **bold**. It has to reach the
    measurer as well as the draw call: measuring a monospace run with the
    proportional metrics is what makes the words run together.

    Returns the height consumed, in figure fractions.
    """
    sp = {False: mz.space(size, mono=False), True: mz.space(size, mono=True)}

    def is_mono(run):
        return run[2] or mono_default

    def gap(prev_word, next_word):
        # A proportional space between two monospace tokens reads as no space
        # at all next to the monospace gaps around it, so a boundary touching
        # a mono run takes the mono advance.
        return sp[is_mono(prev_word[-1]) or is_mono(next_word[0])]

    lines = []                       # [[(word, gap_before), ...], ...]
    for para in md.split("\n"):      # "\n" is a hard break
        cur, curw = [], 0.0
        for w in _words(para):
            ww = sum(mz.run(t, size, b, m or mono_default) for t, b, m in w)
            g = gap(cur[-1][0], w) if cur else 0.0
            if cur and curw + g + ww > width:
                lines.append(cur)
                cur, curw = [(w, 0.0)], ww
            else:
                cur.append((w, g))
                curw += g + ww
        lines.append(cur)

    lh = size * leading / 72.0 / fig.get_figheight()
    yy = y
    for line in lines:
        xx = x
        for w, g in line:
            xx += g
            for txt, bold, mono in w:
                use_mono = mono or mono_default
                fig.text(xx, yy, txt, fontsize=size, color=color,
                         fontfamily=MONO if use_mono else SANS,
                         fontweight="bold" if bold else "normal",
                         va="top", ha="left")
                xx += mz.run(txt, size, bold, use_mono)
        yy -= lh
    return y - yy


# -- data helpers -------------------------------------------------------
def crop(fr, keep=0.72, upscale=3):
    """Drop the hotbar strip, then integer-upscale (see the compact script)."""
    if fr is None:
        return None
    fr = fr[: int(fr.shape[0] * keep)]
    return np.repeat(np.repeat(fr, upscale, axis=0), upscale, axis=1)


def assignment_grid(run_dir: Path, episode: int, hi: int, n_agents: int):
    """(hi+1, n_agents) array of task-id-or-None, from assignments.jsonl.

    Events at the same step are applied in file order, which puts the frees
    before the allocations of a re-decomposition, so the new task wins.
    """
    lines = (run_dir / "orchestrator" / "assignments.jsonl").read_text(
        encoding="utf-8").splitlines()
    ev = [json.loads(l) for l in lines if l.strip()]
    ev = sorted((e for e in ev if e["episode"] == episode),
                key=lambda e: e["t"])            # stable: file order within t
    grid = np.empty((hi + 1, n_agents), dtype=object)
    cur = [None] * n_agents
    ptr = 0
    for t in range(hi + 1):
        while ptr < len(ev) and ev[ptr]["t"] <= t:
            e = ev[ptr]
            a = int(e["agent"].rsplit("_", 1)[1])
            if a < n_agents:
                cur[a] = (e["task_id"] if e["reason"].startswith("allocate")
                          else None)
            ptr += 1
        grid[t] = cur
    return grid


def short_task(tid: str) -> str:
    return (tid.replace("t_ch4_", "").replace("t_ch3_", "")
            .replace("t_ch2_", "").replace("_", " "))


# -- cell painters ------------------------------------------------------
def paint_bonds(fig, rect, spec):
    """Hebbian column: the three pairwise bond trajectories."""
    lo, hi = spec["win"]
    ax = fig.add_axes(rect)
    run = mdt.load_run(spec["run"])
    off = run["_ep_bounds"][spec["ep"] - 1][0]
    ser = mdt.load_bonds(run)
    xs = np.arange(lo, hi + 1)
    ends = []
    for q in mdt.PAIRS:
        ys = np.interp(off + xs, *ser[q])
        ax.plot(xs, ys, color=PAIR_C[q], lw=1.25, solid_capstyle="round",
                zorder=3)
        ends.append((ys[-1], q))
    placed = []
    for y, q in sorted(ends):
        while any(abs(y - p) < 0.045 for p in placed):
            y += 0.045
        placed.append(y)
        ax.annotate(f"a{q[0]}-a{q[1]}", (hi, y), xytext=(2.5, 0),
                    textcoords="offset points", fontsize=FS["lane"],
                    color=PAIR_C[q], va="center", annotation_clip=False)
    star = spec.get("star")
    if star is not None:
        ax.scatter(star, float(np.interp(off + star, *ser[(0, 1)])),
                   marker="*", s=54, color=AGENT_C[spec.get("star_agent", 0)],
                   edgecolor="white", linewidths=0.4, zorder=6)
    ax.set_ylim(0.26, 0.76)
    ax.set_yticks([0.3, 0.5, 0.7])
    ax.set_ylabel("bond $\\bar{W}$", fontsize=FS["axis"], color=INK,
                  labelpad=1.5)
    _finish_axis(ax, lo, hi)


def paint_assignments(fig, rect, spec):
    """Orchestrator column: who holds which subtask, shaded by team size."""
    lo, hi = spec["win"]
    n = spec["n_agents"]
    ax = fig.add_axes(rect)
    grid = assignment_grid(spec["run"], spec["ep"], hi + 1, n)
    rows = {a: n - 1 - a for a in range(n)}
    bar_h = 0.62 if n <= 3 else 0.70
    span = hi - lo
    for a in range(n):
        t0 = lo
        for t in range(lo + 1, hi + 2):
            if t <= hi and grid[t, a] == grid[t0, a]:
                continue
            tid = grid[t0, a]
            if tid is not None:
                mates = sum(1 for b in range(n) if grid[t0, b] == tid)
                ax.broken_barh(
                    [(t0 + 1.0, max(t - t0 - 2.0, 0.6))],
                    (rows[a] - bar_h / 2, bar_h),
                    facecolors=TEAM_CMAP((mates - 1) / max(n - 1, 1)),
                    linewidth=0, zorder=3)
            t0 = t
    # Name the widest block on a lane, but only when it says something new:
    # a lane repeating the lane above it (the whole team on one subtask, or
    # the N solo clones of one subtask) would just tile the same words.
    widest = {}
    for a in range(n):
        t0, best = lo, None
        for t in range(lo + 1, hi + 2):
            if t <= hi and grid[t, a] == grid[t0, a]:
                continue
            if grid[t0, a] is not None:
                width = min(t, hi) - t0
                if best is None or width > best[0]:
                    best = (width, t0, min(t, hi), grid[t0, a])
            t0 = t
        widest[a] = best
    for a in range(n):
        best = widest[a]
        if best is None or best[0] <= span * 0.24:
            continue
        width, s0, s1, tid = best
        prev = widest.get(a - 1)
        if prev is not None and short_task(prev[3]) == short_task(tid):
            continue
        mates = sum(1 for b in range(n) if grid[s0, b] == tid)
        ax.text((s0 + s1) / 2, rows[a], short_task(tid),
                fontsize=FS["lane"],
                color="white" if mates > n * 0.5 else INK,
                ha="center", va="center", zorder=5)
    for t in spec.get("marks", []):
        ax.scatter([t] * n, list(rows.values()), marker="x", s=16, color=INK,
                   linewidths=0.85, zorder=6)
    ax.set_ylim(-0.66, n - 1 + 0.66)
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels([f"a{a}" for a in range(n)], fontsize=FS["lane"])
    for tick, a in zip(ax.get_yticklabels(), range(n)):
        tick.set_color(AGENT_C[a])
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.set_ylabel("assigned\nsubtask", fontsize=FS["axis"], color=INK,
                  labelpad=1.5, linespacing=0.95)
    _finish_axis(ax, lo, hi)


def _finish_axis(ax, lo, hi):
    ax.set_xlim(lo, hi)
    ax.set_xticks(list(range(int(np.ceil(lo / 50.0)) * 50, hi + 1, 50)))
    ax.tick_params(labelsize=FS["tick"], pad=1.2)
    ax.set_xlabel("environment step", fontsize=FS["axis"], color=INK,
                  labelpad=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE_HAIR)


def paint_frames(fig, rect, spec, frames):
    """2 x 2 grid of first-person captures, or labelled placeholders."""
    x0, y0, w, h = rect
    gap_x, gap_y = 0.008, 0.012
    fw = (w - gap_x) / 2
    fh = (h - gap_y) / 2
    for k, (a, t) in enumerate(spec["frames"]):
        fx = x0 + (k % 2) * (fw + gap_x)
        fy = y0 + h - fh - (k // 2) * (fh + gap_y)
        axf = fig.add_axes([fx, fy, fw, fh])
        axf.axis("off")
        fr = crop(frames.get((a, spec["ep"], t)))
        if fr is not None:
            axf.imshow(fr, aspect="auto", interpolation="none")
        else:
            axf.add_patch(Rectangle((0, 0), 1, 1, facecolor=PLACEHOLDER,
                                    edgecolor=RULE_HAIR, linewidth=0.5,
                                    transform=axf.transAxes))
            axf.text(0.5, 0.42, "recording not synced", ha="center",
                     va="center", fontsize=FS["tag"], color=MUTED,
                     transform=axf.transAxes, style="italic")
        axf.text(0.028, 0.93, f"a{a}  t={t}", transform=axf.transAxes,
                 fontsize=FS["tag"], color="white", va="top",
                 fontweight="bold",
                 bbox=dict(facecolor=AGENT_C[a] + "e0", edgecolor="none",
                           pad=1.0))


def paint_chat(fig, mz, rect, spec):
    """Verbatim turns: coloured sender tag, monospace body."""
    x0, y0, w, h = rect
    tag_w = mz.run("a0→a5", FS["chat"], bold=True, mono=True) + 0.014
    y = y0 + h
    for t, src, dst, txt in spec["chat"]:
        fig.text(x0, y, f"a{src}→a{dst}", fontsize=FS["chat"],
                 color=AGENT_C[src], va="top", fontweight="bold",
                 fontfamily=MONO)
        used = rich_text(fig, mz, x0 + tag_w, y, txt, w - tag_w, FS["chat"],
                         color=INK, leading=1.30, mono_default=True)
        y -= used + 0.010


def paint_outcome(fig, mz, rect, spec):
    x0, y0, w, h = rect
    rich_text(fig, mz, x0, y0 + h * 0.84, spec["outcome"], w, FS["mono"],
              color=INK, leading=1.30, mono_default=True)


# -- the table engine ---------------------------------------------------
M_L, M_R, M_T, M_B = 0.12, 0.12, 0.11, 0.14      # inches
GUT_GROUP, GUT_LABEL = 0.60, 1.00
COL_GAP = 0.26
HEAD_H = 0.56
PAD_Y = 0.07                                      # inside a cell
AX_FOOT = 0.34          # room under a plot cell for ticks + x label
AX_LEFT, AX_RIGHT = 0.34, 0.44   # room for the y label / lane end labels


def build(spec, stem: str, dpi: int = 600) -> None:
    cols, rows = spec["cols"], spec["rows"]
    ncol = len(cols)
    fig_w = spec.get("fig_w", 7.2)
    col_w = (fig_w - M_L - GUT_GROUP - GUT_LABEL - M_R
             - COL_GAP * (ncol - 1)) / ncol
    fig_h = M_T + HEAD_H + sum(r["h"] for r in rows) + M_B

    # frames up front: a missing gifs/ layer changes nothing but the pixels
    frames = {}
    for c in cols:
        if not c.get("frames"):
            continue
        got = mff.grab_frames(dict(run=c["run"], exp=c["exp"], seed=SEED),
                              [(a, c["ep"], t) for a, t in c["frames"]])
        frames[id(c)] = got
        miss = len(c["frames"]) - len(got)
        print(f"  {c['head']:<26s} frames {len(got)}/{len(c['frames'])}"
              + (f"   [{miss} placeholder]" if miss else ""))

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")
        mz = Measurer(fig)

        def FX(inches):
            return inches / fig_w

        def FY(inches_from_top):
            return 1.0 - inches_from_top / fig_h

        col_x = [M_L + GUT_GROUP + GUT_LABEL + i * (col_w + COL_GAP)
                 for i in range(ncol)]
        left, right = M_L, fig_w - M_R

        # -- row bands, drawn first so everything sits on top ----------
        y = M_T + HEAD_H
        band = False
        for r in rows:
            if band or r.get("band"):
                fig.patches.append(Rectangle(
                    (FX(left), FY(y + r["h"])), FX(right - left),
                    r["h"] / fig_h, facecolor=BAND, edgecolor="none",
                    zorder=0, transform=fig.transFigure, figure=fig))
            y += r["h"]
            band = not band

        # -- header row ------------------------------------------------
        # Title and provenance stack rather than sitting on one line: right
        # aligned, the provenance of column i butted against the title of
        # column i+1 and read as part of it.
        for i, c in enumerate(cols):
            fig.text(FX(col_x[i]), FY(M_T + 0.23), c["head"],
                     fontsize=FS["head"], color=HEAD_INK, fontweight="bold",
                     va="baseline", ha="left")
            if c.get("sub"):
                fig.text(FX(col_x[i]), FY(M_T + 0.42), c["sub"],
                         fontsize=FS["note"], color=MUTED,
                         va="baseline", ha="left")
        fig.lines.append(plt.Line2D(
            [FX(left), FX(right)], [FY(0.03)] * 2, color=RULE_HAIR,
            lw=0.7, transform=fig.transFigure, figure=fig, zorder=4))
        fig.lines.append(plt.Line2D(
            [FX(left), FX(right)], [FY(M_T + HEAD_H)] * 2, color=RULE_HEAVY,
            lw=2.0, transform=fig.transFigure, figure=fig, zorder=4,
            solid_capstyle="butt"))

        # -- rows ------------------------------------------------------
        y = M_T + HEAD_H
        for ri, r in enumerate(rows):
            top, h = y, r["h"]
            if ri:
                fig.lines.append(plt.Line2D(
                    [FX(left), FX(right)], [FY(top)] * 2, color=RULE_HAIR,
                    lw=0.5, transform=fig.transFigure, figure=fig, zorder=4))
            if r.get("label"):
                shift = 0.055 if r.get("sublabel") else 0.0
                fig.text(FX(M_L + GUT_GROUP), FY(top + h / 2 - shift),
                         r["label"], fontsize=FS["label"], color=INK,
                         va="center", ha="left", linespacing=1.25)
            if r.get("sublabel"):
                n_lab = r.get("label", "").count("\n") + 1
                fig.text(FX(M_L + GUT_GROUP),
                         FY(top + h / 2 + n_lab * 0.055 + 0.065),
                         r["sublabel"], fontsize=FS["note"], color=MUTED,
                         va="center", ha="left", linespacing=1.25)
            for i, c in enumerate(cols):
                rect = [FX(col_x[i]), FY(top + h - PAD_Y),
                        FX(col_w), (h - 2 * PAD_Y) / fig_h]
                if c.get(r["key"]) is None:
                    continue
                kind = r["kind"]
                if kind == "plot":
                    # the axes keeps the cell's top edge; AX_FOOT is carved
                    # off the BOTTOM so the tick labels and the x label stay
                    # inside the row instead of bleeding into the next one
                    inner = [rect[0] + FX(AX_LEFT),
                             rect[1] + AX_FOOT / fig_h,
                             rect[2] - FX(AX_LEFT + AX_RIGHT),
                             rect[3] - AX_FOOT / fig_h]
                    (paint_bonds if c["lane"] == "bonds"
                     else paint_assignments)(fig, inner, c)
                elif kind == "text":
                    used = rich_text(fig, mz, rect[0], rect[1] + rect[3],
                                     c[r["key"]], rect[2], FS["body"])
                    if used > rect[3] + 1e-6:
                        print(f"    [tight] {r['label']!r} col {i}: "
                              f"{used * fig_h:.2f}in in "
                              f"{rect[3] * fig_h:.2f}in")
                elif kind == "frames":
                    paint_frames(fig, rect, c, frames.get(id(c), {}))
                elif kind == "chat":
                    paint_chat(fig, mz, rect, c)
                elif kind == "outcome":
                    paint_outcome(fig, mz, rect, c)
            y += h

        # -- row-group labels in the outer gutter ----------------------
        y = M_T + HEAD_H
        spans, cur = [], None
        for r in rows:
            g = r.get("group") or ""
            if cur and cur[0] == g:
                cur[2] += r["h"]
            else:
                cur = [g, y, r["h"]]
                spans.append(cur)
            y += r["h"]
        for g, y0, hh in spans:
            if g:
                fig.text(FX(M_L), FY(y0 + hh / 2), g, fontsize=FS["group"],
                         color=MUTED, va="center", ha="left", linespacing=1.25)

        fig.lines.append(plt.Line2D(
            [FX(left), FX(right)], [FY(y)] * 2, color=RULE_HEAVY, lw=1.0,
            transform=fig.transFigure, figure=fig, zorder=4,
            solid_capstyle="butt"))

        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi, facecolor="white")
        plt.close(fig)
    print(f"wrote {stem}.(pdf|png|svg)   {fig_w:.2f} x {fig_h:.2f} in")


# -- figure 1: relational plasticity vs central orchestration -----------
ORCH_CH4_CHAT = [
    (653, 2, 0, "Attacking the nearest zombie now; let me know if you need focus fire."),
    (657, 0, 2, "Turning right to center the zombie. Let's focus fire on this one."),
    (659, 2, 0, "Can you confirm which zombie you are targeting?"),
    (664, 1, 0, "Moving forward to join the fight. Which zombie should I focus on?"),
]
ORCH_CH4_FRAMES = [(0, 616), (2, 636), (1, 664), (0, 692)]


def spec_pair():
    heb = dict(
        head="Relational plasticity",
        sub="3 agents · ep 3 · t 580-720",
        lane="bonds", run=HEB_RUN, exp=HEB_EXP, ep=3, win=(580, 720),
        n_agents=3, star=665, star_agent=0, mech=True,
        what=("**No task assignment** at any point in the window. The "
              "**a0-a1 bond** is already the strongest in the run and keeps "
              "rising across it, W **0.55 → 0.61**. a1 names a0, turns "
              "to clear a path for it, and a0 takes the kill."),
        frames=[(0, 655), (1, 657), (1, 659), (0, 663)],
        chat=[(653, 1, 0, "I see agent_0. I'll start targeting the zombies now."),
              (657, 0, 1, "Zombies here. Targeting one now."),
              (659, 0, 1, "Engaging combat now to clear the path to Ch5."),
              (662, 1, 0, "Turning right to find a zombie to help Agent 0.")],
        outcome="`m21_first_mob_kill`  —  **agent 0**, t = 665",
    )
    orch = dict(
        head="Central orchestration",
        sub="3 agents · ep 3 · t 580-720",
        lane="tasks", run=ORCH_RUN, exp=ORCH_EXP, ep=3, win=(580, 720),
        n_agents=3, marks=[660], mech=True,
        what=("All three agents hold **one shared subtask**, "
              "`t_ch4_clear_zombies`, staffed at t=600. It **times out for "
              "all three at t=660** and is reissued as `..._retry`. Zombies "
              "are in view from t=606."),
        frames=ORCH_CH4_FRAMES,
        chat=ORCH_CH4_CHAT,
        outcome="**no milestone** in the window",
    )
    rows = [
        dict(group="Mechanism", label="Coupling\nstate", key="mech",
             kind="plot", h=1.05,
             sublabel="left: bond strength\nright: who holds\nwhich subtask"),
        dict(group="Mechanism", label="What links\nthe agents", key="what",
             kind="text", h=0.86),
        dict(group="Observation", label="First-person\nview", key="frames",
             kind="frames", h=1.18),
        dict(group="Dialogue", label="Messages\nsent", key="chat",
             kind="chat", h=1.16),
        dict(group="Outcome", label="", key="outcome", kind="outcome",
             h=0.40, band=True),
    ]
    return dict(cols=[heb, orch], rows=rows, fig_w=7.2)


# -- figure 2: the orchestrator at 3 agents and at 6 --------------------
def spec_n6():
    a3 = dict(
        head="Orchestrator · 3 agents",
        sub="seed 42 · ep 3 · t 580-740",
        lane="tasks", run=ORCH_RUN, exp=ORCH_EXP, ep=3, win=(580, 740),
        n_agents=3, marks=[660, 720], mech=True,
        what=("**One shared subtask** for the whole team: "
              "`t_ch4_clear_zombies`, with all 3 agents staffed on it. It "
              "times out at **t=660**, is reissued, and times out again at "
              "**t=720**. The retry is written `min_agents = 3` — a "
              "genuinely joint task."),
        frames=ORCH_CH4_FRAMES,
        chat=ORCH_CH4_CHAT,
        outcome="**no milestone** in the window",
    )
    a6 = dict(
        head="Orchestrator · 6 agents",
        sub="seed 42 · ep 1 · t 595-760",
        lane="tasks", run=N6_RUN, exp=N6_EXP, ep=1, win=(595, 760),
        n_agents=6, marks=[665, 743], mech=True,
        what=("The same chamber is split into **six single-agent subtasks**, "
              "`t_ch4_kill_mob_1..6`, one per agent, each `min_agents = 1`. "
              "All six **time out together at t=665** and are reissued as "
              "six more. The only joint task written, `t_ch4_ensure_alive` "
              "(`min_agents = 6`), is **never staffed**."),
        frames=[(5, 680), (0, 682), (5, 683), (3, 683)],
        chat=[(681, 5, 0, "Target locked on zombie, ready to attack."),
              (682, 0, 5, "Agent 5, confirm zombie location so I can join the attack."),
              (682, 5, 0, "I see the zombie centered, starting attack now."),
              (683, 0, 5, "Agent 5, if you have the zombie centered, let me know when you start attacking.")],
        outcome=("`m21_first_mob_kill`  —  **agent 5 alone**, t = 682\n"
                 "→  closes **6 ×** `freed_success` at t = 683"),
    )
    rows = [
        dict(group="Task graph", label="Assigned\nsubtask", key="mech",
             kind="plot", h=1.38,
             sublabel="shade = how many\nagents share it"),
        dict(group="Task graph", label="What the\norchestrator did",
             key="what", kind="text", h=1.04),
        dict(group="Observation", label="First-person\nview", key="frames",
             kind="frames", h=1.18),
        dict(group="Dialogue", label="Messages\nsent", key="chat",
             kind="chat", h=1.28),
        dict(group="Outcome", label="", key="outcome", kind="outcome",
             h=0.54, band=True),
    ]
    return dict(cols=[a3, a6], rows=rows, fig_w=7.2)


def main() -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Teaser-table renderings of the counterfactual windows.")
    ap.add_argument("--only", choices=("pair", "n6"), default=None)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    if args.only in (None, "pair"):
        print("counterfactual_table:")
        build(spec_pair(), "counterfactual_table", args.dpi)
    if args.only in (None, "n6"):
        print("orchestrator_n6_table:")
        build(spec_n6(), "orchestrator_n6_table", args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
