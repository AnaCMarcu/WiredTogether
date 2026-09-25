#!/usr/bin/env python3
r"""make_counterfactual_compact.py - the matched pair, at ICLR figure size.

One 5.5 in figure (the ICLR text width, so it renders 1:1 at \linewidth)
holding the single strongest matched interval from the two full timelines
built by make_counterfactual_story.py.

Interval: seed 42, EPISODE 3, within-episode steps 580-720, Chamber 4. Same
map, same episode, same milestone (m21_first_mob_kill) on both sides - the
only window matched on all three that also has legible frames from both
agents on both sides. (ep 1 Ch4 is matched too, but a0's view is an unlit
passage wall throughout; the anvil/switch pair reads well but they are
different milestones, so it is a mechanism contrast, not a counterfactual.)

The window is deliberately wider than the interaction itself so the
mechanism state is visible BEFORE, AT and AFTER the moment that matters:

  Central orchestration    per-agent Ch3-entry tasks until t=600; all three
                           then hold one decomposed subtask
                           (t_ch4_clear_zombies) which times out for all
                           three at t=660 and is reassigned as
                           t_ch4_clear_zombies_retry. Zombies are in view
                           from t=606. No kill in the window.
  Relational plasticity    no assignment at any point. The a0-a1 bond is
                           already the strongest in the run and keeps
                           rising through the window; a1 names a0, turns to
                           clear a path for it, and a0 kills at t=663.

The figure states no interpretation: the column headers name the mechanism
and everything else is evidence - bond trajectories against assignment
bars, four frames per side in chronological order, four verbatim dialogue
turns. No subtitles, no outcome sentences.

Frames are cropped to 72% height; the bottom of every capture is hotbar
HUD, not scene.

Usage:  python analysis/make_counterfactual_compact.py
Out:    paper_assets/timelines/counterfactual/counterfactual_compact.{pdf,png,svg}
"""

from __future__ import annotations

import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from paths import ASSETS, group  # noqa: E402  (also puts siblings on sys.path)

import make_directive_timelines as mdt  # noqa: E402
import make_final_figures as mff  # noqa: E402
import make_team_tenure as mtt  # noqa: E402

SEED, EP = 42, 3
WIN = (580, 720)                     # within-episode steps
ORCH_RUN = group("orchestrator") / "new_exp_0_gemma_orch_villager_advisory" / f"seed_{SEED}"
HEB_RUN = group("pareto_social_3f") / "new_exp_0_gemma_si3f8" / f"seed_{SEED}"
OUT = ASSETS / "timelines" / "counterfactual"

INK, MUTED, RULE = "#31373f", "#8b95a1", "#c9ced6"
AGENT_C = mff.AGENT_C                       # a0 blue, a1 orange, a2 green
PAIR_C = mff.PAIR_C
TEAM_C = mtt.TEAM_C                         # orchestrator team-size ramp
FIG_W, FIG_H = 5.5, 3.32              # combined (--combined)
# Split panels: half the text width each, so two sit side by side at
# 0.49\linewidth. Both get the SAME canvas and the same reserved chat
# block (sized to whichever side wraps to more lines), so the two PDFs
# are identical in size and their frames align across the gutter.
PANEL_W, PANEL_H = 2.72, 3.12
WRAP = 40
FS = dict(head=9.0, chat=6.0, tag=5.5, tick=5.6, axis=6.0, lane=5.6)
RC = dict(mff.RC)
RC.update({"font.size": FS["tick"], "axes.linewidth": 0.5,
           # pixel art: never resample, never smooth
           "image.interpolation": "none", "image.resample": False,
           "xtick.major.width": 0.5, "ytick.major.width": 0.5,
           "xtick.major.size": 1.8, "ytick.major.size": 1.8})

COLS = {
    "heb": dict(
        head="Relational plasticity",
        run=HEB_RUN, exp="new_exp_0_gemma_si3f8",
        frames=[(0, 655), (1, 657), (1, 659), (0, 663)],
        chat=[(653, 1, 0, "I see agent_0. I'll start targeting the zombies now."),
              (657, 0, 1, "Zombies here. Targeting one now."),
              (659, 0, 1, "Engaging combat now to clear the path to Ch5."),
              (662, 1, 0, "Turning right to find a zombie to help Agent 0.")],
    ),
    "orch": dict(
        head="Central orchestration",
        run=ORCH_RUN, exp="new_exp_0_gemma_orch_villager_advisory",
        # t=662/682 dropped: the engine renders agent2 mid-air in both.
        # 664 is 4 steps after the reassignment, 692 well inside the new slot.
        frames=[(0, 616), (2, 636), (1, 664), (0, 692)],
        chat=[(653, 2, 0, "Attacking the nearest zombie now; let me know if you need focus fire."),
              (657, 0, 2, "Turning right to center the zombie. Let's focus fire on this one."),
              (659, 2, 0, "Can you confirm which zombie you are targeting?"),
              (664, 1, 0, "Moving forward to join the fight. Which zombie should I focus on?")],
    ),
}


def crop(fr, keep=0.72, upscale=3):
    """Drop the hotbar, then integer-upscale.

    Captures are 320x180. A frame is drawn ~1.3 in wide, so the raster is
    embedded at ~248 dpi - and matplotlib resamples images to the FIGURE dpi
    (100) when writing a vector file, i.e. down to 129 px, then smooths them
    with the default 'antialiased' filter. Both steps destroy pixel art.
    Repeating each pixel k times is lossless and pushes the embedded raster
    to ~744 dpi; rcParams['image.interpolation']='none' then tells the pdf
    and svg backends to embed it unresampled, and the viewer scales it
    nearest-neighbour."""
    if fr is None:
        return None
    fr = fr[: int(fr.shape[0] * keep)]
    return np.repeat(np.repeat(fr, upscale, axis=0), upscale, axis=1)


# ─── mechanism strips ──────────────────────────────────────────────────
def strip_heb(ax, lo, hi):
    run = mdt.load_run(HEB_RUN)
    off = run["_ep_bounds"][EP - 1][0]
    ser = mdt.load_bonds(run)
    xs = np.arange(lo, hi + 1)
    ends = []
    for q in mdt.PAIRS:
        ys = np.interp(off + xs, *ser[q])
        ax.plot(xs, ys, color=PAIR_C[q], lw=1.2, solid_capstyle="round",
                zorder=3)
        ends.append((ys[-1], q))
    placed = []
    for y, q in sorted(ends):
        while any(abs(y - p) < 0.042 for p in placed):
            y += 0.042
        placed.append(y)
        ax.annotate(f"a{q[0]}–a{q[1]}", (hi, y), xytext=(2.5, 0),
                    textcoords="offset points", fontsize=FS["lane"],
                    color=PAIR_C[q], va="center", annotation_clip=False)
    # the one quantity worth naming on this side: the bond carrying the pair
    w0 = float(np.interp(off + lo, *ser[(0, 1)]))
    w1 = float(np.interp(off + hi, *ser[(0, 1)]))
    ax.annotate(f"$\\bar{{W}}$ {w0:.2f}→{w1:.2f}", (lo + 5, w1 + 0.03),
                fontsize=FS["lane"], color=PAIR_C[(0, 1)], va="bottom")
    ax.scatter(663, float(np.interp(off + 663, *ser[(0, 1)])), marker="*",
               s=52, color=AGENT_C[0], edgecolor="white", linewidths=0.4,
               zorder=6)
    ax.set_ylim(0.26, 0.76)
    ax.set_yticks([0.3, 0.5, 0.7])
    ax.set_ylabel("bond $\\bar{W}$", fontsize=FS["axis"], color=INK,
                  labelpad=1.5)


def strip_orch(ax, lo, hi):
    run = mdt.load_run(ORCH_RUN)
    off = run["_ep_bounds"][EP - 1][0]
    task = mtt.orch_lane_data(run)["task"]
    rows = {0: 2, 1: 1, 2: 0}
    spans = []                       # (tid, t0, t1, mates) from a0's row
    for a in range(3):
        t0 = lo
        for t in range(lo + 1, hi + 2):
            if t > hi or task[off + min(t, hi), a] != task[off + t0, a]:
                tid = task[off + t0, a]
                if tid is not None:
                    mates = sum(1 for b in range(3) if b != a
                                and task[off + t0, b] == tid)
                    ax.broken_barh([(t0 + 1.2, t - t0 - 2.4)],
                                   (rows[a] - 0.30, 0.60),
                                   facecolors=TEAM_C[mates], linewidth=0,
                                   zorder=3)
                    if a == 0:
                        spans.append((tid, t0, min(t, hi), mates))
                t0 = t
    # assignment identity, only where the block is wide enough to hold it
    for tid, t0, t1, mates in spans:
        if t1 - t0 < (hi - lo) * 0.24:
            continue
        lab = tid.replace("t_ch4_", "").replace("t_ch3_", "").replace("_", " ")
        ax.text((t0 + t1) / 2, 2, lab, fontsize=FS["lane"],
                color="white" if mates == 2 else INK, ha="center",
                va="center", zorder=5)
    ax.scatter([660] * 3, [2, 1, 0], marker="x", s=18, color=INK,
               linewidths=0.9, zorder=6)
    ax.set_ylim(-0.60, 2.60)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["a0", "a1", "a2"], fontsize=FS["lane"])
    for tick, a in zip(ax.get_yticklabels(), (0, 1, 2)):
        tick.set_color(AGENT_C[a])
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.set_ylabel("assigned\nsubtask", fontsize=FS["axis"], color=INK,
                  labelpad=1.5, linespacing=0.95)


def _load_frames():
    for key, c in COLS.items():
        c["_fr"] = mff.grab_frames(dict(run=c["run"], exp=c["exp"], seed=SEED),
                                   [(a, EP, t) for a, t in c["frames"]])
        print(f"  {key}: frames {len(c['_fr'])}/{len(c['frames'])}")


def _chat_lines(key):
    return sum(len(textwrap.wrap(x, WRAP)) for *_h, x in COLS[key]["chat"])


def build_panel(key: str, stem: str):
    """One mechanism, on its own canvas, with no title."""
    c = COLS[key]
    lo, hi = WIN
    max_lines = max(_chat_lines(k) for k in COLS)      # keep both panels equal
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(PANEL_W, PANEL_H))
        L, R = 0.035, 0.985

        ax = fig.add_axes([L + 0.135, 0.815, (R - L) - 0.150 - 0.105, 0.150])
        (strip_heb if key == "heb" else strip_orch)(ax, lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_xticks([600, 650, 700])
        ax.tick_params(labelsize=FS["tick"], pad=1.2)
        ax.set_xlabel("environment step", fontsize=FS["axis"], color=INK,
                      labelpad=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(RULE)

        gap = 0.016
        fw = (R - L - gap) / 2
        fh = fw * PANEL_W / PANEL_H * (9 / 16) * 0.72
        top = 0.735
        for k, (a, t_) in enumerate(c["frames"]):
            fx = L + (k % 2) * (fw + gap)
            fy = top - (k // 2) * (fh + 0.016) - fh
            axf = fig.add_axes([fx, fy, fw, fh])
            axf.axis("off")
            fr = crop(c["_fr"].get((a, EP, t_)))
            if fr is not None:
                axf.imshow(fr, aspect="auto", interpolation="none")
            else:
                axf.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eef0f3",
                                        transform=axf.transAxes))
            axf.text(0.025, 0.93, f"a{a}  t={t_}", transform=axf.transAxes,
                     fontsize=FS["tag"], color="white", va="top",
                     fontweight="bold",
                     bbox=dict(facecolor=AGENT_C[a] + "e0", edgecolor="none",
                               pad=1.0))

        y = top - 2 * (fh + 0.016) - 0.034
        for t_, src, dst, txt in c["chat"]:
            fig.text(L, y, f"a{src}→a{dst}", fontsize=FS["chat"],
                     color=AGENT_C[src], va="top", fontweight="bold",
                     family="DejaVu Sans Mono")
            lines = textwrap.wrap(txt, WRAP)
            fig.text(L + 0.150, y, "\n".join(lines), fontsize=FS["chat"],
                     color=INK, va="top", family="DejaVu Sans Mono",
                     linespacing=1.26)
            y -= (len(lines) * FS["chat"] * 1.30 / 72 + 0.028) / PANEL_H

        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=600,
                        facecolor="white")
        plt.close(fig)
    print(f"wrote {stem}.(pdf|png|svg)   [{_chat_lines(key)}/{max_lines} chat lines]")


def main() -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    _load_frames()
    build_panel("heb", "counterfactual_compact_a")
    build_panel("orch", "counterfactual_compact_b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
