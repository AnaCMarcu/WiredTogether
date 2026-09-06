"""ICLR-sized qualitative figure (single column, 5.5 in wide).

Built at final print size so typography is not shrunk by LaTeX. Layout:
compressed timeline on top (mutual-message strip + bond/co-assignment lane
with numbered zoom windows), then one full-width case panel per example,
stacked. Each panel: numbered badge, title, verdict tag, two frames per
agent (before / after) in agent-pure columns, Minecraft-style chat lines
and an outcome line.

Data loading and the arm configs are reused from make_final_figures; only
the layout is re-done for print size.

Run:  python analysis/make_iclr_figure.py [arm ...]
Out:  paper_assets/timelines/<arm>/final/iclr_overview.{pdf,png,svg}
"""
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

from paths import ASSETS  # noqa: F401  (puts siblings on sys.path)
import make_final_figures as M
from make_final_figures import (AGENT_C, INK, MUTED, PAIR_C, ARMS,
                                chamber_spans_from_steps, grab_frames,
                                load_bond_series, load_messages, load_orch,
                                load_run)

ROOT = ASSETS / "timelines"
PAPER_NAME = {"gemma3f_seed123": "gemma_hebbian3f",
              "orch789": "orchestrator",
              "qwen9b_base789": "qwen9b_base",
              "qwen9b_heb42": "qwen9b_hebbian"}

# ── print-size design constants ─────────────────────────────────────────
FIG_W = 5.50                     # ICLR single column
PANEL_EDGE = "#c3c9d0"
BADGE = "#4a4f57"
FS = dict(axis=6.6, tick=5.8, raster=5.4, chamber=5.0, title=8.0,
          verdict=5.8, agent=6.2, body=5.5, step=4.4, outcome=5.5)
RC = {"font.family": "serif",
      "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
      "axes.edgecolor": "#9aa3ad", "axes.linewidth": 0.6,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6,
      "svg.fonttype": "none", "pdf.fonttype": 42,
      "mathtext.fontset": "stix"}

# verdict tag per (arm, example key)
VERDICT = {
    ("gemma3f_seed123", "switch"): ("SUCCESS", "#1b7f5a"),
    ("gemma3f_seed123", "kill"): ("PARTIAL", "#b3730c"),
    ("gemma3f_seed123", "anvil"): ("FAILURE", "#b03a3a"),
    ("orch789", "kill"): ("SUCCESS", "#1b7f5a"),
    ("orch789", "switch"): ("SUCCESS", "#1b7f5a"),
    ("orch789", "solokill"): ("FAILURE", "#b03a3a"),
    ("qwen9b_base789", "anvil"): ("SUCCESS", "#1b7f5a"),
    ("qwen9b_base789", "combat"): ("SUCCESS", "#1b7f5a"),
    ("qwen9b_base789", "wipe"): ("FAILURE", "#b03a3a"),
    ("qwen9b_heb42", "anvil"): ("SUCCESS", "#1b7f5a"),
    ("qwen9b_heb42", "kill3"): ("SUCCESS", "#1b7f5a"),
    ("qwen9b_heb42", "phantom"): ("FAILURE", "#b03a3a"),
}

# geometry, in inches
M_L, M_R = 0.52, 0.10            # timeline margins
STRIP_H, BOND_H, XLAB_H = 0.42, 1.28, 0.30
PANEL_GAP = 0.11
MSG_LINE, MSG_GAP = 0.082, 0.020
FR_GAP = 0.040
FR_W = ((FIG_W - 0.16 - 3 * FR_GAP) / 4) * 0.86   # four frames across
FR_H = FR_W * 9 / 16
FR_X0 = (FIG_W - (4 * FR_W + 3 * FR_GAP)) / 2


def panel_height(ex):
    """Inches needed by one panel, from its actual content."""
    h = 0.11 + 0.13 + 0.045 + FR_H + 0.075          # title, headers, frames
    for _, _, _, _, txt in messages_for(ex):
        h += MSG_LINE * min(len(textwrap.wrap(txt, 88)), 2) + MSG_GAP
    return h + 0.045


def frames_for(ex):
    """Two frames per agent: first and last of that agent's column."""
    aL, aR = ex["agents"]
    cols = {aL: [], aR: []}
    for row in ex["frames"]:
        if row[0] != "F":
            continue
        for slot in (row[1], row[2]):
            if slot is not None:
                cols[slot[0]].append(slot)
    out = []
    for a in (aL, aR):
        c = cols[a]
        out.append((c[0], c[-1]) if len(c) > 1 else (c[0], c[0]))
    return out


def messages_for(ex, limit=4):
    msgs = [r for r in ex["frames"] if r[0] == "M"]
    if len(msgs) <= limit:
        return msgs
    keep = [msgs[0]]
    step = (len(msgs) - 2) / max(limit - 2, 1)
    for k in range(1, limit - 1):
        keep.append(msgs[min(int(1 + k * step), len(msgs) - 2)])
    keep.append(msgs[-1])
    return keep


def badge(fig, x, y, n, size=6.4):
    fig.text(x, y, str(n), fontsize=size, color="white", ha="center",
             va="center", zorder=6, fontweight="bold",
             bbox=dict(boxstyle="circle,pad=0.30", facecolor=BADGE,
                       edgecolor="none"))


def draw_timeline(fig, arm, run, Wr, messages, spans, H, examples):
    xmax = run["_ep_bounds"][-1][1]
    left, width = M_L / FIG_W, (FIG_W - M_L - M_R) / FIG_W
    top = H - 0.10
    axS = fig.add_axes([left, (top - STRIP_H) / H, width, STRIP_H / H])
    axB = fig.add_axes([left, (top - STRIP_H - BOND_H) / H, width,
                        BOND_H / H], sharex=axS)

    # mutual-message strip
    per_step = {}
    for t, s_, r_ in messages:
        per_step.setdefault(t, set()).add((s_, r_))
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_of = {p: len(pairs) - 1 - i for i, p in enumerate(pairs)}
    for k, (_, lo, hi) in enumerate(spans):
        axS.axvspan(lo, hi, color=M.CHAMBER_TINT[k % 2], zorder=0, lw=0)
    for t, dirs in per_step.items():
        for i, j in pairs:
            if (i, j) in dirs and (j, i) in dirs:
                axS.vlines(t, y_of[(i, j)] + 0.15, y_of[(i, j)] + 0.85,
                           color=PAIR_C[(i, j)], lw=0.32, alpha=0.85)
    axS.set_yticks([y_of[p] + 0.5 for p in pairs])
    axS.set_yticklabels([f"a{i}↔a{j}" for i, j in pairs],
                        fontsize=FS["raster"])
    for tick, p in zip(axS.get_yticklabels(), pairs):
        tick.set_color(PAIR_C[p])
    axS.set_ylim(0, len(pairs))
    axS.set_ylabel("mutual\nmessages", fontsize=FS["axis"], color=INK,
                   linespacing=1.1)
    axS.tick_params(length=0)

    # main lane
    for k, (_, lo, hi) in enumerate(spans):
        axB.axvspan(lo, hi, color=M.CHAMBER_TINT[k % 2], zorder=0, lw=0)
    if arm["lane"] in ("bonds_replay", "bonds_snap"):
        steps = np.arange(len(Wr))
        for q in PAIR_C:
            axB.plot(steps, (Wr[:, q[0], q[1]] + Wr[:, q[1], q[0]]) / 2,
                     color=PAIR_C[q], lw=1.0,
                     label=f"$\\bar{{W}}$(a{q[0]}–a{q[1]})",
                     solid_capstyle="round", zorder=3)
        for t, q in arm["joints"]:
            axB.scatter(t, (Wr[t, q[0], q[1]] + Wr[t, q[1], q[0]]) / 2,
                        marker="o", s=13, facecolor="white",
                        edgecolor=PAIR_C[q], linewidths=0.9, zorder=6)
        for t, killer in arm["kills"]:
            ys = [(Wr[t, q[0], q[1]] + Wr[t, q[1], q[0]]) / 2
                  for q in PAIR_C if killer in q]
            axB.scatter(t, max(ys), marker="*", s=34,
                        facecolor=AGENT_C[killer], edgecolor="white",
                        linewidths=0.3, zorder=6)
        ymax = float(np.nanmax(Wr))
        axB.set_ylim(0.05, ymax * 1.18)
        axB.set_ylabel("bond strength", fontsize=FS["axis"], color=INK)
        axB.legend(fontsize=FS["raster"], ncol=3, frameon=False,
                   loc="upper left", borderaxespad=0.15,
                   handlelength=1.2, columnspacing=1.0)
    elif arm["lane"] == "coassign":
        orch = load_orch(run)
        pair_y = {(0, 1): 2, (0, 2): 1, (1, 2): 0}
        team = {q: [] for q in PAIR_C}
        for tid, lo, hi, q in orch["team_iv"]:
            team[tuple(sorted(q))].append((lo, hi))
        for q, y in pair_y.items():
            for lo, hi in orch["pair_iv"][q]:
                axB.broken_barh([(lo, hi - lo)], (y - 0.22, 0.44),
                                facecolors=PAIR_C[q], alpha=0.28, lw=0)
            for lo, hi in team[q]:
                axB.broken_barh([(lo, hi - lo)], (y - 0.22, 0.44),
                                facecolors=PAIR_C[q], alpha=0.95,
                                edgecolors="white", linewidth=0.2)
        for t, q in arm["joints"]:
            axB.scatter(t, pair_y[q] + 0.36, marker="o", s=13,
                        facecolor="white", edgecolor=PAIR_C[q],
                        linewidths=0.9, zorder=6)
        for t, killer in arm["kills"]:
            axB.scatter(t, 2.74, marker="*", s=34, facecolor=AGENT_C[killer],
                        edgecolor="white", linewidths=0.3, zorder=6)
        axB.set_yticks([2, 1, 0])
        axB.set_yticklabels(["a0–a1", "a0–a2", "a1–a2"],
                            fontsize=FS["raster"])
        for tick, q in zip(axB.get_yticklabels(), PAIR_C):
            tick.set_color(PAIR_C[q])
        axB.set_ylim(-0.55, 3.0)
        axB.set_ylabel("co-assigned\npairs", fontsize=FS["axis"], color=INK,
                       linespacing=1.1)
    else:                                    # message-only arm
        axB.set_yticks([])
        axB.set_ylim(0, 1)

    lo_y, hi_y = axB.get_ylim()
    last = -1e9
    for lab, lo, hi in spans:
        if lo - last > 100 and hi - lo > 60:
            axB.text((lo + hi) / 2, lo_y + (hi_y - lo_y) * 0.012, lab,
                     fontsize=FS["chamber"], color=MUTED, ha="center",
                     va="bottom")
            last = lo
    axB.grid(axis="y", color="#eef1f4", lw=0.5)
    axB.set_axisbelow(True)
    axB.set_xlabel("environment step (cumulative)", fontsize=FS["axis"],
                   labelpad=1.5)
    axB.tick_params(labelsize=FS["tick"], pad=1.5)

    for ax in (axS, axB):
        for e, (o, _) in enumerate(run["_ep_bounds"]):
            if e:
                ax.axvline(o, color=INK, lw=0.6, ls=(0, (4, 3)), alpha=0.35)
        ax.set_xlim(-15, xmax + 15)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    plt.setp(axS.get_xticklabels(), visible=False)

    # numbered zoom windows
    marks = []
    for n, ex in enumerate(examples, 1):
        lo, hi = ex["x"]
        pad = max((hi - lo) * 0.5, xmax * 0.004)
        for ax in (axS, axB):
            ax.axvspan(lo - pad, hi + pad, color="#3f4650", alpha=0.30, lw=0,
                       zorder=4)
        axB.add_patch(Rectangle((lo - pad, 0), (hi - lo) + 2 * pad, 1,
                                transform=axB.get_xaxis_transform(),
                                facecolor="none", edgecolor="#2b3038",
                                lw=0.9, zorder=5))
        xf = left + width * ((lo + hi) / 2 + 15) / (xmax + 30)
        badge(fig, xf, (top + 0.055) / H, n, size=5.6)
        marks.append((axB, lo - pad, hi + pad))
    return marks, (top - STRIP_H - BOND_H - XLAB_H)


def draw_panel(fig, arm_name, ex, n, frames, H, y_in, panel_h):
    """One full-width case panel; y_in = its top edge in inches."""
    x0, w = 0.06 / FIG_W, (FIG_W - 0.12) / FIG_W
    y0 = (y_in - panel_h) / H
    fig.patches.append(Rectangle((x0, y0), w, panel_h / H,
                                 transform=fig.transFigure, facecolor="white",
                                 edgecolor=PANEL_EDGE, lw=0.7, zorder=1))
    pad = 0.055 / FIG_W
    y = y_in - 0.11
    badge(fig, x0 + pad + 0.012, y / H, n, size=5.6)
    fig.text(x0 + pad + 0.036, y / H, ex["title"], fontsize=FS["title"],
             color=INK, va="center", fontweight="bold", zorder=4,
             family="serif")
    tag, tcol = VERDICT.get((arm_name, ex["key"]), ("", INK))
    if tag:
        fig.text(x0 + w - pad, y / H, tag, fontsize=FS["verdict"],
                 color=tcol, va="center", ha="right", fontweight="bold",
                 zorder=4, family="serif")

    (l1, l2), (r1, r2) = frames_for(ex)
    aL, aR = ex["agents"]
    y -= 0.13
    half = 2 * FR_W + 1.5 * FR_GAP
    for a, cx_in in ((aL, FR_X0 + half / 2), (aR, FR_X0 + 1.5 * half)):
        fig.text(cx_in / FIG_W, y / H, f"agent {a}", fontsize=FS["agent"],
                 color=AGENT_C[a], ha="center", va="center",
                 fontweight="bold", zorder=4, family="serif")

    y -= 0.045
    for k, slot in enumerate((l1, l2, r1, r2)):
        a, t = slot[0], slot[1]
        xi = FR_X0 + k * (FR_W + FR_GAP)
        ax = fig.add_axes([xi / FIG_W, (y - FR_H) / H, FR_W / FIG_W,
                           FR_H / H])
        ax.axis("off")
        ax.set_zorder(3)
        fr = frames.get((a, ex["ep"], t))
        if fr is not None:
            ax.imshow(fr)
        ax.text(0.03, 0.94, f"t={t}", transform=ax.transAxes,
                fontsize=FS["step"], color="white", va="top", zorder=5,
                bbox=dict(facecolor="#000000aa", edgecolor="none", pad=0.7))
    y -= FR_H + 0.075

    for _, t, src, dst, txt in messages_for(ex):
        lines = textwrap.wrap(txt, 88) or [""]
        fig.text(x0 + pad, y / H, f"<a{src}→a{dst}>",
                 fontsize=FS["body"], color=AGENT_C[src], va="top",
                 family="monospace", fontweight="bold", zorder=4)
        for i, ln in enumerate(lines[:2]):
            fig.text(x0 + pad + 0.075, (y - i * 0.082) / H, ln,
                     fontsize=FS["body"], color=INK, va="top",
                     family="monospace", zorder=4)
        y -= MSG_LINE * min(len(lines), 2) + MSG_GAP
    return y0


def build(arm_name, arm):
    run = load_run(arm["run"])
    messages = load_messages(run)
    spans = chamber_spans_from_steps(arm, run)
    Wr = load_bond_series(arm)
    examples = sorted(arm["examples"], key=lambda e: e["x"][0])
    wanted = [(s[0], ex["ep"], s[1]) for ex in arm["examples"]
              for r in ex["frames"] if r[0] == "F"
              for s in (r[1], r[2]) if s]
    frames = grab_frames(arm, wanted)

    heights = [panel_height(ex) for ex in examples]
    H = (0.10 + STRIP_H + BOND_H + XLAB_H + 0.10
         + sum(heights) + len(examples) * PANEL_GAP + 0.04)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FIG_W, H))
        marks, y_in = draw_timeline(fig, arm, run, Wr, messages, spans, H,
                                    examples)
        y_in -= 0.10
        top_edge = y_in
        for n, (ex, ph) in enumerate(zip(examples, heights), 1):
            draw_panel(fig, arm_name, ex, n, frames, H, top_edge, ph)
            top_edge -= ph + PANEL_GAP
        outdir = ROOT / arm_name / "final"
        outdir.mkdir(parents=True, exist_ok=True)
        paper = ROOT / "for_paper"
        paper.mkdir(parents=True, exist_ok=True)
        alias = PAPER_NAME.get(arm_name, arm_name)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(outdir / f"iclr_overview.{ext}",
                        dpi=300 if ext == "png" else None, facecolor="white")
            fig.savefig(paper / f"qualitative_{alias}.{ext}",
                        dpi=300 if ext == "png" else None, facecolor="white")
        plt.close(fig)
    print(f"  wrote {arm_name}/final/iclr_overview.*  "
          f"({FIG_W:.2f} x {H:.2f} in, {len(examples)} panels)")


def main(only=None):
    ARMS["gemma3f_seed123"]["delib"] = M.GEMMA_DELIB
    for arm_name, arm in ARMS.items():
        if only and arm_name not in only:
            continue
        print(f"== {arm_name} ==")
        build(arm_name, arm)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
