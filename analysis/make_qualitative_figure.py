"""Publication-quality qualitative timeline figure (v2).

Top (~30%): directed-message raster + the bond plot with chamber bands,
event glyphs and zoom-window boxes drawn directly on the graph. Bottom:
three example panels, each a two-agent parallel storyboard — the two
protagonists' frames side by side in time order with directional message
arrows between them, the social-module deliberation that matches the
window (aligned via referenced-bond values), and the directed bond change.

All content from run logs; bond values from the validated per-step replay.
Outputs: qualitative_figure.png (300 dpi) / .pdf / .svg.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

from paths import ASSETS, RUNS  # noqa: E402  (also puts siblings on sys.path)
from make_directive_timelines import load_run  # noqa: E402
from replay_hebbian_terms import load_inputs  # noqa: E402
from prototype_three_factor_rule import replay_three_factor  # noqa: E402

# ── design constants ────────────────────────────────────────────────────
INK, MUTED, RULE = "#31373f", "#8b95a1", "#d9dee4"
AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}      # Okabe-Ito
PAIR_C = {(0, 1): "#5B4EA4", (0, 2): "#B2652C", (1, 2): "#2E8B6E"}
CHAMBER_TINT = ["#f2f5f8", "#ffffff"]
FS = dict(axis=11, tick=10, panel_title=11.5, caption=8.0, body=8.2,
          header=8.6, raster=7.5, chamber=8.5)
PANEL_EDGE = "#c3c9d0"
DPI = 300
MAXLEN = 62

RUN = RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f/seed_123"
OUTDIR = ASSETS / "timelines/gemma3f_seed123"

JOINT_FIRES = [442, 1407, 2401]                    # m17+m18 joint (pair a0-a1)
KILLS = [(722, 1), (1683, 2), (2642, 2)]           # m21 milestone, killer

# Panels: two protagonists; "rows" interleave frames and messages in time
# order. ("F", (a, t_local, cap)|None left, ...|None right)  /
# ("M", t_local, src, dst, verbatim text)
EXAMPLES = [
    dict(x=(244, 268), ep=1, agents=(1, 2), dpairs=[(1, 2)],
         title="Failure in anvil dig",
         rows=[
             ("F", (1, 246, "a1 at the anvils; a2 in view"),
              (2, 244, "a2 at the anvils; a1 in view")),
             ("M", 250, 1, 2, "I'm moving toward the center anvils now. Let's coordinate the hits soon."),
             ("M", 253, 2, 1, "Approaching center now. Ready to coordinate digs."),
             ("M", 255, 2, 1, "Sheep here, ready to dig."),
             ("F", (1, 258, "a1, 12 steps on: blank corridors"),
              (2, 266, "a2: the farm pen instead")),
         ],
         delib=[(1, 254, "Both agents have communicated their movement towards the anvils, suggesting shared task progress. Since no direct request for help was made, I will focus on my own movement [...]"),
                (2, 257, "Agent 1 is actively moving and communicating — I should stay focused on my own movement [...] rather than interrupting the established coordination.")],
         outcome="anvil talk, no dig — nobody asked for help"),
    dict(x=(714, 726), ep=1, agents=(1, 0), dpairs=[(1, 0)],
         title="Cooperative zombie kill",
         rows=[
             ("M", 717, 1, 0, "Zombie here, focusing fire now."),
             ("F", (1, 716, "a1 engages the zombie"),
              (0, 716, "a0 arrives: zombie + teammates")),
             ("M", 718, 0, 1, "Orientation should be right now; ready to engage zombies."),
             ("M", 720, 1, 0, "Boss is centered; I am attacking it now."),
             ("F", (1, 723, "kill (+60), credited to a1"),
              (0, 720, "a0 holds position, one block away")),
         ],
         delib=[(1, 692, "I will offer assistance if they indicate a need [...] their request is informational rather than a direct ask for help."),
                (0, 700, "Agent_1 has communicated that they are starting combat against a zombie [...] since they haven't explicitly asked for help, I will focus on my own task for now.")],
         outcome="kill succeeds; support never engages — no one asked"),
    dict(x=(2399, 2416), ep=3, agents=(0, 2), dpairs=[(0, 1), (0, 2)],
         title="Switch puzzle solved",
         rows=[
             ("M", 400, 0, 2, "Need agent_2 to press switch C to open my door."),
             ("M", 401, 2, 0, "Agent 0, I am still in the cell. I will look for the switch C now."),
             ("F", (0, 403, "a0 digs its switch (frees a1)"),
              (2, 405, "a2 spots its own switch")),
             ("M", 403, 2, 0, "Agent 0 is pressing switch now, I will turn to press mine."),
             ("M", 406, 2, 0, "Switch is on the south wall, turning right to face it now."),
             ("F", (2, 407, "a2 turns to the locked door"),
              (0, 413, "a0 freed by a2's answering press")),
             ("M", 414, 0, 2, "Door open now, moving out."),
         ],
         delib=[(0, 2497, "Agent 1 [...] has requested me to 'Keep an eye on the door state' — I should respond to offer help / acknowledge the request to maintain the alliance.")],
         outcome="explicit request, answered press — both doors open"),
]


# ── data loading ────────────────────────────────────────────────────────
def load_messages(run):
    out = []
    for e, (s0, _) in enumerate(run["_ep_bounds"]):
        p = run["_run_dir"] / "episodes" / f"ep_{e + 1:04d}" / "messages.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            m = json.loads(line)
            snd = int(re.search(r"(\d+)", m["sender"]).group(1))
            rcv = int(re.search(r"(\d+)", m["receiver"]).group(1))
            out.append((m["t"] + s0, snd, rcv))
    return out


def load_deliberations(run_dir):
    p = run_dir / "llm_logs" / "social_module.log"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    calls = {0: [], 1: [], 2: []}
    for m in re.finditer(r"SocialModule\[agent_(\d)\]:\s+Response:\s*(\{.*?\n\})",
                         txt, re.S):
        try:
            calls[int(m.group(1))].append(json.loads(m.group(2)))
        except json.JSONDecodeError:
            pass
    return calls


def align_delib(calls, Wr, agent, window, tol=0.02):
    """The call whose referenced_bonds best match replayed W near the
    window (cumulative steps). Returns (est_step, response) or None."""
    if not calls:
        return None
    lo, hi = window
    best = None
    for r in calls.get(agent, []):
        refs = r.get("referenced_bonds") or {}
        vals = {}
        for k, v in refs.items():
            m = re.search(r"(\d+)", str(k))
            if m and isinstance(v, (int, float)) and int(m.group(1)) != agent:
                vals[int(m.group(1))] = float(v)
        if not vals:
            continue
        for t in range(max(0, lo - 80), min(len(Wr), hi + 80)):
            err = max(abs(Wr[t, agent, j] - v) for j, v in vals.items())
            if err < tol and (best is None or err < best[0]):
                best = (err, t, r)
    return (best[1], best[2]) if best else None


def grab_frames(wanted):
    import cv2
    out, caps = {}, {}
    for a, ep, t in wanted:
        key = (a, ep)
        if key not in caps:
            p = RUN / f"seed_123_agent_{a}_ep{ep}.mp4"
            caps[key] = cv2.VideoCapture(str(p)) if p.exists() else None
        cap = caps[key]
        if cap is None:
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(t - 1, 0), n - 1))
        ok, fr = cap.read()
        if ok:
            fr = cv2.convertScaleAbs(fr, alpha=1.45, beta=12)
            out[(a, ep, t)] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    return out


# ── chamber segmentation from per-step data ─────────────────────────────
def chamber_spans_from_steps(inp, run, min_len=12):
    """Majority chamber per step (unknowns ignored), segmented; episode
    boundaries always cut segments."""
    ch = inp["chamber"]
    T = len(ch)
    maj = np.zeros(T, dtype=int)
    for t in range(T):
        known = ch[t][ch[t] > 0]
        maj[t] = int(np.median(known)) if len(known) else (maj[t - 1] if t else 0)
    bounds = {b for b, _ in run["_ep_bounds"]} | {T}
    spans, t0 = [], 0
    for t in range(1, T + 1):
        if t == T or maj[t] != maj[t0] or t in bounds:
            if maj[t0] > 0 and t - t0 >= min_len:
                spans.append((f"Ch{maj[t0]}", t0, t))
            elif spans and maj[t0] > 0 and t0 not in bounds:
                spans[-1] = (spans[-1][0], spans[-1][1], t)
            t0 = t
    return spans


def shade_chambers(ax, spans):
    for k, (_, lo, hi) in enumerate(spans):
        ax.axvspan(lo, hi, color=CHAMBER_TINT[k % 2], zorder=0, linewidth=0)
        ax.axvline(lo, color=RULE, lw=0.7, zorder=1)


# ── timeline ────────────────────────────────────────────────────────────
def plot_message_raster(ax, messages, spans):
    """Mutual exchanges only: a tick where the pair messaged EACH OTHER
    in the same step (a→b and b→a)."""
    shade_chambers(ax, spans)
    per_step = {}
    for t, snd, rcv in messages:
        per_step.setdefault(t, set()).add((snd, rcv))
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_of = {pr: len(pairs) - 1 - i for i, pr in enumerate(pairs)}
    for t, dirs in per_step.items():
        for i, j in pairs:
            if (i, j) in dirs and (j, i) in dirs:
                y = y_of[(i, j)]
                ax.vlines(t, y + 0.14, y + 0.86, color=PAIR_C[(i, j)],
                          lw=0.5, alpha=0.85, zorder=2)
    ax.set_yticks([y_of[pr] + 0.5 for pr in pairs])
    ax.set_yticklabels([f"a{i}↔a{j}" for i, j in pairs],
                       fontsize=FS["raster"] + 0.5)
    for tick, pr in zip(ax.get_yticklabels(), pairs):
        tick.set_color(PAIR_C[pr])
    ax.set_ylim(0, len(pairs))
    ax.set_ylabel("mutual\nmessages", fontsize=FS["axis"], color=INK)
    ax.tick_params(length=0)


def plot_bonds(ax, Wr, spans):
    shade_chambers(ax, spans)
    steps = np.arange(len(Wr))
    for q in PAIR_C:
        ys = (Wr[:, q[0], q[1]] + Wr[:, q[1], q[0]]) / 2
        ax.plot(steps, ys, color=PAIR_C[q], lw=2.0,
                label=f"$\\bar{{W}}$(a{q[0]}–a{q[1]})",
                solid_capstyle="round", zorder=3)
    for t in JOINT_FIRES:                       # events on the graph itself
        y = (Wr[t, 0, 1] + Wr[t, 1, 0]) / 2
        ax.scatter(t, y, marker="o", s=34, facecolor="white",
                   edgecolor=PAIR_C[(0, 1)], linewidths=1.5, zorder=6)
    for t, killer in KILLS:
        ys = [(Wr[t, q[0], q[1]] + Wr[t, q[1], q[0]]) / 2
              for q in PAIR_C if killer in q]
        ax.scatter(t, max(ys), marker="*", s=110,
                   facecolor=AGENT_C[killer], edgecolor="white",
                   linewidths=0.5, zorder=6)
    ymax = float(np.nanmax(Wr))
    last = -1e9
    for lab, lo, hi in spans:                   # labels along the bottom edge
        if lo - last > 100 and hi - lo > 60:
            ax.text((lo + hi) / 2, 0.075, lab, fontsize=FS["chamber"],
                    color=MUTED, ha="center", va="bottom")
            last = lo
    ax.grid(axis="y", color="#eef1f4", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("bond strength", fontsize=FS["axis"], color=INK)
    ax.set_xlabel("environment step (cumulative)", fontsize=FS["axis"],
                  labelpad=2)
    ax.tick_params(labelsize=FS["tick"])
    ax.legend(fontsize=9, ncol=3, frameon=False, loc="upper left",
              borderaxespad=0.2)
    ax.set_ylim(0.05, ymax * 1.21)


def zoom_box(ax, Wr, ex):
    """Full-height band at exactly the interaction interval."""
    lo, hi = ex["x"]
    ax.add_patch(Rectangle((lo, 0), hi - lo, 1,
                           transform=ax.get_xaxis_transform(),
                           facecolor="#5b6472", alpha=0.10, edgecolor=INK,
                           lw=1.2, zorder=5))
    return lo, hi


def draw_zoom_connector(fig, ax, box, x0, w, ytop):
    lo, hi = box
    tr = ax.get_xaxis_transform()
    for x_data, x_fig in ((lo, x0), (hi, x0 + w)):
        fig.add_artist(ConnectionPatch(
            xyA=(x_data, 0.0), coordsA=tr,
            xyB=(x_fig, ytop), coordsB=fig.transFigure,
            color="#aab3bd", lw=0.9, ls=(0, (3, 2)), zorder=0))


# ── example panels ──────────────────────────────────────────────────────
def truncate(txt, n=MAXLEN):
    return txt if len(txt) <= n else txt[:n - 1].rstrip() + "…"


def panel(fig, ex, frames, delibs, Wr, x0, w, y0, h):
    fig.patches.append(Rectangle((x0, y0), w, h, transform=fig.transFigure,
                                 facecolor="white", edgecolor=PANEL_EDGE,
                                 lw=1.0, zorder=1))
    pad = 0.008
    aL, aR = ex["agents"]
    y = y0 + h - 0.018
    fig.text(x0 + pad, y, ex["title"], fontsize=FS["panel_title"],
             color=INK, va="center", fontweight="bold", zorder=4)
    y -= 0.019
    fig.text(x0 + pad, y,
             f"episode {ex['ep']}, steps {ex['x'][0]}–{ex['x'][1]}",
             fontsize=8, color=MUTED, va="center", zorder=4)
    y -= 0.021
    colw = (w - 3 * pad) / 2
    for a, cx in ((aL, x0 + pad + colw / 2),
                  (aR, x0 + 2 * pad + colw * 1.5)):
        fig.text(cx, y, f"agent {a}", fontsize=9.5, color=AGENT_C[a],
                 ha="center", va="center", fontweight="bold", zorder=4)
    y -= 0.010

    fr_h, cap_h, msg_h = 0.080, 0.023, 0.0195
    for row in ex["rows"]:
        if row[0] == "F":
            y -= fr_h
            for slot, xleft in ((row[1], x0 + pad),
                                (row[2], x0 + 2 * pad + colw)):
                if slot is None:
                    continue
                a, t, cap = slot
                ax = fig.add_axes([xleft, y, colw, fr_h])
                ax.axis("off")
                ax.set_zorder(3)
                fr = frames.get((a, ex["ep"], t))
                if fr is not None:
                    ax.imshow(fr)
                ax.text(0.02, 0.95, f"t={t}", transform=ax.transAxes,
                        fontsize=6.6, color="white", va="top", zorder=5,
                        bbox=dict(facecolor="#00000099", edgecolor="none",
                                  pad=1.2))
                ax.text(0.5, -0.09, truncate(cap, 46),
                        transform=ax.transAxes, fontsize=FS["caption"],
                        color=INK, ha="center", va="top")
            y -= cap_h + 0.004
        else:
            _, t, src, dst, txt = row
            y -= msg_h
            l2r = src == aL
            xa, xb = x0 + pad + 0.012, x0 + w - pad - 0.012
            fig.patches.append(FancyArrowPatch(
                (xa if l2r else xb, y + 0.004),
                (xb if l2r else xa, y + 0.004),
                transform=fig.transFigure, arrowstyle="-|>",
                mutation_scale=9, color=AGENT_C[src], lw=1.1, zorder=3,
                shrinkA=0, shrinkB=0, alpha=0.85))
            fig.text(x0 + w / 2, y + 0.0085, f"“{truncate(txt)}”",
                     fontsize=FS["body"], color=INK, ha="center",
                     va="bottom", zorder=4,
                     bbox=dict(facecolor="white", edgecolor="none",
                               pad=0.4))
            y -= 0.0045
    y -= 0.014

    fig.text(x0 + pad, y, "Social deliberation", fontsize=FS["header"],
             color=MUTED, va="top", fontweight="bold", zorder=4)
    y -= 0.016
    entries = ex.get("delib")
    if entries is None:
        entries = []
        for a in ex["agents"]:
            hit = align_delib(delibs, Wr, a, ex["x"])
            if hit:
                entries.append((a, hit[0],
                                (hit[1].get("reasoning") or "").strip()))
                break
    if entries:
        import textwrap
        for a, step, reason in entries:
            fig.text(x0 + pad + 0.002, y, f"a{a} (t≈{step}):",
                     fontsize=FS["body"], color=AGENT_C[a], va="top",
                     fontweight="bold", zorder=4)
            y -= 0.0135
            for ln in textwrap.wrap(reason, 70)[:3]:
                fig.text(x0 + pad + 0.006, y, ln, fontsize=FS["body"],
                         color=INK, va="top", zorder=4)
                y -= 0.0122
            y -= 0.005
    else:
        fig.text(x0 + pad + 0.002, y,
                 "(no bond-matched social-module call in this window)",
                 fontsize=FS["body"], color=MUTED, style="italic",
                 va="top", zorder=4)
        y -= 0.017
    y -= 0.004

    # bond-trend snippets: W_{i→j} solid, W_{j→i} dashed, window shaded
    y -= 0.014
    spark_w, spark_h = 0.105, 0.040
    xcur = x0 + pad + 0.004
    for i, j in ex["dpairs"]:
        t0 = max(ex["x"][0] - 60, 0)
        t1 = min(ex["x"][1] + 80, len(Wr) - 1)
        col = PAIR_C[tuple(sorted((i, j)))]
        wb = float(Wr[max(ex["x"][0] - 1, 0), i, j])
        wa = float(Wr[min(ex["x"][1] + 25, len(Wr) - 1), i, j])
        d = wa - wb
        fig.text(xcur, y,
                 f"$W_{{a{i}\\rightarrow a{j}}}$"
                 f"  {wb:.2f}→{wa:.2f} ({'+' if d >= 0 else '−'}{abs(d):.2f})",
                 fontsize=8.4, color=col, va="top", fontweight="bold",
                 zorder=4)
        axS = fig.add_axes([xcur, y - 0.014 - spark_h, spark_w, spark_h])
        axS.set_zorder(3)
        ts = np.arange(t0, t1)
        axS.plot(ts, Wr[t0:t1, i, j], color=col, lw=1.5)
        axS.plot(ts, Wr[t0:t1, j, i], color=col, lw=0.8, ls=(0, (2, 1.5)),
                 alpha=0.6)
        axS.axvspan(ex["x"][0], ex["x"][1], color="#5b6472", alpha=0.12,
                    linewidth=0)
        for s_ in axS.spines.values():
            s_.set_visible(False)
        axS.set_xticks([])
        axS.set_yticks([])
        xcur += spark_w + 0.040
    y -= 0.014 + spark_h + 0.010
    fig.text(x0 + pad, y, ex["outcome"], fontsize=FS["body"], color=INK,
             va="top", zorder=4)


# ── assembly ────────────────────────────────────────────────────────────
def main():
    run = load_run(RUN)
    inp = load_inputs(RUN)
    Wr = replay_three_factor(inp)["W"]
    messages = load_messages(run)
    delibs = load_deliberations(RUN)
    spans = chamber_spans_from_steps(inp, run)
    wanted = [(s[0], ex["ep"], s[1]) for ex in EXAMPLES for r in ex["rows"]
              if r[0] == "F" for s in (r[1], r[2]) if s]
    frames = grab_frames(wanted)
    print(f"frames {len(frames)}/{len(set(wanted))}; "
          f"deliberations {'loaded' if delibs else 'MISSING'}; "
          f"chamber spans {len(spans)}")

    with plt.rc_context({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral",
                                        "DejaVu Serif"],
                         "axes.edgecolor": "#9aa3ad",
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "mathtext.fontset": "stix"}):
        fig = plt.figure(figsize=(13.6, 10.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.40, 1.0], hspace=0.08,
                              left=0.065, right=0.985, top=0.975,
                              bottom=0.70)
        axR = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1], sharex=axR)
        plot_message_raster(axR, messages, spans)
        plot_bonds(axB, Wr, spans)
        boxes = [zoom_box(axB, Wr, ex) for ex in EXAMPLES]
        for ax in (axR, axB):
            for e, (o, _) in enumerate(run["_ep_bounds"]):
                if e:
                    ax.axvline(o, color=INK, lw=0.9, ls=(0, (4, 3)),
                               alpha=0.35)
            ax.set_xlim(-15, run["_ep_bounds"][-1][1] + 15)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        plt.setp(axR.get_xticklabels(), visible=False)

        n = len(EXAMPLES)
        cw, gap = 0.305, 0.009
        left = 0.5 - (n * cw + (n - 1) * gap) / 2
        for k, (ex, box) in enumerate(zip(EXAMPLES, boxes)):
            x0 = left + k * (cw + gap)
            panel(fig, ex, frames, delibs, Wr, x0, cw, 0.005, 0.645)
            draw_zoom_connector(fig, axB, box, x0, cw, 0.653)

        OUTDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUTDIR / f"qualitative_figure.{ext}",
                        dpi=DPI if ext == "png" else None, facecolor="white")
        plt.close(fig)
        print(f"wrote {OUTDIR}/qualitative_figure.(png|pdf|svg)")


if __name__ == "__main__":
    main()
