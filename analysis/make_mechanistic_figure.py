"""Mechanistic case-study figure: long-horizon bond dynamics + magnified
temporal windows (ICLR/NeurIPS style).

Design: a global 3000-step social trajectory on top (chamber bands from the
per-step step_log majority — NOT the unreliable episode-summary
chamber_entry_steps — a communication lane, and the pairwise Hebbian bond
trajectories with a directed-spread envelope), with three case-study windows
(A1/A2/A3) outlined on the trajectory and magnified below. Every zoom is a
temporal story: local directed W dynamics with event markers, semantic
frames, the real logged conversation, the logged social-module routing
decision, and the local Hebbian consequence with per-term attribution
(reward-gated eligibility vs baseline co-activity, from the validated
replay of the deployed rule).

Case studies (all numbers computed live from the replay):
  A1  failed coordination  — anvil approach talk, no joint outcome
  A2  successful support   — a1 requests, a0 presses the switch, m17+m18
                             joint fire (+40/+60), both directed bonds jump
  A3  reciprocal help      — third repetition of the same a0-a1 chain;
                             re-consolidation on an already-strong bond and
                             a2's answering press frees a0 in turn

Variants (same data, genuinely different layouts):
  A  global + three equal zoom panels
  B  global + dominant A3 panel (visual climax)
  C  scientific comic strip: each example is a full-width causal band
  D  one giant A3 zoom + two compact counterexamples
Each × two communication modes: 'raster' (directed message raster) and
'strip' (dominant-pair strip).

Nothing is fabricated: message text is looked up from messages.jsonl by
(t, sender, receiver) keys; missing keys abort. Deliberation text is not in
the local logs, so panels state the logged routing decision instead.

Outputs (into paper_assets/timelines/gemma3f_seed123/):
  mechanistic_v{A,B,C,D}_{raster,strip}.{png,pdf}
  mechanistic_figure_stats.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyArrow, Rectangle

from paths import ASSETS, REPO, RUNS  # noqa: E402  (also puts siblings on sys.path)

from make_directive_timelines import load_run  # noqa: E402
from replay_hebbian_terms import load_inputs  # noqa: E402
from prototype_three_factor_rule import replay_three_factor  # noqa: E402

# ── design constants ────────────────────────────────────────────────────
INK, MUTED, RULE = "#31373f", "#8b95a1", "#d9dee4"
AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}          # Okabe-Ito
PAIR_C = {(0, 1): "#5B4EA4", (0, 2): "#B2652C", (1, 2): "#2E8B6E"}
CHAMBER_TINT = ["#f4f6f9", "#ffffff"]
REWARD_FILL = "#f2b3a0"                                       # reward-gated dW
PANEL_EDGE = "#c3c9d0"
DPI = 250

FS = dict(title=15, axis=10, tick=9, panel_title=11.5, caption=7.3,
          body=8.6, header=8.8, raster=7.5, chamber=8.5, small=7.2,
          dw=9.2, local=8)

RUN = RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f/seed_123"
OUTDIR = ASSETS / "timelines/gemma3f_seed123"
TITLE = ("Long-horizon bond dynamics and mechanistic case studies — "
         "Gemma-E4B + three-factor Hebbian (seed 123)")
MAXLEN = 46

# Case studies. `x` = window (cumulative steps); `wpad` extends the local
# W plot beyond the window so consolidation/decay stays visible; `dpairs` =
# directed bonds shown locally (source agent's colour); `convo` keys are
# (t_cum, sender, receiver) — text is PULLED FROM THE LOGS at load time.
EXAMPLES = [
    dict(tag="A1", x=(248, 272), wpad=20, ep=1,
         dpairs=[(1, 2), (2, 1)],
         short_title="A1. Failed coordination",
         title="A1. Failed coordination: anvil talk, no dig",
         story="approach talk only — nobody digs, no milestone",
         frames=[(0, 254, "a0: blank wall"),
                 (1, 258, "a1: blank wall"),
                 (2, 262, "a2: farm animals"),
                 (0, 266, "a0 at an anvil, alone"),
                 (0, 268, "a0 eyes a sheep")],
         convo=[(250, 1, 2), (253, 2, 1), (255, 2, 1), (266, 0, 1)],
         cause="coordination talk, no joint action, no reward",
         cause_short="talk only, no joint action, no reward"),
    dict(tag="A2", x=(428, 455), wpad=20, ep=1,
         dpairs=[(1, 0), (0, 1)],
         short_title="A2. Successful support",
         title="A2. Successful support: first switch chain",
         story="a1 asks, a0 presses its switch, both milestones fire",
         frames=[(1, 428, "a1: door locked (red)"),
                 (0, 442, "a0 centers the switch"),
                 (1, 443, "door opens; a0 visible"),
                 (0, 446, "a0 works the switch"),
                 (1, 449, "a1 freed, heads north")],
         convo=[(428, 1, 0), (442, 0, 1), (443, 1, 0), (445, 0, 2),
                (452, 2, 0)],
         cause="request + joint switch press + milestone rewards",
         cause_short="request + joint press + rewards"),
    dict(tag="A3", x=(2396, 2420), wpad=20, ep=3,
         dpairs=[(0, 1), (1, 0)],
         short_title="A3. Reciprocal help",
         title="A3. Reciprocal help: the chain repeats",
         story="3rd a0–a1 chain; a2's answering press frees a0",
         frames=[(1, 2401, "a1: door locked (red)"),
                 (0, 2403, "a0 digs its switch"),
                 (1, 2404, "a1 freed, moves north"),
                 (2, 2405, "a2 turns to its switch"),
                 (0, 2413, "a0 freed in turn")],
         convo=[(2401, 0, 2), (2403, 1, 0), (2403, 2, 0), (2413, 0, 1)],
         cause="repeated reciprocal help + milestone rewards",
         cause_short="repeated reciprocal help + rewards"),
]


# ── data assembly ───────────────────────────────────────────────────────
def load_messages(run):
    """All directed messages: dicts with cumulative t, ids and text."""
    out = []
    for e, (s0, _) in enumerate(run["_ep_bounds"]):
        p = run["_run_dir"] / "episodes" / f"ep_{e + 1:04d}" / "messages.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            m = json.loads(line)
            out.append(dict(
                t=m["t"] + s0,
                snd=int(re.search(r"(\d+)", m["sender"]).group(1)),
                rcv=int(re.search(r"(\d+)", m["receiver"]).group(1)),
                text=m["text"], routing=m.get("routing", "?")))
    return out


def chamber_spans_steplog(run):
    """(label, lo, hi) chamber bands from the per-step majority chamber in
    step_log.csv (forward-filled, 15-step hysteresis). The episode-summary
    chamber_entry_steps disagree with the logged per-step chamber (e.g.
    ep1 claims Ch3 at 599 while step_log has the team in ch3 from 399),
    so the step log is the source of truth here."""
    import csv
    spans = []
    for e, (s0, s1) in enumerate(run["_ep_bounds"]):
        per_t = defaultdict(list)
        with open(run["_run_dir"] / "episodes" / f"ep_{e + 1:04d}" /
                  "step_log.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                per_t[int(r["step"])].append(r["chamber"] or "?")
        L = s1 - s0
        seq, prev = [], "ch1"
        for t in range(L):
            c = Counter(per_t.get(t, ["?"])).most_common(1)[0][0]
            if c == "?":
                c = prev
            seq.append(c)
            prev = c
        cur, t0, t = seq[0], 0, 1
        while t < L:
            if seq[t] != cur:
                j = t
                while j < L and seq[j] == seq[t]:
                    j += 1
                if j - t >= 15:
                    spans.append((cur, s0 + t0, s0 + t))
                    cur, t0 = seq[t], t
                t = j
            else:
                t += 1
        spans.append((cur, s0 + t0, s0 + L))
    return [(c.replace("ch", "Ch"), lo, hi) for c, lo, hi in spans]


def milestone_marks(run):
    """Joint switch chains (m17+m18 same step) and first-kill events."""
    by_step = defaultdict(list)
    for ev in run.get("milestone_events", []):
        by_step[int(ev["step"])].append(ev)
    chains, kills = [], []
    for s, evs in sorted(by_step.items()):
        mids = {e["milestone_id"] for e in evs}
        if "m17_switch_pressed" in mids and "m18_door_opened" in mids:
            chains.append(s)
        if "m21_first_mob_kill" in mids:
            kills.append(s)
    return chains, kills


def window_events(run, lo, hi):
    """Milestone events (with reward) inside [lo, hi]."""
    out = []
    for ev in run.get("milestone_events", []):
        if lo <= int(ev["step"]) <= hi and not str(
                ev["milestone_id"]).startswith("m_comm"):
            a = int(re.search(r"(\d+)", str(ev["contributor"])).group(1))
            out.append((int(ev["step"]), ev["milestone_id"], a,
                        float(ev["reward"])))
    return out


def grab_frames(wanted, ep_off):
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
        t_loc = t - ep_off[ep]
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(t_loc - 1, 0), n - 1))
        ok, fr = cap.read()
        if ok:
            fr = cv2.convertScaleAbs(fr, alpha=1.45, beta=12)
            out[(a, ep, t)] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    for cap in caps.values():
        if cap is not None:
            cap.release()
    return out


def build_context():
    run = load_run(RUN)
    ep_off = {e + 1: b[0] for e, b in enumerate(run["_ep_bounds"])}
    inp = load_inputs(RUN)
    rep = replay_three_factor(inp)
    messages = load_messages(run)
    msg_by_key = {}
    for m in messages:
        msg_by_key.setdefault((m["t"], m["snd"], m["rcv"]), m)
    spans = chamber_spans_steplog(run)
    chains, kills = milestone_marks(run)

    examples = []
    for ex in EXAMPLES:
        ex = dict(ex)
        convo = []
        for key in ex["convo"]:
            m = msg_by_key.get(key)
            if m is None:
                raise SystemExit(f"{ex['tag']}: message {key} not in logs")
            convo.append(m)
        ex["convo"] = convo
        lo, hi = ex["x"]
        t1 = min(hi + ex["wpad"], len(rep["W"]) - 1)
        stats = []
        for i, j in ex["dpairs"]:
            wb = float(rep["W"][lo - 1, i, j])
            wa = float(rep["W"][t1, i, j])
            dg = float(rep["growth"][lo:t1, i, j].sum())
            dgr = float(rep["growth_reward"][lo:t1, i, j].sum())
            stats.append(dict(i=i, j=j, wb=wb, wa=wa, dw=wa - wb,
                              rg_frac=dgr / dg if dg > 1e-9 else 0.0))
        ex["stats"] = stats
        ex["events"] = window_events(run, lo, t1)
        examples.append(ex)

    wanted = [(f[0], ex["ep"], f[1]) for ex in examples for f in ex["frames"]]
    frames = grab_frames(wanted, ep_off)
    print(f"frames {len(frames)}/{len(set(wanted))}; messages "
          f"{len(messages)}; spans {[(c, lo, hi) for c, lo, hi in spans]}")
    return dict(run=run, W=rep["W"], gr=rep["growth_reward"],
                growth=rep["growth"], messages=messages, spans=spans,
                chains=chains, kills=kills, examples=examples,
                frames=frames, xmax=run["_ep_bounds"][-1][1])


# ── global section ──────────────────────────────────────────────────────
def shade_chambers(ax, spans):
    for k, (_, lo, hi) in enumerate(spans):
        ax.axvspan(lo, hi, color=CHAMBER_TINT[k % 2], zorder=0, linewidth=0)
        ax.axvline(lo, color=RULE, lw=0.7, zorder=1)


def draw_chamber_ribbon(ax, ctx):
    spans = ctx["spans"]
    for k, (lab, lo, hi) in enumerate(spans):
        ax.axvspan(lo, hi, color=CHAMBER_TINT[k % 2], zorder=0, linewidth=0)
        ax.axvline(lo, color=RULE, lw=0.7, zorder=1)
        if hi - lo > 55:
            ax.text((lo + hi) / 2, 0.5, lab, fontsize=FS["chamber"],
                    color="#5a6472", ha="center", va="center")
    for e, (s0, _) in enumerate(ctx["run"]["_ep_bounds"]):
        if e:
            ax.axvline(s0, color=INK, lw=1.0, ls=(0, (4, 3)), alpha=0.5,
                       zorder=3)
        ax.text(s0 + 8, 1.30, f"episode {e + 1}", fontsize=7,
                color=MUTED, ha="left", va="bottom", clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("chamber", fontsize=FS["axis"], color=INK, rotation=0,
                  ha="right", va="center", labelpad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelbottom=False)


def draw_msg_raster(ax, ctx):
    shade_chambers(ax, ctx["spans"])
    rows = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    y_of = {pr: len(rows) - 1 - i for i, pr in enumerate(rows)}
    for m in ctx["messages"]:
        y = y_of[(m["snd"], m["rcv"])]
        ax.vlines(m["t"], y + 0.12, y + 0.88, color=AGENT_C[m["snd"]],
                  lw=0.35, alpha=0.75, zorder=2)
    ax.set_yticks([y_of[pr] + 0.5 for pr in rows])
    ax.set_yticklabels([f"a{s}→a{r}" for s, r in rows],
                       fontsize=FS["raster"])
    for tick, (snd, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(AGENT_C[snd])
    ax.set_ylim(0, len(rows))
    ax.set_ylabel("messages", fontsize=FS["axis"], color=INK)
    ax.tick_params(length=0, labelbottom=False)


def draw_dominant_strip(ax, ctx, window=25, margin=0.05):
    """Dominant communication pair per step (rolling per-pair share)."""
    shade_chambers(ax, ctx["spans"])
    pairs = [(0, 1), (0, 2), (1, 2)]
    total = ctx["xmax"]
    counts = {q: np.zeros(total) for q in pairs}
    for m in ctx["messages"]:
        q = tuple(sorted((m["snd"], m["rcv"])))
        if q in counts and 0 <= m["t"] < total:
            counts[q][m["t"]] += 1
    for e, (s0, s1) in enumerate(ctx["run"]["_ep_bounds"]):
        roll = {}
        for q in pairs:
            c = np.cumsum(counts[q][s0:s1])
            roll[q] = c - np.concatenate(([0.0] * window,
                                          c[:-window]))[:s1 - s0]
        tot = sum(roll.values())
        for t in range(s1 - s0):
            if tot[t] <= 0:
                continue
            shares = {q: roll[q][t] / tot[t] for q in pairs}
            lead = max(shares, key=shares.get)
            if shares[lead] > 1 / 3 + margin:
                ax.axvspan(s0 + t, s0 + t + 1, ymin=0.15, ymax=0.85,
                           color=PAIR_C[lead], alpha=0.85, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("dominant\npair", fontsize=FS["axis"], color=INK,
                  rotation=0, ha="right", va="center", labelpad=8)
    ax.tick_params(length=0, labelbottom=False)


def draw_global_bonds(ax, ctx, mark_events=True):
    shade_chambers(ax, ctx["spans"])
    W = ctx["W"]
    steps = np.arange(len(W))
    for q in PAIR_C:
        wij, wji = W[:, q[0], q[1]], W[:, q[1], q[0]]
        ax.fill_between(steps, np.minimum(wij, wji), np.maximum(wij, wji),
                        color=PAIR_C[q], alpha=0.18, linewidth=0, zorder=2)
        ax.plot(steps, (wij + wji) / 2, color=PAIR_C[q], lw=2.0,
                label=f"$\\bar{{W}}$(a{q[0]}–a{q[1]})",
                solid_capstyle="round", zorder=3)
    if mark_events:
        for t in ctx["chains"]:
            y = (W[t, 0, 1] + W[t, 1, 0]) / 2
            ax.scatter(t, y, marker="D", s=34, facecolor="white",
                       edgecolor=PAIR_C[(0, 1)], linewidths=1.4, zorder=5)
        for t in ctx["kills"]:
            ax.scatter(t, 0.045, marker="*", s=70, facecolor="#9aa3ad",
                       edgecolor="white", linewidths=0.4, zorder=5,
                       transform=ax.get_xaxis_transform())
    for e, (s0, _) in enumerate(ctx["run"]["_ep_bounds"]):
        if e:
            ax.axvline(s0, color=INK, lw=0.9, ls=(0, (4, 3)), alpha=0.35)
    ax.grid(axis="y", color="#eef1f4", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("bond strength W", fontsize=FS["axis"], color=INK)
    ax.tick_params(labelsize=FS["tick"])
    hs, ls = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    hs += [Line2D([], [], marker="D", ls="", markerfacecolor="white",
                  markeredgecolor=PAIR_C[(0, 1)], markersize=6,
                  label="a0–a1 switch chain (m17+m18)"),
           Line2D([], [], marker="*", ls="", color="#9aa3ad", markersize=9,
                  label="first mob kill (m21)")]
    ax.legend(handles=hs, fontsize=7.4, ncol=5, frameon=False,
              loc="upper left", borderaxespad=0.15, columnspacing=1.0,
              handletextpad=0.45)
    ax.set_ylim(0.05, float(np.nanmax(W)) * 1.30)
    ax.set_xlabel("environment step (3 episodes × 1000)",
                  fontsize=FS["axis"], labelpad=3,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


def draw_global(fig, rect, ctx, comm_mode):
    """Chamber ribbon + comm lane + bond lane inside figure rect.
    Returns (axes_list, bond_ax)."""
    x0, y0, w, h = rect
    hs = [0.115, 0.265, 0.62] if comm_mode == "raster" \
        else [0.14, 0.13, 0.73]
    gaps = 0.012
    axes = []
    y = y0 + h
    for k, hf in enumerate(hs):
        hh = (h - 2 * gaps * h) * hf
        y -= hh + (gaps * h if k else 0)
        axes.append(fig.add_axes([x0, y, w, hh]))
    ax_ch, ax_cm, ax_w = axes
    draw_chamber_ribbon(ax_ch, ctx)
    if comm_mode == "raster":
        draw_msg_raster(ax_cm, ctx)
    else:
        draw_dominant_strip(ax_cm, ctx)
    draw_global_bonds(ax_w, ctx)
    for ax in axes:
        ax.set_xlim(-15, ctx["xmax"] + 15)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    return axes, ax_w


def mark_window(axes, bond_ax, ex):
    lo, hi = ex["x"]
    for ax in axes:
        ax.axvspan(lo, hi, color="#5b6472", alpha=0.10, zorder=1,
                   linewidth=0)
    y0, y1 = bond_ax.get_ylim()
    bond_ax.add_patch(Rectangle((lo, y0), hi - lo, y1 - y0,
                                facecolor="none", edgecolor=INK, lw=1.2,
                                zorder=6))
    axes[0].text((lo + hi) / 2, 1.55, ex["tag"], fontsize=9, color=INK,
                 ha="center", va="center", fontweight="bold", zorder=7,
                 clip_on=False,
                 bbox=dict(facecolor="white", edgecolor=INK, lw=0.9,
                           boxstyle="round,pad=0.22"))


def connect(fig, bond_ax, ex, x_left, x_right, y_top):
    lo, hi = ex["x"]
    y0 = bond_ax.get_ylim()[0]
    for xd, xf in ((lo, x_left), (hi, x_right)):
        fig.add_artist(ConnectionPatch(
            xyA=(xd, y0), coordsA=bond_ax.transData,
            xyB=(xf, y_top), coordsB=fig.transFigure,
            color="#aab3bd", lw=0.9, ls=(0, (3, 2)), zorder=0))


# ── local (zoom) components ─────────────────────────────────────────────
def truncate(txt, n=MAXLEN):
    return txt if len(txt) <= n else txt[:n - 1].rstrip() + "…"


def draw_local_w(fig, rect, ctx, ex, label_events=True, fs=None):
    """Local directed W over the window: focal directions coloured by the
    SOURCE agent, per-step reward-gated growth as a fill, focal-pair
    message rug (one mini-row per direction), milestone markers and
    frame-time ticks."""
    fs = fs or FS["local"]
    lo, hi = ex["x"]
    t1 = min(hi + ex["wpad"], len(ctx["W"]) - 1)
    xa, xb = lo - 6, t1
    ax = fig.add_axes(rect)
    ax.set_zorder(3)
    ts = np.arange(xa, xb)
    seg = np.array([ctx["W"][xa:xb, i, j] for i, j in ex["dpairs"]])
    ylo = max(0.0, float(seg.min()) - 0.03)
    yhi = float(seg.max()) + max(0.10, 0.35 * (seg.max() - seg.min()))
    ax.set_xlim(xa, xb)
    ax.set_ylim(ylo, yhi)
    for i, j in ex["dpairs"]:
        ax.plot(ts, ctx["W"][xa:xb, i, j], color=AGENT_C[i], lw=1.8,
                solid_capstyle="round", zorder=4)
    # staggered direct end labels
    ends = sorted(
        (float(ctx["W"][xb - 1, i, j]), i, j) for i, j in ex["dpairs"])
    gap = 0.10 * (yhi - ylo)
    ys = []
    for y, i, j in ends:
        if ys and y < ys[-1] + gap:
            y = ys[-1] + gap
        ys.append(y)
        ax.annotate(f"W(a{i}→a{j})", (xb - 1, y), xytext=(3, 0),
                    textcoords="offset points", fontsize=fs - 1,
                    color=AGENT_C[i], va="center", fontweight="bold",
                    annotation_clip=False)
    gr = sum(ctx["gr"][xa:xb, i, j] for i, j in ex["dpairs"])
    if gr.max() > 1e-4:
        ax.fill_between(ts, ylo, ylo + gr / gr.max() * 0.30 * (yhi - ylo),
                        color=REWARD_FILL, alpha=0.85, linewidth=0,
                        zorder=2, step="mid")
        ax.text(0.02, 0.035, "reward-gated ΔW/step", fontsize=fs - 1.8,
                color="#b3543a", va="bottom", zorder=5,
                transform=ax.transAxes)
    else:
        ax.text(0.02, 0.035, "reward-gated ΔW = 0", fontsize=fs - 1.8,
                color=MUTED, va="bottom", zorder=5, transform=ax.transAxes)
    rug_rows = {tuple(p): 1.0 - 0.045 * k
                for k, p in enumerate(ex["dpairs"])}
    for m in ctx["messages"]:
        y = rug_rows.get((m["snd"], m["rcv"]))
        if y is not None and xa <= m["t"] < xb:
            ax.vlines(m["t"], y - 0.035, y,
                      transform=ax.get_xaxis_transform(),
                      color=AGENT_C[m["snd"]], lw=0.8, alpha=0.85, zorder=4)
    seen = set()
    for s, mid, a, r in ex["events"]:
        ax.axvline(s, color="#c0392b", lw=1.0, ls=(0, (2, 2)), zorder=3)
        if label_events and (s, a) not in seen:
            n_at = sum(1 for e2 in seen if e2[0] == s)
            ax.text(s + 1, 0.80 - 0.13 * n_at, f"+{r:.0f} (a{a})",
                    transform=ax.get_xaxis_transform(), fontsize=fs - 1.4,
                    color="#c0392b", va="top", zorder=5)
            seen.add((s, a))
    ax.scatter([t for _, t, _ in ex["frames"]],
               [0.0] * len(ex["frames"]), marker="^", s=16, color=INK,
               zorder=5, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.grid(axis="y", color="#eef1f4", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs - 1, length=2.5, pad=1.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return ax


def draw_frame(fig, rect, ctx, a, ep, t, cap, fs_cap=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_zorder(3)
    fr = ctx["frames"].get((a, ep, t))
    if fr is not None:
        ax.imshow(fr)
        for sp in ax.spines.values():
            sp.set_visible(True)
    ax.set_title(f"t={t}", fontsize=7, color=MUTED, pad=1.5)
    if cap:
        ax.text(0.5, -0.09, cap, transform=ax.transAxes,
                fontsize=fs_cap or FS["caption"], color=INK, ha="center",
                va="top")
    return ax


def convo_lines(fig, x, y, ex, dy=0.0165, maxlen=MAXLEN, n=None):
    for m in ex["convo"][:n]:
        fig.text(x, y, f"t={m['t']}", fontsize=FS["body"] - 1.4,
                 color=MUTED, va="top", zorder=4)
        fig.text(x + 0.028, y, f"a{m['snd']}→a{m['rcv']}",
                 fontsize=FS["body"], color=AGENT_C[m["snd"]], va="top",
                 fontweight="bold", zorder=4)
        fig.text(x + 0.064, y, f"“{truncate(m['text'], maxlen)}”",
                 fontsize=FS["body"], color=INK, va="top", zorder=4)
        y -= dy
    return y


def social_line(fig, x, y, ex):
    m = ex["convo"][0]
    fig.text(x, y, f"Social module: a{m['snd']} chose target a{m['rcv']} "
                   f"(logged routing = {m['routing']}).",
             fontsize=FS["body"], color=INK, va="top", zorder=4)
    fig.text(x, y - 0.016, "(deliberation text not in local logs)",
             fontsize=FS["body"] - 0.8,
             color=MUTED, style="italic", va="top", zorder=4)
    return y - 0.0335


def dw_summary(fig, x, y, ex, narrow=False, cause=True, dy=0.0175):
    if cause:
        txt = ex["cause_short"] if narrow else ex["cause"]
        fig.text(x, y, txt + "  →", fontsize=FS["body"],
                 color=INK, va="top", zorder=4, style="italic")
        y -= dy - 0.001
    for st in ex["stats"]:
        i, j = st["i"], st["j"]
        sgn = "+" if st["dw"] >= 0 else "−"
        pct = f"{st['rg_frac'] * 100:.0f}%" if abs(st["dw"]) > 0.02 else "0%"
        if narrow:
            txt = (f"W(a{i}→a{j})  {st['wb']:.2f}→{st['wa']:.2f}  "
                   f"{sgn}{abs(st['dw']):.2f}  ({pct} rg)")
        else:
            txt = (f"W(a{i}→a{j})  {st['wb']:.2f} → {st['wa']:.2f}   "
                   f"(ΔW {sgn}{abs(st['dw']):.2f},  {pct} reward-gated)")
        fig.text(x, y, txt, fontsize=FS["dw"] - (0.6 if narrow else 0),
                 color=AGENT_C[i], va="top", fontweight="bold", zorder=4)
        y -= dy
    return y


# ── standard card panel (variants A, B, D-minis) ────────────────────────
def panel_card(fig, ctx, ex, x0, y0, w, h, n_frame_cols=3, frames=None,
               n_convo=None, compact=False):
    fig.patches.append(Rectangle((x0, y0), w, h, transform=fig.transFigure,
                                 facecolor="white", edgecolor=PANEL_EDGE,
                                 lw=1.0, zorder=1))
    pad = 0.009
    ytop = y0 + h - 0.013
    fig.text(x0 + pad, ytop,
             ex["short_title"] if compact else ex["title"],
             fontsize=FS["panel_title"], color=INK, va="center",
             fontweight="bold", zorder=4)
    fig.text(x0 + pad, ytop - 0.0145, ex["story"], fontsize=FS["small"],
             color=MUTED, va="center", zorder=4)
    y = ytop - 0.027

    # Layer 1 — local W
    wh = 0.082 if compact else 0.100
    y -= wh + 0.008
    draw_local_w(fig, [x0 + pad + 0.022, y, w - 2 * pad - 0.052, wh],
                 ctx, ex, label_events=not compact)
    y -= 0.022

    # Layer 2 — frames
    flist = frames if frames is not None else ex["frames"]
    fw = (w - (n_frame_cols + 1) * pad) / n_frame_cols
    fr_h = fw * (9 / 16) * (fig.get_figwidth() / fig.get_figheight())
    for r0 in range(0, len(flist), n_frame_cols):
        row = flist[r0:r0 + n_frame_cols]
        y -= fr_h + 0.030
        for k, (a, t, cap) in enumerate(row):
            draw_frame(fig, [x0 + pad + k * (fw + pad), y, fw, fr_h],
                       ctx, a, ex["ep"], t, cap)
    y -= 0.022

    # Layer 3 — conversation + social module
    fig.text(x0 + pad, y, "Conversation", fontsize=FS["header"],
             color=MUTED, va="top", zorder=4, fontweight="bold")
    y -= 0.0165
    y = convo_lines(fig, x0 + pad + 0.002, y, ex, dy=0.016,
                    maxlen=int(MAXLEN * (w / 0.31)), n=n_convo)
    y -= 0.005
    if not compact:
        y = social_line(fig, x0 + pad + 0.002, y, ex)
        y -= 0.003

    # Layer 4 — Hebbian consequence
    fig.text(x0 + pad, y, "Hebbian consequence", fontsize=FS["header"],
             color=MUTED, va="top", zorder=4, fontweight="bold")
    y -= 0.0165
    dw_summary(fig, x0 + pad + 0.002, y, ex, narrow=compact)


# ── comic-strip band (variant C) ────────────────────────────────────────
def panel_band(fig, ctx, ex, y0, h, x0=0.035, x1=0.985):
    fig.patches.append(Rectangle((x0, y0), x1 - x0, h,
                                 transform=fig.transFigure,
                                 facecolor="white", edgecolor=PANEL_EDGE,
                                 lw=1.0, zorder=1))
    pad = 0.008
    ytop = y0 + h - 0.013
    fig.text(x0 + pad, ytop, ex["title"], fontsize=FS["panel_title"],
             color=INK, va="center", fontweight="bold", zorder=4)
    fig.text(x1 - pad, ytop, ex["story"], fontsize=FS["small"],
             color=MUTED, va="center", ha="right", zorder=4)

    # left: local W
    wx, ww = x0 + pad + 0.018, 0.155
    wy, wh = y0 + h * 0.30, h * 0.48
    draw_local_w(fig, [wx, wy, ww, wh], ctx, ex, label_events=True)
    y_soc = y0 + h * 0.185
    m0 = ex["convo"][0]
    fig.text(wx, y_soc, f"social module: a{m0['snd']} routed → "
                        f"a{m0['rcv']} (model)", fontsize=FS["body"] - 0.6,
             color=INK, va="top", zorder=4)
    fig.text(wx, y_soc - 0.014, "(deliberation text not in local logs)",
             fontsize=FS["body"] - 1.2, color=MUTED, style="italic",
             va="top", zorder=4)

    # middle: filmstrip at temporal positions with messages beneath
    sx0, sx1 = wx + ww + 0.030, 0.842
    lo, hi = ex["x"][0] - 4, ex["x"][1] + 6
    fw = 0.088
    fr_h = fw * (9 / 16) * (fig.get_figwidth() / fig.get_figheight())
    fy = y0 + h * 0.44
    xs_used, x_of_t = [], {}
    for a, t, cap in ex["frames"]:
        xc = sx0 + (t - lo) / (hi - lo) * (sx1 - sx0 - fw)
        if xs_used and xc < xs_used[-1] + fw + 0.004:
            xc = xs_used[-1] + fw + 0.004
        xs_used.append(xc)
        x_of_t[t] = xc
        draw_frame(fig, [xc, fy, fw, fr_h], ctx, a, ex["ep"], t, cap,
                   fs_cap=FS["caption"] - 0.4)
    for k in range(len(xs_used) - 1):
        xa = xs_used[k] + fw
        xb = xs_used[k + 1]
        fig.patches.append(FancyArrow(
            xa + 0.001, fy + fr_h / 2, max(xb - xa - 0.004, 0.001), 0,
            transform=fig.transFigure, width=0.0004, head_width=0.004,
            head_length=0.0035, color=MUTED, zorder=4))
    # messages anchored under the nearest frame, packed into stagger rows
    ty = fy - 0.021
    row_end = [sx0 - 1.0, sx0 - 1.0, sx0 - 1.0]
    for m in ex["convo"][:4]:
        near = min(x_of_t, key=lambda t: abs(t - m["t"]))
        txt = (f"t={m['t']}  a{m['snd']}→a{m['rcv']} "
               f"“{truncate(m['text'], 42)}”")
        xc = min(max(x_of_t[near] - 0.015, sx0), sx1 - 0.19)
        for r in range(3):
            if xc >= row_end[r] + 0.006:
                fig.text(xc, ty - r * 0.0155, txt,
                         fontsize=FS["body"] - 0.8, color=AGENT_C[m["snd"]],
                         va="top", zorder=4)
                row_end[r] = xc + 0.0042 * len(txt)
                break

    # right: consequence box
    import textwrap
    bx = 0.856
    yb = y0 + h * 0.82
    fig.text(bx, yb, "Hebbian consequence", fontsize=FS["header"],
             color=MUTED, va="top", fontweight="bold", zorder=4)
    yb -= 0.0165
    for ln in textwrap.wrap(ex["cause"] + " →", 30)[:3]:
        fig.text(bx, yb, ln, fontsize=FS["body"] - 0.6, color=INK,
                 style="italic", va="top", zorder=4)
        yb -= 0.0135
    yb -= 0.003
    dw_summary(fig, bx, yb, ex, narrow=True, cause=False, dy=0.0155)


# ── variant assemblers ──────────────────────────────────────────────────
def _new_fig(hgt):
    fig = plt.figure(figsize=(11.0, hgt))
    fig.text(0.035, 1 - 0.10 / hgt, TITLE, fontsize=13.5,
             color=INK, va="top", fontweight="bold")
    return fig


def _save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}",
                    dpi=DPI if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUTDIR / name}.png/.pdf")


def variant_a(ctx, comm_mode):
    """Global + three equal zoom panels."""
    fig = _new_fig(10.2)
    axes, ax_w = draw_global(fig, [0.065, 0.640, 0.92, 0.295], ctx,
                             comm_mode)
    for ex in ctx["examples"]:
        mark_window(axes, ax_w, ex)
    y_top, y0, ph = 0.578, 0.012, 0.566
    cw, gap = 0.308, 0.010
    left = 0.5 - (3 * cw + 2 * gap) / 2
    for k, ex in enumerate(ctx["examples"]):
        x0 = left + k * (cw + gap)
        panel_card(fig, ctx, ex, x0, y0, cw, ph)
        connect(fig, ax_w, ex, x0, x0 + cw, y_top)
    _save(fig, f"mechanistic_vA_{comm_mode}")


def variant_b(ctx, comm_mode):
    """Global + dominant A3 panel."""
    fig = _new_fig(9.6)
    axes, ax_w = draw_global(fig, [0.065, 0.619, 0.92, 0.313], ctx,
                             comm_mode)
    for ex in ctx["examples"]:
        mark_window(axes, ax_w, ex)
    y_top, y0, ph = 0.553, 0.012, 0.541
    widths = [0.215, 0.215, 0.492]
    gap = 0.011
    x0 = 0.5 - (sum(widths) + 2 * gap) / 2
    for ex, w in zip(ctx["examples"], widths):
        if w > 0.4:
            panel_card(fig, ctx, ex, x0, y0, w, ph, n_frame_cols=5)
        else:
            panel_card(fig, ctx, ex, x0, y0, w, ph, n_frame_cols=2,
                       frames=ex["frames"][:4], n_convo=3, compact=True)
        connect(fig, ax_w, ex, x0, x0 + w, y_top)
        x0 += w + gap
    _save(fig, f"mechanistic_vB_{comm_mode}")


def variant_c(ctx, comm_mode):
    """Global + three full-width causal bands (comic strip)."""
    fig = _new_fig(11.8)
    axes, ax_w = draw_global(fig, [0.065, 0.740, 0.92, 0.205], ctx,
                             comm_mode)
    for ex in ctx["examples"]:
        mark_window(axes, ax_w, ex)
    bh, bgap = 0.228, 0.010
    y = 0.705
    for ex in ctx["examples"]:
        y -= bh
        panel_band(fig, ctx, ex, y, bh - 0.004)
        connect(fig, ax_w, ex, 0.035, 0.985, y + bh - 0.004)
        y -= bgap
    _save(fig, f"mechanistic_vC_{comm_mode}")


def variant_d(ctx, comm_mode):
    """Global + one giant A3 zoom + two compact counterexamples."""
    fig = _new_fig(11.6)
    axes, ax_w = draw_global(fig, [0.065, 0.755, 0.92, 0.19], ctx,
                             comm_mode)
    for ex in ctx["examples"]:
        mark_window(axes, ax_w, ex)
    a1, a2, a3 = ctx["examples"]

    # primary: A3, full width
    px0, pw = 0.035, 0.95
    py0, ph = 0.392, 0.318
    fig.patches.append(Rectangle((px0, py0), pw, ph,
                                 transform=fig.transFigure,
                                 facecolor="white", edgecolor=INK, lw=1.3,
                                 zorder=1))
    connect(fig, ax_w, a3, px0, px0 + pw, py0 + ph)
    pad = 0.010
    ytop = py0 + ph - 0.015
    fig.text(px0 + pad, ytop, a3["title"], fontsize=FS["panel_title"] + 1,
             color=INK, va="center", fontweight="bold", zorder=4)
    fig.text(px0 + pad, ytop - 0.015, a3["story"], fontsize=FS["small"],
             color=MUTED, va="center", zorder=4)
    wy = ytop - 0.125
    draw_local_w(fig, [px0 + pad + 0.022, wy, 0.40, 0.095], ctx, a3)
    yc = wy + 0.085
    fig.text(px0 + 0.50, yc, "Conversation", fontsize=FS["header"],
             color=MUTED, va="top", fontweight="bold", zorder=4)
    yc = convo_lines(fig, px0 + 0.50, yc - 0.0165, a3, dy=0.0155,
                     maxlen=80)
    yc = social_line(fig, px0 + 0.50, yc - 0.002, a3)
    fw = (pw - 6 * pad) / 5
    fr_h = fw * (9 / 16) * (fig.get_figwidth() / fig.get_figheight())
    fy = wy - fr_h - 0.042
    for k, (a, t, cap) in enumerate(a3["frames"]):
        draw_frame(fig, [px0 + pad + k * (fw + pad), fy, fw, fr_h],
                   ctx, a, a3["ep"], t, cap)
    yq = fy - 0.030
    fig.text(px0 + pad, yq, "Hebbian consequence",
             fontsize=FS["header"], color=MUTED, va="top",
             fontweight="bold", zorder=4)
    fig.text(px0 + pad + 0.132, yq, a3["cause"] + "  →",
             fontsize=FS["body"], color=INK, va="top", style="italic",
             zorder=4)
    for k, st in enumerate(a3["stats"]):
        i, j = st["i"], st["j"]
        sgn = "+" if st["dw"] >= 0 else "−"
        pct = f"{st['rg_frac'] * 100:.0f}%"
        fig.text(px0 + pad + 0.378 + k * 0.295, yq,
                 f"W(a{i}→a{j})  {st['wb']:.2f} → {st['wa']:.2f}  "
                 f"({sgn}{abs(st['dw']):.2f}, {pct} reward-gated)",
                 fontsize=FS["dw"], color=AGENT_C[i], va="top",
                 fontweight="bold", zorder=4)

    # two compact counterexamples
    cw, ch_, gap = 0.468, 0.368, 0.014
    y0 = 0.008
    for k, ex in enumerate((a1, a2)):
        x0 = 0.035 + k * (cw + gap)
        panel_card(fig, ctx, ex, x0, y0, cw, ch_, n_frame_cols=4,
                   frames=ex["frames"][:4], n_convo=2, compact=True)
        connect(fig, ax_w, ex, x0, x0 + cw, y0 + ch_)
    _save(fig, f"mechanistic_vD_{comm_mode}")


# ── stats sidecar ───────────────────────────────────────────────────────
def write_stats(ctx):
    st = dict(
        run=str(RUN.relative_to(REPO)),
        total_steps=int(ctx["xmax"]),
        chamber_spans=[(c, int(lo), int(hi)) for c, lo, hi in ctx["spans"]],
        chamber_source="step_log.csv per-step majority (15-step "
                       "hysteresis); episode_summary chamber_entry_steps "
                       "disagreed with the logged per-step chamber",
        switch_chains=ctx["chains"], first_kills=ctx["kills"],
        n_messages=len(ctx["messages"]),
        deliberation_logs="absent locally (no llm_logs/); figures show the "
                          "logged routing decision instead",
        examples=[dict(tag=ex["tag"], window=list(ex["x"]),
                       wpad=ex["wpad"], episode=ex["ep"],
                       events=[list(e) for e in ex["events"]],
                       bonds=[dict(dir=f"a{s['i']}->a{s['j']}",
                                   w_before=round(s["wb"], 4),
                                   w_after=round(s["wa"], 4),
                                   dW=round(s["dw"], 4),
                                   reward_gated_frac=round(s["rg_frac"], 3))
                              for s in ex["stats"]])
                  for ex in ctx["examples"]])
    (OUTDIR / "mechanistic_figure_stats.json").write_text(
        json.dumps(st, indent=2), encoding="utf-8")
    print(f"wrote {OUTDIR / 'mechanistic_figure_stats.json'}")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ctx = build_context()
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "axes.edgecolor": "#9aa3ad",
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"}):
        builders = dict(A=variant_a, B=variant_b, C=variant_c, D=variant_d)
        for tag, fn in builders.items():
            if only and tag.lower() != only.lower():
                continue
            for mode in ("raster", "strip"):
                fn(ctx, mode)
    write_stats(ctx)


if __name__ == "__main__":
    main()
