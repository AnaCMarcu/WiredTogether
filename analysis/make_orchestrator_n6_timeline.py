#!/usr/bin/env python3
r"""make_orchestrator_n6_timeline.py - the whole N=6 orchestrator run, one
lane per agent: what each agent was told to do, where it was, what it did,
and who it was actually with.

WHY A SEPARATE FIGURE
---------------------
``make_counterfactual_table.py`` shows a 165-step Ch4 window side by side
with the 3-agent run. That window is where the orchestrator's decomposition
is most legible, but it is 7% of the run and it cannot answer "what is agent
3 doing for the other 93%" or "which agents ever work together". This figure
covers all three episodes and answers both.

FOUR THINGS PER AGENT, stacked in one block per agent:

  chamber   background tint - which chamber the agent is in
  told      the subtask the orchestrator assigned, shaded by the size of the
            team it assigned to that subtask (pale = alone, navy = all six)
  with      the agents actually within reach, shaded by cluster size, from
            positions in step_log.csv
  did       the action it took, by category (move / turn / dig / look-slot /
            no-op)

"WITH" IS NOT "TOLD"
--------------------
The ``with`` band is a co-location cluster: same chamber and within
``RADIUS`` blocks, transitively closed, so it yields groups of any size
rather than pairs. RADIUS defaults to 5.0, which is ``hebbian_radius`` - the
distance the Hebbian rule uses to decide two agents are interacting - so the
orchestrator arm is measured on the coupling criterion of the arm it is the
control for. Cluster-size shares over the run at that radius: 53% alone, 19%
in a pair, 12% in a triple, 9% four, 5% five, 3% all six.

Member sets churn from step to step, so the band carries cluster SIZE and
the persistent groups are called out separately: every maximal span in which
the exact same member set holds for >= MIN_GROUP_STEPS steps is marked with
a vertical tie, one dot per member, labelled with the membership. Over the
whole run only five such groups exist (ep1 a0a1a2a3, a0a1, a4a5; ep3
a1a2a3a4a5, a0a1) - ep2 has none at all.

Joint completion is rarer still: of the 40 milestone events in the run
exactly ONE has more than one contributor (ep3 t=391 m8_anvil_A1, agents 1
and 2). It is ringed in the milestone lane. Note that m14_sword_equipped
fires six times at that same step - once per agent, separately - which is
what makes raw milestone counts misleading at N=6; see the counting note in
make_counterfactual_table.py.

Usage:  python analysis/make_orchestrator_n6_timeline.py
        python analysis/make_orchestrator_n6_timeline.py --episode 1
Out:    paper_assets/timelines/counterfactual/orchestrator_n6_timeline.*
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from paths import ASSETS  # noqa: E402  (also puts siblings on sys.path)

from make_counterfactual_table import (  # noqa: E402  (shared design)
    AGENT_C, HEAD_INK, INK, MONO, MUTED, N6_RUN, RULE_HAIR, RULE_HEAVY,
    SANS, TEAM_CMAP, assignment_grid, short_task,
)

OUT = ASSETS / "timelines" / "counterfactual"
N_AGENTS = 6
EPISODES = (1, 2, 3)
RADIUS = 5.0            # = hebbian_radius, see the module docstring
MIN_GROUP_STEPS = 30    # a member set has to hold this long to be called out
BIN_W = 20              # steps per cell in the "with" and "did" bands

CHAMBER_TINT = {"ch1": "#e4efdc", "ch2": "#dde9f5", "ch3": "#e9e2f3",
                "ch4": "#f7e4d2", "ch5": "#f5d9d9", "": "#ffffff"}

ACT_CAT = {
    "MoveForward": "move", "MoveBackward": "move", "MoveLeft": "move",
    "MoveRight": "move",
    "TurnLeft": "turn", "TurnRight": "turn", "TurnAround": "turn",
    "Dig": "dig",
    "LookUp": "aim", "LookDown": "aim", "Slot1": "aim", "Slot2": "aim",
    "NoOp": "idle",
}
ACT_C = {"move": "#6f95bf", "turn": "#c6d5e4", "dig": "#D55E00",
         "aim": "#c9a227", "idle": "#b9c0c8", None: "#ffffff"}
ACT_LABEL = {"move": "move", "turn": "turn", "dig": "dig / strike",
             "aim": "look, hotbar", "idle": "no-op"}

FS = dict(head=10.5, note=6.2, lane=5.8, agent=7.0, tick=5.8, axis=6.6,
          band=5.2, ms=5.4, legend=6.0)
RC = {"svg.fonttype": "none", "pdf.fonttype": 42,
      "font.family": "sans-serif", "font.sans-serif": SANS,
      "axes.linewidth": 0.5,
      "xtick.major.width": 0.5, "xtick.major.size": 2.0}


# -- data ---------------------------------------------------------------
def load_steps(run: Path, ep: int):
    """(chamber, pos, action) per (step, agent) from step_log.csv."""
    ch, pos, act = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    path = run / "episodes" / f"ep_{ep:04d}" / "step_log.csv"
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            t, a = int(r["step"]), int(r["agent_id"])
            ch[t][a] = r["chamber"]
            act[t][a] = ACT_CAT.get(r["action"])
            try:
                pos[t][a] = (float(r["pos_x"]), float(r["pos_z"]))
            except (TypeError, ValueError):
                pass
    return ch, pos, act


def clusters(pos_t, ch_t, radius=RADIUS):
    """Transitive co-location groups: same chamber, within *radius*.

    Transitive rather than pairwise on purpose - three agents strung out
    along a corridor are one group of three, not two overlapping pairs.
    """
    adj = {a: set() for a in pos_t}
    for a in pos_t:
        for b in pos_t:
            if a >= b:
                continue
            if not ch_t.get(a) or ch_t.get(a) != ch_t.get(b):
                continue
            if math.dist(pos_t[a], pos_t[b]) <= radius:
                adj[a].add(b)
                adj[b].add(a)
    seen, out = set(), []
    for a in sorted(adj):
        if a in seen:
            continue
        stack, comp = [a], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(y for y in adj[x] if y not in comp)
        seen |= comp
        out.append(frozenset(comp))
    return out


def milestones(run: Path, ep: int, merge_window: int = 20):
    """Milestone events, collapsed for display but NOT for jointness.

    Two different things land on the same step here and must not be
    confused: one event credited to several agents (genuinely joint), and
    several single-agent events of the same milestone that happen to fire
    together. m1_move_5 fires once per agent in the first few steps and
    m14_sword_equipped fires six times at ep3 t=391 - none of those are
    joint. Jointness is therefore read off the ORIGINAL event's contributor
    list; the merge only decides how many labels get drawn.
    """
    path = run / "episodes" / f"ep_{ep:04d}" / "event_log.jsonl"
    raw = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") == "milestone":
            raw.append((d["step"], d["id"], d.get("contributors") or []))
    by_id = defaultdict(list)
    for step, mid, contrib in sorted(raw):
        by_id[mid].append((step, contrib))
    out = []
    for mid, evs in by_id.items():
        bucket = []
        for step, contrib in evs:
            if bucket and step - bucket[0][0] > merge_window:
                out.append(_collapse(mid, bucket))
                bucket = []
            bucket.append((step, contrib))
        if bucket:
            out.append(_collapse(mid, bucket))
    return sorted(out)


def _collapse(mid, bucket):
    step = bucket[len(bucket) // 2][0]
    joint = any(len(c) > 1 for _s, c in bucket)     # per-EVENT, not pooled
    agents = sorted({a for _s, c in bucket for a in c})
    return (step, mid, len(bucket), joint, agents)


def bin_modal(seq, bin_w, prefer=None):
    """Down-sample a per-step sequence to one value per *bin_w* steps.

    Co-location and the action choice both flicker every step; drawn raw
    they are a barcode with no readable structure. *prefer* names a value
    that wins its bin if it occurs at all - used for `dig`, which is 4% of
    actions but the only one that changes the world.
    """
    out = []
    for i in range(0, len(seq), bin_w):
        chunk = [v for v in seq[i:i + bin_w] if v is not None]
        if not chunk:
            val = None
        elif prefer is not None and prefer in chunk:
            val = prefer
        else:
            val = max(set(chunk), key=chunk.count)
        out.extend([val] * len(seq[i:i + bin_w]))
    return out


def runs_of(seq):
    """Contiguous (start, end_exclusive, value) runs - one patch per run
    instead of one per step, or the figure carries 14 490 rectangles."""
    if not seq:
        return
    t0, prev = 0, seq[0]
    for i, v in enumerate(seq[1:], 1):
        if v != prev:
            yield t0, i, prev
            t0, prev = i, v
    yield t0, len(seq), prev


def build_episode(run: Path, ep: int):
    """Everything one episode contributes, on that episode's own clock."""
    ch, pos, act = load_steps(run, ep)
    steps = sorted(ch)
    T = steps[-1] + 1
    told = assignment_grid(run, ep, T - 1, N_AGENTS)

    chamber = [[ch.get(t, {}).get(a, "") for t in range(T)]
               for a in range(N_AGENTS)]
    action = [bin_modal([act.get(t, {}).get(a) for t in range(T)], BIN_W)
              for a in range(N_AGENTS)]
    digs = [[t for t in range(T) if act.get(t, {}).get(a) == "dig"]
            for a in range(N_AGENTS)]
    task = [[told[t, a] for t in range(T)] for a in range(N_AGENTS)]
    # team size the ORCHESTRATOR assigned to that agent's task
    team = [[(sum(1 for b in range(N_AGENTS) if told[t, b] == told[t, a])
              if told[t, a] is not None else 0) for t in range(T)]
            for a in range(N_AGENTS)]

    memb = [[frozenset({a}) for t in range(T)] for a in range(N_AGENTS)]
    for t in range(T):
        for g in clusters(pos.get(t, {}), ch.get(t, {})):
            for a in g:
                memb[a][t] = g
    with_n = [bin_modal([len(memb[a][t]) for t in range(T)], BIN_W)
              for a in range(N_AGENTS)]

    # persistent exact member sets, per agent, deduplicated across members
    groups = {}
    for a in range(N_AGENTS):
        for t0, t1, g in runs_of(memb[a]):
            if len(g) > 1 and t1 - t0 >= MIN_GROUP_STEPS:
                groups[(t0, t1, g)] = True
    return dict(T=T, chamber=chamber, action=action, task=task, team=team,
                with_n=with_n, digs=digs, groups=sorted(groups),
                milestones=milestones(run, ep))


# -- drawing ------------------------------------------------------------
FIG_W = 7.6
M_L, M_R, M_T, M_B = 0.60, 0.14, 0.52, 0.30
MS_H = 0.66              # milestone lane
BLOCK_H = 0.56           # one agent
GAP_EP = 62              # blank steps drawn between episodes
LEG_H = 0.62


def draw(run: Path, eps, stem: str, dpi: int):
    data = {ep: build_episode(run, ep) for ep in eps}
    for ep in eps:
        d = data[ep]
        print(f"  ep{ep}: {d['T']} steps, "
              f"{len(d['groups'])} persistent group span(s), "
              f"{len(d['milestones'])} milestone event(s)")

    # one continuous axis; each episode keeps its own step numbers via
    # relabelled ticks, because every other artefact cites (ep, t)
    offs, x = {}, 0
    for ep in eps:
        offs[ep] = x
        x += data[ep]["T"] + GAP_EP
    xmax = x - GAP_EP

    fig_h = (M_T + MS_H + N_AGENTS * BLOCK_H + 0.34 + LEG_H + M_B)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FIG_W, fig_h))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([M_L / FIG_W, (M_B + LEG_H) / fig_h,
                           1 - (M_L + M_R) / FIG_W,
                           (MS_H + N_AGENTS * BLOCK_H) / fig_h])
        ax.set_xlim(-GAP_EP * 0.3, xmax + GAP_EP * 0.3)
        top = MS_H + N_AGENTS * BLOCK_H          # y in inches, 0 at bottom
        ax.set_ylim(0, top)
        ax.axis("off")

        gutter = blended_transform_factory(ax.transAxes, ax.transData)

        def block_y(a):
            """Bottom edge of agent a's block, in axis (inch) units."""
            return top - MS_H - (a + 1) * BLOCK_H

        # ---- per-agent blocks -----------------------------------------
        for a in range(N_AGENTS):
            y0 = block_y(a)
            d_band = dict(chamber=(y0 + 0.02, 0.48),
                          told=(y0 + 0.31, 0.16),
                          with_=(y0 + 0.16, 0.11),
                          did=(y0 + 0.05, 0.08))
            for ep in eps:
                d, off = data[ep], offs[ep]
                yb, hb = d_band["chamber"]
                for t0, t1, c in runs_of(d["chamber"][a]):
                    ax.broken_barh([(off + t0, t1 - t0)], (yb, hb),
                                   facecolors=CHAMBER_TINT.get(c, "#ffffff"),
                                   linewidth=0, zorder=1)
                yb, hb = d_band["told"]
                for t0, t1, tid in runs_of(d["task"][a]):
                    if tid is None:
                        continue
                    n = d["team"][a][t0]
                    ax.broken_barh([(off + t0, t1 - t0)], (yb, hb),
                                   facecolors=TEAM_CMAP(
                                       (n - 1) / (N_AGENTS - 1)),
                                   linewidth=0, zorder=3)
                yb, hb = d_band["with_"]
                for t0, t1, n in runs_of(d["with_n"][a]):
                    if not n:
                        continue        # no position logged; leave it blank
                    # solo is drawn in the palest tone rather than skipped,
                    # so an empty cell means "no data" and nothing else, and
                    # the legend's "alone" swatch covers both bands
                    ax.broken_barh([(off + t0, t1 - t0)], (yb, hb),
                                   facecolors=TEAM_CMAP(
                                       (n - 1) / (N_AGENTS - 1)),
                                   linewidth=0, zorder=3)
                yb, hb = d_band["did"]
                for t0, t1, c in runs_of(d["action"][a]):
                    if c is None:
                        continue
                    ax.broken_barh([(off + t0, t1 - t0)], (yb, hb),
                                   facecolors=ACT_C[c], linewidth=0, zorder=3)
                for t in d["digs"][a]:
                    ax.plot([off + t, off + t], [yb, yb + hb],
                            color=ACT_C["dig"], lw=0.45, zorder=4,
                            solid_capstyle="butt")
            # agent label + band keys in the gutter
            ax.text(-0.070, y0 + 0.28, f"a{a}", fontsize=FS["agent"],
                    color=AGENT_C[a], fontweight="bold", ha="right",
                    va="center", clip_on=False, transform=gutter)
            for key, lab in (("told", "told"), ("with_", "with"),
                             ("did", "did")):
                yb, hb = d_band[key]
                ax.text(-0.010, yb + hb / 2, lab, fontsize=FS["band"],
                        color=MUTED, ha="right", va="center", clip_on=False,
                        transform=gutter)
            ax.axhline(y0, color=RULE_HAIR, lw=0.4, zorder=2)

        # ---- subtask names on the widest blocks -----------------------
        # ONE name per agent per episode. The 60-step node timeout splits
        # every objective into a chain of ..._retry / _v2 / _3_run tasks, so
        # labelling each span buries the band under overlapping text; the
        # longest span is the only one with room, and the chain it belongs
        # to is what the reader needs to identify the lane.
        for ep in eps:
            d, off = data[ep], offs[ep]
            for a in range(N_AGENTS):
                spans = [(t1 - t0, t0, t1, tid)
                         for t0, t1, tid in runs_of(d["task"][a])
                         if tid is not None]
                if not spans:
                    continue
                width, t0, t1, tid = max(spans)
                if width < d["T"] * 0.055:
                    continue
                lab = short_task(tid)
                if len(lab) > 26:
                    lab = lab[:25] + "…"
                # keep the whole label inside this episode's frame: the
                # longest span is often the last one, and a label centred on
                # it would otherwise run into the next episode
                w = len(lab) * d["T"] * 0.0175
                xc = min(max((t0 + t1) / 2, w / 2), d["T"] - w / 2)
                ax.text(off + xc, block_y(a) + 0.39, lab,
                        fontsize=FS["band"],
                        color="white" if d["team"][a][t0] > 3 else INK,
                        ha="center", va="center", zorder=5)

        # ---- persistent co-location groups ----------------------------
        for ep in eps:
            d, off = data[ep], offs[ep]
            for t0, t1, g in d["groups"]:
                xc = off + (t0 + t1) / 2
                ys = [block_y(a) + 0.215 for a in sorted(g)]
                ax.plot([xc, xc], [min(ys), max(ys)], color=HEAD_INK, lw=0.7,
                        zorder=6, solid_capstyle="butt")
                ax.scatter([xc] * len(ys), ys, s=7, color=HEAD_INK, zorder=7)
                ax.text(xc, block_y(min(g)) + 0.495,
                        "".join(f"a{a}" for a in sorted(g)),
                        fontsize=FS["band"], color=HEAD_INK, ha="center",
                        va="bottom", zorder=7, fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=0.8, alpha=0.88))

        # ---- milestone lane -------------------------------------------
        y_ms = top - MS_H
        ax.axhline(y_ms, color=RULE_HAIR, lw=0.5, zorder=2)
        for ep in eps:
            d, off = data[ep], offs[ep]
            placed = []           # (x0, x1, row) of labels already drawn
            for step, mid, count, joint, agents in d["milestones"]:
                lab = mid.split("_", 1)[1].replace("_", " ")
                if count > 1:
                    lab += f" ×{count}"
                # crude but adequate width in step units: the lane is drawn
                # at a known figure width, so a character is a fixed share
                wid = len(lab) * d["T"] * 0.019 + d["T"] * 0.012
                flip = step + wid > d["T"] * 0.98      # label would overrun
                x0 = (step - wid) if flip else step
                row = 0
                while any(r == row and not (x0 + wid < p0 or x0 > p1)
                          for p0, p1, r in placed):
                    row += 1
                placed.append((x0, x0 + wid, row))
                yy = y_ms + MS_H - 0.11 - row * 0.108
                if joint:
                    ax.scatter(off + step, yy, s=22, marker="o",
                               facecolor="white", edgecolor=HEAD_INK,
                               linewidths=1.1, zorder=7)
                else:
                    c = AGENT_C[int(agents[0].replace("agent", ""))] \
                        if agents else MUTED
                    ax.plot([off + step, off + step], [yy - 0.026, yy + 0.026],
                            color=c, lw=0.9, zorder=6, solid_capstyle="butt")
                pad = d["T"] * (0.021 if joint else 0.010)
                ax.text(off + step + (-1 if flip else 1) * pad, yy,
                        lab, fontsize=FS["ms"],
                        color=HEAD_INK if joint else MUTED,
                        va="center", ha="right" if flip else "left", zorder=7,
                        fontweight="bold" if joint else "normal")

        # ---- episode frames, separators, ticks ------------------------
        for ep in eps:
            d, off = data[ep], offs[ep]
            ax.add_patch(plt.Rectangle(
                (off, 0), d["T"], top, fill=False, edgecolor=RULE_HAIR,
                lw=0.6, zorder=8))
            ax.text(off + d["T"] / 2, top + 0.06, f"Episode {ep}",
                    fontsize=FS["note"], color=HEAD_INK, ha="center",
                    va="bottom", fontweight="bold", clip_on=False)
            for t in range(0, d["T"], 200):
                ax.plot([off + t, off + t], [-0.045, 0], color=MUTED,
                        lw=0.5, clip_on=False, zorder=8)
                ax.text(off + t, -0.075, str(t), fontsize=FS["tick"],
                        color=MUTED, ha="center", va="top", clip_on=False)
        ax.text(offs[eps[0]] + xmax / 2 if len(eps) > 1 else
                offs[eps[0]] + data[eps[0]]["T"] / 2, -0.22,
                "environment step (within episode)", fontsize=FS["axis"],
                color=INK, ha="center", va="top", clip_on=False)

        # ---- legend ----------------------------------------------------
        handles = [
            Patch(facecolor=TEAM_CMAP(0.0), label="alone"),
            Patch(facecolor=TEAM_CMAP(0.4), label="3 together"),
            Patch(facecolor=TEAM_CMAP(1.0), label="all 6 together"),
            Patch(facecolor="none", edgecolor="none", label="   "),
        ] + [Patch(facecolor=ACT_C[c], label=ACT_LABEL[c])
             for c in ("move", "turn", "dig", "aim", "idle")] + [
            Patch(facecolor="none", edgecolor="none", label="   "),
            Line2D([], [], marker="o", markerfacecolor="none",
                   markeredgecolor=HEAD_INK, linestyle="none", markersize=4.5,
                   label="joint milestone"),
            Line2D([], [], marker="|", color=MUTED, linestyle="none",
                   markersize=5, label="solo milestone"),
        ]
        leg = fig.legend(handles=handles, loc="lower center", ncol=6,
                         frameon=False, fontsize=FS["legend"],
                         bbox_to_anchor=(0.5, 0.012), handlelength=1.1,
                         handleheight=0.8, columnspacing=1.2,
                         handletextpad=0.45, labelspacing=0.55)
        for t in leg.get_texts():
            t.set_color(INK)
        fig.text(M_L / FIG_W, 1 - 0.16 / fig_h,
                 "Orchestrator · 6 agents", fontsize=FS["head"],
                 color=HEAD_INK, fontweight="bold", va="top", ha="left")
        fig.text(1 - M_R / FIG_W, 1 - 0.17 / fig_h,
                 "villager DAG · advisory · cadence 8 · "
                 "node timeout 60 · seed 42", fontsize=FS["note"],
                 color=MUTED, va="top", ha="right")
        fig.lines.append(Line2D(
            [M_L / FIG_W, 1 - M_R / FIG_W],
            [1 - 0.34 / fig_h] * 2, color=RULE_HEAVY, lw=1.6,
            transform=fig.transFigure, figure=fig, solid_capstyle="butt"))

        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi, facecolor="white")
        plt.close(fig)
    print(f"wrote {stem}.(pdf|png|svg)   {FIG_W:.2f} x {fig_h:.2f} in")


def main() -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Per-agent timeline of the N=6 orchestrator run.")
    ap.add_argument("--episode", type=int, choices=EPISODES, default=None,
                    help="draw one episode instead of all three")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    eps = (args.episode,) if args.episode else EPISODES
    stem = ("orchestrator_n6_timeline" if not args.episode
            else f"orchestrator_n6_timeline_ep{args.episode}")
    print(f"{stem}:")
    draw(N6_RUN, eps, stem, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
