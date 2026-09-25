#!/usr/bin/env python3
r"""make_counterfactual_compact_n6.py - the six-agent matched pair, at ICLR
figure size, in the counterfactual_compact_{a,b} layout.

Two panels on identical canvases (half the text width each), one mechanism
per panel, no titles: the column headers in LaTeX name the mechanism and
everything else is evidence - the mechanism state before, at and after the
moment that matters, four first-person frames in chronological order, four
verbatim dialogue turns.

THE PAIR. Both sides are Gemma-4 E4B, six agents, ``--team-scaling``, Ch4
pinned to three zombies (``--ch4-mob-count 3``), the same milestone
(m21_first_mob_kill), the same chamber, and - by luck - the same two agents,
a0 and a5, in the same roles:

  (a) Relational plasticity  agent_scaling_3f / scale_gemma_hebbian_n6 /
      seed 456, episode 2, steps 330-410. Three-factor rule (Hebbian 2.0).
      a5's strongest bond is a0 (Wbar 0.36) and the two exchange 19 mutual
      messages in the 30 steps before the kill. Three agents swing at the
      same zombies in turn (Dig actions within 5 blocks of a5 in the 10
      steps before the kill, from step_log.csv - assist_inference.py's
      rule): a2 at t=388 and 391 (4.8 / 3.6 blocks from a5; it also swung
      at t=382), a0 at t=394 (2.4 blocks; "Zombies are centered now. I'm engaging
      the closest one"), a5 at t=396 and 397 - the kill lands at 396 ("I
      see a zombie centered now. I'm engaging it to help clear the mobs");
      a0: "I see the zombie, I'll join the attack sequence on the right"
      (t=398); a0 and a5 cross into Ch5 together at t=399. Bonds over the
      window (snapshots at t=351 and t=401, interpolated between): a5-a0
      0.32 -> 0.47, a5-a2 0.12 -> 0.19, a0-a2 0.19 flat; a5's other three
      bonds move by <= 0.03. The milestone credits a5 alone - it records
      the killing blow - and nobody assigned anything.

      No strictly joint kill (two contributors on one m21) exists in any of
      the 17 six-agent Hebbian runs on disk (63 kill events); the milestone
      credits one agent by construction. Under assist_inference.py's rule
      (another agent within 5 blocks with >= 1 attack in the 10 steps
      before) 27 of the 63 are assisted. The strongest with footage is the
      SHUFFLED control, seed 42, ep 1: a3 kills at t=553 and a1 at t=556 on
      the same tile, each the other's #1 bond (0.21) - an unseeded pair -
      with 22-24 mutual messages ("I'm on the nearest zombie now, focusing
      fire" / "Focusing on the nearest zombie now, ready for backup"). It
      is not used here because the pair figure must stay matched to the
      scaling setup on the orchestrator side.
  (b) Central orchestration  agent_scaling_orch / scale_gemma_orch_villager_n6
      / seed 42, episode 1, steps 600-700. The DAG splits Ch4 into six
      single-agent subtasks t_ch4_kill_mob_1..6 at t=605, all six time out
      at t=665 and are reissued, a0 asks a5 to "confirm zombie location so
      I can join" (t=682), a5 kills alone at t=682, a0 never arrives, and
      at t=683 all six solo tasks close as freed_success.

Horizons differ (3 x 500 vs 3 x 1000 steps); the window is inside one Ch4
fight on both sides, so the comparison is between the two mechanism states
around one kill, not between run-level outcomes.

Bond source for (a): the recorded 50-step ``graph_snapshots`` (6 x 6),
interpolated - the per-step three-factor replay reads a fixed N = 3 layout.
Frames come from ``<run>/gifs/<exp>/seed_<S>_agent_<a>_ep<E>.mp4``; a run
pulled without its gifs/ layer renders labelled placeholders and the script
says so. To fill (a):

    printf 'agent_scaling_3f/scale_gemma_hebbian_n6/seed_456\n' > /tmp/k.txt
    LIST=/tmp/k.txt bash pull_media.sh

Usage:  python analysis/make_counterfactual_compact_n6.py
Out:    paper_assets/timelines/counterfactual/counterfactual_compact_n6_{a,b}.{pdf,png,svg}
"""

from __future__ import annotations

import csv
import math
import sys
import textwrap
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from paths import ASSETS, group  # noqa: E402  (also puts siblings on sys.path)

import make_final_figures as mff  # noqa: E402
from make_counterfactual_n6 import (  # noqa: E402  (cumulative-clock grid)
    AGENT_C, CROSS, assignment_grid, load_bond_series,
)
from make_counterfactual_table import TEAM_CMAP, short_task  # noqa: E402
from make_directive_timelines import load_run  # noqa: E402

N = 6
HEB_RUN = group("agent_scaling_3f") / "scale_gemma_hebbian_n6" / "seed_456"
HEB_EXP = "scale_gemma_hebbian_n6"
ORCH_RUN = group("agent_scaling_orch") / "scale_gemma_orch_villager_n6" / "seed_42"
ORCH_EXP = "scale_gemma_orch_villager_n6"
OUT = ASSETS / "timelines" / "counterfactual"

INK, MUTED, RULE = "#31373f", "#8b95a1", "#c9ced6"
FOCAL = "#B2652C"                      # the a0-a5 pair colour of the n6 family
PANEL_W, PANEL_H = 2.72, 3.80          # 11 chat lines on (a), 10 on (b)
WRAP = 40
FS = dict(chat=6.0, tag=5.5, tick=5.6, axis=6.0, lane=5.4)
RC = dict(mff.RC)
RC.update({"font.size": FS["tick"], "axes.linewidth": 0.5,
           "image.interpolation": "none", "image.resample": False,
           "xtick.major.width": 0.5, "ytick.major.width": 0.5,
           "xtick.major.size": 1.8, "ytick.major.size": 1.8})

COLS = {
    "heb": dict(
        run=HEB_RUN, exp=HEB_EXP, seed=456, ep=2, win=(330, 410),
        killer=5, partner=0, trio=(5, 0, 2), kill=396,
        outcome=("Cooperative milestone completed: a0–a5", "#2E7D4F", "#e3f1e8"),
        # "hits" only where the footage shows the blow land (a2's view at
        # t=391 flashes red); a0's swing at t=394 is at 4-5 blocks from the
        # zombie and no frame shows it connect, so it stays a swing
        swing_labels={2: "a2 hits", 0: "a0 swings"},
        extra_pairs=[(2, 4)],
        ticks=[350, 375, 400],
        # One frame per swing, from the swinger's own eyes, in order: a2 390
        # is the third-person view of a0 and a5 already at the zombie; a2 391
        # a2's hit landing (the view flashes red, a3 a5 named); a0 394 a0's
        # swing, zombie dead ahead at 2.4 blocks; a5 398 a5's hits landing,
        # a2 beside. The kill itself is credited at 396.
        frames=[(2, 390), (2, 391), (0, 394), (5, 398)],
        # a2 narrates its hits to a4, not to the pair it is fighting beside;
        # a0 and a5 talk to each other. In swing order.
        chat=[(388, 2, 4, "Zombie is centered, I'm attacking it now; keep an eye on the door state."),
              (391, 2, 4, "Zombie is centered, I'm attacking it now."),
              (394, 0, 5, "Zombies are centered now. I'm engaging the closest one to start the kill sequence."),
              (396, 5, 0, "Agent0, I see a zombie centered now. I'm engaging it to help clear the mobs for the door."),
              (398, 0, 5, "I see the zombie, I'll join the attack sequence on the right.")],
    ),
    "orch": dict(
        run=ORCH_RUN, exp=ORCH_EXP, seed=42, ep=1, win=(600, 700),
        killer=5, kill=682, timeouts=[665], ticks=[625, 650, 675, 700],
        outcome=("Only a5 completed the milestone: cooperation failed",
                 "#B23A3A", "#f7e3e3"),
        # a5 680: zombie centred, a0 a1 a3 a4 named; a0 682: the red door,
        # a1 beside it; a5 684: the zombie flashing red under the hits, a0
        # named; a0 688: a2 a3 a5 in the lit room, no zombie
        frames=[(5, 680), (0, 682), (5, 684), (0, 688)],
        chat=[(681, 5, 0, "Target locked on zombie, ready to attack."),
              (682, 0, 5, "Agent 5, confirm zombie location so I can join the attack."),
              (682, 5, 0, "I see the zombie centered, starting attack now."),
              (683, 0, 5, "Agent 5, if you have the zombie centered, let me know when you start attacking so I can move closer to assist.")],
    ),
}


def crop(fr, keep=0.72, upscale=3):
    """Drop the hotbar strip, then integer-upscale (pixel art, no resampling)."""
    if fr is None:
        return None
    fr = fr[: int(fr.shape[0] * keep)]
    return np.repeat(np.repeat(fr, upscale, axis=0), upscale, axis=1)


# ─── mechanism strips ──────────────────────────────────────────────────
TRIO_C = {(0, 5): FOCAL, (2, 5): "#2E8B6E", (0, 2): "#5B4EA4"}
EXTRA_C = "#B8860B"                    # a2-a4, the bond a2 reports to


def swings_near(run_dir, ep, lo, hi, killer, radius=5.0):
    """{agent: [t, ...]} - Dig actions within *radius* of the killer.

    assist_inference.py's rule: Dig doubles as the attack action, so a swing
    in reach of the killer during the fight is an attack on the same mobs.
    (Near a Ch3 switch the same action is a press - not a concern in Ch4.)
    """
    rows = defaultdict(dict)
    path = run_dir / "episodes" / f"ep_{ep:04d}" / "step_log.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["step"])][int(r["agent_id"])] = r
    out = defaultdict(list)
    for t in range(lo, hi + 1):
        rk = rows.get(t, {}).get(killer)
        if rk is None:
            continue
        try:
            pk = (float(rk["pos_x"]), float(rk["pos_z"]))
        except ValueError:
            continue
        for a, r in rows[t].items():
            if r["action"] != "Dig":
                continue
            try:
                d = math.dist((float(r["pos_x"]), float(r["pos_z"])), pk)
            except ValueError:
                continue
            if d <= radius:
                out[a].append(t)
    return out


def strip_heb(ax, c, lo, hi):
    """The three bonds among the agents that fought, in colour, with each
    agent's first swing in reach of the killer marked on its bond to the
    killer."""
    run = load_run(c["run"])
    off = run["_ep_bounds"][c["ep"] - 1][0]
    W = load_bond_series(dict(run=c["run"], lane="bonds_snap"))
    k, p, trio = c["killer"], c["partner"], c["trio"]
    xs = np.arange(lo, hi + 1)

    def wbar(i, j):
        return (W[off + xs, i, j] + W[off + xs, j, i]) / 2

    ends = []
    for i, j in combinations(sorted(trio), 2):
        col = TRIO_C[(i, j)]
        ys = wbar(i, j)
        ax.plot(xs, ys, color=col, lw=1.4 if k in (i, j) else 1.1,
                solid_capstyle="round", zorder=4)
        lab = f"a{k}–a{j if i == k else i}" if k in (i, j) else f"a{i}–a{j}"
        ends.append((ys[-1], lab, col))
    # bonds outside the trio that explain who an agent talks to (a2's
    # strongest bond is a4, and its hit reports go to a4)
    for i, j in c.get("extra_pairs", []):
        ys = wbar(i, j)
        ax.plot(xs, ys, color=EXTRA_C, lw=1.1, solid_capstyle="round", zorder=3)
        ends.append((ys[-1], f"a{i}–a{j}", EXTRA_C))
    placed = []
    for y, lab, col in sorted(ends):
        while any(abs(y - q) < 0.062 for q in placed):
            y += 0.032
        placed.append(y)
        ax.annotate(lab, (hi, y), xytext=(2.5, 0), textcoords="offset points",
                    fontsize=FS["lane"], color=col, va="center",
                    annotation_clip=False)

    # The sequence of swings, read off the step log, not the dialogue, and
    # confined to assist_inference.py's window: the 10 steps before the
    # kill. Over the whole strip a0, a1 and a3 also swing in reach of a5
    # (t=333, 355, 365) - earlier skirmishes, not this kill.
    sw = swings_near(c["run"], c["ep"], c["kill"] - 10, c["kill"] + 1, k)
    c["_swings"] = {a: ts for a, ts in sw.items()}
    for a, ts in sorted(sw.items(), key=lambda kv: kv[1][0]):
        if a == k:
            continue
        # every swing in reach gets a dot; the label sits on the first
        for t in ts:
            ax.scatter(t, float(wbar(k, a)[t - lo]), marker="o", s=15,
                       color=AGENT_C[a], edgecolor="white", linewidths=0.4,
                       zorder=6)
        t = ts[0]
        y = float(wbar(k, a)[t - lo])
        below = a == p                    # a0's dot sits under the kill star
        ax.annotate(c.get("swing_labels", {}).get(a, f"a{a} swings"), (t, y),
                    xytext=(0, -5.5 if below else 4),
                    textcoords="offset points", fontsize=FS["lane"] - 0.4,
                    color=AGENT_C[a], ha="center",
                    va="top" if below else "bottom")
    t = c["kill"]
    y = float(wbar(k, p)[t - lo])
    ax.scatter(t, y, marker="*", s=52, color=AGENT_C[k], edgecolor="white",
               linewidths=0.4, zorder=7)
    ax.annotate(f"a{k} kills", (t, y), xytext=(0, 4.5), textcoords="offset points",
                fontsize=FS["lane"] - 0.4, color=AGENT_C[k], ha="center",
                va="bottom")

    w0, w1 = float(wbar(k, p)[0]), float(wbar(k, p)[-1])
    ax.annotate(f"$\\bar{{W}}$(a{k}–a{p}) {w0:.2f}→{w1:.2f}", (lo + 3, 0.565),
                fontsize=FS["lane"], color=FOCAL, va="top")
    ax.set_ylim(0.05, 0.58)
    ax.set_yticks([0.1, 0.3, 0.5])
    ax.set_ylabel("bond $\\bar{W}$", fontsize=FS["axis"], color=INK, labelpad=1.5)


def strip_orch(ax, c, lo, hi):
    """Six tenure rows on the team-size ramp; the six solo slots named."""
    run = load_run(c["run"])
    grid = assignment_grid(run)
    off = run["_ep_bounds"][c["ep"] - 1][0]
    rows = {a: N - 1 - a for a in range(N)}
    bh = 0.36
    for a in range(N):
        t0 = lo
        for t in range(lo + 1, hi + 2):
            if t <= hi and grid[off + t, a] == grid[off + t0, a]:
                continue
            tid = grid[off + t0, a]
            if tid is not None:
                mates = sum(1 for b in range(N) if grid[off + t0, b] == tid)
                ax.broken_barh([(t0 + 0.8, max(t - t0 - 1.6, 0.6))],
                               (rows[a] - bh, 2 * bh),
                               facecolors=TEAM_CMAP((mates - 1) / (N - 1)),
                               linewidth=0, zorder=3)
                if t - t0 >= (hi - lo) * 0.3:
                    ax.text((t0 + min(t, hi)) / 2, rows[a], short_task(tid),
                            fontsize=FS["lane"] - 0.4,
                            color="white" if mates > N / 2 else INK,
                            ha="center", va="center", zorder=5)
            t0 = t
    for t in c.get("timeouts", []):
        ax.scatter([t] * N, list(rows.values()), marker="x", s=14, color=INK,
                   linewidths=0.8, zorder=6)
    # the same swing rule as panel (a): a dot on an agent's row for each
    # Dig within 5 blocks of the killer in the 10 steps before the kill.
    # Here that set is empty - a2 stands in reach from t=672 to 678 and
    # never swings, a0's one swing (t=676) is 8 blocks away at the door -
    # so the strip says so instead of leaving the reader to infer it.
    k = c["killer"]
    sw = swings_near(c["run"], c["ep"], c["kill"] - 10, c["kill"] + 1, k)
    c["_swings"] = dict(sw)
    others = [a for a in sw if a != k]
    for a in others:
        ax.scatter(sw[a], [rows[a]] * len(sw[a]), marker="o", s=15,
                   color=AGENT_C[a], edgecolor="white", linewidths=0.4,
                   zorder=6)
    ax.scatter(c["kill"], rows[k], marker="*", s=52, color=AGENT_C[k],
               edgecolor="white", linewidths=0.4, zorder=7)
    note = (f"a{k} kills; no other swing in reach" if not others
            else f"a{k} kills")
    ax.text(lo + (hi - lo) * 0.02, N - 1 + 1.25, note, fontsize=FS["lane"] - 0.4,
            color=AGENT_C[k], ha="left", va="center")
    ax.set_ylim(-0.7, N - 1 + 1.85)
    ax.set_yticks([rows[a] for a in range(N)])
    ax.set_yticklabels([f"a{a}" for a in range(N)], fontsize=FS["lane"])
    for tick, a in zip(ax.get_yticklabels(), range(N)):
        tick.set_color(AGENT_C[a])
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.set_ylabel("assigned\nsubtask", fontsize=FS["axis"], color=INK,
                  labelpad=1.5, linespacing=0.95)


def _load_frames():
    for key, c in COLS.items():
        c["_fr"] = mff.grab_frames(dict(run=c["run"], exp=c["exp"], seed=c["seed"]),
                                   [(a, c["ep"], t) for a, t in c["frames"]])
        miss = len(c["frames"]) - len(c["_fr"])
        print(f"  {key}: frames {len(c['_fr'])}/{len(c['frames'])}"
              + (f"   [{miss} placeholder - pull this seed's gifs/]" if miss else ""))


def _chat_lines(key):
    return sum(len(textwrap.wrap(x, WRAP)) for *_h, x in COLS[key]["chat"])


def build_panel(key: str, stem: str):
    c = COLS[key]
    lo, hi = c["win"]
    max_lines = max(_chat_lines(k) for k in COLS)      # both panels equal
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(PANEL_W, PANEL_H))
        L, R = 0.035, 0.985

        ax = fig.add_axes([L + 0.135, 0.812, (R - L) - 0.150 - 0.105, 0.160])
        (strip_heb if key == "heb" else strip_orch)(ax, c, lo, hi)
        ax.set_xlim(lo, hi)
        ax.set_xticks(c["ticks"])
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
            fr = crop(c["_fr"].get((a, c["ep"], t_)))
            if fr is not None:
                axf.imshow(fr, aspect="auto", interpolation="none")
            else:
                axf.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eef0f3",
                                        edgecolor=RULE, linewidth=0.5,
                                        transform=axf.transAxes))
                axf.text(0.5, 0.42, "recording not synced", ha="center",
                         va="center", fontsize=FS["tag"], color=MUTED,
                         style="italic", transform=axf.transAxes)
            axf.text(0.025, 0.93, f"a{a}  t={t_}", transform=axf.transAxes,
                     fontsize=FS["tag"], color="white", va="top",
                     fontweight="bold",
                     bbox=dict(facecolor=AGENT_C[a] + "e0", edgecolor="none",
                               pad=1.0))

        # outcome banner in the gap between the frames and the dialogue
        fb = top - 2 * (fh + 0.016) + 0.016          # frames' bottom edge
        txt, fg, bg = c["outcome"]
        bh = 0.030
        fig.add_artist(Rectangle((L, fb - 0.012 - bh), R - L, bh,
                                 transform=fig.transFigure, facecolor=bg,
                                 edgecolor=fg, linewidth=0.6))
        fig.text((L + R) / 2, fb - 0.012 - bh / 2, txt, fontsize=FS["chat"] + 0.5,
                 color=fg, fontweight="bold", ha="center", va="center")

        y = fb - 0.012 - bh - 0.022
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
            fig.savefig(OUT / f"{stem}.{ext}", dpi=600, facecolor="white")
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
    build_panel("heb", "counterfactual_compact_n6_a")
    sw = COLS["heb"].get("_swings", {})
    print("  swings in reach of a%d: %s" % (COLS["heb"]["killer"],
          ", ".join(f"a{a}@{ts}" for a, ts in sorted(sw.items(), key=lambda kv: kv[1][0]))))
    build_panel("orch", "counterfactual_compact_n6_b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
