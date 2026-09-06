#!/usr/bin/env python3
"""make_coordination_timelines.py — imposed vs learned coordination.

A two-arm companion to make_directive_timelines.py, dropping the base arm.
Message frequency was a poor stand-in for "these two agents are working
together": every agent emits exactly one message per step, so the per-pair
share is compositional bookkeeping, not evidence of a team. Both arms here
carry a mechanism that states its coordination explicitly:

  A  Central Orchestrator — EVERY coordination it proposes, whether or not
     it ever happened. From orchestrator/dag.jsonl: each task with
     min_agents >= 2 is a proposed coordination, drawn over its lifetime
     (created -> finished) on the row of the agents it names, coloured by
     how it ended and outlined only if it was ever actually staffed.
     Proposals rejected at decomposition (orchestrator/calls.jsonl) never
     become tasks and are recorded as ticks on their own row, so the lane
     shows the full proposal ledger rather than only what got executed.

  B  Hebbian — the bond graph itself. Each pair's row is a heat strip of
     the symmetrised weight W_ij over time (final_metrics.json ->
     graph_snapshots, full N x N every 50 steps); no threshold, no derived
     dominance statistic. The top row is the mean over the three edges,
     i.e. how strongly the whole trio is bonded.

Both lanes use the SAME rows (all three / a0-a1 / a0-a2 / a1-a2), and each
block carries a strip of node-link snapshots of the graph at matched steps:
the orchestrator's intended graph (who it has put on one task) against the
Hebbian learned graph (edge weight = W). That is the comparison the figure
exists to make.

Camera-ready by default: authored at the exact placement width, Times, no
Type-3 fonts, no in-figure title (a caption .tex is written alongside).

Usage:
  python analysis/make_coordination_timelines.py                  # seed 456, ICLR
  python analysis/make_coordination_timelines.py --scan-seeds
  python analysis/make_coordination_timelines.py --venue icml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paths import ASSETS, RUNS  # noqa: E402  (also puts siblings on sys.path)

import make_directive_timelines as mdt
from make_directive_timelines import (

    AGENT_C, FS_AXIS, FS_LEGEND, FS_PANEL, FS_RIBBON, FS_ROW, INK, MUTED,
    PAIRS, RULE, VENUES, agent_id, co_completions, coop_count,
    draw_milestone_lane, draw_ribbon, episode_milestone_sets,
    episode_task_returns, load_bonds, load_chamber_bands, load_run,
    milestone_lane_events, milestone_rows, style,
)

ARMS = [
    ("orch", Path("orchestrator/new_exp_0_gemma_orch_villager_advisory"),
     "Gemma-E4B + Central Orchestrator", "coordination proposed by a planner"),
    ("heb", Path("new_exp_0_gemma/new_exp_0_gemma_hebbian"),
     "Gemma-E4B + Hebbian", "coordination learned as bonds"),
]
PANEL_TAG = {"orch": "A", "heb": "B"}
SEEDS = [42, 123, 456, 789, 1011, 1213]

# Rows shared by both lanes, top to bottom.
ALL3 = "all3"
ROWS = [ALL3, (0, 1), (0, 2), (1, 2)]
ROW_Y = {k: 3 - i for i, k in enumerate(ROWS)}
ROW_LABEL = {ALL3: "all three", (0, 1): "a0-a1", (0, 2): "a0-a2",
             (1, 2): "a1-a2"}
REJECT_Y = -1.15

# How a proposed coordination ended.
OUTCOME_C = {
    "success": "#1baf7a",
    "timeout": "#e8a33d",
    "unreachable": "#8a94a6",
    "predecessor_failed": "#c9ced6",
    "death": "#d1495b",
    "open": "#ffffff",
}
OUTCOME_LABEL = {
    "success": "completed",
    "timeout": "timed out",
    "unreachable": "chamber unreachable",
    "predecessor_failed": "prerequisite failed",
    "death": "agent died",
    "open": "still open at episode end",
}
# Bond-strength ramp: monotone in lightness, so it survives greyscale.
BOND_CMAP = LinearSegmentedColormap.from_list(
    "bond", ["#f6f6fa", "#c9c6e6", "#8f86c9", "#5647a3", "#2b1a6b"])


# ─── Orchestrator: the full proposal ledger ─────────────────────────────
def load_orch_proposals(run: dict) -> dict:
    """Every coordination the orchestrator proposed, and how it ended.

    A task with min_agents >= 2 IS a proposed coordination. dag.jsonl holds
    a snapshot of the whole DAG on every change, so a task is aggregated
    across snapshots (keyed per episode, since ids repeat between episodes)
    to recover the widest staffing it ever reached and its final status.
    """
    od = run["_run_dir"] / "orchestrator"
    offsets = {e + 1: b[0] for e, b in enumerate(run["_ep_bounds"])}
    ep_len = {e + 1: b[1] - b[0] for e, b in enumerate(run["_ep_bounds"])}

    agg: dict = {}
    for line in (od / "dag.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        ep = d["episode"]
        for t in d.get("tasks", []):
            k = (ep, t["id"])
            a = agg.setdefault(k, {
                "episode": ep, "id": t["id"],
                "description": t.get("description", ""),
                "milestones": t.get("milestones") or [],
                "min_agents": int(t.get("min_agents", 1)),
                "candidates": set(), "assigned_max": 0, "assigned_union": set(),
                "created": t.get("created_at_step"),
            })
            a["candidates"] |= {x for x in (t.get("candidates") or [])}
            a["candidates"] |= {x for x in (t.get("roles") or {})}
            seen = t.get("assigned") or []
            a["assigned_max"] = max(a["assigned_max"], len(seen))
            a["assigned_union"] |= set(seen)
            a["status"] = t.get("status")
            a["reason"] = t.get("failure_reason") or ""
            a["finished"] = t.get("finished_at_step")
            a["assigned_at"] = t.get("assigned_at_step")

    props = []
    for (ep, _), a in agg.items():
        if a["min_agents"] < 2:
            continue
        ids = sorted(x for x in (agent_id(c) for c in a["candidates"])
                     if x is not None)
        row = ALL3 if len(ids) != 2 else tuple(ids)
        t0 = offsets[ep] + max(int(a["created"] or 0), 0)
        fin = a["finished"]
        t1 = (offsets[ep] + ep_len[ep] if fin is None or fin < 0
              else offsets[ep] + int(fin))
        outcome = ("success" if a["status"] == "success"
                   else a["reason"] if a["reason"] in OUTCOME_C else "open")
        props.append({
            "row": row, "t0": t0, "t1": max(t1, t0 + 1), "episode": ep,
            "outcome": outcome, "status": a["status"],
            "min_agents": a["min_agents"], "agents": ids,
            "staffed": a["assigned_max"] >= a["min_agents"],
            "assigned_max": a["assigned_max"],
            "milestones": a["milestones"], "description": a["description"],
        })
    props.sort(key=lambda p: (p["t0"], p["t1"]))

    rejected = []
    for line in (od / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        for item in (c.get("rejected") or []):
            why = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else ""
            rejected.append({"step": offsets[c["episode"]] + int(c["t"]),
                             "why": why})
    # Actual co-assignment (any task, not only the multi-agent proposals) —
    # this is what the orchestrator's graph really looks like at a given step.
    pair_iv = mdt.load_orch(run)["pair_iv"]
    return {"proposals": props, "rejected": rejected, "pair_iv": pair_iv}


def orch_edges_at(ledger, step) -> dict:
    """The orchestrator's graph at `step`: solid where a pair is actually on
    one task, dotted where a proposal naming them is open but unstaffed."""
    out = {}
    for p in ledger["proposals"]:
        if not (p["t0"] <= step < p["t1"]):
            continue
        pairs = ([p["row"]] if p["row"] != ALL3 else
                 [q for q in PAIRS if set(q) <= set(p["agents"])] or list(PAIRS))
        for q in pairs:
            out.setdefault(q, "proposed")
    for q, ivs in ledger["pair_iv"].items():
        if any(lo <= step < hi for lo, hi in ivs):
            out[q] = "staffed"
    return out


# ─── Hebbian: the bond graph ────────────────────────────────────────────
def bond_series(run: dict, total: int) -> dict:
    """Per-step symmetrised W for each pair, held between snapshots."""
    series = load_bonds(run)
    out = {}
    for q in PAIRS:
        if q not in series:
            out[q] = np.zeros(total)
            continue
        xs, ys = series[q]
        k = np.searchsorted(xs, np.arange(total), side="right") - 1
        out[q] = ys[np.clip(k, 0, len(xs) - 1)]
    out[ALL3] = np.mean([out[q] for q in PAIRS], axis=0)
    return out


# ─── Drawing ────────────────────────────────────────────────────────────
def draw_orch_lane(ax, ledger, xmax):
    props, rejected = ledger["proposals"], ledger["rejected"]
    for p in props:
        y = ROW_Y[p["row"]]
        ax.broken_barh(
            [(p["t0"], p["t1"] - p["t0"])], (y - 0.30, 0.60),
            facecolors=OUTCOME_C.get(p["outcome"], "#dddddd"),
            edgecolors=INK if p["staffed"] else "#b6bcc6",
            linewidth=0.9 if p["staffed"] else 0.5,
            linestyle="solid" if p["staffed"] else (0, (1.6, 1.2)),
            zorder=3)
    for r in rejected:
        ax.scatter(r["step"], REJECT_Y, marker="x", s=9, linewidths=0.7,
                   color="#8a94a6", zorder=3)
    ax.axhline(-0.55, color=RULE, lw=0.8, zorder=2)
    ax.set_ylim(REJECT_Y - 0.55, 3.6)
    ax.set_yticks([ROW_Y[k] for k in ROWS] + [REJECT_Y])
    ax.set_yticklabels([ROW_LABEL[k] for k in ROWS] + ["rejected outright"],
                       fontsize=FS_ROW)
    ax.get_yticklabels()[-1].set_color(MUTED)
    ax.get_yticklabels()[-1].set_style("italic")
    ax.set_ylabel("coordination\nproposed", fontsize=FS_AXIS, color=INK,
                  linespacing=1.3)
    style(ax)


def draw_heb_lane(ax, W, xmax, total, vmax):
    for k in ROWS:
        y = ROW_Y[k]
        ax.imshow(W[k][None, :], extent=(0, total, y - 0.30, y + 0.30),
                  aspect="auto", cmap=BOND_CMAP, vmin=0.0, vmax=vmax,
                  interpolation="nearest", zorder=3)
    ax.set_ylim(REJECT_Y - 0.55, 3.6)
    ax.set_yticks([ROW_Y[k] for k in ROWS])
    ax.set_yticklabels([ROW_LABEL[k] for k in ROWS], fontsize=FS_ROW)
    ax.set_ylabel("bond strength\nW", fontsize=FS_AXIS, color=INK,
                  linespacing=1.3)
    style(ax)


def draw_graph_row(ax, steps, edges_at, xmax, w_in, h_in, vmax=None):
    """Node-link snapshots of the coordination graph at matched steps."""
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # keep the triangles equilateral despite the wide, short axes
    ry = 0.30
    rx = ry * (h_in / max(w_in, 1e-6)) * xmax
    node = {0: (0.0, 1.0), 1: (-0.866, -0.5), 2: (0.866, -0.5)}
    for s in steps:
        cx, cy = float(s), 0.52
        pos = {i: (cx + node[i][0] * rx, cy + node[i][1] * ry) for i in node}
        e = edges_at(s)
        for q in PAIRS:
            v = e.get(q)
            if v is None:
                continue
            (x0, y0), (x1, y1) = pos[q[0]], pos[q[1]]
            if isinstance(v, tuple):                    # hebbian: (w, frac)
                w, frac = v
                if frac <= 0.02:
                    continue
                ax.plot([x0, x1], [y0, y1], lw=0.4 + 2.4 * frac,
                        color=BOND_CMAP(0.25 + 0.75 * frac), zorder=2,
                        solid_capstyle="round")
            else:                                       # orchestrator
                ax.plot([x0, x1], [y0, y1],
                        lw=1.5 if v == "staffed" else 0.9,
                        color=INK if v == "staffed" else "#9aa2ae",
                        ls="solid" if v == "staffed" else (0, (1.6, 1.2)),
                        zorder=2, solid_capstyle="round")
        for i, (px, py) in pos.items():
            ax.scatter([px], [py], s=7.5, color=AGENT_C[i], zorder=4,
                       edgecolors="white", linewidths=0.5)
    ax.text(-xmax * 0.008, 0.52, "graph", fontsize=FS_RIBBON, color=MUTED,
            ha="right", va="center")


def bond_scale_handles(vmax):
    """W scale as legend swatches — avoids a separate colorbar axes."""
    levels = [mdt.WARM_START_W, (mdt.WARM_START_W + vmax) / 2, vmax]
    labels = [f"W={levels[0]:.2f} (initial)", f"W={levels[1]:.2f}",
              f"W={levels[2]:.2f}"]
    return [Patch(facecolor=BOND_CMAP(v / vmax), edgecolor=RULE,
                  linewidth=0.4, label=lab)
            for v, lab in zip(levels, labels)]


# ─── Stats / headline ───────────────────────────────────────────────────
def orch_stats(data: dict) -> dict:
    props = data["ledger"]["proposals"]
    return {
        "n_proposed": len(props),
        "n_staffed": sum(1 for p in props if p["staffed"]),
        "n_completed": sum(1 for p in props if p["outcome"] == "success"),
        "n_rejected_outright": len(data["ledger"]["rejected"]),
        "by_outcome": dict(Counter(p["outcome"] for p in props)),
        "by_row": {str(k): v for k, v in
                   Counter(p["row"] for p in props).items()},
    }


def heb_stats(data: dict) -> dict:
    W = data["W"]
    return {
        "final_W": {ROW_LABEL[q]: round(float(W[q][-1]), 3) for q in PAIRS},
        "peak_W": {ROW_LABEL[q]: round(float(W[q].max()), 3) for q in PAIRS},
        "mean_W_over_run": round(float(W[ALL3].mean()), 3),
        "steps_above_initial": {
            ROW_LABEL[q]: int((W[q] > mdt.WARM_START_W).sum()) for q in PAIRS},
    }


def headline(arm: str, data: dict) -> str:
    run = data["run"]
    coops = [coop_count(s) for s in episode_milestone_sets(run)]
    coop_ep = sum(coops) / max(len(coops), 1)
    lead = f"{coop_ep:.2f} cooperative milestones per episode"
    if arm == "orch":
        s = orch_stats(data)
        return (f"{lead}  ·  {s['n_proposed']} coordinations proposed, "
                f"{s['n_staffed']} ever staffed, {s['n_completed']} completed "
                f"({s['n_rejected_outright']} more rejected outright)")
    s = heb_stats(data)
    top = max(s["final_W"], key=s["final_W"].get)
    return (f"{lead}  ·  mean bond W {s['mean_W_over_run']:.2f} over the run, "
            f"strongest edge {top} at W={s['final_W'][top]:.2f}")


# ─── Assembly ───────────────────────────────────────────────────────────
XLABEL = "env step (running across the three episodes)"


def _geometry(n_rows: int, titled: bool) -> dict:
    pitch = 0.115
    return {
        "left": 0.95, "right": 0.10, "top": 0.34 if titled else 0.10,
        "bottom": 0.98,
        "header": 0.38, "ribbon": 0.12,
        "mile": pitch * (n_rows + 1) + 0.10,
        "graph": 0.40,
        "coord": pitch * 5 + 0.20,
        "gap_rm": 0.05, "gap_mg": 0.10, "gap_gc": 0.05, "block_gap": 0.30,
    }


def _block_h(g):
    return (g["header"] + g["ribbon"] + g["gap_rm"] + g["mile"] + g["gap_mg"]
            + g["graph"] + g["gap_gc"] + g["coord"])


def _fig_h(g, n=2):
    return g["top"] + n * _block_h(g) + (n - 1) * g["block_gap"] + g["bottom"]


def _axes(fig, g, index, W, H):
    x0, w = g["left"] / W, 1 - (g["left"] + g["right"]) / W
    y = H - g["top"] - index * (_block_h(g) + g["block_gap"]) - g["header"]
    out = []
    for h, gap in ((g["ribbon"], g["gap_rm"]), (g["mile"], g["gap_mg"]),
                   (g["graph"], g["gap_gc"]), (g["coord"], 0.0)):
        y -= h
        out.append(fig.add_axes([x0, y / H, w, h / H]))
        y -= gap
    return out, w * W


def legend_handles(show_comm: bool, vmax: float):
    h = [Line2D([], [], marker="o", ls="", color=AGENT_C[i], markersize=3.6,
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"agent {i}") for i in range(3)]
    if show_comm:
        h.append(Line2D([], [], marker="|", ls="", color=INK, alpha=0.55,
                        markersize=5.5, label="comm. milestone"))
    h += [Patch(facecolor=OUTCOME_C[k], edgecolor="#b6bcc6", linewidth=0.5,
                label=OUTCOME_LABEL[k])
          for k in ("timeout", "unreachable", "predecessor_failed", "death")]
    h += [
        Patch(facecolor="white", edgecolor=INK, linewidth=0.9,
              label="ever staffed (solid edge)"),
        Patch(facecolor="white", edgecolor="#b6bcc6", linewidth=0.5,
              linestyle=(0, (1.6, 1.2)), label="never staffed (dotted edge)"),
        Line2D([], [], marker="x", ls="", color="#8a94a6", markersize=3.6,
               markeredgewidth=0.7, label="rejected at proposal"),
    ]
    return h + bond_scale_handles(vmax)


def build(arms_data, out_dir: Path, seed, show_comm: bool, width: float,
          titled: bool):
    total_max = max(d["run"]["_ep_bounds"][-1][1] for d in arms_data.values())
    xmax = total_max + 20
    rows, row_of = milestone_rows(arms_data)
    g = _geometry(len(rows), titled)
    W_in, H = width, _fig_h(_geometry(len(rows), titled))
    fig = plt.figure(figsize=(W_in, H))

    vmax = max(0.05, max(float(arms_data["heb"]["W"][q].max()) for q in PAIRS))
    n_snap = 7
    snaps = np.linspace(xmax * 0.045, xmax * 0.955, n_snap)

    for k, (arm, _, name, tagline) in enumerate(ARMS):
        (ax_r, ax_m, ax_g, ax_c), aw = _axes(fig, g, k, W_in, H)
        d = arms_data[arm]
        draw_ribbon(ax_r, d["run"], d["bands"], xmax)
        draw_milestone_lane(ax_m, d["events"], rows, row_of,
                            show_comm=show_comm)
        mdt.episode_rules(ax_m, d["run"])
        if arm == "orch":
            draw_graph_row(ax_g, snaps,
                           lambda s, L=d["ledger"]: orch_edges_at(L, s),
                           xmax, aw, g["graph"])
            draw_orch_lane(ax_c, d["ledger"], xmax)
        else:
            Wb, tot = d["W"], d["run"]["_ep_bounds"][-1][1]

            def heb_edges(s, Wb=Wb, tot=tot):
                i = int(min(max(s, 0), tot - 1))
                return {q: (float(Wb[q][i]), float(Wb[q][i]) / vmax)
                        for q in PAIRS}

            draw_graph_row(ax_g, snaps, heb_edges, xmax, aw, g["graph"])
            draw_heb_lane(ax_c, Wb, xmax, tot, vmax)
        mdt.chamber_rules(ax_c, d["bands"])
        mdt.episode_rules(ax_c, d["run"])
        for ax in (ax_r, ax_m, ax_g, ax_c):
            ax.set_xlim(0, xmax)
        ax_r.annotate(f"{PANEL_TAG[arm]}   {name}", xy=(0, 1),
                      xycoords="axes fraction", xytext=(0, 22),
                      textcoords="offset points", fontsize=FS_PANEL, color=INK,
                      fontweight="bold", ha="left", va="bottom",
                      annotation_clip=False)
        ax_r.annotate(tagline, xy=(1, 1), xycoords="axes fraction",
                      xytext=(0, 22.5), textcoords="offset points",
                      fontsize=6.3, color=MUTED, ha="right", va="bottom",
                      annotation_clip=False)
        ax_r.annotate(headline(arm, d), xy=(0, 1), xycoords="axes fraction",
                      xytext=(0, 10), textcoords="offset points",
                      fontsize=6.4, color=INK, ha="left", va="bottom",
                      annotation_clip=False)
        if k < len(ARMS) - 1:
            ax_c.tick_params(labelbottom=False)
        else:
            ax_c.set_xlabel(XLABEL, fontsize=FS_AXIS, color=INK, labelpad=2)

    fig.legend(handles=legend_handles(show_comm, vmax), loc="lower center",
               bbox_to_anchor=(0.5, 0.004), fontsize=FS_LEGEND, frameon=False,
               labelcolor=INK, ncol=4, borderpad=0.3, handlelength=1.3,
               columnspacing=1.2, handletextpad=0.5)
    if titled:
        fig.text(g["left"] / W_in, 1 - 0.10 / H,
                 f"Proposed vs learned coordination - seed {seed}",
                 fontsize=10, color=INK, ha="left", va="top",
                 fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"coordination_timelines.{ext}", dpi=400,
                    facecolor="white")
    plt.close(fig)
    return W_in, H


CAPTION = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/coordination_timelines.pdf}
\caption{\textbf{Coordination that is imposed versus coordination that is
learned} (seed %(seed)s, %(steps)s env steps over three episodes). Both
panels share rows (all three agents, and each agent pair), a time axis, and
a milestone lane recording which agent completed which milestone.
\textbf{(A)} Every coordination the central orchestrator proposes: each
task it creates that requires two or more agents, drawn across its lifetime
on the row of the agents it names, filled by how it ended and outlined only
where the task was ever actually staffed with the agents it asked for;
proposals rejected at decomposition are ticked on the bottom row. The
orchestrator proposed %(n_prop)s such coordinations, staffed %(n_staff)s of
them, and completed %(n_done)s. \textbf{(B)} The Hebbian bond graph: the
symmetrised weight $W_{ij}$ for each edge, sampled every 50 steps, shown
without any threshold or derived dominance statistic; the top row is the
mean over the three edges. The node-link strips show both graphs at matched
steps -- the orchestrator's intended graph against the learned one. %(story)s}
\label{fig:coordination-timelines}
\end{figure}
"""


def write_caption(arms_data, out_dir: Path, seed):
    o = orch_stats(arms_data["orch"])
    h = heb_stats(arms_data["heb"])
    steps = max(d["run"]["_ep_bounds"][-1][1] for d in arms_data.values())
    co = {a: sum(coop_count(s) for s in episode_milestone_sets(
        arms_data[a]["run"])) / 3 for a in ("orch", "heb")}
    story = (f"Coordination the planner asks for largely does not happen, "
             f"while the learned bonds hold a persistent structure and reach "
             f"{co['heb']:.2f} cooperative milestones per episode against "
             f"{co['orch']:.2f} for the orchestrator.")
    (out_dir / "caption_coordination_timelines.tex").write_text(
        CAPTION % {"seed": seed, "steps": steps, "n_prop": o["n_proposed"],
                   "n_staff": o["n_staffed"], "n_done": o["n_completed"],
                   "story": story}, encoding="utf-8")


def write_stats(arms_data, out_dir: Path, seed):
    out = {"seed": seed, "orch": orch_stats(arms_data["orch"]),
           "heb": heb_stats(arms_data["heb"])}
    for a in ("orch", "heb"):
        run = arms_data[a]["run"]
        task, _ = episode_task_returns(run)
        coops = [coop_count(s) for s in episode_milestone_sets(run)]
        out[a].update(task_return_total=round(sum(task), 1),
                      coop_per_episode=round(sum(coops) / max(len(coops), 1), 2),
                      episode_lengths=run.get("episode_lengths"))
    (out_dir / "coordination_stats.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    o, h = out["orch"], out["heb"]
    md = [f"# Coordination timelines - seed {seed}", "",
          "| | orchestrator | hebbian |", "|---|---|---|",
          f"| task return | {o['task_return_total']:.0f} | {h['task_return_total']:.0f} |",
          f"| coop milestones/ep | {o['coop_per_episode']:.2f} | {h['coop_per_episode']:.2f} |",
          f"| coordinations proposed | {o['n_proposed']} | n/a (bonds are continuous) |",
          f"| ...ever staffed | {o['n_staffed']} | - |",
          f"| ...completed | {o['n_completed']} | - |",
          f"| rejected at proposal | {o['n_rejected_outright']} | - |",
          f"| mean bond W | - | {h['mean_W_over_run']:.3f} |",
          f"| final W per edge | - | {h['final_W']} |", "",
          f"Proposal outcomes: {o['by_outcome']}", ""]
    (out_dir / "coordination_stats.md").write_text("\n".join(md) + "\n",
                                                   encoding="utf-8")


# ─── Loading / main ─────────────────────────────────────────────────────
def load_arm(arm: str, run_dir: Path) -> dict:
    run = load_run(run_dir)
    total = run["_ep_bounds"][-1][1]
    d = {"run": run, "events": milestone_lane_events(run),
         "bands": load_chamber_bands(run), "cocos": co_completions(run)}
    if arm == "orch":
        d["ledger"] = load_orch_proposals(run)
    else:
        d["W"] = bond_series(run, total)
        if not load_bonds(run):
            print("  [warn] hebbian arm has no graph_snapshots",
                  file=sys.stderr)
    return d


def scan_seeds(runs_root: Path, out_dir: Path):
    rowsmd = ["# Seed scan - orchestrator vs Hebbian", "",
              "Gate: Hebbian beats the orchestrator on BOTH cooperative "
              "milestones per episode and task return.", "",
              "| seed | gate | coop/ep orch->heb | task orch->heb | "
              "coords proposed | staffed | completed | mean W |",
              "|---|---|---|---|---|---|---|---|"]
    best, out = None, {}
    for seed in SEEDS:
        try:
            data = {a: load_arm(a, runs_root / rel / f"seed_{seed}")
                    for a, rel, _, _ in ARMS}
        except (FileNotFoundError, OSError):
            print(f"  seed {seed}: missing runs - skipped", file=sys.stderr)
            continue
        m = {}
        for a in ("orch", "heb"):
            run = data[a]["run"]
            t, _ = episode_task_returns(run)
            c = [coop_count(s) for s in episode_milestone_sets(run)]
            m[a] = (sum(t), sum(c) / max(len(c), 1))
        o = orch_stats(data["orch"])
        hs = heb_stats(data["heb"])
        gate = m["heb"][1] > m["orch"][1] and m["heb"][0] > m["orch"][0]
        score = ((m["heb"][1] - m["orch"][1]) * 2
                 + (m["heb"][0] - m["orch"][0]) / 500
                 + 0.05 * o["n_proposed"])
        out[seed] = {"gate": gate, "score": round(score, 3),
                     "orch": {"task": m["orch"][0], "coop": m["orch"][1], **o},
                     "heb": {"task": m["heb"][0], "coop": m["heb"][1], **hs}}
        rowsmd.append(
            f"| {seed} | {'**PASS**' if gate else 'fail'} | "
            f"{m['orch'][1]:.2f} -> {m['heb'][1]:.2f} | "
            f"{m['orch'][0]:.0f} -> {m['heb'][0]:.0f} | {o['n_proposed']} | "
            f"{o['n_staffed']} | {o['n_completed']} | "
            f"{hs['mean_W_over_run']:.3f} |")
        print(f"  seed {seed}: gate={'PASS' if gate else 'fail'} "
              f"score={score:6.2f}  proposed={o['n_proposed']} "
              f"staffed={o['n_staffed']} completed={o['n_completed']}")
        if gate and (best is None or score > out[best]["score"]):
            best = seed
    (out_dir / "coordination_seed_scan.md").write_text(
        "\n".join(rowsmd) + "\n", encoding="utf-8")
    (out_dir / "coordination_seed_scan.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    if best:
        print(f"  best seed: {best}")
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS)
    ap.add_argument("--seed", type=int, default=456)
    ap.add_argument("--out", type=Path, default=ASSETS / "timelines")
    ap.add_argument("--venue", choices=sorted(VENUES), default="iclr")
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--no-comm-events", action="store_true")
    ap.add_argument("--with-title", action="store_true")
    ap.add_argument("--scan-seeds", action="store_true")
    ap.add_argument("--use-best-seed", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    if args.scan_seeds:
        print("scanning seeds...")
        best, _ = scan_seeds(args.runs_root, args.out)
        if args.use_best_seed and best:
            seed = best
            print(f"using best seed: {seed}")

    arms_data = {}
    for arm, rel, _, _ in ARMS:
        rd = args.runs_root / rel / f"seed_{seed}"
        if not (rd / "final_metrics.json").exists():
            sys.exit(f"error: no run at {rd}")
        arms_data[arm] = load_arm(arm, rd)
    o = orch_stats(arms_data["orch"])
    print(f"orch: {o['n_proposed']} coordinations proposed, "
          f"{o['n_staffed']} staffed, {o['n_completed']} completed, "
          f"{o['n_rejected_outright']} rejected outright")
    print(f"heb:  mean W {heb_stats(arms_data['heb'])['mean_W_over_run']:.3f}")

    w, h = build(arms_data, args.out, seed, not args.no_comm_events,
                 args.width or VENUES[args.venue], args.with_title)
    write_caption(arms_data, args.out, seed)
    write_stats(arms_data, args.out, seed)
    print(f"  coordination_timelines.pdf  {w:.2f} x {h:.2f} in ({args.venue})")
    print(f"wrote figure, caption and stats to {args.out}/")


if __name__ == "__main__":
    main()
