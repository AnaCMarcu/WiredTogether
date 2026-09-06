#!/usr/bin/env python3
"""prototype_three_factor_rule.py — what the bonds would do under a proper
three-factor rule, replayed on the SAME logged inputs as the real run.

Current rule (mode reward_modulated) uses reward only on the step it arrives,
as a one-step gain on the learning rate:

    dW = (eta0 + eta+ |r_i|/R) * c_ij * (1-W) - lambda * W

Proposed rule: co-activity leaves an eligibility trace; reward converts the
accumulated trace into a lasting weight change (classic three-factor /
R-STDP form). Because reward is sparse it cannot saturate the weights, so
lambda can be small and reward-earned credit persists:

    e_ij <- rho * e_ij + c_ij
    dW   = eta0 * c_ij * (1-W) + eta+ * (|r_i|/R) * e_ij * (1-W) - lambda * W

Also fixes the co-activity definition so "together" is monotone:
    c_ij = clip( near * max(g_i g_j, c_floor) + 0.5 * messaged , 0, 1 )
(near + messaging 0.75 > far + messaging 0.5 > near + silent 0.25 > 0),
instead of the current c_comm * (1 - near), under which a far-apart
messaging pair has a HIGHER equilibrium bond than a co-located one.

This is an OPEN-LOOP counterfactual: the agents' messages and movements are
the logged ones and would themselves change under different bonds. It shows
the rule's dynamics on realistic inputs, not a rerun.

Usage:
  python analysis/prototype_three_factor_rule.py            # seed 456
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths  # noqa: F401,E402  (puts sibling modules on sys.path)
from replay_hebbian_terms import (  # noqa: E402
    AGENT_C, INK, MUTED, PAIR_C, PAIRS, RULE, load_inputs, replay, sym,
)


def replay_three_factor(inp, eta0=0.001, eta_plus=0.05, R=50.0, rho=0.9,
                        lam=0.001, c_floor=0.25, monotone_coact=True,
                        alpha=0.5, radius=5.0, w0=0.1):
    N, T = inp["N"], inp["total"]
    W = np.full((N, N), w0); np.fill_diagonal(W, 0.0)
    E = np.zeros((N, N))
    M = 0.0
    out = {k: np.zeros((T, N, N)) for k in ("W", "growth", "growth_reward", "homeo", "c", "e")}
    out["rb"] = np.zeros((T, N))
    task, cb, cm = inp["rew"]["task"], inp["rew"]["comm_base"], inp["rew"]["comm_milestone"]
    for t in range(T):
        ch = (inp["chamber"][t] >= 2).astype(float)
        resid = task[t] - inp["envstep"][t] - inp["milestone"][t]
        death = np.where(resid <= -9.0, resid, 0.0)
        rb = ch * (inp["milestone"][t] + cb[t] + cm[t] + (resid - death))
        M = max(M, float(np.abs(rb).max()))
        social = set()
        for s, r in inp["comm"][t]:
            social.add(s); social.add(r)
        g = np.clip(alpha * np.abs(rb) / (M + 1e-8)
                    + np.array([(1 - alpha) if i in social else 0.0 for i in range(N)]), 0, 1)
        p = inp["pos"][t]
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
        S = ((d <= radius) & np.isfinite(d)).astype(float); np.fill_diagonal(S, 0.0)
        msg = np.zeros((N, N))
        for s, r in inp["comm"][t]:
            if s != r:
                msg[s, r] = msg[r, s] = 1.0
        if monotone_coact:
            c = np.clip(S * np.maximum(np.outer(g, g), c_floor) + 0.5 * msg, 0, 1)
        else:
            c = np.clip(S * np.outer(g, g) + 0.5 * msg * (1 - S), 0, 1)
        np.fill_diagonal(c, 0.0)
        E = rho * E + c
        sal = eta_plus * np.abs(rb) / R
        growth_base = eta0 * c * (1 - W)
        growth_reward = sal[:, None] * E * (1 - W)
        homeo = lam * W
        delta = growth_base + growth_reward - homeo
        np.fill_diagonal(delta, 0.0)
        W = np.clip(W + delta, 0, 1); np.fill_diagonal(W, 0.0)
        out["W"][t], out["growth"][t] = W, growth_base + growth_reward
        out["growth_reward"][t], out["homeo"][t], out["c"][t], out["e"][t] = growth_reward, homeo, c, E
        out["rb"][t] = rb
    return out


def summarise(name, out, inp):
    g, gr, h = out["growth"].sum(), out["growth_reward"].sum(), out["homeo"].sum()
    Wf = out["W"][-1]
    lines = [f"{name}: growth {g:+.2f} (reward-driven {gr:+.2f} = {100 * gr / max(g, 1e-9):.0f}%), "
             f"decay -{h:.2f}; max W {out['W'].max():.2f}; final W "
             + ", ".join(f"a{i}-a{j}={0.5 * (Wf[i, j] + Wf[j, i]):.2f}" for i, j in PAIRS)]
    for c in inp["cocos"]:
        s, (i, j) = c["step"], c["pair"]
        w = sym(out["W"], i, j)
        e50, e100 = min(s + 50, inp["total"] - 1), min(s + 100, inp["total"] - 1)
        lines.append(f"   step {s:4d} a{i}-a{j} {c['mid']:16s} W={w[s]:.2f} "
                     f"dW+50={w[e50] - w[s]:+.3f} dW+100={w[e100] - w[s]:+.3f}")
    return lines


PAPER_TITLES = {
    0: ("A", "rule used in the experiments"),
    1: ("B", "with an eligibility trace over co-activity, same decay"),
    2: ("C", "with the trace and the homeostatic decay reduced"),
}


def build_paper(inp, variants, out_dir: Path, seed: int, width: float = 5.5):
    """Camera-ready three-panel version: exact width, Times, Type-42 fonts,
    descriptive panel headers, no in-figure title or percentages. Palette,
    envelope fill, gridlines and legend styling match the final qualitative
    figures (make_final_figures.py / make_mechanistic_figure.py), so this
    plot reads as part of the same figure family as e.g. wide_gemma_hebbian3f."""
    import make_directive_timelines as mdt  # applies the camera-ready rcParams
    plt.rcParams["mathtext.fontset"] = "stix"
    ink, muted, rule = mdt.INK, "#8b95a1", "#9aa3ad"
    grid_c = "#eef1f4"
    pair_c = {(0, 1): "#5B4EA4", (0, 2): "#B2652C", (1, 2): "#2E8B6E"}
    T = inp["total"]
    steps = np.arange(T)

    panel_h, header, left, right, top, bottom = 1.28, 0.20, 0.55, 0.10, 0.06, 0.36
    H = top + 3 * (header + panel_h) + bottom
    fig = plt.figure(figsize=(width, H))
    axes, y = [], H - top
    for _ in range(3):
        y -= header + panel_h
        axes.append(fig.add_axes([left / width, y / H,
                                  1 - (left + right) / width, panel_h / H]))

    for k, (ax, (_, out)) in enumerate(zip(axes, variants)):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=grid_c, lw=0.8, zorder=0)
        for q in PAIRS:
            wij, wji = out["W"][:, q[0], q[1]], out["W"][:, q[1], q[0]]
            ax.fill_between(steps, np.minimum(wij, wji), np.maximum(wij, wji),
                            color=pair_c[q], alpha=0.18, linewidth=0, zorder=2)
            ax.plot(steps, sym(out["W"], *q), color=pair_c[q], lw=1.6,
                    solid_capstyle="round", zorder=3,
                    label=f"$\\bar{{W}}$(a{q[0]}–a{q[1]})")
        for c in inp["cocos"]:
            q = c["pair"]
            ax.scatter(c["step"], sym(out["W"], *q)[c["step"]], marker="D",
                       s=28, facecolor="white", edgecolor=pair_c[q],
                       linewidths=1.2, zorder=6)
            ax.axvline(c["step"], color=pair_c[q], lw=0.5, alpha=0.3)
        for e, o in enumerate(inp["off"]):
            if e:
                ax.axvline(o, color=ink, lw=0.7, ls=(0, (4, 3)), alpha=0.35)
        tag, desc = PAPER_TITLES[k]
        ax.annotate(f"{tag}   {desc}", xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 3), textcoords="offset points",
                    fontsize=mdt.FS_PANEL - 1.0, color=ink, fontweight="bold",
                    ha="left", va="bottom", annotation_clip=False)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("bond $W_{ij}$", fontsize=mdt.FS_AXIS, color=ink)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(rule)
        ax.tick_params(colors=ink, labelsize=mdt.FS_ROW, length=2.5, width=0.7,
                       color=rule, pad=1.6)
        ax.set_xlim(0, T)
        if k < 2:
            ax.tick_params(labelbottom=False)
    axes[0].scatter([], [], marker="D", s=28, facecolor="white", edgecolor=ink,
                    linewidths=1.0, label="pair completed a milestone together")
    axes[0].legend(fontsize=6.0, ncol=4, loc="upper left", frameon=False,
                   handlelength=1.4, columnspacing=1.0, borderaxespad=0.2,
                   labelcolor=ink)
    axes[-1].set_xlabel("env step (running across the three episodes)",
                        fontsize=mdt.FS_AXIS, color=ink, labelpad=2)
    name = f"three_factor_prototype_seed{seed}_paper"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    return name, width, H


def build_split(inp, variants, out_dir: Path, seed: int,
                width: float = 6.4, height: float = 4.3):
    """Three STANDALONE figures, one per rule variant (A/B/C), styled like
    the paper's pareto figures (make_pareto_perception_fig.py /
    pareto_partner.pdf) rather than the camera-ready multi-panel version:
    plain matplotlib sans-serif, full box, framed legend, no grid, chunky
    lines/markers in the default tab10 palette."""
    T = inp["total"]
    steps = np.arange(T)
    pair_c = {(0, 1): "#1f77b4", (0, 2): "#ff7f0e", (1, 2): "#2ca02c"}
    names = {(0, 1): "a0–a1", (0, 2): "a0–a2", (1, 2): "a1–a2"}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        for k, (_, out) in enumerate(variants):
            fig, ax = plt.subplots(figsize=(width, height))
            for q in PAIRS:
                ax.plot(steps, sym(out["W"], *q), color=pair_c[q], lw=2.0,
                        label=names[q], zorder=3)
            for c in inp["cocos"]:
                q = c["pair"]
                ax.scatter(c["step"], sym(out["W"], *q)[c["step"]], marker="D",
                           s=55, facecolor="white", edgecolor=pair_c[q],
                           linewidths=1.6, zorder=6)
            for e, o in enumerate(inp["off"]):
                if e:
                    ax.axvline(o, color="#333333", lw=1.0, ls=(0, (4, 3)),
                               alpha=0.4, zorder=1)
            ax.scatter([], [], marker="D", s=55, facecolor="white",
                       edgecolor="#333333", linewidths=1.4,
                       label="milestone together")
            tag, desc = PAPER_TITLES[k]
            title = "\n".join(textwrap.wrap(f"{tag}   {desc}", width=36,
                                            break_long_words=False))
            ax.set_title(title, fontsize=12.5, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, T)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylabel("Bond strength $W_{ij}$", fontsize=12)
            ax.set_xlabel("Env step (running across the three episodes)",
                          fontsize=12)
            ax.tick_params(labelsize=10)
            ax.legend(fontsize=10, loc="upper left", frameon=True,
                      framealpha=0.95, edgecolor="#999999")
            fig.tight_layout()
            name = f"three_factor_prototype_seed{seed}_{tag}"
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"{name}.{ext}",
                           dpi=(300 if ext == "pdf" else 200))
            plt.close(fig)
            written.append(out_dir / f"{name}.pdf")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("runs_from_daic"))
    ap.add_argument("--arm", default="new_exp_0_gemma/new_exp_0_gemma_hebbian")
    ap.add_argument("--seed", type=int, default=456)
    ap.add_argument("--out", type=Path, default=Path("paper_assets/hebbian_terms"))
    ap.add_argument("--paper", action="store_true",
                    help="also write the camera-ready *_paper.pdf/.png")
    ap.add_argument("--width", type=float, default=5.5)
    ap.add_argument("--split", action="store_true",
                    help="also write three standalone A/B/C pareto-styled "
                         "PDFs (see --split-out)")
    ap.add_argument("--split-out", type=Path, default=None,
                    help="output dir for --split (default: --out)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inp = load_inputs(args.runs_root / args.arm / f"seed_{args.seed}")
    variants = [
        ("A  current rule (as run)", replay(inp)),
        ("B  three-factor, same decay 0.005",
         replay_three_factor(inp, lam=0.005)),
        ("C  three-factor, decay 0.001 (proposed)",
         replay_three_factor(inp, lam=0.001)),
    ]
    report = []
    for name, out in variants:
        report += summarise(name, out, inp) + [""]
    print("\n".join(report))
    (args.out / f"three_factor_prototype_seed{args.seed}.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8")

    T = inp["total"]
    steps = np.arange(T)
    fig, axes = plt.subplots(len(variants), 1, figsize=(8.0, 7.6), sharex=True,
                             gridspec_kw=dict(hspace=0.18))
    for ax, (name, out) in zip(axes, variants):
        for q in PAIRS:
            ax.plot(steps, sym(out["W"], *q), color=PAIR_C[q], lw=1.3,
                    label=f"a{q[0]}-a{q[1]}")
        for c in inp["cocos"]:
            q = c["pair"]
            ax.scatter(c["step"], sym(out["W"], *q)[c["step"]], marker="D", s=40,
                       facecolor="white", edgecolor=PAIR_C[q], linewidths=1.3, zorder=6)
            ax.axvline(c["step"], color=PAIR_C[q], lw=0.6, alpha=0.3)
        for e, o in enumerate(inp["off"]):
            if e:
                ax.axvline(o, color=INK, lw=0.7, ls=(0, (4, 3)), alpha=0.35)
        g, gr = out["growth"].sum(), out["growth_reward"].sum()
        ax.set_title(f"{name}   —   reward-driven share of bond growth: "
                     f"{100 * gr / max(g, 1e-9):.0f}%", fontsize=9, loc="left", color=INK)
        ax.set_ylabel("bond W_ij", fontsize=8)
        ax.set_ylim(0, 1.0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, ncol=3, loc="upper left", frameon=False)
    axes[-1].set_xlabel("env step (running across the three episodes) — "
                        "diamonds: the pair completed a milestone together", fontsize=8)
    fig.suptitle(f"Same logged inputs (seed {args.seed}), three update rules",
                 fontsize=10.5, x=0.125, ha="left", color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"three_factor_prototype_seed{args.seed}.{ext}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}/three_factor_prototype_seed{args.seed}.pdf/.png/.txt")
    if args.paper:
        name, w, h = build_paper(inp, variants, args.out, args.seed, args.width)
        print(f"  {name}.pdf  {w:.2f} x {h:.2f} in (camera-ready)")
    if args.split:
        split_out = args.split_out or args.out
        for p in build_split(inp, variants, split_out, args.seed):
            print(f"  {p}")


if __name__ == "__main__":
    main()
