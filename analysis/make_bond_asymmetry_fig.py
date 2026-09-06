#!/usr/bin/env python3
"""make_bond_asymmetry_fig.py — figure for the W[i,j] vs W[j,i] measurement.

Reads the CSVs written by make_bond_asymmetry.py and draws a three-panel
figure:

  A  relative asymmetry |W_ij - W_ji| / Wbar per arm, per-seed dots + arm
     mean, grouped into the four channel regimes
  B  W_ij vs W_ji for every pair in every run, against the identity line —
     the direct answer to "is the bond matrix symmetric?"
  C  relative asymmetry per episode — does the gap accumulate with training?

Palette: two hues only (blue #2a78d6 = slot 1, orange #eb6834 = slot 2),
validated all-pairs in light mode (CVD dE 24.7 protan, normal-vision dE
33.6, both >= floor). Regime identity is carried by axis grouping and
direct labels, not by a third hue — a gray third series fails the
normal-vision floor against blue (dE 14.2 < 15).

Usage:
  python analysis/make_bond_asymmetry_fig.py
  python analysis/make_bond_asymmetry_fig.py --assets paper_assets/bond_asymmetry
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paths import ASSETS  # noqa: E402  (also puts siblings on sys.path)



# Validated categorical slots 1 and 2 (see module docstring).
BLUE = "#2a78d6"        # regimes whose social channels are all symmetric
ORANGE = "#eb6834"      # regimes with a directed (obs / imit) channel
INK = "#374151"         # primary text
MUTED = "#9ca3af"       # axis / grid ink, and frozen-control context marks
SOFT_BLUE = "#a9c9ee"   # per-seed dots
SOFT_ORANGE = "#f6bfa6"

# Regime is computed once in make_bond_asymmetry.py (keyed on group AND arm,
# because cofiring_bidi reuses the cofiring arm names with a different wiring
# rule) and carried in every CSV as a column. Read it from there.
REGIME_ORDER = ["frozen", "symmetric", "mixed", "bidi", "directed"]
REGIME_TITLE = {
    "frozen": "Frozen topology (never updated)",
    "symmetric": "Learned — symmetric channels only (proximity, messages)",
    "mixed": "Learned — symmetric + directed channels",
    "bidi": "Learned — directed channel, delivery symmetrised "
            "(--social-bidirectional)",
    "directed": "Learned — directed channel only (observation / imitation)",
}

# Unidirectional counterpart value, for the direct labels on the bidi rows.
BIDI_COUNTERPART = ("cofiring_final", "cofiring_bidi")

# (group, arm) -> row label, in the order they should appear. Restricted to
# the primary suite; the CSVs carry every arm.
ROWS = [
    ("medium_runs", "exp05_mappo_hebbian", "MAPPO+Heb"),
    ("medium_runs", "exp06_ippo_hebbian", "IPPO+Heb"),
    ("medium_runs", "exp07_llm_2b_social_prompt", "LLM-2B+Heb"),
    ("medium_runs", "exp08_llm_9b_social_prompt", "LLM-9B+Heb"),
    ("new_exp_0_gemma", "new_exp_0_gemma_hebbian", "Gemma-E4B+Heb"),
    ("pareto_social", "new_exp_0_gemma_si2", "Gemma si2"),
    ("pareto_social", "new_exp_0_gemma_si20", "Gemma si20"),
    ("pareto_social", "new_exp_0_gemma_si50", "Gemma si50"),
    ("pareto_social", "new_exp_0_gemma_si100", "Gemma si100"),
    ("pair_bonding", "expA_pair_bonding", "Pair (N=2)"),
    ("pair_bonding", "expB_merged_transplant", "Transplant (N=6)"),
    ("pair_bonding", "expB_merged_shuffled", "Shuffled (N=6)"),
    ("cofiring_final", "exp20_cofire_prc", "cofire prc"),
    ("cofiring_final", "exp27_cofire_anchor", "cofire anchor"),
    ("cofiring_final", "exp28_cofire_null", "cofire null (pr)"),
    ("cofiring_final", "exp29_cofire_prco", "cofire prco"),
    ("cofiring_final", "exp23_cofire_prcoi", "cofire prcoi"),
    ("cofiring_final", "exp21_cofire_pro", "cofire pro"),
    ("cofiring_final", "exp22_cofire_pri", "cofire pri"),
    ("agent_scaling", "scale_gemma_hebbian_n2", "Scale N=2"),
    ("agent_scaling", "scale_gemma_hebbian_n3", "Scale N=3"),
    ("agent_scaling", "scale_gemma_hebbian_n4", "Scale N=4"),
    ("agent_scaling", "scale_gemma_hebbian_n5", "Scale N=5"),
    ("agent_scaling", "scale_gemma_hebbian_n6", "Scale N=6"),
    ("agent_scaling", "scale_gemma_hebbian_n9", "Scale N=9"),
    ("pareto_gemma4", "pareto_e2b_hebbian", "Pareto E2B+Heb"),
    ("pareto_gemma4", "pareto_12b_hebbian", "Pareto 12B+Heb"),
    ("social_replay_gemma4", "exp30_mappo_hebbian_replay", "Replay MAPPO+Heb"),
    ("social_replay_gemma4", "exp31_ippo_hebbian_replay", "Replay IPPO+Heb"),
    ("cofiring_bidi", "exp29_cofire_prco", "bidi prco"),
    ("cofiring_bidi", "exp23_cofire_prcoi", "bidi prcoi"),
    ("cofiring_bidi", "exp21_cofire_pro", "bidi pro"),
    ("cofiring_bidi", "exp22_cofire_pri", "bidi pri"),
    ("medium_runs", "exp09_llm_9b_allied_all", "Allied-all"),
    ("medium_runs", "exp10_llm_9b_allied_pair", "Allied-pair"),
    ("medium_runs", "exp11_llm_9b_allied_none", "No-bonds"),
]


def regime_of(row):
    """Regime for a CSV row (written by make_bond_asymmetry.py)."""
    return row["regime"]


def read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=8, length=3, width=0.8)


def panel_a(ax, runs):
    """Per-arm relative asymmetry, grouped by channel regime."""
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[(r["group"], r["arm"])].append(r)
    reg = {(r["group"], r["arm"]): r["regime"] for r in runs}

    def rg_of(group, arm):
        return reg.get((group, arm), "symmetric")

    ordered = sorted(ROWS, key=lambda t: REGIME_ORDER.index(rg_of(t[0], t[1])))
    y, labels, ticks = 0.0, [], []
    seps, headers = [], []
    last_regime = None

    for group, arm, label in ordered:
        rg = rg_of(group, arm)
        if rg != last_regime:
            if last_regime is not None:
                y += 1.15
                seps.append(y - 0.55)
            headers.append((y - 0.05, REGIME_TITLE[rg]))
            y += 0.75
            last_regime = rg
        rows = by_arm.get((group, arm), [])
        vals = [fnum(r["mean_rel_asym"]) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        col = ORANGE if rg == "directed" else BLUE if rg != "frozen" else MUTED
        soft = (SOFT_ORANGE if rg == "directed"
                else SOFT_BLUE if rg != "frozen" else "#dfe3e8")

        n_seeds = len(rows)
        if rg == "frozen":
            # W is symmetric by construction: exactly 0, direct-labelled.
            ax.scatter([0], [y], s=44, facecolors="white", edgecolors=MUTED,
                       linewidths=1.4, zorder=3)
            ax.text(0.075, y, "0 exactly, all seeds", va="center", ha="left",
                    fontsize=7.5, color=MUTED, style="italic")
        elif len(vals) < n_seeds and not vals:
            # Every seed's bonds sat below the floor: the ratio is 0/0.
            ax.scatter([0], [y], s=44, facecolors="white", edgecolors=MUTED,
                       linewidths=1.4, zorder=3)
            ax.text(0.075, y, "bonds ≈ 0 — ratio undefined", va="center",
                    ha="left", fontsize=7.5, color=MUTED, style="italic")
        elif vals:
            ax.scatter(vals, [y] * len(vals), s=22, color=soft,
                       edgecolors="none", zorder=2)
            m = st.mean(vals)
            ax.scatter([m], [y], s=52, color=col, edgecolors="white",
                       linewidths=1.0, zorder=4)
            note = "" if len(vals) == n_seeds else f"  ({n_seeds - len(vals)} of {n_seeds} seeds undefined)"
            if rg == "bidi":
                uni = [fnum(r["mean_rel_asym"])
                       for r in by_arm.get((BIDI_COUNTERPART[0], arm), [])]
                uni = [v for v in uni if not math.isnan(v)]
                if uni:
                    note += f"   (unidirectional: {st.mean(uni):.3f})"
            mw = [fnum(r["mean_W"]) for r in rows]
            mw = [v for v in mw if not math.isnan(v)]
            if mw and st.mean(mw) < 0.01:
                note += "  — bonds ≈ 0, ratio unstable"
            ax.text(m + 0.055, y, f"{m:.3f}{note}", va="center",
                    ha="left", fontsize=7.5, color=col, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.82, pad=1.2))
        ticks.append(y)
        labels.append(f"{label}  (n={n_seeds})")
        y += 1.0

    ax.axvline(0.05, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(2.0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.06, y - 0.35, "5%", fontsize=7, color=MUTED, va="top")
    ax.text(1.98, y - 0.35, "2.0 = one-way\n(weaker side is 0)", fontsize=7,
            color=MUTED, va="top", ha="right")

    for sy in seps:
        ax.axhline(sy, color="#e5e7eb", lw=0.8, zorder=0)
    for hy, text in headers:
        ax.text(-0.008, hy, text, fontsize=8, color=INK, fontweight="bold",
                ha="right", va="center", transform=ax.get_yaxis_transform(),
                clip_on=False)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 2.30)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("relative asymmetry   |W$_{ij}$ − W$_{ji}$| / mean(W$_{ij}$, W$_{ji}$)",
                  fontsize=9, color=INK)
    ax.set_title("A · Directed-trust gap per experiment arm",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.grid(axis="x", color="#eef0f2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    style(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=7,
               label="symmetric channels"),
        Line2D([], [], marker="o", ls="", color=ORANGE, markersize=7,
               label="directed channel"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=MUTED, color=MUTED, markersize=7,
               label="frozen control"),
        Line2D([], [], marker="o", ls="", color=SOFT_BLUE, markersize=5,
               label="individual seed"),
    ], loc="upper right", fontsize=7.5, frameon=True, framealpha=0.95,
        edgecolor="#e5e7eb", labelcolor=INK)


def panel_b(ax, pairs):
    """Every bond pair as (W_ij, W_ji) against the identity line."""
    groups = {"symmetric": ([], []), "directed": ([], []), "mixed": ([], [])}
    for p in pairs:
        rg = p["regime"]
        if rg == "frozen":
            continue                     # exactly on the line by construction
        key = ("directed" if rg == "directed"
               else "symmetric" if rg == "symmetric" else "mixed")
        w_ij, w_ji = fnum(p["w_ij"]), fnum(p["w_ji"])
        # Plot both orderings so the cloud is not artificially triangular.
        groups[key][0].extend([w_ij, w_ji])
        groups[key][1].extend([w_ji, w_ij])

    lim = 0.62
    ax.plot([0, lim], [0, lim], color=INK, lw=1.2, zorder=1)
    ax.text(lim * 0.80, lim * 0.86, "W$_{ij}$ = W$_{ji}$", fontsize=8,
            color=INK, rotation=45, rotation_mode="anchor", ha="center")

    for key, col in (("symmetric", BLUE), ("mixed", BLUE), ("directed", ORANGE)):
        xs, ys = groups[key]
        ax.scatter(xs, ys, s=13, color=col, alpha=0.45, edgecolors="none",
                   zorder=2 if key != "directed" else 3,
                   label=("directed channel" if key == "directed"
                          else "symmetric / mixed / bidi channels" if key == "symmetric"
                          else None))

    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("W$_{ij}$  (i's bond toward j)", fontsize=9, color=INK)
    ax.set_ylabel("W$_{ji}$  (j's bond toward i)", fontsize=9, color=INK)
    ax.set_title("B · Every bond pair against the identity line",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.grid(color="#eef0f2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    style(ax)
    handles = [h for h in ax.get_legend_handles_labels()[0]]
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95,
              edgecolor="#e5e7eb", labelcolor=INK, markerscale=1.6)


def panel_c(ax_dir, ax_sym, eps):
    """Relative asymmetry per episode, faceted by magnitude regime.

    Two facets rather than one axis: the directed arms run ~20x higher than
    the rest, and a shared scale would flatten the symmetric trace to a line
    on the floor. (A second y-scale on one axis is never the answer.)
    """
    by = defaultdict(lambda: defaultdict(list))
    for e in eps:
        rg = e["regime"]
        if rg == "frozen":
            continue
        v = fnum(e["mean_rel_asym"])
        if math.isnan(v):
            continue
        by[rg][int(float(e["episode"]))].append(v)

    def draw(ax, rg, col, marker, ls, label):
        pts = sorted(by[rg].items())
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [st.mean(p[1]) for p in pts]
        ax.plot(xs, ys, color=col, lw=2.0, ls=ls, marker=marker,
                markersize=6.5, markeredgecolor="white", markeredgewidth=1.0,
                zorder=3)
        ax.text(xs[-1] + 0.08, ys[-1], label, fontsize=8, color=col,
                va="center", ha="left")

    draw(ax_dir, "directed", ORANGE, "D", "-", "directed")
    draw(ax_sym, "bidi", BLUE, "^", (0, (1, 2)), "bidi")
    draw(ax_sym, "mixed", BLUE, "s", (0, (4, 2)), "mixed")
    draw(ax_sym, "symmetric", BLUE, "o", "-", "symmetric")

    ax_dir.set_ylim(0, 1.25)
    ax_dir.set_yticks([0, 0.5, 1.0])
    ax_sym.set_ylim(0, 0.15)
    ax_sym.set_yticks([0, 0.05, 0.10, 0.15])

    for ax in (ax_dir, ax_sym):
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.85, 3.75)
        ax.grid(color="#eef0f2", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        style(ax)
    ax_dir.set_xticklabels([])
    ax_sym.set_xlabel("episode", fontsize=9, color=INK)
    ax_dir.set_ylabel("rel. asym.", fontsize=8.5, color=INK)
    ax_sym.set_ylabel("rel. asym.", fontsize=8.5, color=INK)
    ax_dir.set_title("C · Per-episode drift  "
                     "(note the two y-scales — separate facets)",
                     fontsize=10.5, color=INK, loc="left", pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=ASSETS / "bond_asymmetry", type=Path)
    args = ap.parse_args()

    runs = read(args.assets / "bond_asymmetry_runs.csv")
    pairs = read(args.assets / "bond_asymmetry_pairs.csv")
    eps = read(args.assets / "bond_asymmetry_episodes.csv")

    fig = plt.figure(figsize=(14.6, 14.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.42, 1.0],
                          height_ratios=[1.0, 1.0, 0.34, 0.34],
                          wspace=0.34, hspace=0.30,
                          left=0.185, right=0.975, top=0.925, bottom=0.048)
    panel_a(fig.add_subplot(gs[:, 0]), runs)
    panel_b(fig.add_subplot(gs[0:2, 1]), pairs)
    panel_c(fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[3, 1]), eps)

    fig.suptitle("Is the learned Hebbian social graph actually directed?",
                 fontsize=13.5, color=INK, x=0.185, ha="left", y=0.975)
    fig.text(0.185, 0.955,
             "The update rule is asymmetric by construction (egocentric "
             f"modulator, one-way obs/imit channels). Measured over "
             f"{len(runs)} runs / {len(pairs)} pairs / "
             f"{len({(r['group'], r['arm']) for r in runs})} arms.",
             fontsize=9, color=MUTED, ha="left")

    for ext in ("pdf", "png"):
        fig.savefig(args.assets / f"bond_asymmetry.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.assets}/bond_asymmetry.pdf and .png")


if __name__ == "__main__":
    main()
