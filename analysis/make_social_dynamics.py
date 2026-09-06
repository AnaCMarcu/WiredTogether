"""Two standalone ICLR figures: pairwise communication and bond strength.

Replaces the stacked "mutual-message raster over bond lane" block. At three
thousand steps the per-step raster fuses into solid blocks, so the exchange
signal is re-expressed as a rate: mutual exchanges per 50-step window, the
same cadence as the Hebbian graph snapshots.

  fig 1  mutual_messages : reciprocal exchanges per pair over time.
  fig 2  bond_strength   : symmetrised bond weight per pair over time,
                           with the cooperative joint milestones marked.

Conventional formatting only -- no background shading, no decorative bands;
episode boundaries are thin dashed rules. Chamber context belongs in the
caption.

Run:  python analysis/make_social_dynamics.py [--arm gemma3f] [--seed seed_123]
Out:  paper_assets/timelines/social_dynamics/{mutual_messages,bond_strength}.*
"""
import json
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import ASSETS, RUNS            # noqa: F401 (adds siblings to path)
from make_directive_timelines import load_run
from make_results import MILESTONE_TRACK
from replay_hebbian_terms import load_inputs
from prototype_three_factor_rule import replay_three_factor

COOP = {"ch2_anvils", "ch3_switches", "ch4_combat", "ch5_boss"}
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_C = {(0, 1): "#5B4EA4", (0, 2): "#B2652C", (1, 2): "#2E8B6E"}
INK, MUTED, RULE = "#1a1a1a", "#666666", "#b8bec6"
WINDOW = 50

ARMS = {
    "gemma3f": dict(
        run=RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f",
        label="Gemma-4B + Hebbian 2.0", bonds="replay"),
    "gemma_heb": dict(
        run=RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian",
        label="Gemma-4B + Hebbian 1.0", bonds="snapshots"),
    "qwen9b_heb": dict(
        run=RUNS / "new_exp_0_qwen9b/new_exp_0_qwen9b_hebbian",
        label="Qwen-9B + Hebbian", bonds="snapshots"),
}
OUT = ASSETS / "timelines" / "social_dynamics"

RC = {"font.family": "serif",
      "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
      "mathtext.fontset": "stix",
      "axes.edgecolor": INK, "axes.linewidth": 0.7,
      "xtick.direction": "out", "ytick.direction": "out",
      "xtick.major.width": 0.7, "ytick.major.width": 0.7,
      "xtick.major.size": 3.0, "ytick.major.size": 3.0,
      "svg.fonttype": "none", "pdf.fonttype": 42}
FS = dict(label=9, tick=8, legend=8, note=7.5)


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


def mutual_rate(messages, T, window=WINDOW):
    """Reciprocal exchanges per pair, binned into non-overlapping windows."""
    per_step = defaultdict(set)
    for t, snd, rcv in messages:
        per_step[t].add((snd, rcv))
    nbin = int(np.ceil(T / window))
    counts = {q: np.zeros(nbin) for q in PAIRS}
    for t, dirs in per_step.items():
        for i, j in PAIRS:
            if (i, j) in dirs and (j, i) in dirs:
                counts[(i, j)][min(t // window, nbin - 1)] += 1
    centres = (np.arange(nbin) + 0.5) * window
    return centres, counts


def bond_series(cfg, run_dir):
    if cfg["bonds"] == "replay":
        return replay_three_factor(load_inputs(run_dir))["W"]
    fm = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    snaps = sorted((s["step"], np.array(s["W"]))
                   for s in fm.get("graph_snapshots", []) if s.get("W"))
    if not snaps:
        raise SystemExit("no graph snapshots in " + str(run_dir))
    T = snaps[-1][0] + 1
    W = np.zeros((T, 3, 3))
    ss = [s for s, _ in snaps]
    for i in range(3):
        for j in range(3):
            W[:, i, j] = np.interp(np.arange(T), ss, [m[i, j] for _, m in snaps])
    return W


def joint_fires(run):
    by, step = defaultdict(set), {}
    for e in run["milestone_events"]:
        if MILESTONE_TRACK.get(e["milestone_id"]) not in COOP:
            continue
        a = int(re.search(r"(\d+)", e["contributor"]).group(1))
        by[e["lua_step"]].add(a)
        step[e["lua_step"]] = min(step.get(e["lua_step"], 10 ** 9), e["step"])
    out = []
    for k, ags in by.items():
        if len(ags) >= 2:
            for q in PAIRS:
                if set(q) <= ags:
                    out.append((step[k], q))
    return sorted(out)


def episode_rules(ax, run, T, y):
    """Thin dashed rules at episode boundaries, labelled once at the top."""
    for e, (s0, _) in enumerate(run["_ep_bounds"][1:], start=2):
        ax.axvline(s0, color=RULE, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.text(s0 + T * 0.004, y, "episode %d" % e, fontsize=FS["note"],
                color=MUTED, ha="left", va="top")
    ax.text(T * 0.004, y, "episode 1", fontsize=FS["note"], color=MUTED,
            ha="left", va="top")


def finish(ax, T, xlabel=True):
    ax.set_xlim(0, T)
    ax.tick_params(labelsize=FS["tick"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xlabel:
        ax.set_xlabel("Environment step", fontsize=FS["label"])


def build(arm="gemma3f", seed="seed_123"):
    cfg = ARMS[arm]
    run_dir = cfg["run"] / seed
    run = load_run(run_dir)
    T = run["_ep_bounds"][-1][1]

    OUT.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(RC):
        # ---- figure 1: reciprocal communication -----------------------
        centres, counts = mutual_rate(load_messages(run), T)
        fig, ax = plt.subplots(figsize=(5.5, 2.15))
        fig.subplots_adjust(left=0.115, right=0.972, top=0.855, bottom=0.205)
        top = max(c.max() for c in counts.values())
        episode_rules(ax, run, T, top * 1.14)
        for q in PAIRS:
            ax.plot(centres, counts[q], color=PAIR_C[q], lw=1.2,
                    label="a%d–a%d" % q, zorder=3)
        ax.set_ylim(0, top * 1.18)
        ax.set_ylabel("Reciprocal exchanges\nper %d steps" % WINDOW,
                      fontsize=FS["label"])
        ax.legend(fontsize=FS["legend"], frameon=False, ncol=3,
                  loc="lower left", bbox_to_anchor=(0, 1.005),
                  handlelength=1.5, columnspacing=1.6, borderaxespad=0)
        finish(ax, T)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / ("mutual_messages." + ext),
                        dpi=400 if ext == "png" else None, facecolor="white")
        plt.close(fig)

        # ---- figure 2: bond strength ----------------------------------
        W = bond_series(cfg, run_dir)
        wbar = {q: (W[:, q[0], q[1]] + W[:, q[1], q[0]]) / 2 for q in PAIRS}
        fires = joint_fires(run)
        fig, ax = plt.subplots(figsize=(5.5, 2.15))
        fig.subplots_adjust(left=0.115, right=0.972, top=0.855, bottom=0.205)
        top = max(w.max() for w in wbar.values())
        episode_rules(ax, run, T, top * 1.18)
        for q in PAIRS:
            ax.plot(np.arange(len(W)), wbar[q], color=PAIR_C[q], lw=1.2,
                    label="a%d–a%d" % q, zorder=3)
        for t, q in fires:
            ax.plot(t, wbar[q][min(t, len(W) - 1)], marker="o", ms=4.5,
                    mfc="white", mec=PAIR_C[q], mew=1.1, zorder=5)
        ax.plot([], [], marker="o", ls="none", ms=4.5, mfc="white",
                mec=INK, mew=1.1, label="Joint milestone")
        ax.set_ylim(0, top * 1.22)
        ax.set_ylabel("Bond strength $\\bar{W}$", fontsize=FS["label"])
        ax.legend(fontsize=FS["legend"], frameon=False, ncol=4,
                  loc="lower left", bbox_to_anchor=(0, 1.005),
                  handlelength=1.5, columnspacing=1.6, borderaxespad=0)
        finish(ax, T)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / ("bond_strength." + ext),
                        dpi=400 if ext == "png" else None, facecolor="white")
        plt.close(fig)

    tot = {q: int(counts[q].sum()) for q in PAIRS}
    print("  %s %s, %d steps" % (cfg["label"], seed, T))
    print("  reciprocal exchanges: " +
          ", ".join("a%d-a%d %d" % (q[0], q[1], tot[q]) for q in PAIRS))
    print("  joint milestones: %s" % (fires,))
    print("  wrote %s/{mutual_messages,bond_strength}.(pdf|png|svg)" % OUT)


if __name__ == "__main__":
    a = sys.argv
    build(arm=a[a.index("--arm") + 1] if "--arm" in a else "gemma3f",
          seed=a[a.index("--seed") + 1] if "--seed" in a else "seed_123")
