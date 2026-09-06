#!/usr/bin/env python3
"""replay_hebbian_terms.py — replay the gated Hebbian rule offline, term by term.

The LLM runs log W only every 50 steps (graph_snapshots) and at 2 dp per
step (the agents' 'social_bonds' belief text); none of the update's terms
are logged. This script re-runs HebbianGraph._update_gated exactly, from
the per-step inputs that ARE logged (positions + chamber from step_log.csv,
routed messages from messages.jsonl, per-agent reward streams from
final_metrics.json), validates the replay against both ground truths, and
plots every term of

    dW_ij = [ coeff_i * c_ij * (1 - W_ij)          growth
              or  -eta_minus * W_ij                 failure decay (gated)  ]
            - lambda * W_ij                         homeostatic decay (always)

    coeff_i = eta_0 + eta_plus * |r_bond_i| / R
    c_ij    = 1[d_ij <= radius] * g_i * g_j  +  0.5 * 1[i<->j messaged] * 1[d_ij > radius]
    g_i     = clip(0.5 * |r_bond_i| / M_running_max + 0.5 * 1[i messaged], 0, 1)

so that "the bond drops after the pair succeeds" can be traced to the term
responsible. Mirrors src/hebbian/graph.py::_update_gated (mode
reward_modulated); the equation and call-site signals are documented in
the module docstring of that file and in the summary printed here.

Usage:
  python analysis/replay_hebbian_terms.py                          # seed 456, hebbian arm
  python analysis/replay_hebbian_terms.py --seed 1213
  python analysis/replay_hebbian_terms.py --zoom 529 --zoom 2032   # event windows
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paths import ASSETS  # noqa: E402  (also puts siblings on sys.path)
from make_results import MILESTONE_TRACK  # noqa: E402

PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_C = {(0, 1): "#4a3aa7", (0, 2): "#008300", (1, 2): "#d55181"}
AGENT_C = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
INK, MUTED, RULE = "#31373f", "#98a1ad", "#d7dce3"
CH_INT = {"ch1": 1, "ch2": 2, "ch3": 3, "ch4": 4, "ch5": 5}


# ─── Inputs ─────────────────────────────────────────────────────────────
def load_inputs(run_dir: Path) -> dict:
    fm = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["cli_args"]
    lens = [int(x) for x in fm["episode_lengths"]]
    off = [sum(lens[:e]) for e in range(len(lens))]
    total = sum(lens)
    N = 3

    pos = np.full((total, N, 3), np.nan)
    chamber = np.zeros((total, N), dtype=int)
    envstep = np.zeros((total, N))
    for e, L in enumerate(lens):
        with open(run_dir / "episodes" / f"ep_{e + 1:04d}" / "step_log.csv",
                  newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                t = int(r["step"])
                a = int(str(r["agent_id"]).replace("agent_", ""))
                if not (0 <= t < L):
                    continue
                k = off[e] + t
                pos[k, a] = (float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"]))
                chamber[k, a] = CH_INT.get(str(r.get("chamber") or ""), 0)
                try:
                    envstep[k, a] = float(r.get("reward_task") or 0.0)
                except ValueError:
                    pass

    comm = [[] for _ in range(total)]          # routed (sender, receiver) per step
    for e, L in enumerate(lens):
        p = run_dir / "episodes" / f"ep_{e + 1:04d}" / "messages.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            try:
                i = int(str(m["sender"]).replace("agent_", ""))
                j = int(str(m["receiver"]).replace("agent_", ""))
            except (ValueError, TypeError, KeyError):
                continue
            if i != j and 0 <= m["t"] < L:
                comm[off[e] + m["t"]].append((i, j))

    comps = ["task", "comm_base", "comm_milestone", "hebbian_diffuse"]
    rew = {c: np.zeros((total, N)) for c in comps}
    for a, recs in enumerate(fm["reward_history_decomposed"]):
        # records are per metric timestep; the per-episode logger drops the
        # final step of each episode, so map by episode-local 't' when it resets
        k_global = 0
        prev_t = -1
        ep = 0
        for r in recs:
            t = int(r.get("t", k_global))
            if t < prev_t:                       # new episode
                ep += 1
                k_global = off[ep] if ep < len(off) else k_global
            elif prev_t == -1:
                k_global = t
            else:
                k_global = off[ep] + t if ep < len(off) else k_global + 1
            prev_t = t
            if 0 <= k_global < total:
                for c in comps:
                    rew[c][k_global, a] = float(r.get(c, 0.0))

    # env-milestone drain per agent per step (global steps, per contributor)
    milestone = np.zeros((total, N))
    for ev in fm.get("milestone_events", []):
        mid = ev.get("milestone_id") or ev.get("milestone")
        if MILESTONE_TRACK.get(mid) == "communication":
            continue
        s = int(ev["step"])
        a = int("".join(ch for ch in str(ev.get("contributor")) if ch.isdigit()) or -1)
        if 0 <= s < total and 0 <= a < N:
            milestone[s, a] += float(ev.get("reward", 0.0))

    snaps = [(int(s["step"]), np.array(s["W"], dtype=float))
             for s in fm.get("graph_snapshots", []) if s.get("W")]
    W_text = None
    cache = ASSETS / "hebbian_terms" / \
        f"W_text_seed{cfg.get('seed')}.npy"
    if cache.exists():
        W_text = np.load(cache)

    seen, cocos = set(), []
    for ev in fm.get("co_completion_events", []):
        pair = tuple(sorted((int(ev["agent_i"]), int(ev["agent_j"]))))
        mid = ev.get("milestone", "")
        if MILESTONE_TRACK.get(mid) == "communication":
            continue
        key = (int(ev["step"]), pair, mid)
        if key not in seen:
            seen.add(key)
            cocos.append({"step": int(ev["step"]), "pair": pair, "mid": mid})

    hp = dict(
        eta_plus=float(cfg.get("hebbian_eta_plus", 0.05)),
        eta_0=float(cfg.get("hebbian_eta_0", 0.01)),
        eta_minus=float(cfg.get("hebbian_eta_minus", 0.025)),
        coop_eps=float(cfg.get("hebbian_coop_eps", 0.05)),
        coop_window=int(cfg.get("hebbian_coop_window", 50)),
        neg_theta=float(cfg.get("hebbian_neg_theta", 5.0)),
        R=float(cfg.get("hebbian_reward_norm", 300.0)),
        alpha=float(cfg.get("hebbian_alpha", 0.5)),
        radius=float(cfg.get("hebbian_radius", 5.0)),
        lam=float(cfg.get("hebbian_decay", 0.0003)),
        delta_comm=0.0 if cfg.get("hebbian_no_comm_bond") else 0.5,
        w0=float(cfg.get("hebbian_init_weight", 0.1)),
        mode=cfg.get("hebbian_mode", "reward_modulated"),
    )
    return dict(total=total, lens=lens, off=off, N=N, pos=pos, chamber=chamber,
                envstep=envstep, comm=comm, rew=rew, milestone=milestone,
                snaps=snaps, W_text=W_text, cocos=cocos, hp=hp, seed=cfg.get("seed"))


# ─── The rule, replayed ─────────────────────────────────────────────────
def replay(inp: dict) -> dict:
    """Exact re-implementation of HebbianGraph._update_gated for mode
    reward_modulated (src/hebbian/graph.py:552-677)."""
    hp, N, T = inp["hp"], inp["N"], inp["total"]
    W = np.full((N, N), hp["w0"]); np.fill_diagonal(W, 0.0)
    M = 0.0
    coact_win = deque(maxlen=max(1, hp["coop_window"]))
    rew_win = deque(maxlen=max(1, hp["coop_window"]))
    eps = 1e-8

    out = {k: np.zeros((T, N, N)) for k in
           ("W", "growth", "growth_reward", "growth_base", "homeo", "fail",
            "c", "c_spat", "c_comm", "coop")}
    out.update({k: np.zeros((T, N)) for k in ("g", "coeff", "rb", "rt", "neg", "ch")})
    out["M"] = np.zeros(T)

    task, cb, cm = inp["rew"]["task"], inp["rew"]["comm_base"], inp["rew"]["comm_milestone"]
    for t in range(T):
        ch = (inp["chamber"][t] >= 2).astype(float)
        # bond reward: milestone drain + comm (base+milestone) + pitch penalty.
        # pitch + death = task - envstep - milestone; a residual <= -9 is a death
        # (excluded from the bond reward by construction), the rest is pitch.
        resid = task[t] - inp["envstep"][t] - inp["milestone"][t]
        death = np.where(resid <= -9.0, resid, 0.0)
        pitch = resid - death
        rb = ch * (inp["milestone"][t] + cb[t] + cm[t] + pitch)
        rt = ch * (task[t] + cb[t] + cm[t])                 # step_rewards_raw, gated

        # engagement
        cur_max = float(np.abs(rb).max())
        M = max(M, cur_max)
        social = set()
        for s, r in inp["comm"][t]:
            social.add(s); social.add(r)
        g = np.clip(hp["alpha"] * np.abs(rb) / (M + eps)
                    + np.array([(1 - hp["alpha"]) if i in social else 0.0 for i in range(N)]),
                    0.0, 1.0)

        # co-activity
        p = inp["pos"][t]
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        S = ((dist <= hp["radius"]) & np.isfinite(dist)).astype(float)
        np.fill_diagonal(S, 0.0)
        c_spat = S * np.outer(g, g)
        c_comm = np.zeros((N, N))
        for s, r in inp["comm"][t]:
            if 0 <= s < N and 0 <= r < N and s != r and hp["delta_comm"] > 0:
                for a, b in ((s, r), (r, s)):
                    c_comm[a, b] = hp["delta_comm"] * (1.0 - S[a, b])
        c = np.clip(c_spat + c_comm, 0.0, 1.0)
        np.fill_diagonal(c, 0.0)
        c[c < hp["coop_eps"]] = 0.0

        # windows
        coact_win.append(c.copy()); rew_win.append(rt.copy())
        coop = np.maximum.reduce(list(coact_win))
        neg = np.sum(rew_win, axis=0) < -hp["neg_theta"]

        # growth / decay
        if hp["mode"] == "coactivity":
            coeff = np.full(N, hp["eta_plus"])
            salience = np.zeros(N)
        else:
            salience = hp["eta_plus"] * np.abs(rb) / hp["R"]
            coeff = hp["eta_0"] + salience
        growth = coeff[:, None] * c * (1.0 - W)
        growth_reward = salience[:, None] * c * (1.0 - W)
        decay_mask = (coop < hp["coop_eps"]) & neg[:, None]
        fail = hp["eta_minus"] * W
        homeo = hp["lam"] * W
        delta = np.where(decay_mask, -fail, growth) - homeo
        np.fill_diagonal(delta, 0.0)

        out["growth"][t] = np.where(decay_mask, 0.0, growth)
        out["growth_reward"][t] = np.where(decay_mask, 0.0, growth_reward)
        out["growth_base"][t] = out["growth"][t] - out["growth_reward"][t]
        out["fail"][t] = np.where(decay_mask, fail, 0.0)
        out["homeo"][t] = homeo
        out["c"][t], out["c_spat"][t], out["c_comm"][t], out["coop"][t] = c, c_spat, c_comm, coop
        out["g"][t], out["coeff"][t], out["rb"][t], out["rt"][t] = g, coeff, rb, rt
        out["neg"][t], out["ch"][t], out["M"][t] = neg, ch, M

        W = np.clip(W + delta, 0.0, 1.0)
        np.fill_diagonal(W, 0.0)
        out["W"][t] = W                     # W AFTER the update of step t
    return out


def validate(inp, out) -> dict:
    W = out["W"]
    e_snap = [abs(W[G - 1, i, j] - Wt[i, j]) for G, Wt in inp["snaps"]
              if 1 <= G <= inp["total"] for i in range(3) for j in range(3) if i != j]
    res = dict(snap_mae=float(np.mean(e_snap)), snap_max=float(np.max(e_snap)),
               n_snap=len(e_snap))
    if inp["W_text"] is not None:
        Wt = inp["W_text"]
        m = np.isfinite(Wt[1:])
        res["text_mae"] = float(np.abs(W[:-1] - Wt[1:])[m].mean())
    return res


# ─── Plots ──────────────────────────────────────────────────────────────
def sym(A, i, j):
    return 0.5 * (A[:, i, j] + A[:, j, i])


def plot_terms(inp, out, out_dir: Path, x0=None, x1=None, tag=""):
    T = inp["total"]
    x0 = 0 if x0 is None else max(0, x0)
    x1 = T if x1 is None else min(T, x1)
    steps = np.arange(T)
    sl = slice(x0, x1)
    hp = inp["hp"]

    fig, axes = plt.subplots(7, 1, figsize=(8.0, 11.0), sharex=True,
                             gridspec_kw=dict(hspace=0.10, height_ratios=[1.6, 1, 1, 1, 1, 1, 1]))
    ax = axes[0]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["W"], *q)[sl], color=PAIR_C[q], lw=1.4,
                label=f"a{q[0]}-a{q[1]} (replay)")
    for G, Wt in inp["snaps"]:
        if x0 <= G - 1 < x1:
            for q in PAIRS:
                ax.scatter(G - 1, 0.5 * (Wt[q[0], q[1]] + Wt[q[1], q[0]]), s=12,
                           color=PAIR_C[q], edgecolors="white", linewidths=0.4, zorder=5)
    for c in inp["cocos"]:
        if x0 <= c["step"] < x1:
            q = c["pair"]
            y = sym(out["W"], *q)[c["step"]]
            ax.scatter(c["step"], y, marker="D", s=40, facecolor="white",
                       edgecolor=PAIR_C[q], linewidths=1.3, zorder=6)
    # no-reward fixed points of the rule
    for c_val, lab in ((0.25, "W* near + messaging (c=0.25)"),
                       (0.5, "W* far + messaging (c=0.5)")):
        wstar = hp["eta_0"] * c_val / (hp["eta_0"] * c_val + hp["lam"])
        ax.axhline(wstar, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.text(x1 - (x1 - x0) * 0.005, wstar + 0.004, f"{lab} = {wstar:.2f}",
                fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.set_ylabel("bond W_ij\n(symmetrised)", fontsize=8)
    ax.legend(fontsize=6.5, ncol=3, loc="upper left", frameon=False)
    ax.set_title(f"Hebbian update replayed term by term — seed {inp['seed']}"
                 + (f"  ({tag})" if tag else ""), fontsize=10, loc="left", color=INK)

    ax = axes[1]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["growth_reward"], *q)[sl], color=PAIR_C[q], lw=1.0)
    ax.set_ylabel("growth from\nreward\n(eta+ |r|/R) c (1-W)", fontsize=7.5)

    ax = axes[2]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["growth_base"], *q)[sl], color=PAIR_C[q], lw=1.0)
        ax.plot(steps[sl], -sym(out["homeo"], *q)[sl], color=PAIR_C[q], lw=1.0, ls=(0, (2, 1.5)))
    ax.axhline(0, color=RULE, lw=0.7)
    ax.set_ylabel("baseline growth\neta0 c (1-W)  (solid)\nvs  -lambda W  (dotted)", fontsize=7.5)

    ax = axes[3]
    for q in PAIRS:
        net = sym(out["growth"], *q) - sym(out["homeo"], *q) - sym(out["fail"], *q)
        ax.plot(steps[sl], net[sl], color=PAIR_C[q], lw=1.0)
        f = sym(out["fail"], *q)
        if (f[sl] > 0).any():
            ax.plot(steps[sl], -f[sl], color=PAIR_C[q], lw=1.6, alpha=0.6)
    ax.axhline(0, color=RULE, lw=0.7)
    ax.set_ylabel("net dW per step", fontsize=7.5)

    ax = axes[4]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["c_spat"], *q)[sl], color=PAIR_C[q], lw=0.9)
        ax.plot(steps[sl], sym(out["c_comm"], *q)[sl], color=PAIR_C[q], lw=0.9, ls=(0, (2, 1.5)))
    ax.set_ylabel("co-activity\nspatial g_i g_j (solid)\ncomm 0.5(1-near) (dotted)", fontsize=7.5)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[5]
    for a in range(3):
        ax.plot(steps[sl], out["g"][sl, a], color=AGENT_C[a], lw=0.9, label=f"g agent {a}")
    ax2 = ax.twinx()
    ax2.plot(steps[sl], out["M"][sl], color=INK, lw=0.9, ls=(0, (4, 2)))
    ax2.set_ylabel("running max |r_bond|  M", fontsize=7, color=INK)
    ax.set_ylabel("engagement g_i", fontsize=7.5)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6.5, ncol=3, loc="upper left", frameon=False)

    ax = axes[6]
    for a in range(3):
        rb = out["rb"][sl, a]
        ax.vlines(steps[sl][rb != 0], 0, rb[rb != 0], color=AGENT_C[a], lw=1.0, alpha=0.9)
        ax.plot(steps[sl], out["coeff"][sl, a] * 100, color=AGENT_C[a], lw=0.8, ls=(0, (1, 1.5)))
    ax.set_ylabel("bond reward r_bond\n(bars)  and\ncoeff x100 (dotted)", fontsize=7.5)
    ax.set_xlabel("env step (running across the three episodes)", fontsize=8)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=7)
        for e, o in enumerate(inp["off"]):
            if e and x0 <= o < x1:
                a.axvline(o, color=INK, lw=0.7, ls=(0, (4, 3)), alpha=0.35)
        for c in inp["cocos"]:
            if x0 <= c["step"] < x1:
                a.axvline(c["step"], color=PAIR_C[c["pair"]], lw=0.6, alpha=0.35)
    axes[0].set_xlim(x0, x1)
    name = f"hebbian_terms_seed{inp['seed']}" + (f"_{tag}" if tag else "")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def plot_terms_paper(inp, out, out_dir: Path, x0: int, x1: int, name: str,
                     width: float = 5.5):
    """Camera-ready five-panel version: exact width, Times, Type-42 fonts,
    no in-figure title (the caption carries it). Panels, top to bottom:
    bond W with stored snapshots; reward-driven growth; co-activity growth
    against homeostatic decay; net update; co-activity by channel."""
    import make_directive_timelines as mdt  # applies the camera-ready rcParams
    plt.rcParams["mathtext.fontset"] = "stix"
    FS_ROW, FS_AXIS, FS_LEG = mdt.FS_ROW, mdt.FS_AXIS, 6.0
    ink, muted, rule = mdt.INK, mdt.MUTED, mdt.RULE
    T = inp["total"]
    x0, x1 = max(0, x0), min(T, x1)
    steps = np.arange(T)
    sl = slice(x0, x1)
    hp = inp["hp"]

    heights = [1.30, 0.62, 0.84, 0.62, 0.84]          # inches
    gap, left, right, top, bottom = 0.07, 0.70, 0.10, 0.08, 0.36
    H = top + sum(heights) + gap * (len(heights) - 1) + bottom
    fig = plt.figure(figsize=(width, H))
    axes, y = [], H - top
    for h in heights:
        y -= h
        axes.append(fig.add_axes([left / width, y / H,
                                  1 - (left + right) / width, h / H]))
        y -= gap

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(rule)
        ax.tick_params(colors=ink, labelsize=FS_ROW, length=2.5, width=0.7,
                       color=rule, pad=1.6)
        ax.set_xlim(x0, x1)
        for e, o in enumerate(inp["off"]):
            if e and x0 <= o < x1:
                ax.axvline(o, color=ink, lw=0.7, ls=(0, (4, 3)), alpha=0.35)
        for c in inp["cocos"]:
            if x0 <= c["step"] < x1:
                ax.axvline(c["step"], color=PAIR_C[c["pair"]], lw=0.6,
                           alpha=0.35)

    # 1. bonds
    ax = axes[0]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["W"], *q)[sl], color=PAIR_C[q], lw=1.2,
                label=f"a{q[0]}-a{q[1]}")
    for G, Wt in inp["snaps"]:
        if x0 <= G - 1 < x1:
            for q in PAIRS:
                ax.scatter(G - 1, 0.5 * (Wt[q[0], q[1]] + Wt[q[1], q[0]]), s=9,
                           color=PAIR_C[q], edgecolors="white", linewidths=0.4,
                           zorder=5)
    for c in inp["cocos"]:
        if x0 <= c["step"] < x1:
            q = c["pair"]
            ax.scatter(c["step"], sym(out["W"], *q)[c["step"]], marker="D",
                       s=28, facecolor="white", edgecolor=PAIR_C[q],
                       linewidths=1.1, zorder=6)
    for c_val, lab in ((0.25, "$W^*$, near + messaging"),
                       (0.5, "$W^*$, far + messaging")):
        wstar = hp["eta_0"] * c_val / (hp["eta_0"] * c_val + hp["lam"])
        ax.axhline(wstar, color=muted, lw=0.7, ls=(0, (3, 2)))
        ax.text(x1 - (x1 - x0) * 0.006, wstar + 0.004, lab, fontsize=5.6,
                color=muted, ha="right", va="bottom")
    ax.scatter([], [], s=9, color=ink, label="stored snapshot")
    ax.set_ylabel("bond $W_{ij}$", fontsize=FS_AXIS, color=ink)
    wlo = min(float(np.nanmin(sym(out["W"], *q)[sl])) for q in PAIRS)
    whi = max(float(np.nanmax(sym(out["W"], *q)[sl])) for q in PAIRS)
    ax.set_ylim(wlo - 0.07, max(whi, 0.34) + 0.03)   # free band for the legend
    ax.legend(fontsize=FS_LEG, ncol=4, loc="lower left", frameon=False,
              handlelength=1.4, columnspacing=1.0, borderaxespad=0.3)
    style(ax)

    # 2. reward-driven growth
    ax = axes[1]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["growth_reward"], *q)[sl], color=PAIR_C[q],
                lw=0.9)
    ax.set_ylabel("reward growth\n$\\eta_+ |r|\\,c\\,(1-W)/R$",
                  fontsize=FS_AXIS, color=ink, linespacing=1.3)
    style(ax)

    # 3. baseline growth vs homeostatic decay
    ax = axes[2]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["growth_base"], *q)[sl], color=PAIR_C[q],
                lw=0.9)
        ax.plot(steps[sl], -sym(out["homeo"], *q)[sl], color=PAIR_C[q],
                lw=0.9, ls=(0, (2, 1.5)))
    ax.axhline(0, color=rule, lw=0.7)
    ax.plot([], [], color=ink, lw=0.9, label="$\\eta_0\\,c\\,(1-W)$")
    ax.plot([], [], color=ink, lw=0.9, ls=(0, (2, 1.5)), label="$-\\lambda W$")
    glo = min(float(np.nanmin(-sym(out["homeo"], *q)[sl])) for q in PAIRS)
    ghi = max(float(np.nanmax(sym(out["growth_base"], *q)[sl])) for q in PAIRS)
    ax.set_ylim(glo * 1.15, ghi * 1.75)             # headroom for the legend
    ax.legend(fontsize=FS_LEG, ncol=2, loc="upper right", frameon=False,
              handlelength=1.6, borderaxespad=0.2)
    ax.set_ylabel("growth vs\ndecay", fontsize=FS_AXIS, color=ink,
                  linespacing=1.3)
    style(ax)

    # 4. net
    ax = axes[3]
    for q in PAIRS:
        net = (sym(out["growth"], *q) - sym(out["homeo"], *q)
               - sym(out["fail"], *q))
        ax.plot(steps[sl], net[sl], color=PAIR_C[q], lw=0.9)
    ax.axhline(0, color=rule, lw=0.7)
    ax.set_ylabel("net $\\Delta W$\nper step", fontsize=FS_AXIS, color=ink,
                  linespacing=1.3)
    style(ax)

    # 5. co-activity channels
    ax = axes[4]
    for q in PAIRS:
        ax.plot(steps[sl], sym(out["c_spat"], *q)[sl], color=PAIR_C[q], lw=0.8)
        ax.plot(steps[sl], sym(out["c_comm"], *q)[sl], color=PAIR_C[q], lw=0.8,
                ls=(0, (2, 1.5)))
    ax.plot([], [], color=ink, lw=0.8, label="spatial $g_i g_j$")
    ax.plot([], [], color=ink, lw=0.8, ls=(0, (2, 1.5)),
            label="communication")
    ax.legend(fontsize=FS_LEG, ncol=2, loc="upper right", frameon=False,
              handlelength=1.6, borderaxespad=0.2)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("co-activity\n$c_{ij}$", fontsize=FS_AXIS, color=ink,
                  linespacing=1.3)
    style(ax)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("env step (running across the three episodes)",
                        fontsize=FS_AXIS, color=ink, labelpad=2)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    return width, H


# ─── Report ─────────────────────────────────────────────────────────────
def event_report(inp, out, horizon=(10, 25, 50, 100)) -> list[str]:
    lines = []
    for c in inp["cocos"]:
        s, q = c["step"], c["pair"]
        i, j = q
        w = sym(out["W"], i, j)
        gr = sym(out["growth"], i, j); gr_r = sym(out["growth_reward"], i, j)
        ho = sym(out["homeo"], i, j); fa = sym(out["fail"], i, j)
        cc = sym(out["c"], i, j)
        row = [f"step {s:4d} pair a{i}-a{j} {c['mid']}: W={w[s]:.3f}; kick at event step "
               f"= +{gr[s]:.4f} (reward part +{gr_r[s]:.4f}), homeo -{ho[s]:.4f}"]
        for h in horizon:
            e = min(s + h, inp['total'] - 1)
            G, H, F = gr[s + 1:e + 1].sum(), ho[s + 1:e + 1].sum(), fa[s + 1:e + 1].sum()
            act = (cc[s + 1:e + 1] > 0).mean() * 100
            row.append(f"   +{h:3d}: dW={w[e] - w[s]:+.3f}  = growth {G:+.3f} - homeo {H:.3f}"
                       f" - fail {F:.3f} | pair co-active {act:.0f}% of steps")
        lines.extend(row)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("runs_from_daic"))
    ap.add_argument("--arm", default="new_exp_0_gemma/new_exp_0_gemma_hebbian")
    ap.add_argument("--seed", type=int, default=456)
    ap.add_argument("--out", type=Path, default=Path("paper_assets/hebbian_terms"))
    ap.add_argument("--zoom", type=int, action="append", default=None,
                    help="event step(s) to zoom on ([-100, +200] window)")
    ap.add_argument("--paper", action="store_true",
                    help="also write camera-ready *_paper.pdf/.png zooms "
                         "(5.5in, Times, Type-42 fonts, five panels)")
    ap.add_argument("--width", type=float, default=5.5,
                    help="paper figure width in inches (ICLR textwidth)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rd = args.runs_root / args.arm / f"seed_{args.seed}"
    inp = load_inputs(rd)
    out = replay(inp)
    v = validate(inp, out)
    print(f"replay vs exact 50-step snapshots: MAE={v['snap_mae']:.5f} max={v['snap_max']:.5f} "
          f"(n={v['n_snap']})" + (f"; vs 2-dp per-step text: MAE={v['text_mae']:.4f}"
                                  if "text_mae" in v else ""))
    hp = inp["hp"]
    print(f"rule: mode={hp['mode']} eta0={hp['eta_0']} eta+={hp['eta_plus']} R={hp['R']} "
          f"lambda={hp['lam']} eta-={hp['eta_minus']} window={hp['coop_window']} "
          f"radius={hp['radius']} alpha={hp['alpha']}")
    fires = int((out["fail"] > 0).any(axis=(1, 2)).sum())
    print(f"failure-decay branch active on {fires} of {inp['total']} steps; "
          f"reward-driven growth nonzero on {int((out['growth_reward'] > 1e-9).any(axis=(1,2)).sum())} steps; "
          f"running max M final = {out['M'][-1]:.1f}")
    tot_g, tot_h, tot_f = (out["growth"].sum(), out["homeo"].sum(), out["fail"].sum())
    tot_gr = out["growth_reward"].sum()
    print(f"cumulative over run (all pairs): growth {tot_g:+.2f} (of which reward-driven {tot_gr:+.2f}, "
          f"baseline {tot_g - tot_gr:+.2f}), homeostatic decay -{tot_h:.2f}, failure decay -{tot_f:.2f}")

    rep = event_report(inp, out)
    print("\n".join(rep))
    (args.out / f"hebbian_terms_seed{args.seed}_report.txt").write_text(
        "\n".join([f"validation: {v}", f"rule: {hp}", ""] + rep) + "\n", encoding="utf-8")

    names = [plot_terms(inp, out, args.out)]
    for z in (args.zoom or []):
        names.append(plot_terms(inp, out, args.out, z - 100, z + 200, tag=f"zoom{z}"))
    np.savez(args.out / f"hebbian_terms_seed{args.seed}.npz",
             **{k: v for k, v in out.items()})
    print("wrote", ", ".join(f"{args.out}/{n}.pdf/.png" for n in names))
    if args.paper:
        for z in (args.zoom or [529]):
            pname = f"hebbian_terms_seed{args.seed}_zoom{z}_paper"
            w, h = plot_terms_paper(inp, out, args.out, z - 100, z + 200,
                                    pname, width=args.width)
            print(f"  {pname}.pdf  {w:.2f} x {h:.2f} in (camera-ready)")


if __name__ == "__main__":
    main()
