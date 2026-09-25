#!/usr/bin/env python
"""Do the agents that fire together stay wired together?  (Fig. 11 / 13 follow-up)

Delta-bond analysis of the two relational-plasticity timelines in the paper:

  Fig. 11   N=6, Gemma-E4B +trace   agent_scaling_3f/scale_gemma_hebbian_n6/seed_456
  Fig. 13   N=3, Gemma-E4B +trace   pareto_social_3f/new_exp_0_gemma_si3f8/seed_42

Both figures draw the symmetrised bond W̄ from the 50-step graph_snapshots; Fig. 11
shows only five of the fifteen pairs.  This script looks at ALL pairs through the
CHANGE of the bond, ΔW̄ per snapshot interval (the "delta graph" over time), and:

  1. replays the deployed three-factor rule from the logged inputs (positions,
     messages, chambers, bondable rewards) inside every interval, starting from
     the recorded W, to attribute the observed ΔW̄ to
        - milestone impulses  η₊·(r_ms/R)·e·(1−W)   (a kill / switch / door
          converting the eligibility trace — "firing together"),
        - message pay         η₊·(r_comm/R)·e·(1−W) (the +0.5 per valid message
          and the m_comm milestones converting the same trace, every step),
        - baseline co-activity η₀·c·(1−W),
        - homeostatic decay   −λ·W   and signed death LTD;
  2. splits the pairs into bonds that CHANGE (at least one interval whose |ΔW̄|
     clears τ, by default 0.08 ≈ the jump one milestone credited through a live
     trace produces) and bonds that DON'T, and replots the Fig. 11/13 lane with
     that split + the ΔW̄ strip + one delta graph per episode;
  3. measures RETENTION: after every potentiation interval, how much of the gain
     is still there 50–400 steps later — raw, and as the pair's ADVANTAGE over
     the team-mean bond (immune to the field-wide drift) — against the
     decay-only reference (1−λ)^h;
  4. pools the same statistics over every seed of both arms.

Outputs → paper_assets/delta_bonds/
  delta_bonds_<tag>.pdf/.png          replot (lane split + strip + delta graphs)
  delta_bonds_retention.pdf/.png      retention after potentiation, both runs
  delta_bonds_<tag>_intervals.csv     one row per (pair, interval)
  delta_bonds_stats.md                every number quoted in the text

Run:  python analysis/make_delta_bonds.py [--tau 0.08] [--only n6] [--no-pool]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ASSETS, group  # noqa: E402

OUT = ASSETS / "delta_bonds"
HS = (50, 100, 200, 400)

# ── the two runs behind the paper's figures (+ their arms, for pooling) ──
RUNS = {
    "n6": dict(arm=group("agent_scaling_3f") / "scale_gemma_hebbian_n6", seed=456,
               fig="Fig. 11", label="N = 6, seed 456"),
    "n3": dict(arm=group("pareto_social_3f") / "new_exp_0_gemma_si3f8", seed=42,
               fig="Fig. 13", label="N = 3, seed 42"),
}
for _r in RUNS.values():
    _r["run"] = _r["arm"] / f"seed_{_r['seed']}"

# Deployed three-factor constants (config.json cli_args of both runs).
RULE = dict(eta0=0.001, eta_plus=0.05, lam=0.001, rho=0.9, R=50.0, radius=5.0,
            floor=0.25, delta_comm=0.5, eps=0.05, death_ltd=0.05, death_cap=10.0)

# Milestones one event can credit to two agents (same set as the timeline
# scripts); m21 kills are solo-credited but the trace credits the partners.
SHAREABLE = {"m8_anvil_A1", "m9_anvil_B1", "m17_switch_pressed", "m18_door_opened",
             "m21_first_mob_kill", "m22_all_mobs_killed", "m25_first_boss_dmg",
             "m26_boss_half_hp", "m27_boss_defeated"}

# ── design tokens (Okabe–Ito, as in the paper's timeline scripts) ───────
AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73",
           3: "#CC79A7", 4: "#E69F00", 5: "#56B4E9"}
# Fixed categorical order for the changing pairs (never cycled past 8).
CHANGE_C = ["#5B4EA4", "#B2652C", "#2E8B6E", "#B8860B", "#a0567a",
            "#0072B2", "#4f7f7a", "#c0862f"]
STABLE_C = ["#5f6b78", "#7d8894", "#98a2ad", "#b0b8c1", "#c4cad1",
            "#6b7683", "#8a949f", "#a3acb6", "#77828e"]
INK = "#2b2f36"
GRID = "#eef1f4"
CHAMBER_TINT = ["#f2f5f8", "#ffffff"]
# Diverging ΔW̄: blue (weakens) – neutral grey – orange (strengthens).
DIV = LinearSegmentedColormap.from_list(
    "dW", ["#2166ac", "#7fa8d0", "#e8eaee", "#f0a06a", "#c8501a"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#c3c9d0", "axes.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})
FS = dict(axis=8.6, tick=7.6, title=9.0, small=7.0, label=7.4)


def aid(name):
    return int(re.search(r"(\d+)", name).group(1))


def pairs_of(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def pname(p):
    return f"a{p[0]}–a{p[1]}"


def topk(N):
    """'Still a leading bond' = rank within the top fifth of the pairs
    (top 3 of 15 at N=6, the single strongest of 3 at N=3)."""
    return max(1, (N * (N - 1) // 2) // 5)


# ── loading ─────────────────────────────────────────────────────────────
def load(run_dir: Path) -> dict:
    fm = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    N = int(json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["num_agents"])
    lens = [int(x) for x in fm["episode_lengths"]]
    bounds, c = [], 0
    for L in lens:
        bounds.append((c, c + L))
        c += L
    T = bounds[-1][1]

    by_step = {}                                             # last snapshot per step
    for s in fm["graph_snapshots"]:
        if s.get("W"):
            by_step[int(s["step"])] = np.array(s["W"], dtype=float)
    snaps = sorted(by_step.items())
    S = np.array([s for s, _ in snaps])
    Wd = np.stack([w for _, w in snaps])                     # (K, N, N) directed
    T = max(T, int(S[-1]) + 1)

    pos = np.full((T, N, 3), np.nan)
    ch = np.zeros((T, N), dtype=int)
    msg = np.zeros((T, N, N), dtype=bool)
    ch_int = {"ch1": 1, "ch2": 2, "ch3": 3, "ch4": 4, "ch5": 5}
    for e, (s0, _s1) in enumerate(bounds):
        ep = run_dir / "episodes" / f"ep_{e + 1:04d}"
        with (ep / "step_log.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                t = s0 + int(r["step"])
                if t >= T:
                    continue
                a = int(r["agent_id"])
                try:
                    pos[t, a] = (float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"]))
                except ValueError:
                    pass
                ch[t, a] = ch_int.get(r.get("chamber") or "", 0)
        for line in (ep / "messages.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            t = s0 + int(m["t"])
            if t >= T or not m.get("receiver"):
                continue
            s, r = aid(m["sender"]), aid(m["receiver"])
            if s != r and 0 <= r < N:
                msg[t, s, r] = True

    # Bondable reward per agent per step: milestone drain + comm pay
    # (reward_history_decomposed carries comm_base / comm_milestone per step;
    # the milestone drain is rebuilt from milestone_events on the same clock).
    comm = np.zeros((T, N))
    death = np.zeros((T, N))
    for a, hist in enumerate(fm["reward_history_decomposed"]):
        for rec in hist:
            t = int(rec["t"])
            if t < T:
                comm[t, a] = float(rec.get("comm_base", 0.0)) + float(rec.get("comm_milestone", 0.0))
                task = float(rec.get("task", 0.0))
                if task <= -9.0:          # −10 would-die / −50 death impulse
                    death[t, a] = min(abs(task), RULE["death_cap"])
    ms = np.zeros((T, N))
    for ev in fm["milestone_events"]:
        t, a, r = int(ev["step"]), aid(ev["contributor"]), float(ev["reward"])
        if t < T and r > 0 and not ev["milestone_id"].startswith("m_comm"):
            ms[t, a] += r

    # Genuine joint fires: two agents credited on the same lua_step for a
    # shareable milestone.  Kills: (step, killer).
    by, st = defaultdict(set), {}
    kills = []
    for ev in fm["milestone_events"]:
        mid = ev["milestone_id"]
        if mid not in SHAREABLE:
            continue
        by[ev["lua_step"]].add(aid(ev["contributor"]))
        st[ev["lua_step"]] = min(st.get(ev["lua_step"], 10 ** 9), int(ev["step"]))
        if mid == "m21_first_mob_kill":
            kills.append((int(ev["step"]), aid(ev["contributor"])))
    joints = []
    for k, ags in by.items():
        ags = sorted(ags)
        for x in range(len(ags)):
            for y in range(x + 1, len(ags)):
                joints.append((st[k], (ags[x], ags[y])))

    return dict(N=N, T=T, bounds=bounds, S=S, Wd=Wd, pos=pos, ch=ch, msg=msg,
                comm=comm, ms=ms, death=death, kills=sorted(kills), joints=sorted(joints),
                fm=fm, run_dir=run_dir)


# ── the rule, replayed per interval from the recorded W ─────────────────
def coactivity(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """c_ij(t) as the deployed rule computes it (engagement at its 0.25 floor,
    which binds whenever both agents message — every agent does, every step)
    and the eligibility trace e_ij(t) = ρ e(t−1) + c(t)."""
    T, N = d["T"], d["N"]
    pos, msg = d["pos"], d["msg"]
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    near = np.isfinite(dist) & (dist <= RULE["radius"])
    c = RULE["floor"] * near + RULE["delta_comm"] * (msg | msg.transpose(0, 2, 1))
    c = np.clip(c, 0.0, 1.0)
    for i in range(N):
        c[:, i, i] = 0.0
    c[c < RULE["eps"]] = 0.0
    e = np.zeros_like(c)
    for t in range(T):
        e[t] = (RULE["rho"] * e[t - 1] if t else 0.0) + c[t]
    return c, e


def sym(M):
    return (M + M.T) / 2


def replay_intervals(d: dict, c: np.ndarray, e: np.ndarray) -> list[dict]:
    """For every snapshot interval, integrate the rule forward from the
    recorded W and keep the per-term sums (symmetrised)."""
    S, Wd, N = d["S"], d["Wd"], d["N"]
    gate = (d["ch"] >= 2).astype(float)                       # Ch2–5 gate
    r_ms = d["ms"] * gate
    r_comm = d["comm"] * gate
    rows = []
    for k in range(len(S) - 1):
        t0, t1 = int(S[k]), int(S[k + 1])
        W = Wd[k].copy()
        acc = {n: np.zeros((N, N)) for n in ("base", "imp_ms", "imp_comm", "decay", "death")}
        for t in range(t0, t1):
            one_m_w = 1.0 - W
            base = RULE["eta0"] * c[t] * one_m_w
            sal_ms = RULE["eta_plus"] * (r_ms[t] / RULE["R"])[:, None] * e[t] * one_m_w
            sal_cm = RULE["eta_plus"] * (r_comm[t] / RULE["R"])[:, None] * e[t] * one_m_w
            dec = RULE["lam"] * W
            dl = RULE["death_ltd"] * (d["death"][t] / RULE["R"])[:, None] * e[t] * W
            for n, v in (("base", base), ("imp_ms", sal_ms), ("imp_comm", sal_cm),
                         ("decay", -dec), ("death", -dl)):
                acc[n] += v
            W = np.clip(W + base + sal_ms + sal_cm - dec - dl, 0.0, 1.0)
            np.fill_diagonal(W, 0.0)
        obs = sym(Wd[k + 1] - Wd[k])
        pred = sym(W - Wd[k])
        mutual = (d["msg"][t0:t1] & d["msg"][t0:t1].transpose(0, 2, 1)).sum(0)
        near = (c[t0:t1] >= RULE["floor"] - 1e-9).sum(0)   # steps within radius
        for (i, j) in pairs_of(N):
            rows.append(dict(
                k=k, t0=t0, t1=t1, i=i, j=j, W0=sym(Wd[k])[i, j], W1=sym(Wd[k + 1])[i, j],
                obs=obs[i, j], pred=pred[i, j],
                base=sym(acc["base"])[i, j], imp_ms=sym(acc["imp_ms"])[i, j],
                imp_comm=sym(acc["imp_comm"])[i, j], decay=sym(acc["decay"])[i, j],
                death=sym(acc["death"])[i, j], mutual=int(mutual[i, j]),
                near=int(near[i, j])))
    return rows


def credited_kills(d: dict, e: np.ndarray, thresh: float = 1.0) -> list[tuple[int, int, int]]:
    """(step, killer, partner): partners whose trace with the killer was live
    (e_ij ≥ thresh, i.e. co-active within the last ~10 steps) when the kill
    reward landed — the pairs the rule potentiates for that kill."""
    out = []
    for t, k in d["kills"]:
        t = min(t, d["T"] - 1)
        for j in range(d["N"]):
            if j != k and e[t, k, j] >= thresh:
                out.append((t, k, j))
    return out


# ── classification + retention ──────────────────────────────────────────
def wbar_series(d: dict) -> dict[tuple[int, int], np.ndarray]:
    return {(i, j): (d["Wd"][:, i, j] + d["Wd"][:, j, i]) / 2 for (i, j) in pairs_of(d["N"])}


def classify(rows: list[dict], N: int, tau: float, min_len: int = 25) -> tuple[list, list, dict]:
    """A pair CHANGES if any interval of ≥ min_len steps moved |ΔW̄| ≥ tau."""
    peak = defaultdict(float)
    for r in rows:
        if r["t1"] - r["t0"] >= min_len:
            peak[(r["i"], r["j"])] = max(peak[(r["i"], r["j"])], abs(r["obs"]))
    changing = [p for p in pairs_of(N) if peak[p] >= tau]
    stable = [p for p in pairs_of(N) if peak[p] < tau]
    return changing, stable, dict(peak)


def retention(d: dict, rows: list[dict], tau: float) -> list[dict]:
    """After every potentiation interval (obs ≥ tau): the fraction of the raw
    gain still present h steps after the interval closed (ret_h), the same
    for the pair's ADVANTAGE over the team-mean bond (adv_h; the field's
    common drift cancels), and the pair's rank among all pairs (1 = strongest)
    at the close and h steps later."""
    S, wb = d["S"], wbar_series(d)
    ps = list(wb)
    Wall = np.stack([wb[p] for p in ps])                     # (P, K)
    mean_all = Wall.mean(0)
    out = []
    for r in rows:
        if r["obs"] < tau or r["t1"] - r["t0"] < 25:
            continue
        p = (r["i"], r["j"])
        pre, post = r["W0"], r["W1"]
        gain = post - pre
        adv = wb[p] - mean_all
        adv0, adv1 = float(np.interp(r["t0"], S, adv)), float(np.interp(r["t1"], S, adv))
        again = adv1 - adv0
        rank1 = 1 + int((Wall[:, np.searchsorted(S, r["t1"])] > post).sum())
        rec = dict(pair=p, t0=r["t0"], t1=r["t1"], pre=pre, post=post, gain=gain,
                   adv_gain=again, rank1=rank1, imp_ms=r["imp_ms"], imp_comm=r["imp_comm"])
        for h in HS:
            th = r["t1"] + h
            if th <= S[-1]:
                w_h = float(np.interp(th, S, wb[p]))
                rec[f"ret_{h}"] = (w_h - pre) / gain
                rec[f"adv_{h}"] = ((float(np.interp(th, S, adv)) - adv0) / again
                                   if again >= tau / 2 else np.nan)
                col = np.array([np.interp(th, S, wb[q]) for q in ps])
                rec[f"rank_{h}"] = 1 + int((col > w_h).sum())
            else:
                rec[f"ret_{h}"] = rec[f"adv_{h}"] = rec[f"rank_{h}"] = np.nan
        out.append(rec)
    return out


def control_drift(d: dict, rows: list[dict], tau: float) -> dict:
    """Same look-ahead for the pair-intervals that were NOT potentiated:
    the field's own drift, in W̄ units, to compare with the retained gain."""
    S, wb = d["S"], wbar_series(d)
    out = {h: [] for h in HS}
    for r in rows:
        if r["obs"] >= tau or r["t1"] - r["t0"] < 25:
            continue
        p = (r["i"], r["j"])
        for h in HS:
            th = r["t1"] + h
            if th <= S[-1]:
                out[h].append(float(np.interp(th, S, wb[p])) - r["W1"])
    return {h: (float(np.median(v)) if v else np.nan, len(v)) for h, v in out.items()}


# ── chamber spans on the cumulative clock ───────────────────────────────
def chamber_spans(d: dict, min_len: int = 12) -> list[tuple[str, int, int]]:
    T, ch = d["T"], d["ch"]
    maj = np.zeros(T, dtype=int)
    for t in range(T):
        v = ch[t][ch[t] > 0]
        maj[t] = np.bincount(v).argmax() if v.size else (maj[t - 1] if t else 0)
    starts = {b[0] for b in d["bounds"]}
    spans, t0 = [], 0
    for t in range(1, T + 1):
        if t == T or maj[t] != maj[t0] or t in starts:
            if maj[t0] > 0 and t - t0 >= min_len:
                spans.append((f"Ch{maj[t0]}", t0, t))
            elif spans and maj[t0] > 0 and t0 not in starts:
                spans[-1] = (spans[-1][0], spans[-1][1], t)
            t0 = t
    out = []
    for lab, lo, hi in spans:
        if out and out[-1][0] == lab and lo <= out[-1][2] and lo not in starts:
            out[-1] = (lab, out[-1][1], hi)
        else:
            out.append((lab, lo, hi))
    return out


def draw_backdrop(ax, d: dict, spans, label_chambers=True, label_episodes=True):
    for n, (lab, lo, hi) in enumerate(spans):
        ax.axvspan(lo, hi, color=CHAMBER_TINT[n % 2], lw=0, zorder=0)
    for b0, _b1 in d["bounds"][1:]:
        ax.axvline(b0, color="#9aa4af", lw=0.8, ls=(0, (2, 2)), zorder=1)
    ymax = ax.get_ylim()[1]
    if label_chambers:
        last = -1e9
        for lab, lo, hi in spans:
            cx = (lo + hi) / 2
            if hi - lo < 0.025 * d["T"] or cx - last < 0.045 * d["T"]:
                continue
            ax.text(cx, ymax * 0.02, lab, ha="center", va="bottom", fontsize=FS["small"],
                    color="#6b7683", zorder=4)
            last = cx
    if label_episodes:
        for n, (b0, b1) in enumerate(d["bounds"]):
            ax.text((b0 + b1) / 2, ymax * 0.975, f"episode {n + 1}", ha="center", va="top",
                    fontsize=FS["small"], color="#6b7683", zorder=4)


# ── the replot ──────────────────────────────────────────────────────────
def _repel(ys, gap):
    """Push label y-positions apart (sorted order preserved)."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def lane(ax, d, wb, pairs, colors, glyphs, title, ylim, lw=1.6, label_end=True):
    S = d["S"]
    for n, p in enumerate(pairs):
        col = colors[n % len(colors)]
        ax.plot(S, wb[p], color=col, lw=lw, solid_capstyle="round", zorder=3)
    if label_end and pairs:
        ys = _repel([wb[p][-1] for p in pairs], 0.055 * ylim[1])
        for n, p in enumerate(pairs):
            ax.annotate(pname(p), (S[-1], ys[n]), xytext=(3, 0), textcoords="offset points",
                        fontsize=FS["label"], color=colors[n % len(colors)], va="center",
                        ha="left", zorder=5, annotation_clip=False)
    for kind, t, p, who in glyphs:
        if p not in pairs:
            continue
        y = float(np.interp(t, S, wb[p]))
        col = colors[pairs.index(p) % len(colors)]
        if kind == "kill":
            ax.scatter(t, y, marker="*", s=95, facecolor=AGENT_C[who], edgecolor="white",
                       linewidths=0.5, zorder=6)
        else:
            ax.scatter(t, y, marker="o", s=42, facecolor="white", edgecolor=col,
                       linewidths=1.4, zorder=6)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, d["T"])
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel("bond $\\bar{W}$", fontsize=FS["axis"])
    ax.tick_params(labelsize=FS["tick"], length=2)
    ax.set_title(title, loc="left", fontsize=FS["title"], fontweight="bold", pad=3)


def strip(ax, d, rows, order):
    """ΔW̄ per pair per interval; ★ where a milestone impulse credited the pair."""
    S = d["S"]
    K = len(S) - 1
    M = np.full((len(order), K), np.nan)
    idx = {p: n for n, p in enumerate(order)}
    for r in rows:
        M[idx[(r["i"], r["j"])], r["k"]] = r["obs"]
    vmax = float(np.nanmax(np.abs(M)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    y_edges = np.arange(len(order) + 1)
    pc = ax.pcolormesh(S, y_edges, M[::-1], cmap=DIV, norm=norm, shading="flat",
                       edgecolors="white", linewidth=0.3, zorder=2)
    for r in rows:
        if r["imp_ms"] >= 0.01:
            n = len(order) - 1 - idx[(r["i"], r["j"])]
            ax.scatter((r["t0"] + r["t1"]) / 2, n + 0.5, marker="*", s=22,
                       facecolor="white", edgecolor=INK, linewidths=0.4, zorder=4)
    for b0, _ in d["bounds"][1:]:
        ax.axvline(b0, color=INK, lw=0.8, ls=(0, (2, 2)), zorder=3)
    ax.set_yticks(y_edges[:-1] + 0.5)
    ax.set_yticklabels([pname(p) for p in order[::-1]], fontsize=FS["label"])
    ax.set_xlim(0, d["T"])
    ax.set_ylim(0, len(order))
    ax.tick_params(length=0, labelsize=FS["tick"])
    ax.set_ylabel("$\\Delta\\bar{W}$ per interval", fontsize=FS["axis"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return pc, vmax


def delta_graphs(axes, d, wb, order, vmax):
    """One node–link delta graph per episode: edge = ΔW̄ over the episode."""
    N, S = d["N"], d["S"]
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, N, endpoint=False)
    xy = np.c_[np.cos(ang), np.sin(ang)]
    for ax, (e, (b0, b1)) in zip(axes, enumerate(d["bounds"])):
        dW = {p: float(np.interp(min(b1, S[-1]), S, wb[p]) - np.interp(b0, S, wb[p]))
              for p in order}
        top = sorted(order, key=lambda p: -abs(dW[p]))[:3]
        for p in order:
            v = dW[p]
            col = DIV(0.5 + 0.5 * np.clip(v / vmax, -1, 1) * 0.95)
            ax.plot(xy[list(p), 0], xy[list(p), 1], color=col,
                    lw=0.4 + 6.0 * abs(v) / max(vmax, 1e-9), solid_capstyle="round", zorder=2)
            if p in top and abs(v) >= 0.02:
                mx, my = xy[list(p)].mean(0)
                ax.text(mx, my, f"{v:+.2f}", fontsize=FS["small"], ha="center", va="center",
                        color=INK, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
        for a in range(N):
            ax.scatter(*xy[a], s=110, facecolor=AGENT_C[a], edgecolor="white", linewidths=1.0, zorder=6)
            ax.text(*xy[a], f"{a}", fontsize=FS["small"], ha="center", va="center",
                    color="white", fontweight="bold", zorder=7)
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.axis("off")
        ax.set_title(f"episode {e + 1}: $\\Delta\\bar{{W}}$ start→end", fontsize=FS["label"], pad=2)


def replot(tag, d, rows, changing, stable, glyphs, tau, out: Path):
    wb = wbar_series(d)
    spans = chamber_spans(d)
    N = d["N"]
    n_pairs = len(pairs_of(N))
    order = changing + stable
    heights = [1.6] + ([1.15] if stable else []) + [0.115 * n_pairs + 0.25, 1.2]
    fig_h = sum(heights) + 0.45 * len(heights) + 0.75
    fig = plt.figure(figsize=(5.5, fig_h))
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.42,
                          left=0.12, right=0.89, top=1 - 0.55 / fig_h, bottom=0.45 / fig_h)
    ymax = max(float(v.max()) for v in wb.values()) * 1.12
    row = 0
    ax1 = fig.add_subplot(gs[row]); row += 1
    lane(ax1, d, wb, changing, CHANGE_C, glyphs,
         f"bonds that change: {len(changing)} of {n_pairs} pairs (some ≥25-step interval |$\\Delta\\bar{{W}}$| ≥ {tau:g})",
         (0, ymax))
    draw_backdrop(ax1, d, spans)
    ax1.tick_params(labelbottom=False)
    if stable:
        ax2 = fig.add_subplot(gs[row], sharex=ax1); row += 1
        lane(ax2, d, wb, stable, STABLE_C, glyphs,
             f"bonds that don't: {len(stable)} of {n_pairs} pairs (every interval |$\\Delta\\bar{{W}}$| < {tau:g})",
             (0, ymax), lw=1.1, label_end=False)
        draw_backdrop(ax2, d, spans, label_chambers=False, label_episodes=False)
        ax2.tick_params(labelbottom=False)
    ax3 = fig.add_subplot(gs[row], sharex=ax1); row += 1
    pc, vmax = strip(ax3, d, rows, order)
    if stable:
        ax3.axhline(len(stable), color=INK, lw=0.9, zorder=5)
    ax3.set_xlabel("environment step (cumulative over episodes)", fontsize=FS["axis"])
    cb = fig.colorbar(pc, ax=ax3, fraction=0.025, pad=0.012, aspect=12)
    cb.ax.tick_params(labelsize=FS["small"], length=2)
    cb.outline.set_visible(False)
    sub = gs[row].subgridspec(1, len(d["bounds"]), wspace=0.05)
    gaxes = [fig.add_subplot(sub[0, e]) for e in range(len(d["bounds"]))]
    delta_graphs(gaxes, d, wb, order, vmax)
    handles = [
        Line2D([], [], marker="*", ls="none", mfc=AGENT_C[1], mec="white", ms=9,
               label="mob kill (killer's colour) on each pair its trace credits"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=CHANGE_C[0], mew=1.3, ms=6,
               label="joint milestone (same lua step)"),
        Line2D([], [], marker="*", ls="none", mfc="white", mec=INK, mew=0.5, ms=6,
               label="strip: milestone impulse ≥ 0.01 credited to the pair"),
    ]
    fig.legend(handles=handles, fontsize=FS["small"], frameon=False, loc="lower center",
               ncol=2, borderaxespad=0.2, handletextpad=0.3, columnspacing=1.2)
    fig.suptitle(f"{RUNS[tag]['fig']} replotted by bond change — {RUNS[tag]['label']}",
                 fontsize=FS["title"], y=0.985)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"delta_bonds_{tag}.{ext}", dpi=220)
    plt.close(fig)


def retention_figure(res: dict, out: Path):
    lam = RULE["lam"]
    fig, axes = plt.subplots(2, len(res), figsize=(5.5, 4.0), sharey="row", squeeze=False)
    for col, (tag, r) in enumerate(res.items()):
        recs = r["retention"]
        for row, key, ylab in ((0, "ret", "raw gain retained"),
                               (1, "adv", "advantage over team mean retained")):
            ax = axes[row, col]
            for rec in recs:
                ys = [rec[f"{key}_{h}"] for h in HS]
                if np.all(np.isnan(ys)):
                    continue
                ax.plot([0] + list(HS), [1.0] + ys, color="#b8c0c8", lw=0.7, alpha=0.9, zorder=2)
            med = [float(np.nanmedian([rec[f"{key}_{h}"] for rec in recs])) if recs else np.nan for h in HS]
            n_ev = sum(1 for rec in recs if not np.isnan(rec[f"{key}_50"]))
            ax.plot([0] + list(HS), [1.0] + med, color=CHANGE_C[0], lw=2.0, marker="o", ms=4, zorder=4,
                    label=f"median of {n_ev} potentiations")
            ax.plot([0] + list(HS), [(1 - lam) ** h for h in (0,) + HS], color=INK, lw=1.0,
                    ls=(0, (3, 2)), zorder=3, label="decay only, $(1-\\lambda)^h$")
            ax.axhline(0, color="#9aa4af", lw=0.7, zorder=1)
            ax.axhline(1, color="#9aa4af", lw=0.7, zorder=1)
            ax.tick_params(labelsize=FS["tick"], length=2)
            ax.grid(axis="y", color=GRID, lw=0.7)
            ax.set_axisbelow(True)
            ax.set_ylim(-1.0, 2.5)
            ax.legend(fontsize=FS["small"], frameon=False, loc="upper left")
            if col == 0:
                ax.set_ylabel(ylab, fontsize=FS["axis"])
            if row == 0:
                ax.set_title(f"{RUNS[tag]['fig']} run — {RUNS[tag]['label']}", fontsize=FS["title"])
            else:
                ax.set_xlabel("steps after the potentiation interval, $h$", fontsize=FS["axis"])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"delta_bonds_retention.{ext}", dpi=220)
    plt.close(fig)


# ── stats ───────────────────────────────────────────────────────────────
def episode_end_table(d: dict) -> tuple[list[str], list[float]]:
    wb = wbar_series(d)
    S = d["S"]
    lines, rhos = [], []
    prev = None
    for e, (b0, b1) in enumerate(d["bounds"]):
        end = min(b1, S[-1])
        vals = {p: float(np.interp(end, S, wb[p])) for p in wb}
        rank = sorted(vals, key=lambda p: -vals[p])
        top = ", ".join(f"{pname(p)} {vals[p]:.2f}" for p in rank[:3])
        line = f"episode {e + 1} end (t={end}): top pairs {top}"
        if prev is not None:
            rho = spearmanr([prev[p] for p in wb], [vals[p] for p in wb]).correlation
            rhos.append(rho)
            line += f"; rank correlation with the previous episode end {rho:+.2f}"
        lines.append(line)
        prev = vals
    return lines, rhos


def _median_iqr(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "n/a"
    return f"{np.median(v):.2f} (IQR {np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f}, n={v.size})"


def run_stats(d: dict, rows: list[dict], tau: float) -> dict:
    """Everything the text quotes, for one run."""
    long = [r for r in rows if r["t1"] - r["t0"] >= 25]
    obs = np.array([r["obs"] for r in long])
    pred = np.array([r["pred"] for r in long])
    share = {n: float(np.sum([abs(r[n]) for r in long])) for n in ("base", "imp_ms", "imp_comm", "decay", "death")}
    tot = sum(share.values())
    pot = [r for r in long if r["obs"] >= tau]
    wb = wbar_series(d)
    final = {p: wb[p][-1] for p in wb}
    cred = defaultdict(lambda: defaultdict(float))
    for r in rows:
        for n in ("imp_ms", "imp_comm", "base", "decay", "mutual", "near"):
            cred[(r["i"], r["j"])][n] += r[n]
        # Effective drive rate k (per step): growth / Σ(1−W).  With the
        # always-on decay the bond relaxes toward W* = k/(k+λ) at rate k+λ.
        cred[(r["i"], r["j"])]["headroom"] += (r["t1"] - r["t0"]) * (1.0 - (r["W0"] + r["W1"]) / 2)
    for p, cr in cred.items():
        cr["k"] = (cr["imp_ms"] + cr["imp_comm"] + cr["base"]) / max(cr["headroom"], 1e-9)
        cr["wstar"] = cr["k"] / (cr["k"] + RULE["lam"])
        cr["tau_relax"] = 1.0 / (cr["k"] + RULE["lam"])
    ps = list(wb)
    sp = {}
    for n in ("imp_ms", "imp_comm", "mutual", "near"):
        sp[n] = (spearmanr([cred[p][n] for p in ps], [final[p] for p in ps]).correlation
                 if len(ps) > 3 else np.nan)
    ret = retention(d, rows, tau)
    ctrl = control_drift(d, rows, tau)
    _lines, rhos = episode_end_table(d)
    return dict(n_long=len(long), r=pearsonr(obs, pred)[0], mae=float(np.mean(np.abs(obs - pred))),
                share={n: v / tot for n, v in share.items()}, n_pot=len(pot),
                n_dep=sum(1 for r in long if r["obs"] <= -tau),
                n_pot_ms=sum(1 for r in pot if r["imp_ms"] >= 0.01),
                cred=cred, final=final, sp=sp, ret=ret, ctrl=ctrl, ep_rhos=rhos)


def analyse(tag: str, tau: float, out: Path) -> dict:
    d = load(RUNS[tag]["run"])
    c, e = coactivity(d)
    rows = replay_intervals(d, c, e)
    N = d["N"]
    changing, stable, peak = classify(rows, N, tau)
    ck = credited_kills(d, e)
    glyphs = [("kill", t, (min(k, j), max(k, j)), k) for t, k, j in ck]
    glyphs += [("joint", t, p, None) for t, p in d["joints"]]
    st = run_stats(d, rows, tau)

    with (out / f"delta_bonds_{tag}_intervals.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["changing"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "changing": int((r["i"], r["j"]) in changing)})

    replot(tag, d, rows, changing, stable, glyphs, tau, out)

    wb = wbar_series(d)
    sh = st["share"]
    L = [f"## {RUNS[tag]['fig']} run — {RUNS[tag]['label']}  (`{d['run_dir'].relative_to(ASSETS.parent)}`)", ""]
    L += [f"- {len(d['S'])} snapshots → {st['n_long']} pair-intervals of ≥25 steps over {N * (N - 1) // 2} pairs; τ = {tau:g}",
          f"- replay of the deployed rule inside each interval vs. the recorded ΔW̄: Pearson r = {st['r']:.2f}, MAE = {st['mae']:.3f} "
          "(the attribution below is therefore exact up to the 0.25 engagement floor and the reconstructed death impulses)",
          f"- |term| share of the predicted change: baseline co-activity {sh['base']:.0%}, milestone impulses {sh['imp_ms']:.0%}, "
          f"message-pay impulses {sh['imp_comm']:.0%}, decay {sh['decay']:.0%}, death LTD {sh['death']:.0%}",
          f"- intervals: {st['n_pot']} potentiation (ΔW̄ ≥ τ), {st['n_dep']} depression (≤ −τ), "
          f"{st['n_long'] - st['n_pot'] - st['n_dep']} within ±τ; {st['n_pot_ms']}/{st['n_pot']} potentiations carry a milestone impulse ≥ 0.01",
          f"- pairs that change ({len(changing)}): " + ", ".join(f"{pname(p)} (peak {peak[p]:.2f})" for p in changing),
          f"- pairs that don't ({len(stable)}): " + (", ".join(f"{pname(p)} (peak {peak[p]:.2f})" for p in stable) or "none"),
          f"- kills: {len(d['kills'])}; partners whose trace was live at a kill (e ≥ 1): {len(ck)} → "
          + (", ".join(f"t={t} a{k}→a{j} (e={e[min(t, d['T'] - 1), k, j]:.1f})" for t, k, j in ck) or "none"),
          f"- joint milestone fires: {len(d['joints'])} → " + (", ".join(f"t={t} {pname(p)}" for t, p in d["joints"]) or "none"),
          ""]
    L += ["| pair | changes? | final W̄ | range | peak \\|ΔW̄\\| | milestone credit | message-pay credit | baseline | decay | mutual msgs | steps ≤ 5 blocks | drive k (10⁻³/step) | W* = k/(k+λ) | relax. time (steps) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(wb, key=lambda q: -st["final"][q]):
        cr = st["cred"][p]
        L.append(f"| {pname(p)} | {'yes' if p in changing else 'no'} | {st['final'][p]:.2f} | {wb[p].max() - wb[p].min():.2f} | "
                 f"{peak[p]:.2f} | {cr['imp_ms']:+.2f} | {cr['imp_comm']:+.2f} | {cr['base']:+.2f} | {cr['decay']:+.2f} | "
                 f"{int(cr['mutual'])} | {int(cr['near'])} | {1e3 * cr['k']:.2f} | {cr['wstar']:.2f} | {cr['tau_relax']:.0f} |")
    L.append("")
    ret, ctrl = st["ret"], st["ctrl"]
    if ret:
        L.append("Retention after the potentiation intervals (fraction of the gain still present h steps after the interval closed):")
        L.append("")
        kk = topk(N)
        L.append(f"| h | raw gain retained | advantage over team mean retained | still top-{kk} | decay-only (1−λ)^h | drift of the non-potentiated pairs |")
        L.append("|---|---|---|---|---|---|")
        for h in HS:
            raw = [r[f"ret_{h}"] for r in ret]
            adv = [r[f"adv_{h}"] for r in ret]
            top = [(r[f"rank_{h}"] <= kk) for r in ret if not np.isnan(r[f"rank_{h}"]) and r["rank1"] <= kk]
            L.append(f"| {h} | {_median_iqr(raw)} | {_median_iqr(adv)} | "
                     f"{(np.mean(top) if top else np.nan):.0%} of {len(top)} | {(1 - RULE['lam']) ** h:.2f} | "
                     f"{ctrl[h][0]:+.3f} W̄ (n={ctrl[h][1]}) |")
        L.append("")
    sp = st["sp"]
    L += [f"- end-of-run W̄ vs. cumulative credit, Spearman over pairs: milestone impulses {sp['imp_ms']:+.2f}, "
          f"message-pay impulses {sp['imp_comm']:+.2f}, mutual-message count {sp['mutual']:+.2f}, steps within 5 blocks {sp['near']:+.2f}"]
    L += ["- " + s for s in episode_end_table(d)[0]]
    L.append("")
    return dict(d=d, rows=rows, changing=changing, stable=stable, retention=ret, control=ctrl, lines=L)


def pooled(tag: str, tau: float) -> list[str]:
    """The same statistics over every seed of the arm."""
    arm = RUNS[tag]["arm"]
    seeds = sorted(p for p in arm.iterdir() if re.fullmatch(r"seed_\d+", p.name))
    agg = defaultdict(list)
    per_seed = []
    for sd in seeds:
        d = load(sd)
        c, e = coactivity(d)
        rows = replay_intervals(d, c, e)
        changing, stable, _peak = classify(rows, d["N"], tau)
        st = run_stats(d, rows, tau)
        n_pairs = len(pairs_of(d["N"]))
        per_seed.append(f"  - {sd.name}: replay r={st['r']:.2f}; changing {len(changing)}/{n_pairs}; "
                        f"potentiations {st['n_pot']} ({st['n_pot_ms']} milestone-credited); "
                        f"shares base {st['share']['base']:.0%} / milestone {st['share']['imp_ms']:.0%} / "
                        f"message-pay {st['share']['imp_comm']:.0%} / decay {st['share']['decay']:.0%}; "
                        f"Spearman(final W̄, credit): milestone {st['sp']['imp_ms']:+.2f}, message-pay {st['sp']['imp_comm']:+.2f}; "
                        f"episode-end rank correlations {', '.join(f'{x:+.2f}' for x in st['ep_rhos'])}; "
                        f"joint fires {len(d['joints'])}, kills {len(d['kills'])}")
        kk = topk(d["N"])
        for h in HS:
            agg[f"ret_{h}"] += [r[f"ret_{h}"] for r in st["ret"]]
            agg[f"adv_{h}"] += [r[f"adv_{h}"] for r in st["ret"]]
            agg[f"top_{h}"] += [(r[f"rank_{h}"] <= kk) for r in st["ret"]
                                if not np.isnan(r[f"rank_{h}"]) and r["rank1"] <= kk]
        for n, v in st["share"].items():
            agg[f"share_{n}"].append(v)
        agg["n_pot"].append(st["n_pot"])
        agg["n_pot_ms"].append(st["n_pot_ms"])
        agg["sp_ms"].append(st["sp"]["imp_ms"])
        agg["sp_comm"].append(st["sp"]["imp_comm"])
        agg["ep_rhos"] += st["ep_rhos"]
        agg["changing"].append(len(changing))
        agg["n_pairs"].append(n_pairs)
    L = [f"### Pooled over {len(seeds)} seeds of `{arm.relative_to(ASSETS.parent)}`", ""]
    L += per_seed
    L += ["",
          f"- pairs that change: {sum(agg['changing'])}/{sum(agg['n_pairs'])}; potentiation intervals {sum(agg['n_pot'])}, "
          f"of which {sum(agg['n_pot_ms'])} carry a milestone impulse ≥ 0.01",
          f"- mean |term| shares: baseline {np.mean(agg['share_base']):.0%}, milestone {np.mean(agg['share_imp_ms']):.0%}, "
          f"message-pay {np.mean(agg['share_imp_comm']):.0%}, decay {np.mean(agg['share_decay']):.0%}, death {np.mean(agg['share_death']):.0%}",
          f"- Spearman(final W̄, cumulative credit), per seed: milestone {', '.join(f'{x:+.2f}' for x in agg['sp_ms'])}; "
          f"message-pay {', '.join(f'{x:+.2f}' for x in agg['sp_comm'])}",
          f"- episode-end rank correlation (consecutive episodes), pooled: {_median_iqr(agg['ep_rhos'])}",
          "", f"| h | raw gain retained | advantage retained | still top-{topk(load(seeds[0])['N'])} | decay-only |", "|---|---|---|---|---|"]
    for h in HS:
        top = agg[f"top_{h}"]
        L.append(f"| {h} | {_median_iqr(agg[f'ret_{h}'])} | {_median_iqr(agg[f'adv_{h}'])} | "
                 f"{(np.mean(top) if top else np.nan):.0%} of {len(top)} | {(1 - RULE['lam']) ** h:.2f} |")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.08,
                    help="|ΔW̄| per ≥25-step interval that counts as a change (default 0.08 ≈ one "
                         "milestone credited through a live trace)")
    ap.add_argument("--only", choices=list(RUNS), default=None)
    ap.add_argument("--no-pool", action="store_true", help="skip the all-seeds pooling")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tags = [args.only] if args.only else list(RUNS)
    res = {tag: analyse(tag, args.tau, OUT) for tag in tags}
    retention_figure(res, OUT)
    md = ["# Do the agents that fire together stay wired together?", "",
          f"Generated by `analysis/make_delta_bonds.py --tau {args.tau:g}`. ΔW̄ = change of the symmetrised "
          "bond between consecutive 50-step graph snapshots; the rule is replayed inside each interval from "
          "the recorded W with the deployed three-factor constants (η₀=0.001, η₊=0.05, λ=0.001, ρ=0.9, R=50, "
          "radius 5, engagement floor 0.25, δ_comm 0.5, death LTD 0.05 capped at 10).",
          ""]
    for tag in tags:
        md += res[tag]["lines"]
        if not args.no_pool:
            md += pooled(tag, args.tau)
    (OUT / "delta_bonds_stats.md").write_text("\n".join(md), encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    print("\n".join(md))


if __name__ == "__main__":
    main()
