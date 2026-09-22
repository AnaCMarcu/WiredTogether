"""Communication-budget sweep: outcome curves, budget utilisation, per-cell table.

Reads the ``comm_budget`` run group (``budget_gemma_<arm>_n<N>_b<B>/seed_*``,
see hpc/daic/experiments/submit_comm_budget.sh) and writes, under
``paper_assets/comm_budget/`` by default:

  budget_outcomes.{png,pdf}     one column per team size N; rows = cooperative
                                milestones (% of COOP_MAX) and non-comm
                                milestones (% of NONCOMM_MAX) against the
                                budget (0 / low / medium / high on a
                                categorical axis — the ladder is log-spaced,
                                so equal spacing IS the log axis with a slot
                                for zero); one line per arm, one small dot
                                per seed
  budget_utilisation.{png,pdf}  rows = N, columns = non-zero budgets; mean
                                fraction of the budget still unspent against
                                the step, per arm, with the median exhaustion
                                step marked
  budget_cells.csv              one row per (arm, N, budget): seeds, episodes,
                                both milestone percentages (mean, sd), messages
                                sent per agent-episode, spend fraction, share of
                                agent-episodes that exhausted, median exhaustion
                                step, request share and exact-repeat share of
                                the sent messages

Performance numbers come from make_results (same honesty-filtered episode
sets as every paper table); budget numbers come from the per-episode
``summary.json`` ``comm_budget`` block, the ``budget_left`` field stamped on
``messages.jsonl`` records, and the ``comm_budget_exhausted`` events.

    python analysis/make_budget_fig.py
    python analysis/make_budget_fig.py --runs-root runs_from_daic/comm_budget --ns 3 5 7 \
        --budgets 0 800 3200 12800
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paths import ASSETS, RUNS, group  # noqa: E402  (also puts siblings on sys.path)

from make_results import (  # noqa: E402
    COOP_MAX,
    MILESTONE_TRACK,
    NONCOMM_MAX,
    SOCIAL_ACT_TRACKS,
    aggregate,
    coop_count,
    episode_milestone_sets,
    load_runs,
    mean_std,
)

# Same arm colours/markers as make_scaling_fig.py so the two figures read as
# one family (blue/orange is the paper's two-arm pair).
ARMS = [  # (arm key, label, colour, marker, linestyle)
    ("base", "baseline", "#2c7fb8", "o", "-"),
    ("hebbian", "+Hebbian", "#d95f0e", "D", "--"),
]
DEFAULT_NS = (3, 5, 7)
DEFAULT_BUDGETS = (0, 800, 3200, 12800)
DEFAULT_PREFIX = "budget_gemma"
CURVE_BIN = 25  # steps per utilisation-curve sample

_REQUEST = re.compile(
    r"\?|\b(can you|could you|please|help me|need you|come to|join me|"
    r"meet me|wait for|follow me|press|hold|attack the|go to)\b", re.I)


def cell_dir(prefix: str, arm: str, n: int, b: int) -> str:
    return f"{prefix}_{arm}_n{n}_b{b}"


def budget_label(b: int) -> str:
    if b == 0:
        return "0"
    return f"{b / 1000:g}k" if b >= 1000 else str(b)


# ── per-run numbers ──────────────────────────────────────────────────────

def run_outcome_means(run) -> tuple[float, float]:
    """(coop %, non-comm milestone %) averaged over this run's episodes."""
    sets_ = episode_milestone_sets(run)
    if not sets_:
        return float("nan"), float("nan")
    coop = [100.0 * coop_count(s) / COOP_MAX for s in sets_]
    nc = [100.0 * sum(1 for m in s if MILESTONE_TRACK.get(m) not in SOCIAL_ACT_TRACKS)
          / NONCOMM_MAX for s in sets_]
    return statistics.fmean(coop), statistics.fmean(nc)


def run_budget_stats(run_dir: Path, budget: int, max_steps: int) -> dict:
    """Budget bookkeeping for one run, pooled over its agent-episodes."""
    spent_frac, exhausted_steps, sent, texts_by_ae = [], [], [], defaultdict(list)
    n_agent_eps = 0
    n_bins = max_steps // CURVE_BIN + 1
    curve_sum = np.zeros(n_bins)
    curve_cnt = np.zeros(n_bins)
    for ep_dir in sorted(run_dir.glob("episodes/ep_*")):
        try:
            summ = json.loads((ep_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        final_step = int(summ.get("final_step") or max_steps)
        cb = summ.get("comm_budget") or {}
        per_agent = cb.get("per_agent") or {}
        for agent, st in per_agent.items():
            n_agent_eps += 1
            total = float(st.get("budget") or budget or 0)
            spent_frac.append(float(st.get("spent", 0)) / total if total > 0 else 1.0)
            sent.append(int(st.get("sent", 0)))
            if st.get("exhausted_step") is not None:
                exhausted_steps.append(int(st["exhausted_step"]))
        # Utilisation curve: budget_left is stamped on each sent message.
        left_by_agent: dict[str, list[tuple[int, int]]] = defaultdict(list)
        mpath = ep_dir / "messages.jsonl"
        if mpath.exists():
            with open(mpath, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    texts_by_ae[(ep_dir.name, m.get("sender"))].append(m.get("text") or "")
                    if "budget_left" in m:
                        left_by_agent[m["sender"]].append((int(m["t"]), int(m["budget_left"])))
        if budget > 0 and per_agent:
            for agent in per_agent:
                pts = sorted(left_by_agent.get(agent, []))
                frac, j = 1.0, 0
                for k in range(n_bins):
                    t = k * CURVE_BIN
                    if t > final_step:
                        break
                    while j < len(pts) and pts[j][0] <= t:
                        frac = pts[j][1] / budget
                        j += 1
                    curve_sum[k] += frac
                    curve_cnt[k] += 1
    n_msgs = sum(len(v) for v in texts_by_ae.values())
    n_req = sum(1 for v in texts_by_ae.values() for t in v if _REQUEST.search(t))
    n_rep = sum(1 for v in texts_by_ae.values()
                for a, b in zip(v, v[1:]) if a == b)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(curve_cnt > 0, curve_sum / np.maximum(curve_cnt, 1), np.nan)
    return {
        "n_agent_eps": n_agent_eps,
        "spent_frac": spent_frac,
        "exhausted_steps": exhausted_steps,
        "sent": sent,
        "n_msgs": n_msgs,
        "request_share": (n_req / n_msgs) if n_msgs else float("nan"),
        "repeat_share": (n_rep / n_msgs) if n_msgs else float("nan"),
        "curve": curve,
        "curve_cnt": curve_cnt,
    }


# ── collection ───────────────────────────────────────────────────────────

def collect(runs_root: Path, prefix: str, ns, budgets, arms, max_steps: int):
    rows, curves = [], {}
    for arm, label, colour, marker, ls in arms:
        for n in ns:
            for b in budgets:
                dir_name = cell_dir(prefix, arm, n, b)
                runs = load_runs(runs_root, dir_name)
                if not runs:
                    print(f"  [skip] {dir_name}: no finished seed", file=sys.stderr)
                    continue
                agg = aggregate(runs)
                cm, cs = agg["coop"]
                mm, ms = agg["allms_nc"]
                per_seed = [run_outcome_means(r) for r in runs]
                bs = [run_budget_stats(Path(r["_path"]).parent, b, max_steps) for r in runs]
                spent = [v for s in bs for v in s["spent_frac"]]
                exh = [v for s in bs for v in s["exhausted_steps"]]
                sent = [v for s in bs for v in s["sent"]]
                n_ae = sum(s["n_agent_eps"] for s in bs)
                n_msgs = sum(s["n_msgs"] for s in bs)
                req = sum(s["request_share"] * s["n_msgs"] for s in bs
                          if s["n_msgs"]) / n_msgs if n_msgs else float("nan")
                rep = sum(s["repeat_share"] * s["n_msgs"] for s in bs
                          if s["n_msgs"]) / n_msgs if n_msgs else float("nan")
                rows.append({
                    "arm": arm, "label": label, "colour": colour, "marker": marker,
                    "ls": ls, "n_agents": n, "budget": b,
                    "n_seeds": len(runs), "n_episodes": agg["n_eps"],
                    "seeds": ";".join(Path(r["_path"]).parent.name for r in runs),
                    "coop_pct_mean": 100.0 * cm / COOP_MAX,
                    "coop_pct_sd": 100.0 * cs / COOP_MAX,
                    "ms_pct_mean": 100.0 * mm / NONCOMM_MAX,
                    "ms_pct_sd": 100.0 * ms / NONCOMM_MAX,
                    "per_seed_coop": [p[0] for p in per_seed],
                    "per_seed_ms": [p[1] for p in per_seed],
                    "msgs_sent_per_agent_ep": (statistics.fmean(sent) if sent else float("nan")),
                    "spent_frac_mean": (statistics.fmean(spent) if spent else float("nan")),
                    "exhausted_share": (len(exh) / n_ae) if n_ae else float("nan"),
                    "exhausted_step_median": (statistics.median(exh) if exh else float("nan")),
                    "request_share": req, "repeat_share": rep,
                    "n_msgs": n_msgs,
                })
                if b > 0:
                    tot = np.nansum([np.nan_to_num(s["curve"]) * s["curve_cnt"] for s in bs], axis=0)
                    cnt = np.sum([s["curve_cnt"] for s in bs], axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        curves[(arm, n, b)] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return rows, curves


# ── figures ──────────────────────────────────────────────────────────────

def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_outcomes(rows, ns, budgets, arms, out_stem: Path):
    metrics = [("coop_pct", f"Cooperative milestones (% of {COOP_MAX})", "per_seed_coop"),
               ("ms_pct", f"Milestones, non-comm (% of {NONCOMM_MAX})", "per_seed_ms")]
    fig, axes = plt.subplots(len(metrics), len(ns), figsize=(3.1 * len(ns), 5.2),
                             sharex=True, sharey="row", squeeze=False)
    xs = np.arange(len(budgets))
    for ci, n in enumerate(ns):
        for ri, (key, ylabel, seed_key) in enumerate(metrics):
            ax = axes[ri][ci]
            _style(ax)
            for arm, label, colour, marker, ls in arms:
                cell = {r["budget"]: r for r in rows if r["arm"] == arm and r["n_agents"] == n}
                pts = [(i, cell[b]) for i, b in enumerate(budgets) if b in cell]
                if not pts:
                    continue
                ax.plot([i for i, _ in pts], [r[f"{key}_mean"] for _, r in pts],
                        color=colour, ls=ls, lw=1.6, marker=marker, ms=6,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=label if (ri == 0 and ci == 0) else None, zorder=3)
                for i, r in pts:
                    ax.plot([i] * len(r[seed_key]), r[seed_key], ls="none", marker=".",
                            ms=4, color=colour, alpha=0.55, zorder=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([budget_label(b) for b in budgets])
            if ri == 0:
                ax.set_title(f"N = {n}", fontsize=10)
            if ri == len(metrics) - 1:
                ax.set_xlabel("budget (tokens / agent / episode)", fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            ax.tick_params(labelsize=8)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def fig_utilisation(rows, curves, ns, budgets, arms, max_steps: int, out_stem: Path):
    nz = [b for b in budgets if b > 0]
    if not nz or not curves:
        print("  [skip] utilisation figure: no non-zero budget cells", file=sys.stderr)
        return
    fig, axes = plt.subplots(len(ns), len(nz), figsize=(3.0 * len(nz), 2.4 * len(ns)),
                             sharex=True, sharey=True, squeeze=False)
    t = np.arange(max_steps // CURVE_BIN + 1) * CURVE_BIN
    for ri, n in enumerate(ns):
        for ci, b in enumerate(nz):
            ax = axes[ri][ci]
            _style(ax)
            for arm, label, colour, marker, ls in arms:
                c = curves.get((arm, n, b))
                if c is None:
                    continue
                ax.plot(t, c, color=colour, ls=ls, lw=1.6,
                        label=label if (ri == 0 and ci == 0) else None, zorder=3)
                med = next((r["exhausted_step_median"] for r in rows
                            if r["arm"] == arm and r["n_agents"] == n and r["budget"] == b),
                           float("nan"))
                if med == med:  # not NaN
                    ax.axvline(med, color=colour, ls=":", lw=1.0, alpha=0.8, zorder=2)
            ax.set_ylim(0, 1.02)
            if ri == 0:
                ax.set_title(f"budget {budget_label(b)}", fontsize=10)
            if ri == len(ns) - 1:
                ax.set_xlabel("step", fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(f"N = {n}\nfraction unspent", fontsize=8.5)
            ax.tick_params(labelsize=8)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def write_csv(rows, out: Path):
    cols = ["arm", "n_agents", "budget", "n_seeds", "n_episodes", "seeds",
            "coop_pct_mean", "coop_pct_sd", "ms_pct_mean", "ms_pct_sd",
            "msgs_sent_per_agent_ep", "spent_frac_mean", "exhausted_share",
            "exhausted_step_median", "request_share", "repeat_share", "n_msgs"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["arm"], r["n_agents"], r["budget"])):
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in cols})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="Run group root (default: the comm_budget group)")
    ap.add_argument("--out", type=Path, default=ASSETS / "comm_budget")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--arms", nargs="+", default=[a[0] for a in ARMS],
                    choices=[a[0] for a in ARMS])
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()

    runs_root = args.runs_root or group("comm_budget", RUNS)
    if not runs_root.is_dir():
        sys.exit(f"no such run root: {runs_root}")
    arms = [a for a in ARMS if a[0] in args.arms]
    rows, curves = collect(runs_root, args.prefix, args.ns, args.budgets, arms,
                           args.max_steps)
    if not rows:
        sys.exit(f"no finished cells under {runs_root} "
                 f"(expected {args.prefix}_<arm>_n<N>_b<B>/seed_*/final_metrics.json)")
    args.out.mkdir(parents=True, exist_ok=True)
    fig_outcomes(rows, args.ns, args.budgets, arms, args.out / "budget_outcomes")
    fig_utilisation(rows, curves, args.ns, args.budgets, arms, args.max_steps,
                    args.out / "budget_utilisation")
    write_csv(rows, args.out / "budget_cells.csv")
    for r in sorted(rows, key=lambda r: (r["arm"], r["n_agents"], r["budget"])):
        print(f"{r['arm']:8s} N={r['n_agents']} b={r['budget']:>6d}  seeds={r['n_seeds']}  "
              f"coop {r['coop_pct_mean']:5.1f}%  ms {r['ms_pct_mean']:5.1f}%  "
              f"sent/agent-ep {r['msgs_sent_per_agent_ep']:6.1f}  "
              f"spent {100 * r['spent_frac_mean']:5.1f}%  "
              f"exhausted {100 * r['exhausted_share']:5.1f}% @ {r['exhausted_step_median']:.0f}  "
              f"requests {100 * r['request_share']:4.1f}%")
    print(f"wrote {args.out}/budget_outcomes.png, budget_utilisation.png, budget_cells.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
