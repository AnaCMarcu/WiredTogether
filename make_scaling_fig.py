"""Compute-scaling figure: cooperative milestones vs. inference compute.

The agent-count scaling suite (submit_agent_scaling.sh) runs the IDENTICAL
five-chambers environment (Ch4 pinned to 3 zombies via --ch4-mob-count; only
Ch3 grows one cell per agent; milestone reward values unchanged) with
N ∈ {2,3,4,5,6,9} Gemma agents for 500-step episodes. Team size therefore
traces out a compute axis — more agents means proportionally more LLM
forward passes over the same task — giving the classic scaling-law plot:

  x — total inference FLOPs per episode (log scale), Kaplan-style forward
      cost C ≈ 2 · N_eff · tokens (Kaplan et al. 2020, arXiv:2001.08361),
      computed by scripts/compute_flops.py from the run logs: decode tokens
      are exact; prefill is exact on runs with "[LocalModel usage]" lines
      and char-estimated on older runs. N_eff defaults to Gemma-4-E4B's
      4.5e9 active parameters.
  y — distinct cooperative milestones per episode: team-level union over
      agents, tracks ch2_anvils/ch3_switches/ch4_combat/ch5_boss
      (make_results.COOP_TRACKS). The maximum (COOP_MAX) is the same for
      every team size because the environment is pinned across N — the
      shared-axis requirement that inference-time multi-agent scaling work
      calls out (cf. the collaborative scaling law of Qian et al., ICLR
      2025, arXiv:2406.07155; Ringelmann-effect team-size scaling,
      arXiv:2606.02646).

Each plotted point is one team size: mean ± sd over seeds of the per-episode
values. Base and +Hebbian arms are separate series (house colors, matching
fig:milestone_timeline).

Usage:
    python make_scaling_fig.py                # runs/agent_scaling → paper_assets_scaling/
    python make_scaling_fig.py --runs-root runs/agent_scaling \
        --out paper_assets_scaling --n-eff 4.5e9
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_results import (  # noqa: E402  (repo-root module, same dir)
    COOP_MAX,
    coop_count,
    episode_milestone_sets,
    load_runs,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import compute_flops  # noqa: E402

TEAM_SIZES = [2, 3, 4, 5, 6, 9]
ARMS = [  # (arm key, dir pattern, label, color, marker, linestyle)
    ("base", "scale_gemma_base_n{n}", "baseline", "#2c7fb8", "o", "-"),
    ("hebbian", "scale_gemma_hebbian_n{n}", "+Hebbian", "#d95f0e", "D", "--"),
]


def flops_for_run(run_path: Path, flops_args) -> dict | None:
    """FLOPs row for one seed dir (needs log.txt + llm_logs/), else None."""
    return compute_flops.analyze_run(run_path, flops_args)


def collect_points(runs_root: Path, flops_args, exclude=frozenset()):
    """One row per (arm, N): per-episode coop milestones and FLOPs across seeds."""
    rows = []
    for arm, pattern, label, color, marker, ls in ARMS:
        for n in TEAM_SIZES:
            dir_name = pattern.format(n=n)
            runs = load_runs(runs_root, dir_name, exclude=exclude)
            if not runs:
                continue
            coop_means, flops_means, seeds = [], [], []
            for run in runs:
                ep_sets = episode_milestone_sets(run)
                if not ep_sets:
                    continue
                coop_means.append(
                    sum(coop_count(s) for s in ep_sets) / len(ep_sets))
                run_dir = Path(run["_path"]).parent
                fl = flops_for_run(run_dir, flops_args)
                if fl is None:
                    print(f"  [warn] {run_dir}: no log.txt/llm_logs — "
                          f"skipping this seed's compute (point uses the "
                          f"remaining seeds)", file=sys.stderr)
                else:
                    n_eps = max(1, len(run.get("_ep_bounds", [])) or 1)
                    flops_means.append(fl["flops"] / n_eps)
                seeds.append(run_dir.name)
            if not coop_means or not flops_means:
                print(f"  [warn] {dir_name}: incomplete (coop={len(coop_means)} "
                      f"flops={len(flops_means)}) — skipped", file=sys.stderr)
                continue
            rows.append({
                "arm": arm, "label": label, "color": color,
                "marker": marker, "ls": ls, "n_agents": n,
                "n_seeds": len(coop_means), "seeds": ";".join(seeds),
                "coop_mean": statistics.mean(coop_means),
                "coop_sd": (statistics.stdev(coop_means)
                            if len(coop_means) > 1 else 0.0),
                "flops_mean": statistics.mean(flops_means),
                "flops_sd": (statistics.stdev(flops_means)
                             if len(flops_means) > 1 else 0.0),
            })
    return rows


def fig_scaling(rows, out_dir: Path):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    arms_present = [a for a in ARMS if any(r["arm"] == a[0] for r in rows)]
    for arm, _, label, color, marker, ls in arms_present:
        pts = sorted((r for r in rows if r["arm"] == arm),
                     key=lambda r: r["n_agents"])
        x = [r["flops_mean"] for r in pts]
        y = [r["coop_mean"] for r in pts]
        yerr = [r["coop_sd"] for r in pts]
        ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, ls=ls,
                    lw=1.8, ms=6, capsize=2.5, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=label, zorder=3)
        for r in pts:
            ax.annotate(f"N={r['n_agents']}",
                        (r["flops_mean"], r["coop_mean"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color="#666666")

    ax.axhline(COOP_MAX, color="#c9c9c9", ls=":", lw=0.8, zorder=1)
    ax.text(0.99, COOP_MAX, f"max = {COOP_MAX}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5, color="#888888")

    ax.set_xscale("log")
    ax.set_xlabel(r"Inference compute per episode (FLOPs, $2\,N_{\rm eff}\,D$)")
    ax.set_ylabel("Cooperative milestones / episode")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", color="#ececec", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if len(arms_present) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"compute_scaling.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / 'compute_scaling.pdf'} (+.png)")


def write_csv(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compute_scaling.csv"
    fields = ["arm", "n_agents", "n_seeds", "seeds", "coop_mean", "coop_sd",
              "flops_mean", "flops_sd"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["arm"], r["n_agents"])):
            w.writerow(r)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=Path("runs/agent_scaling"))
    ap.add_argument("--out", type=Path, default=Path("paper_assets_scaling"))
    ap.add_argument("--n-eff", type=float, default=4.5e9,
                    help="active params for FLOPs = 2*N*tokens "
                         "[default 4.5e9, Gemma-4-E4B]")
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="EXP/seed_N", help="drop one run (repeatable)")
    args = ap.parse_args()

    flops_args = SimpleNamespace(
        n_eff=args.n_eff, image_tokens=args.image_tokens,
        overhead_tokens=args.overhead_tokens,
        chars_per_token=args.chars_per_token)

    rows = collect_points(args.runs_root, flops_args,
                          exclude=frozenset(args.exclude))
    if not rows:
        sys.exit(f"no scale_gemma_* runs with final_metrics.json under "
                 f"{args.runs_root}")
    write_csv(rows, args.out)
    fig_scaling(rows, args.out)


if __name__ == "__main__":
    main()
