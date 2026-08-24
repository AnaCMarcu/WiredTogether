"""Pareto figures: performance vs. social-module (and total) inference compute.

The social-interval sweep (submit_pareto_social.sh) reruns the Gemma-4-E4B
hebbian configuration varying ONLY the social module's deliberation interval,
interval ∈ {2, 8, 20, 50, 100}, plus the no-module baseline. This script
renders THREE performance metrics on TWO compute axes (6 figures + overview):

Metrics (y), each mean ± sd over seeds of per-episode values:
  - task milestones, % of the 25 physical-task milestones (all tracks except
    make_results.SOCIAL_ACT_TRACKS — comm/obs/imit milestones are pay for
    social acts, not task progress, and are excluded from comparable counts);
  - cooperative milestones, % of COOP_MAX = 17
    (ch2_anvils/ch3_switches/ch4_combat/ch5_boss, team union over agents);
  - team task return per episode (make_results.episode_task_returns:
    decomposed task + comm streams, hebbian_diffuse excluded).

Axes (x):
  - social-module FLOPs per episode (log scale) — the swept knob's own cost,
    Kaplan-style 2 · N_eff · tokens over llm_logs/social_module.log calls
    only. Prefill AND decode are char-estimated with the per-run chars/token
    calibration from scripts/compute_flops.py — deliberately for every run,
    even those whose log.txt carries exact "[LocalModel usage]" totals,
    because usage lines are not attributable per module; one method
    everywhere keeps anchors (2026-08-04, no usage lines) and sweep runs on
    a bias-consistent axis. The no-module baseline sits at exactly x = 0,
    which a log axis cannot show — it is drawn as a horizontal reference
    band (mean ± sd) instead of a point.
  - total-run FLOPs per episode (all modules, linear axis, exact prefill
    where usage lines exist). Companion for honesty, with its caveats
    visible by construction: the sweep spans only ~1.25x because the
    every-step action module dominates (~95% of compute), so episode-length
    variance (x error bars) swamps the interval's compute delta for every
    arm but Δ=2 — the Δ ordering scrambles along x — and the anchors'
    char-estimated prefill shifts them left ~10-15% relative to the sweep.

Seeds without final_metrics.json (still running / died) are skipped
automatically; points annotate their n when below 3.

Usage:
    python make_pareto_social_fig.py            # runs_from_daic/* → paper_assets_pareto_social/
    python make_pareto_social_fig.py --anchors-root runs_from_daic/new_exp_0_gemma \
        --sweep-root runs_from_daic/pareto_social --out paper_assets_pareto_social
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
    MILESTONE_TRACK,
    SOCIAL_ACT_TRACKS,
    coop_count,
    episode_milestone_sets,
    episode_task_returns,
    load_runs,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import compute_flops  # noqa: E402

# (interval, root key, exp dir) — interval None = no social module (x = 0).
POINTS = [
    (None, "anchors", "new_exp_0_gemma_base"),
    (2,    "sweep",   "new_exp_0_gemma_si2"),
    (8,    "anchors", "new_exp_0_gemma_hebbian"),
    (20,   "sweep",   "new_exp_0_gemma_si20"),
    (50,   "sweep",   "new_exp_0_gemma_si50"),
    (100,  "sweep",   "new_exp_0_gemma_si100"),
]
SERIES_COLOR = "#d95f0e"   # hebbian house color (matches make_scaling_fig)
BASE_COLOR = "#2c7fb8"     # baseline house color
TASK_MAX = sum(1 for v in MILESTONE_TRACK.values()
               if v not in SOCIAL_ACT_TRACKS)  # 25 physical-task milestones

# (row key, y label, filename stem)
METRICS = [
    ("total_pct", f"Task milestones (% of {TASK_MAX})", "milestones_pct"),
    ("coop_pct", f"Cooperative milestones (% of {COOP_MAX})", "coop_pct"),
    ("task_return", "Team task return / episode", "return"),
]
XSCALE_TOTAL = 1e17  # keep matplotlib's offset text out of the xlabel


def task_milestone_count(ms_set) -> int:
    return sum(1 for m in ms_set
               if MILESTONE_TRACK.get(m) is not None
               and MILESTONE_TRACK[m] not in SOCIAL_ACT_TRACKS)


def social_flops_for_run(run_dir: Path, flops_args) -> float | None:
    """Social-module-only FLOPs for one seed dir; 0.0 if the module never ran.

    Char-based on purpose for every run — see module docstring. Calibration
    (chars/token) uses ALL modules' response chars against the exact decode
    total, exactly like compute_flops's estimate path.
    """
    log_txt = run_dir / "log.txt"
    llm_logs = sorted((run_dir / "llm_logs").glob("*.log"))
    if not log_txt.is_file() or not llm_logs:
        return None
    decode_total, _, _, _ = compute_flops.parse_log_txt(log_txt)
    modules = {p.stem: compute_flops.parse_llm_log(p) for p in llm_logs}

    if flops_args.chars_per_token:
        cpt = flops_args.chars_per_token
    else:
        response_chars = sum(m["response_chars"] for m in modules.values())
        cpt = response_chars / decode_total \
            if decode_total > 0 and response_chars > 0 else 4.0
        if not (2.0 <= cpt <= 8.0):
            print(f"  !! {run_dir}: calibrated chars/token={cpt:.2f} out of "
                  f"range, falling back to 4.0", file=sys.stderr)
            cpt = 4.0

    soc = modules.get("social_module")
    if soc is None or soc["calls"] == 0:
        return 0.0
    tokens = ((soc["sys_chars"] + soc["user_chars"] + soc["response_chars"])
              / cpt
              + soc["calls"] * flops_args.overhead_tokens
              + soc["frames"] * flops_args.image_tokens)
    return 2.0 * flops_args.n_eff * tokens


def _mean_sd(vals):
    return (statistics.mean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0)


def collect_points(anchors_root: Path, sweep_root: Path, flops_args,
                   exclude=frozenset()):
    """One row per interval: per-episode metrics and FLOPs across seeds."""
    roots = {"anchors": anchors_root, "sweep": sweep_root}
    rows = []
    for interval, root_key, dir_name in POINTS:
        runs = load_runs(roots[root_key], dir_name, exclude=exclude)
        if not runs:
            print(f"  [warn] {dir_name}: no finished seeds — skipped",
                  file=sys.stderr)
            continue
        per_seed = {k: [] for k, _, _ in METRICS}
        soc_means, tot_means, seeds = [], [], []
        for run in runs:
            ep_sets = episode_milestone_sets(run)
            if not ep_sets:
                continue
            run_dir = Path(run["_path"]).parent
            n_eps = len(ep_sets)
            soc = social_flops_for_run(run_dir, flops_args)
            if soc is None:
                print(f"  [warn] {run_dir}: no log.txt/llm_logs — skipping "
                      f"this seed (compute axis would be missing)",
                      file=sys.stderr)
                continue
            returns, used_dec = episode_task_returns(run)
            if not used_dec:
                print(f"  [warn] {run_dir}: no reward decomposition — "
                      f"task return falls back to logged totals "
                      f"(includes hebbian_diffuse)", file=sys.stderr)
            per_seed["total_pct"].append(
                100.0 * sum(task_milestone_count(s) for s in ep_sets)
                / (n_eps * TASK_MAX))
            per_seed["coop_pct"].append(
                100.0 * sum(coop_count(s) for s in ep_sets)
                / (n_eps * COOP_MAX))
            per_seed["task_return"].append(
                sum(returns) / len(returns) if returns else 0.0)
            soc_means.append(soc / n_eps)
            tot = compute_flops.analyze_run(run_dir, flops_args)
            tot_means.append(tot["flops"] / n_eps if tot else float("nan"))
            seeds.append(run_dir.name)
        if not seeds:
            print(f"  [warn] {dir_name}: no usable seeds — skipped",
                  file=sys.stderr)
            continue
        row = {"interval": interval, "exp": dir_name,
               "n_seeds": len(seeds), "seeds": ";".join(seeds)}
        for key, _, _ in METRICS:
            row[f"{key}_mean"], row[f"{key}_sd"] = _mean_sd(per_seed[key])
        row["social_flops_mean"], row["social_flops_sd"] = _mean_sd(soc_means)
        row["total_flops_mean"], row["total_flops_sd"] = _mean_sd(tot_means)
        rows.append(row)
    return rows


def draw_panel(ax, rows, ykey, ylabel, xmode, legend=False):
    """One metric on one axis. xmode: 'social' (log, base=band) or 'total'
    (linear /1e17, base=point with x error bars)."""
    base = next((r for r in rows if r["interval"] is None), None)
    pts = sorted((r for r in rows if r["interval"] is not None),
                 key=lambda r: r[f"{xmode}_flops_mean"])
    ym, ysd = f"{ykey}_mean", f"{ykey}_sd"

    if xmode == "social":
        if base is not None:
            ax.axhspan(base[ym] - base[ysd], base[ym] + base[ysd],
                       color=BASE_COLOR, alpha=0.12, zorder=1)
            ax.axhline(base[ym], color=BASE_COLOR, ls="--", lw=1.4, zorder=2,
                       label=f"no social module (n={base['n_seeds']})")
        if pts:
            ax.errorbar([r["social_flops_mean"] for r in pts],
                        [r[ym] for r in pts], yerr=[r[ysd] for r in pts],
                        color=SERIES_COLOR, marker="D", ls="-", lw=1.8, ms=6,
                        capsize=2.5, elinewidth=0.9,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label="+Hebbian, interval swept", zorder=3)
            for r in pts:
                tag = f"Δ={r['interval']}"
                if r["n_seeds"] < 3:
                    tag += f" (n={r['n_seeds']})"
                ax.annotate(tag, (r["social_flops_mean"], r[ym]),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7.5, color="#666666")
        ax.set_xscale("log")
        ax.set_xlabel(r"Social-module compute per episode"
                      "\n" r"(FLOPs, $2\,N_{\rm eff}\,D$)")
    else:
        if base is not None:
            ax.errorbar([base["total_flops_mean"] / XSCALE_TOTAL], [base[ym]],
                        xerr=[base["total_flops_sd"] / XSCALE_TOTAL],
                        yerr=[base[ysd]], color=BASE_COLOR, marker="o",
                        ls="none", ms=6, capsize=2.5, elinewidth=0.9,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=f"no social module (n={base['n_seeds']})",
                        zorder=3)
            ax.annotate("none",
                        (base["total_flops_mean"] / XSCALE_TOTAL, base[ym]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color="#666666")
        if pts:
            ax.errorbar([r["total_flops_mean"] / XSCALE_TOTAL for r in pts],
                        [r[ym] for r in pts],
                        xerr=[r["total_flops_sd"] / XSCALE_TOTAL for r in pts],
                        yerr=[r[ysd] for r in pts],
                        color=SERIES_COLOR, marker="D", ls="-", lw=1.8, ms=6,
                        capsize=2.5, elinewidth=0.9,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label="+Hebbian, interval swept", zorder=3)
            for i, r in enumerate(pts):  # alternate to avoid label pileups
                tag = f"Δ={r['interval']}"
                if r["n_seeds"] < 3:
                    tag += f" (n={r['n_seeds']})"
                ax.annotate(tag,
                            (r["total_flops_mean"] / XSCALE_TOTAL, r[ym]),
                            textcoords="offset points",
                            xytext=(0, 7 if i % 2 == 0 else -15),
                            ha="center", fontsize=7.5, color="#666666")
        ax.set_xlabel(r"Total compute per episode"
                      "\n" r"($\times 10^{17}$ FLOPs, all modules)")

    ymax = max(r[f"{ykey}_mean"] + r[f"{ykey}_sd"] for r in rows)
    ax.set_ylim(0, ymax * 1.35)
    ax.margins(x=0.09)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#ececec", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="upper left")


def write_figures(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ykey, ylabel, stem in METRICS:
        for xmode in ("social", "total"):
            fig, ax = plt.subplots(figsize=(4.6, 3.2))
            draw_panel(ax, rows, ykey, ylabel, xmode, legend=True)
            fig.tight_layout()
            name = f"pareto_social_{stem}" + \
                   ("" if xmode == "social" else "_totalflops")
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"{name}.{ext}", dpi=200)
            plt.close(fig)
            print(f"wrote {out_dir / (name + '.pdf')} (+.png)")

    # contact sheet: metrics as columns, axes as rows
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6))
    for col, (ykey, ylabel, _) in enumerate(METRICS):
        for row_i, xmode in enumerate(("social", "total")):
            draw_panel(axes[row_i][col], rows, ykey, ylabel, xmode,
                       legend=(row_i == 0 and col == 0))
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_social_overview.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / 'pareto_social_overview.png'}")


def write_csv(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pareto_social.csv"
    fields = (["interval", "exp", "n_seeds", "seeds"]
              + [f"{k}_{s}" for k, _, _ in METRICS for s in ("mean", "sd")]
              + ["social_flops_mean", "social_flops_sd",
                 "total_flops_mean", "total_flops_sd"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["interval"] is None,
                                             r["interval"] or 0)):
            w.writerow(r)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anchors-root", type=Path,
                    default=Path("runs_from_daic/new_exp_0_gemma"))
    ap.add_argument("--sweep-root", type=Path,
                    default=Path("runs_from_daic/pareto_social"))
    ap.add_argument("--out", type=Path, default=Path("paper_assets_pareto_social"))
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

    rows = collect_points(args.anchors_root, args.sweep_root, flops_args,
                          exclude=frozenset(args.exclude))
    if not rows:
        sys.exit("no runs with final_metrics.json under the given roots")
    write_csv(rows, args.out)
    write_figures(rows, args.out)


if __name__ == "__main__":
    main()
