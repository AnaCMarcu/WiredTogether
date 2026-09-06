"""Pareto figures: task performance vs. inference compute, by team size.

The agent-count scaling suite (submit_agent_scaling.sh) runs the IDENTICAL
WIRE environment (Ch4 pinned to 3 zombies via --ch4-mob-count; only
Ch3 grows one cell per agent; milestone reward values unchanged) with
N ∈ {2,3,4,5,6,9} Gemma agents for 500-step episodes. Team size therefore
traces out a compute axis — more agents means proportionally more LLM
forward passes over the same task — giving the classic scaling-law plot.

  x — total inference FLOPs per episode (log scale), Kaplan-style forward
      cost C ≈ 2 · N_eff · tokens (Kaplan et al. 2020, arXiv:2001.08361),
      computed by analysis/compute_flops.py from the run logs: decode tokens
      are exact; prefill is exact on runs carrying "[LocalModel usage]"
      lines and char-estimated otherwise. N_eff defaults to Gemma-4-E4B's
      4.5e9 active parameters.
  y — one figure per PAPER performance metric. These are not recomputed
      here: make_results.aggregate() is called directly and the same
      denominators are applied, so every point is the quantity
      tab:final_comparison reports (the compiled paper's main table, built
      by make_final_table.py) —

        Coop. %      coop     / COOP_MAX    — 17 Ch2-Ch5 milestones
                     (ch2_anvils, ch3_switches, ch4_combat, ch5_boss),
                     team union over agents
        Milest. %    allms_nc / NONCOMM_MAX — 25 non-social milestones;
                     the comm/obs/imit tracks are stripped because only
                     reward-flagged arms can earn them
        Task return  reward-weighted TEAM-UNION completion, % of the
                     1880-point ceiling (episode_union_rewards below).
                     NOT the paper's team-summed return: that credits every
                     agent separately (once=true PER AGENT), so its ceiling
                     is N x the per-agent max and cannot compare team
                     sizes. The team sum and its per-agent mean are still
                     written to the CSV.
        Switch       % of episodes in which the Ch3 switch puzzle was
                     solved — the mechanism panel (episode_switch_solved).

      The paper reports milestones as PERCENTAGE COMPLETION, never a
      count, and aggregate() POOLS every episode of every seed reporting
      mean ± population sd — not a mean over seed means, which would give
      different error bars.

Because the environment is pinned across N, every plotted ceiling is the
same for every team size — the shared-axis requirement that inference-time
multi-agent scaling work calls out (cf. the collaborative scaling law of
Qian et al., ICLR 2025, arXiv:2406.07155; Ringelmann-effect team-size
scaling, arXiv:2606.02646). Each headline panel also carries a per-arm
logistic-in-log-compute trend line (the collaborative-scaling-law form)
drawn under the points; --no-fit omits it.

Usage:
    python analysis/make_scaling_fig.py                # runs/agent_scaling → paper_assets/scaling/
    python analysis/make_scaling_fig.py --runs-root runs_from_daic/agent_scaling \
        --out paper_assets/scaling --n-eff 4.5e9
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from paths import ASSETS, REPO  # noqa: E402  (also puts siblings on sys.path)

from make_results import (  # noqa: E402  (repo-root module, same dir)
    COOP_MAX,
    MILESTONE_TRACK,
    NONCOMM_MAX,
    SOCIAL_ACT_TRACKS,
    aggregate,
    episode_milestone_sets,
    load_runs,
    mean_std,
)

import compute_flops  # noqa: E402

# The milestone reward table lives in the metric module (mirrors
# milestones.lua). It uses the runtime import path (agent_modules.*), hence
# src/mindforge on sys.path rather than src.
sys.path.insert(0, str(REPO / "src" / "mindforge"))
from mindforge.agent_modules.craftium_metric import TRACKS  # noqa: E402


MILESTONE_REWARD = {mid: r for items in TRACKS.values() for mid, r in items}
# Team-union, reward-weighted ceiling over the 25 non-social milestones.
# Identical for every team size: 360 (Ch1) + 160 + 220 + 340 + 800 = 1880.
UNION_REWARD_MAX = sum(
    MILESTONE_REWARD[m] for m, t in MILESTONE_TRACK.items()
    if t not in SOCIAL_ACT_TRACKS)


def episode_union_rewards(run) -> list[float]:
    """Reward-weighted team milestone completion per episode.

    Sum of each milestone's reward value over the DISTINCT milestones the
    team reached (union over agents), social-act tracks excluded — the
    reward-weighted analogue of the paper's "Milest. %". The team-summed
    task return credits every agent separately for the same milestone
    (once=true PER AGENT), so its ceiling is N × the per-agent maximum and it
    cannot compare team sizes; this metric credits a milestone once per
    episode regardless of how many agents earned it, so a team of 3 and a
    team of 9 face the same UNION_REWARD_MAX. Uses the honesty-filtered
    per-episode sets from load_runs, like every other paper number.
    """
    return [
        sum(MILESTONE_REWARD.get(m, 0.0) for m in s
            if MILESTONE_TRACK.get(m) not in SOCIAL_ACT_TRACKS)
        for s in episode_milestone_sets(run)
    ]


SWITCH_MILESTONE = "m17_switch_pressed"


def episode_switch_solved(run) -> list[float]:
    """1.0 for each episode in which the Ch3 switch puzzle was solved
    (any agent pressed its switch, freeing a teammate), else 0.0.

    This single event is the mechanism behind the scaling curves: across the
    whole suite the anvils broke twice and Ch4 was never cleared, so the
    team-level milestone metrics are essentially "was a switch pressed" plus
    first-mob-kills. Pressing one's own switch is a SOLO act with a
    cooperative consequence, and every agent has its own cell — so the chance
    that at least one agent presses grows with N like coverage / pass@N,
    which is what the N>=5 step is.
    """
    return [1.0 if SWITCH_MILESTONE in s else 0.0
            for s in episode_milestone_sets(run)]


def _logistic_logx(x, L, k, x0):
    """Logistic in log-compute, drawn on the linear axis: L / (1 + (x0/x)^k).

    The collaborative-scaling-law form (Qian et al., ICLR 2025): performance
    follows logistic growth in log(#agents), and here compute ∝ N.
    """
    return L / (1.0 + (x0 / x) ** k)


def fit_logistic(xs, ys):
    """Least-squares logistic fit through one arm's points; None on failure.

    Descriptive only: three parameters on six points. L is bounded to
    [max(y), 100] (a completion percentage cannot saturate below the best
    observed point or above 100), k > 0 keeps it monotone, x0 inside the
    observed compute range.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 4 or ys.max() <= 0:
        return None
    lo, hi = xs.min(), xs.max()
    # L's window must stay open even when the best point is already 100%
    # (the switch panel's baseline at N=5): a lower bound equal to the upper
    # bound makes curve_fit reject the problem as infeasible.
    L_lo = min(float(ys.max()), 99.0)
    try:
        params, _ = curve_fit(
            _logistic_logx, xs, ys,
            p0=[min(100.0, max(L_lo, ys.max() * 1.2)), 2.0,
                float(np.median(xs))],
            bounds=([L_lo, 0.05, lo * 0.5], [100.0, 20.0, hi * 2.0]),
            maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    return lambda x: _logistic_logx(x, *params)

TEAM_SIZES = [2, 3, 4, 5, 6, 7, 9]
ARMS = [  # (arm key, dir pattern, label, color, marker, linestyle)
    ("base", "scale_gemma_base_n{n}", "baseline", "#2c7fb8", "o", "-"),
    ("hebbian", "scale_gemma_hebbian_n{n}", "+Hebbian", "#d95f0e", "D", "--"),
]

# (aggregate() key, denominator or None, output stem, y-axis label).
#
# These mirror tab:final_comparison in the COMPILED paper (built by
# make_final_table.py), which reports milestones as PERCENTAGE COMPLETION,
# never a count:
#     Milest. % = allms_nc / NONCOMM_MAX   (25 non-communication milestones)
#     Coop.   % = coop     / COOP_MAX      (17 Ch2-Ch5 milestones)
#     Task return = team-summed undiffused reward (include_comm=True)
# Note allms_nc, NOT allms: the paper's milestone percentage strips the
# communication/observation/imitation tracks, which only reward-flagged arms
# can earn.
#
# Denominators come from make_results (17 / 25). make_final_table.py computes
# its own NONCOMM_MAX and currently gets 35, because it filters only the
# "communication" track while MILESTONE_TRACK has since gained the obs/imit
# tracks; the published table's numbers correspond to 25.
#
# The paper's "Task return" is the team SUM, whose ceiling grows with N (each
# milestone is once=true PER AGENT), so it cannot serve as a Pareto axis
# across team sizes. The return panel therefore plots the reward-weighted
# team-union completion instead (episode_union_rewards above): the reward
# the team could collect is the same 1880 points at N=2 and N=9. The raw
# team sum and its per-agent mean are still written to the CSV.
# (aggregate() key, denominator or None, output stem, y label, draw fit?)
# denom None  -> plotted in raw units, y-axis not clamped to 0-100.
METRICS = [
    ("coop", COOP_MAX, "pareto_coop",
     "Cooperative milestones (% of 17)", True),
    ("allms_nc", NONCOMM_MAX, "pareto_milestones",
     "Milestones (% of 25)", True),
    ("ureturn", UNION_REWARD_MAX, "pareto_return",
     "Task return, team-union (% of 1880)", True),
    # The mechanism panel: fraction of episodes (0/1 indicator, so the
    # "denominator" 1.0 just turns the pooled mean into a percentage).
    ("switch", 1.0, "pareto_switch",
     "Episodes with Ch3 switch solved (%)", True),
    # The paper's own "Task return": team-summed undiffused reward, in raw
    # points. Its ceiling is N x the per-agent maximum (every milestone is
    # once=true PER AGENT), so the rise with compute is partly mechanical
    # and the arms are only comparable WITHIN a team size — pareto_return
    # above is the N-invariant version. Plotted without a trend line: a
    # logistic fit would assert a saturating law over an axis whose ceiling
    # is itself moving.
    ("task", None, "pareto_task_return",
     "Task return / episode (team sum)", False),
]


def parse_alias(spec: str):
    """``N=ROOT,PREFIX`` → (N, Path(ROOT), PREFIX).

    Lets a team size be supplied from a run group that does not follow the
    sweep's naming, e.g. reusing the 3-agent 1000-step runs as the N=3
    point:  ``--alias 3=runs_from_daic/new_exp_0_gemma,new_exp_0_gemma``
    resolves to ``<root>/new_exp_0_gemma_{base,hebbian}/seed_*``.

    '=' and ',' rather than ':' so Windows drive letters survive.
    """
    n_str, rest = spec.split("=", 1)
    root_str, prefix = rest.rsplit(",", 1)
    return int(n_str), Path(root_str.strip()), prefix.strip()


def collect_points(runs_root: Path, flops_args, exclude=frozenset(),
                   aliases=None):
    """One row per (arm, N): the paper's metrics plus per-episode FLOPs.

    ``aliases`` maps N → (root, prefix) for points sourced from another run
    group; those rows are tagged source="alias" so the caveat survives into
    the CSV and the figure label.
    """
    aliases = aliases or {}
    rows = []
    for arm, pattern, label, color, marker, ls in ARMS:
        for n in TEAM_SIZES:
            if n in aliases:
                a_root, a_prefix = aliases[n]
                root, dir_name, source = a_root, f"{a_prefix}_{arm}", "alias"
            else:
                root, dir_name, source = runs_root, pattern.format(n=n), "sweep"
            runs = load_runs(root, dir_name, exclude=exclude)
            if not runs:
                continue

            # Paper-exact performance numbers (pools episodes, population sd).
            agg = aggregate(runs)
            # N-invariant return: pooled over every episode of every seed,
            # same convention as aggregate().
            agg["ureturn"] = mean_std(
                [v for run in runs for v in episode_union_rewards(run)])
            agg["switch"] = mean_std(
                [v for run in runs for v in episode_switch_solved(run)])

            # FLOPs per EPISODE, averaged over seeds. compute_flops totals a
            # whole run, so divide by that run's episode count to match the
            # per-episode performance metrics.
            flops, seeds = [], []
            for run in runs:
                run_dir = Path(run["_path"]).parent
                seeds.append(run_dir.name)
                fl = compute_flops.analyze_run(run_dir, flops_args)
                if fl is None:
                    print(f"  [warn] {run_dir}: no log.txt/llm_logs — this "
                          f"seed contributes no compute (point uses the "
                          f"remaining seeds)", file=sys.stderr)
                    continue
                n_eps = max(1, len(run.get("_ep_bounds", [])) or 1)
                flops.append(fl["flops"] / n_eps)
            if not flops:
                print(f"  [warn] {dir_name}: no FLOPs data — skipped",
                      file=sys.stderr)
                continue

            row = {
                "arm": arm, "label": label, "color": color,
                "marker": marker, "ls": ls, "n_agents": n,
                "n_seeds": len(runs), "n_episodes": agg["n_eps"],
                "seeds": ";".join(seeds), "source": source,
                "exp_dir": f"{root}/{dir_name}",
            }
            for key, denom, _stem, _ylabel, _fit in METRICS:
                m, s = agg[key]
                # Percentage completion when the metric has a ceiling; the
                # sd is scaled by the same denominator, exactly as
                # make_final_table.pct() does.
                if denom:
                    m, s = 100.0 * m / denom, 100.0 * s / denom
                row[f"{key}_mean"], row[f"{key}_sd"] = m, s
            # Reference columns (CSV only): raw union reward in points, the
            # paper's team-summed task return, and that sum per agent.
            row["ureturn_pts_mean"], row["ureturn_pts_sd"] = agg["ureturn"]
            row["task_pa_mean"] = agg["task"][0] / n
            row["task_pa_sd"] = agg["task"][1] / n
            row["flops_mean"], row["flops_sd"] = mean_std(flops)
            rows.append(row)
    return rows


XSCALE = 1e17  # linear compute axis in units of 1e17 FLOPs (no offset text)


def _free_corner(ax, xs, ys, frac_w=0.36, frac_h=0.34):
    """Legend corner whose box holds no marker or N-label.

    matplotlib's loc="best" only avoids artists, not annotations, so it
    happily parks the legend on a point's label (it did, on the coop
    panel's N=9). Test the four corners in preference order against every
    point and the label slot just above/below it (~7% of the y-range);
    fall back to "best" if all four are occupied.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = (x1 - x0) * frac_w, (y1 - y0) * frac_h
    dy = 0.07 * (y1 - y0)
    occupied = [(x, y + s) for x, y in zip(xs, ys) for s in (-dy, 0.0, dy)]
    corners = [
        ("lower right", x1 - w, y0),
        ("upper right", x1 - w, y1 - h),
        ("upper left", x0, y1 - h),
        ("lower left", x0, y0),
    ]
    for name, bx, by in corners:
        if not any(bx <= x <= bx + w and by <= y <= by + h
                   for x, y in occupied):
            return name
    return "best"


def fig_pareto_paper(rows, key: str, stem: str, ylabel: str, out_dir: Path,
                     fit: bool = True, pct: bool = True):
    """Headline Pareto frontier, styled after make_pareto_social_fig's
    fig_pareto_paper so the scaling and social-interval figures read as one
    family: open markers, framed legend, LINEAR compute axis, no error bars
    (the shape of the frontier is the point; --diagnostics emits the
    error-bar panels for judging individual gaps).

    With ``fit`` (default) a logistic-in-log-compute curve is fitted per arm
    and drawn as a thin dashed line UNDER the points — the collaborative
    scaling-law form — so the trend is visible without hiding the scatter.
    Points and their joining line are always kept.

    Baseline = hollow blue squares, +Hebbian = hollow orange circles, each
    joined in order of compute. Every point is labelled with its team size.
    The two arms share (almost) the same x at each N, so for every team size
    the HIGHER of the two points takes its label above the marker and the
    lower one below — whichever arm that is — so labels never write over
    the other series' marker or text.
    """
    series = [  # (arm, legend label, marker, ms, mew)
        ("hebbian", "+Hebbian", "o", 7, 1.7),
        ("base", "baseline", "s", 8, 1.9),
    ]
    present = [s for s in series if any(r["arm"] == s[0] for r in rows)]
    if not present:
        return

    # Label side per (arm, N): above for the higher point of the pair.
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_agents"], {})[r["arm"]] = r[f"{key}_mean"]
    above = {}
    for n, arms_y in by_n.items():
        top = max(arms_y, key=arms_y.get)
        for arm in arms_y:
            above[(arm, n)] = (arm == top)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ys_all, xs_all = [], []
    for arm, label, marker, ms, mew in present:
        color = next(a[3] for a in ARMS if a[0] == arm)
        pts = sorted((r for r in rows if r["arm"] == arm),
                     key=lambda r: r["flops_mean"])
        xs = [r["flops_mean"] / XSCALE for r in pts]
        ys = [r[f"{key}_mean"] for r in pts]
        xs_all += xs
        ys_all += ys
        if fit:
            f = fit_logistic(xs, ys)
            if f is None:
                print(f"  [warn] {stem}/{arm}: logistic fit did not "
                      f"converge — no trend line", file=sys.stderr)
            else:
                xx = np.linspace(min(xs) * 0.92, max(xs) * 1.06, 200)
                ax.plot(xx, f(xx), color=color, ls="--", lw=1.1, alpha=0.55,
                        zorder=1)
        ax.plot(xs, ys, color=color, ls="-", lw=1.6, zorder=2)
        ax.plot(xs, ys, color=color, ls="none", marker=marker, ms=ms,
                mfc="none", mew=mew, label=label, zorder=3)
        for r, xv, yv in zip(pts, xs, ys):
            tag = f"N={r['n_agents']}"
            if r.get("source") == "alias":
                tag += "*"          # sourced from another run group
            if r["n_seeds"] < 3:
                tag += f" (n={r['n_seeds']})"
            up = above[(arm, r["n_agents"])]
            ax.annotate(tag, (xv, yv), textcoords="offset points",
                        xytext=(0, 9) if up else (0, -17),
                        ha="center", fontsize=8.5, color="#555555")

    lo, hi = min(ys_all), max(ys_all)
    pad = max(1.5, 0.25 * (hi - lo))
    # Generous headroom (the framed 3-row legend lives inside the axes), but
    # Percentage panels must never pad past the meaningful range — the
    # switch panel spans 0..100 and would otherwise show -20..140. Raw-unit
    # panels (pct=False, e.g. the team-summed task return) have no such
    # ceiling, so they only get the headroom.
    if pct:
        ax.set_ylim(max(lo - pad, -4.0), min(hi + 2.6 * pad, 106.0))
    else:
        ax.set_ylim(max(lo - pad, 0.0), hi + 2.6 * pad)
    ax.margins(x=0.10)
    ax.set_xlabel(r"Compute  ($\times 10^{17}$ FLOPs / episode, whole model)")
    ax.set_ylabel(ylabel)
    if fit:
        # One neutral legend entry for the trend lines (per-arm colour is
        # already carried by the markers).
        ax.plot([], [], color="#777777", ls="--", lw=1.1,
                label="logistic fit (log-compute)")
    ax.legend(loc=_free_corner(ax, xs_all, ys_all), fontsize=8.5,
              framealpha=0.95)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def fig_pareto(rows, key: str, denom, stem: str, ylabel: str, out_dir: Path):
    """Diagnostic variant (--diagnostics): same metric with pooled-episode
    error bars on a log compute axis, for judging whether a gap is real."""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    arms_present = [a for a in ARMS if any(r["arm"] == a[0] for r in rows)]
    for arm, _, label, color, marker, ls in arms_present:
        pts = sorted((r for r in rows if r["arm"] == arm),
                     key=lambda r: r["n_agents"])
        ax.errorbar([r["flops_mean"] for r in pts],
                    [r[f"{key}_mean"] for r in pts],
                    yerr=[r[f"{key}_sd"] for r in pts],
                    color=color, marker=marker, ls=ls, lw=1.8, ms=6,
                    capsize=2.5, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=label, zorder=3)
        for r in pts:
            ax.annotate(f"N={r['n_agents']}",
                        (r["flops_mean"], r[f"{key}_mean"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color="#666666")

    if denom:
        # Percentage metric: fix the axis to the full 0-100 range so the
        # low-progress regime is not visually inflated.
        ax.set_ylim(0, 100)

    ax.set_xscale("log")
    ax.set_xlabel(r"Inference compute per episode (FLOPs, $2\,N_{\rm eff}\,D$)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", color="#ececec", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if len(arms_present) > 1:
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def write_csv(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compute_scaling.csv"
    fields = ["arm", "n_agents", "n_seeds", "n_episodes", "seeds",
              "source", "exp_dir"]
    for key, _denom, _stem, _ylabel, _fit in METRICS:
        fields += [f"{key}_mean", f"{key}_sd"]
    fields += ["ureturn_pts_mean", "ureturn_pts_sd",
               "task_pa_mean", "task_pa_sd",
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
    ap.add_argument("--out", type=Path, default=ASSETS / "scaling")
    ap.add_argument("--n-eff", type=float, default=4.5e9,
                    help="active params for FLOPs = 2*N*tokens "
                         "[default 4.5e9, Gemma-4-E4B]")
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="EXP/seed_N", help="drop one run (repeatable)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also emit the error-bar / log-axis panels "
                         "(<stem>_errbars.png) for judging individual gaps")
    ap.add_argument("--no-fit", action="store_true",
                    help="omit the per-arm logistic trend lines")
    ap.add_argument("--alias", action="append", default=[],
                    metavar="N=ROOT,PREFIX",
                    help="source team size N from another run group, e.g. "
                         "3=runs_from_daic/new_exp_0_gemma,new_exp_0_gemma "
                         "(repeatable). Such points are marked N=x* on the "
                         "figure and source=alias in the CSV.")
    args = ap.parse_args()

    flops_args = SimpleNamespace(
        n_eff=args.n_eff, image_tokens=args.image_tokens,
        overhead_tokens=args.overhead_tokens,
        chars_per_token=args.chars_per_token)

    aliases = {}
    for spec in args.alias:
        n, root, prefix = parse_alias(spec)
        aliases[n] = (root, prefix)
        print(f"alias: N={n} <- {root}/{prefix}_{{base,hebbian}}")

    rows = collect_points(args.runs_root, flops_args,
                          exclude=frozenset(args.exclude), aliases=aliases)
    if not rows:
        sys.exit(f"no scale_gemma_* runs with final_metrics.json under "
                 f"{args.runs_root}")
    write_csv(rows, args.out)
    for key, denom, stem, ylabel, fit in METRICS:
        fig_pareto_paper(rows, key, stem, ylabel, args.out,
                         fit=fit and not args.no_fit,
                         pct=denom is not None)
        if args.diagnostics:
            fig_pareto(rows, key, denom, f"{stem}_errbars", ylabel, args.out)


if __name__ == "__main__":
    main()
