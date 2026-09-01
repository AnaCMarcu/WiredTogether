"""Pareto figures for the model-size sweep (compute vs performance).

Produces every combination of

    family  in {gemma, qwen, both}      default: gemma only (Qwen excluded
                                        from the paper's plots by decision)
    y-axis  in {reward, milestone_pct, coop_pct, completions, coop_completions}
    x-axis  in {flops, grounding, partner_loc}

= one PNG per combination plus one composite (rows = y, cols = x) per family,
and a CSV with every plotted number.

Definitions (all mirror make_results.py so the plots agree with the paper
tables — the helpers are IMPORTED from it, not re-implemented):

    reward         team task return per episode (make_results.episode_task_returns:
                   task + comm streams, hebbian_diffuse excluded, chamber-entry
                   honesty filter applied), mean over episodes, then over seeds
    milestone_pct  100 * |distinct team milestones outside the social-act
                   tracks| / NONCOMM_MAX (= 25), per episode -> mean
    coop_pct       100 * |distinct team milestones in Ch2-Ch5| / COOP_MAX
                   (= 17), per episode -> mean   ("Coop. milestones" in the paper)
    flops          inference compute 2*N_eff*(prefill+decode tokens), summed over
                   all agents/modules/retries (scripts/compute_flops.py); shown
                   in units of 1e17 on a LINEAR axis to match the reference
                   figure (the range here is ~5x, so log adds nothing)
    grounding      perception-grounding rate: fraction of the agents' own
                   perception statements naming no object impossible for the
                   agent's current chamber (tab:belief_quality; qualitative
                   pipeline, beliefs.csv, one value per run). Per ARM.
    partner_loc    partner-location accuracy: fraction of partner-location
                   claims matching the partner's true chamber (same source).
                   Both perception axes carry +-1 sd across seeds as x error
                   bars. (An external benchmark such as MMMU-Pro was used in
                   an earlier version; it is a backbone property, identical
                   for both arms, and is no longer plotted.)

Style: identical to make_pareto_social_fig's social-interval frontier (itself
after the compute-vs-reward frontiers in the inference-time-scaling
literature) -- OPEN markers in the house colours (base #2c7fb8 squares,
+Hebbian #d95f0e circles, shared with make_scaling_fig), solid connecting line
within an arm, framed legend lower-right, no grid, LINEAR compute axis, no
error bars (--errorbars adds them; the table carries the sd). Each size is
direct-labelled once. Families are NEVER joined to each other.

Default --ys is coop_pct,milestone_pct,reward, so a bare run emits the three
paper figures + the composite + the two-panel --paper variant.

Usage:
    python scripts/make_pareto_grid.py runs_from_daic/pareto_gemma4 \
        runs_from_daic/new_exp_0_gemma --out-dir paper_assets_pareto/grid --paper
    python scripts/make_pareto_grid.py ... --beliefs analysis_qualitative/out_pareto/tables/beliefs.csv
    python scripts/make_pareto_grid.py ... --families gemma --ys coop_pct --xs flops

FLOPs need a pass over each run's 50+ MB log.txt; results are cached in
<out-dir>/flops_cache.json keyed by run path + log mtime, so re-plotting is
instant.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))            # make_results.py lives at the root
sys.path.insert(0, str(REPO / "scripts"))

import make_results as MR                                  # noqa: E402
from compute_flops import analyze_run                      # noqa: E402
from make_pareto_fig import (  # noqa: E402
    SIZES, exp_to_size_arm, mean_sd, run_metrics,
)

FAMILIES = {
    "gemma": ["gemma"],
    "qwen":  ["qwen"],
    "both":  ["gemma", "qwen"],
}
# Axis labels are terse like the reference ("Compute", "(Normalized) Reward");
# the full definitions live in the caption / tab:final_comparison.
Y_METRICS = {
    "reward":           "Task return",
    # Same quantity AND same label as make_pareto_social_fig's frontier
    # (its total_pct = 100 * task_milestone_count / (n_eps * TASK_MAX=25)),
    # so the two Pareto figures in the paper are directly comparable.
    "milestone_pct":    "Task milestones (% of 25)",
    "coop_pct":         "Coop. milestone completion [%]",
    "completions":      "Milestone completions",
    "coop_completions": "Coop. milestone completions",
    # Per-agent framing. "Milestones per agent" is TASK milestones / N -- it is
    # NOT final_metrics.mean_milestone_count_per_agent, which is ~2/3
    # communication milestones and would contradict tab:final_comparison if
    # labelled "cooperative" (6.78/agent vs the table's 1.44 distinct for the
    # whole team).
    "completions_pa":      "Milestones per agent",
    "coop_completions_pa": "Coop. milestones per agent",
    # Perception as an OUTCOME (y-axis), from the qualitative pipeline's
    # beliefs.csv (one value per run; sd across seeds). grounding_strict is
    # the tightened rate: nothing impossible AND >=1 chamber-diagnostic
    # object named, so vague/empty statements no longer pass for free.
    "grounding":        "Perception grounding rate",
    "grounding_strict": "Strict perception grounding rate",
    "specificity":      "Chamber-diagnostic mention rate",
    "net_grounding":    "Net perception grounding (grounded − hallucinated)",
}
X_AXES = {
    "flops":            "Compute  [$10^{17}$ FLOPs]",
    "grounding":        "Perception grounding rate",
    "grounding_strict": "Strict perception grounding rate",
    "specificity":      "Chamber-diagnostic mention rate",
    "net_grounding":    "Net perception grounding (grounded − hallucinated)",
    "partner_loc":      "Partner-location accuracy",
}
# Perception axes come from the QUALITATIVE pipeline's beliefs.csv (one row
# per run): the paper's tab:belief_quality metrics, computed IN the
# environment from the agents' own belief statements. They are per-arm
# values (the Hebbian arm has its own grounding rate), unlike an external
# benchmark score, which would be a property of the backbone only.
BELIEF_COLS = {"grounding": "perception_grounding_rate",
               "grounding_strict": "perception_grounding_strict",
               "specificity": "perception_specificity",
               "net_grounding": "perception_net_grounding",
               "partner_loc": "partner_loc_accuracy"}
# Aesthetics: identical to make_pareto_social_fig.fig_pareto_paper (the
# social-interval frontier), which is itself styled after the compute-vs-reward
# frontiers in the inference-time-scaling literature -- OPEN markers, house
# colours, solid connecting line, framed legend lower-right, no grid, no error
# bars, linear compute axis. House colours are shared with make_scaling_fig so
# every compute figure in the paper reads as one family.
BASE_COLOR = "#2c7fb8"     # baseline house colour
SERIES_COLOR = "#d95f0e"   # hebbian house colour
ARM_STYLE = {
    "base":    dict(color=BASE_COLOR,   marker="s", ms=8, mew=1.9,
                    label="Base"),
    "hebbian": dict(color=SERIES_COLOR, marker="o", ms=7, mew=1.7,
                    label="+Hebbian"),
}
# Families: Gemma open markers (the house look); Qwen filled, only relevant
# when --families is used to bring Qwen back.
FAMILY_OPEN = {"gemma": True, "qwen": False}
FAMILY_LABEL = {"gemma": "Gemma 4", "qwen": "Qwen3.5"}
# Point labels: "Name-<params>B" (user decision 2026-08-31). Gemma E-series
# uses EFFECTIVE params (the same N the FLOPs axis uses), so E4B -> Gemma-4B.
SHORT_LABEL = {"e2b": "Gemma-2B", "e4b": "Gemma-4B", "12b": "Gemma-12B",
               "26b": "Gemma-26B", "31b": "Gemma-31B",
               "qwen2b": "Qwen-2B", "qwen9b": "Qwen-9B"}
# --sizes filter (None = all). Lets a figure set include a single foreign
# point, e.g. the Gemma curve plus Qwen-9B, without dragging in Qwen-2B.
ACTIVE_SIZES = None
INK = dict(surface="#ffffff", primary="#1a1a19", secondary="#55554e",
           grid="#e4e4e0")


# ─── per-run metrics ────────────────────────────────────────────────────
# run_metrics is imported from make_pareto_fig (single source of truth).


def run_flops(run_dir: Path, n_eff: float, cache: dict, flops_args: dict):
    """FLOPs for one run, cached on (path, log.txt mtime)."""
    log = run_dir / "log.txt"
    key = str(run_dir)
    mtime = os.path.getmtime(log) if log.is_file() else None
    hit = cache.get(key)
    if hit and hit.get("mtime") == mtime and hit.get("n_eff") == n_eff:
        return hit["flops"], hit["decode"]
    row = analyze_run(run_dir, SimpleNamespace(n_eff=n_eff, **flops_args))
    if row is None:
        return None, None
    cache[key] = {"mtime": mtime, "n_eff": n_eff,
                  "flops": row["flops"], "decode": row["decode_tokens_exact"]}
    return row["flops"], row["decode_tokens_exact"]


def collect(roots, flops_args, cache):
    """{(size, arm): {"flops": [...], "reward": [...], ...}} over finished runs."""
    out = defaultdict(lambda: defaultdict(list))
    for root in roots:
        if not root.is_dir():
            print("  (skipping missing root {})".format(root))
            continue
        for cond in sorted(p for p in root.iterdir() if p.is_dir()):
            sa = exp_to_size_arm(cond.name)
            if sa is None:
                continue
            size, arm = sa
            for run in MR.load_runs(root, cond.name):
                run_dir = Path(run["_path"]).parent
                flops, decode = run_flops(run_dir, SIZES[size]["n_eff"],
                                          cache, flops_args)
                if flops is None:
                    print("  !! {} — no log.txt/llm_logs, skipped".format(run_dir))
                    continue
                if decode == 0:
                    print("  !! {} — ZERO generated tokens (model never "
                          "answered), excluded".format(run_dir))
                    continue
                m = run_metrics(run)
                if not m:
                    continue
                out[(size, arm)]["flops"].append(flops)
                out[(size, arm)]["_runs"].append(str(run_dir))
                for k, vals in m.items():        # pool episodes across seeds
                    out[(size, arm)][k].extend(vals)
                cfg = run.get("config") or {}
                out[(size, arm)]["_proto"].append(
                    (cfg.get("num_episodes"), cfg.get("max_steps_per_episode")))
    # A 1x50 smoke run and a 3x1000 production run sit ~100x apart on the
    # compute axis; pooling them draws a "trend" that is just protocol. Be loud.
    protos = {p for d in out.values() for p in d["_proto"]}
    if len(protos) > 1:
        print("  !! MIXED PROTOCOLS (num_episodes, max_steps): {} — you are "
              "probably pointing at a *_smoke tree and a production tree at "
              "once".format(sorted(protos)))
    return out


def load_beliefs(paths) -> dict:
    """{(size, arm): {"grounding": [per-run], "partner_loc": [per-run]}}.

    Reads the qualitative pipeline's beliefs.csv files (columns include
    run="<exp_dir>/seed_N", perception_grounding_rate, partner_loc_accuracy,
    quarantined). Quarantined runs are dropped, as in make_qual_tables.py.
    """
    out = defaultdict(lambda: {k: [] for k in BELIEF_COLS})
    for path in paths:
        path = Path(path)
        if not path.is_file():
            print("  (no beliefs.csv at {})".format(path))
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("quarantined", "")).strip().lower() == "true":
                    continue
                sa = exp_to_size_arm(row.get("run", "").split("/")[0])
                if sa is None:
                    continue
                for axis, col in BELIEF_COLS.items():
                    try:
                        out[sa][axis].append(float(row[col]))
                    except (KeyError, ValueError, TypeError):
                        continue
    return out


# ─── drawing ────────────────────────────────────────────────────────────
def series_points(data, family, arm, y, x, perception):
    """Sorted [(x, y_mean, y_sd, label, n, x_sd)] for one (family, arm) series."""
    pts = []
    for size, meta in SIZES.items():
        if meta["family"] != family:
            continue
        if ACTIVE_SIZES is not None and size not in ACTIVE_SIZES:
            continue
        d = data.get((size, arm))
        if not d:
            continue
        if x == "flops":
            if not d.get("flops"):
                continue
            xv, xe = mean_sd(d["flops"])
            xv, xe = xv / 1e17, xe / 1e17
        else:
            vals = perception.get((size, arm), {}).get(x) or []
            if not vals:
                continue
            xv, xe = mean_sd(vals)          # mean +- sd across runs (seeds)
        # y: run_metrics pools episodes; belief metrics are one value per run.
        yvals = d.get(y) or perception.get((size, arm), {}).get(y) or []
        if not yvals:
            continue
        ym, ys = mean_sd(yvals)
        pts.append((xv, ym, ys, SHORT_LABEL.get(size, meta["label"].split()[-1]),
                    len(d["_runs"]), xe))
    pts.sort(key=lambda p: p[0])
    return pts


def draw_panel(ax, data, families, y, x, perception, label_points=True,
               errorbars=False, normalize=False, legend_loc="best",
               join_families=False):
    """One panel in the CoDe Fig. 10 look. Returns the number of series drawn.

    normalize: divide every series by the BASE arm's value at its smallest-x
    point (the cheapest configuration), so that point reads 1.0 -- the
    reference's "(Normalized) Reward" convention. Per family.

    join_families: ONE connecting line per arm through the points of every
    family, sorted by x. Only legitimate on a MEASURED-capability x-axis
    (strict grounding etc.), where the claim is the cross-model correlation
    "models that perceive better perform better"; on the compute axis a
    joined line would fake a scaling law across architectures, so the flag is
    refused there. Family identity stays on the markers (open Gemma / filled
    Qwen). Ignores normalize.
    """
    join = join_families and len(families) > 1
    if join and x == "flops":
        raise SystemExit("--join-families is not valid on the compute axis: "
                         "across families, size is confounded with "
                         "architecture; use a perception x-axis")
    if join:
        for arm in ("base", "hebbian"):
            allp = []
            for family in families:
                allp += series_points(data, family, arm, y, x, perception)
            if len(allp) > 1:
                allp.sort(key=lambda p: p[0])
                ax.plot([p[0] for p in allp], [p[1] for p in allp],
                        ls="-", lw=1.6, color=ARM_STYLE[arm]["color"],
                        zorder=2)
    n = 0
    tops = {}
    all_y = []
    for family in families:
        ref = None
        if normalize:
            base_pts = series_points(data, family, "base", y, x, perception)
            if base_pts and base_pts[0][1]:
                ref = base_pts[0][1]
        for arm in ("base", "hebbian"):
            pts = series_points(data, family, arm, y, x, perception)
            if not pts:
                continue
            scale = ref if ref else 1.0
            xs = [p[0] for p in pts]
            ys = [p[1] / scale for p in pts]
            es = [p[2] / scale for p in pts]
            st = ARM_STYLE[arm]
            open_marker = FAMILY_OPEN.get(family, True)
            label = st["label"] if len(families) == 1 else \
                "{} — {}".format(FAMILY_LABEL[family], st["label"])
            if errorbars:
                ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=st["color"],
                            elinewidth=0.9, capsize=2.5, alpha=0.6, zorder=2)
                all_y += [v + e for v, e in zip(ys, es)]
                all_y += [v - e for v, e in zip(ys, es)]
            all_y += ys
            ax.plot(xs, ys, ls="none" if join else "-", lw=1.6,
                    color=st["color"],
                    marker=st["marker"], ms=st["ms"],
                    mfc="none" if open_marker else st["color"],
                    mec=st["color"], mew=st["mew"],
                    label=label, zorder=3)
            n += 1
            if label_points:
                for xv, ym, ys_, lab, _, _ in pts:
                    key = (family, lab)
                    top = (ym + (ys_ if errorbars else 0.0)) / scale
                    if key not in tops or top > tops[key][1]:
                        tops[key] = [xv, top]
    if label_points:
        # One label per size, above the higher arm; same grey/size as the
        # Delta labels on the social-interval frontier.
        for (_, lab), (xv, top) in tops.items():
            ax.annotate(lab, (xv, top), textcoords="offset points",
                        xytext=(0, 11), ha="center", va="bottom",
                        fontsize=11, color="#555555", zorder=4)
    ax.set_xlabel(X_AXES[x])
    ylabel = Y_METRICS[y]
    if normalize:
        ylabel = "(Normalized) " + ylabel[0].lower() + ylabel[1:]
    ax.set_ylabel(ylabel)
    # Same framing as the social frontier: no grid, generous headroom above so
    # the size labels clear the top marker.
    ax.grid(False)
    if all_y:
        lo, hi = min(all_y), max(all_y)
        pad = max(1e-9, 0.12 * (hi - lo)) if hi > lo else max(abs(hi) * 0.1, 1.0)
        ax.set_ylim(lo - pad, hi + 2.0 * pad)
    ax.margins(x=0.10)
    if n:
        if n > 2:
            # Four series never have a guaranteed-free spot inside the axes
            # (matplotlib's "best" avoids data but not the size annotations),
            # so multi-family panels put the legend BELOW the axes, one row.
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
                      ncol=min(n, 4), fontsize=10.5, framealpha=0.95,
                      columnspacing=1.0, handletextpad=0.5)
        else:
            ax.legend(loc=legend_loc, fontsize=10.5, framealpha=0.95)
    return n


def save(fig, path_stem: Path):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" keeps a below-axes legend inside the canvas.
    fig.savefig(str(path_stem) + ".png", dpi=200, facecolor="white",
                bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets_pareto/grid"))
    ap.add_argument("--beliefs", type=Path, nargs="+",
                    default=[Path("analysis_qualitative/out_gemma/tables/beliefs.csv"),
                             Path("analysis_qualitative/out_pareto/tables/beliefs.csv"),
                             # medium runs: the Qwen belief values. Without
                             # this the Qwen points silently vanish from every
                             # perception axis (empty belief columns).
                             Path("analysis_qualitative/out/tables/beliefs.csv")],
                    help="qualitative-pipeline beliefs.csv files (one row per "
                         "run) supplying the perception axes")
    ap.add_argument("--families", default="gemma",
                    help="comma-separated subset of gemma,qwen,both — Qwen is "
                         "excluded from the paper's plots by decision, so the "
                         "default is gemma only")
    ap.add_argument("--ys", default="milestone_pct,reward,coop_pct",
                    help="comma-separated subset of {}".format(",".join(Y_METRICS)))
    ap.add_argument("--xs", default="flops",
                    help="comma-separated subset of {}; the perception axes "
                         "are available but off by default (dropped from the "
                         "paper: a precision-style rate that rewards terseness "
                         "and is non-monotonic in size)".format(",".join(X_AXES)))
    ap.add_argument("--sizes", default=None,
                    help="comma-separated subset of {} — e.g. "
                         "e2b,e4b,12b,qwen9b to add one Qwen point to the "
                         "Gemma curve".format(",".join(SIZES)))
    ap.add_argument("--no-point-labels", action="store_true")
    ap.add_argument("--legend-loc", default="best",
                    help="matplotlib legend loc (e.g. 'upper left', 'best') "
                         "for panels where lower-right collides with data")
    ap.add_argument("--join-families", action="store_true",
                    help="one line per arm through all families' points, "
                         "sorted by x — perception axes only (refused on the "
                         "compute axis, where it would fake a cross-"
                         "architecture scaling law)")
    ap.add_argument("--errorbars", action="store_true",
                    help="draw +-1 sd over pooled episodes (off by default to "
                         "match the reference figure; the table carries the sd)")
    ap.add_argument("--normalize", action="store_true",
                    help="divide each series by the base arm's cheapest point, "
                         "the reference's '(Normalized) Reward' convention")
    ap.add_argument("--paper", action="store_true",
                    help="also emit pareto_<family>_paper.png: task return and "
                         "cooperative coverage vs compute, side by side")
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()   # start from a clean sheet; styling is explicit below
    matplotlib.rcParams.update({"font.size": 12, "axes.labelsize": 13,
                                "xtick.labelsize": 11.5,
                                "ytick.labelsize": 11.5})
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "flops_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    flops_args = dict(image_tokens=args.image_tokens,
                      overhead_tokens=args.overhead_tokens,
                      chars_per_token=args.chars_per_token)

    global ACTIVE_SIZES
    if args.sizes:
        ACTIVE_SIZES = {s.strip() for s in args.sizes.split(",") if s.strip()}
        bad = ACTIVE_SIZES - set(SIZES)
        if bad:
            sys.exit("unknown --sizes: {}".format(sorted(bad)))

    data = collect(args.roots, flops_args, cache)
    cache_path.write_text(json.dumps(cache, indent=1))
    if not data:
        sys.exit("no finished runs found")

    perception = load_beliefs(args.beliefs)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    ys = [y.strip() for y in args.ys.split(",") if y.strip()]
    xs = [x.strip() for x in args.xs.split(",") if x.strip()]
    if not perception:
        dropped = [x for x in xs if x in BELIEF_COLS]
        if dropped:
            print("  !! no beliefs.csv rows matched ({}) — skipping the "
                  "perception axes {}".format(
                      ", ".join(str(p) for p in args.beliefs), dropped))
        xs = [x for x in xs if x not in BELIEF_COLS]

    print("\npoints (size, arm): n | FLOPs/1e17 | " + " | ".join(Y_METRICS))
    for (size, arm), d in sorted(data.items()):
        fx, _ = mean_sd(d["flops"])
        cells = []
        for k in Y_METRICS:
            if d.get(k):
                m, s = mean_sd(d[k])
                cells.append("{:6.1f}±{:4.1f}".format(m, s))
            else:
                cells.append("     --     ")
        print("  {:<7} {:<8} {} | {:5.2f} | {}".format(
            size, arm, len(d["flops"]), fx / 1e17, " | ".join(cells)))
    missing = [s for s in SIZES if not any(k[0] == s for k in data)]
    if missing:
        print("  (no runs for: {})".format(", ".join(missing)))

    # CSV of everything plotted
    with open(args.out_dir / "points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Header MUST be derived from BELIEF_COLS: hardcoding it once shifted
        # every Y column four places when BELIEF_COLS grew (silent corruption
        # of any analysis reading points.csv; the figures were unaffected).
        w.writerow(["size", "family", "arm", "n", "n_eff", "flops_mean"]
                   + [c for k in BELIEF_COLS for c in (k + "_mean", k + "_sd")]
                   + [c for k in Y_METRICS if k not in BELIEF_COLS
                      for c in (k + "_mean", k + "_sd")])
        for (size, arm), d in sorted(data.items()):
            fx, _ = mean_sd(d["flops"])
            row = [size, SIZES[size]["family"], arm, len(d["flops"]),
                   SIZES[size]["n_eff"], "{:.4e}".format(fx)]
            for axis in BELIEF_COLS:
                vals = perception.get((size, arm), {}).get(axis) or []
                if vals:
                    pm, psd = mean_sd(vals)
                    row += ["{:.4f}".format(pm), "{:.4f}".format(psd)]
                else:
                    row += ["", ""]
            for k in Y_METRICS:
                if k in BELIEF_COLS:
                    continue   # already written from the beliefs block above
                if d.get(k):
                    m, s = mean_sd(d[k])
                    row += ["{:.4f}".format(m), "{:.4f}".format(s)]
                else:
                    row += ["", ""]
            w.writerow(row)

    # singles
    n_written = 0
    for fam in families:
        for y in ys:
            for x in xs:
                fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
                n = draw_panel(ax, data, FAMILIES[fam], y, x, perception,
                               label_points=not args.no_point_labels,
                               errorbars=args.errorbars,
                               normalize=args.normalize,
                               join_families=args.join_families,
                               legend_loc=args.legend_loc)
                if n == 0:
                    plt.close(fig)
                    continue
                fig.tight_layout()
                save(fig, args.out_dir / "pareto_{}_{}_vs_{}".format(fam, y, x))
                plt.close(fig)
                n_written += 1
        # composite: rows = y metrics, cols = x axes
        if len(ys) > 1 or len(xs) > 1:
            fig, axes = plt.subplots(len(ys), len(xs),
                                     figsize=(6.6 * len(xs), 4.6 * len(ys)),
                                     dpi=200, squeeze=False)
            any_drawn = False
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    ax = axes[i][j]
                    if draw_panel(ax, data, FAMILIES[fam], y, x, perception,
                                  label_points=not args.no_point_labels,
                                  errorbars=args.errorbars,
                                  normalize=args.normalize,
                                  join_families=args.join_families,
                               legend_loc=args.legend_loc):
                        any_drawn = True
            if any_drawn:
                # Extra row gap so below-axes legends clear the next row.
                fig.tight_layout(h_pad=3.5)
                save(fig, args.out_dir / "pareto_{}_grid".format(fam))
                n_written += 1
            plt.close(fig)
        # The paper figure: task milestones (the SAME metric and label as the
        # social-interval frontier, so the two Pareto figures are directly
        # comparable) beside task return, one legend on the right panel.
        if args.paper and "flops" in xs:
            fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=200)
            drawn = 0
            for ax, y in zip(axes, ("milestone_pct", "reward")):
                drawn += draw_panel(ax, data, FAMILIES[fam], y, "flops",
                                    perception,
                                    label_points=not args.no_point_labels,
                                    errorbars=args.errorbars,
                                    normalize=args.normalize)
            if drawn:
                axes[0].get_legend().remove()
                fig.tight_layout(w_pad=2.0)
                save(fig, args.out_dir / "pareto_{}_paper".format(fam))
                n_written += 1
            plt.close(fig)
    print("\nwrote {} figures (png) + points.csv to {}".format(
        n_written, args.out_dir))


if __name__ == "__main__":
    main()
