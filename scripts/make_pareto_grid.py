"""Pareto plot GRID for the model-size sweep, CoDe-style (dashed lines + markers).

Produces every combination of

    family  in {gemma, qwen, both}
    y-axis  in {reward, milestone_pct, coop_pct}
    x-axis  in {flops, perception}

= 18 single-panel figures (PNG + PDF) plus one composite 3x2 panel per family,
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
    perception     an external, model-level visual-reasoning score: MMMU-Pro,
                   NON-thinking mode (our agents run with thinking off), from
                   the official model cards -- see perception_scores.csv, which
                   carries the source per row. Same value for both arms of a
                   size (it is a property of the backbone).

Style (dataviz skill + the CoDe Fig. 10 reference): colour = arm (base blue /
hebbian orange, the two palette slots that validate all-pairs in both modes),
marker shape = family (circle Gemma / square Qwen), dashed connecting lines
within a (family, arm) series, families NEVER joined to each other, boxed
legend inside the axes, thin +-1 sd error bars over seeds, each point labelled
with its size.

Usage:
    python scripts/make_pareto_grid.py runs_from_daic/pareto_gemma4 \
        runs_from_daic/new_exp_0_gemma runs_from_daic/medium_runs \
        --out-dir paper_assets_pareto/grid
    python scripts/make_pareto_grid.py ... --perception paper_assets_pareto/perception_scores.csv
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
Y_METRICS = {
    "reward":           "Task return (team, per episode)",
    "milestone_pct":    "Milestone coverage (team-distinct) [%]",
    "coop_pct":         "Coop. milestone coverage (team-distinct) [%]",
    "completions":      "Milestone completions (all agents, per episode)",
    "coop_completions": "Coop. milestone completions (all agents, per episode)",
}
X_AXES = {
    "flops":      "Inference compute  [$10^{17}$ FLOPs]",
    "perception": "Perception  [MMMU-Pro, non-thinking, %]",
}
ARM_COLOR = {"base": "#2a78d6", "hebbian": "#eb6834"}
FAMILY_MARKER = {"gemma": "o", "qwen": "s"}
FAMILY_LABEL = {"gemma": "Gemma 4", "qwen": "Qwen3.5"}
# Point labels: the family is already carried by marker shape + legend, so
# the label only needs the size. Gemma keeps E2B/E4B/12B; Qwen gets a short
# prefix so "2B" is not confused with "E2B".
SHORT_LABEL = {"qwen2b": "Q-2B", "qwen9b": "Q-9B"}
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


def load_perception(path: Path | None) -> dict:
    """size -> MMMU-Pro score (float) from a CSV with columns size,score,...."""
    if path is None or not path.is_file():
        return {}
    scores = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                scores[row["size"].strip()] = float(row["score"])
            except (KeyError, ValueError):
                continue
    return scores


# ─── drawing ────────────────────────────────────────────────────────────
def series_points(data, family, arm, y, x, perception):
    """Sorted [(x, y_mean, y_sd, label, n)] for one (family, arm) series."""
    pts = []
    for size, meta in SIZES.items():
        if meta["family"] != family:
            continue
        d = data.get((size, arm))
        if not d or not d.get(y):
            continue
        if x == "flops":
            xv, _ = mean_sd(d["flops"])
            xv /= 1e17
        else:
            xv = perception.get(size)
            if xv is None:
                continue
        ym, ys = mean_sd(d[y])
        pts.append((xv, ym, ys, SHORT_LABEL.get(size, meta["label"].split()[-1]),
                    len(d["_runs"])))
    pts.sort(key=lambda p: p[0])
    return pts


def draw_panel(ax, data, families, y, x, perception, label_points=True):
    """One CoDe-style panel. Returns the number of series drawn."""
    n = 0
    tops = {}
    for family in families:
        for arm in ("base", "hebbian"):
            pts = series_points(data, family, arm, y, x, perception)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=ARM_COLOR[arm],
                        elinewidth=0.9, capsize=2.5, alpha=0.55, zorder=2)
            ax.plot(xs, ys, linestyle="--", linewidth=1.6,
                    color=ARM_COLOR[arm], marker=FAMILY_MARKER[family],
                    markersize=7, markerfacecolor=ARM_COLOR[arm],
                    markeredgecolor=INK["surface"], markeredgewidth=1.0,
                    label="{} — {}".format(FAMILY_LABEL[family], arm), zorder=3)
            n += 1
            if label_points:
                for xv, ym, ys_, lab, _ in pts:
                    key = (family, lab)
                    top, bot = ym + ys_, ym - ys_
                    if key not in tops:
                        tops[key] = [xv, top, bot]
                    else:
                        tops[key][1] = max(tops[key][1], top)
                        tops[key][2] = min(tops[key][2], bot)
    if label_points:
        # One label per size, above the taller arm's error bar. Labels are
        # SHORT ("Q-2B", not "Qwen3.5-2B"): on the shared canvas the families
        # interleave in x, and the long form collided with the E4B markers.
        # (Placing Qwen labels below instead was tried and collided with the
        # other family's line -- below-placement is data-dependent, short
        # text is not.)
        for (_, lab), (xv, top, _) in tops.items():
            ax.annotate(lab, (xv, top), textcoords="offset points",
                        xytext=(0, 6), ha="center", va="bottom",
                        fontsize=7, color=INK["secondary"], zorder=4)
    ax.set_xlabel(X_AXES[x], fontsize=9.5, color=INK["primary"])
    ax.set_ylabel(Y_METRICS[y], fontsize=9.5, color=INK["primary"])
    ax.tick_params(colors=INK["secondary"], labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK["grid"])
    ax.grid(True, color=INK["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.18)
    if n:
        leg = ax.legend(frameon=True, fontsize=8, loc="best",
                        edgecolor=INK["grid"], framealpha=0.95)
        for t in leg.get_texts():
            t.set_color(INK["primary"])
    return n


def save(fig, path_stem: Path):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_stem) + ".png", dpi=200, facecolor=INK["surface"])
    fig.savefig(str(path_stem) + ".pdf", facecolor=INK["surface"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets_pareto/grid"))
    ap.add_argument("--perception", type=Path,
                    default=Path("paper_assets_pareto/perception_scores.csv"))
    ap.add_argument("--families", default="gemma,qwen,both")
    ap.add_argument("--ys", default=",".join(Y_METRICS))
    ap.add_argument("--xs", default=",".join(X_AXES))
    ap.add_argument("--no-point-labels", action="store_true")
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "flops_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    flops_args = dict(image_tokens=args.image_tokens,
                      overhead_tokens=args.overhead_tokens,
                      chars_per_token=args.chars_per_token)

    data = collect(args.roots, flops_args, cache)
    cache_path.write_text(json.dumps(cache, indent=1))
    if not data:
        sys.exit("no finished runs found")

    perception = load_perception(args.perception)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    ys = [y.strip() for y in args.ys.split(",") if y.strip()]
    xs = [x.strip() for x in args.xs.split(",") if x.strip()]
    if "perception" in xs and not perception:
        print("  !! no perception scores loaded ({}) — skipping the "
              "perception x-axis".format(args.perception))
        xs = [x for x in xs if x != "perception"]

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
        w.writerow(["size", "family", "arm", "n", "n_eff", "flops_mean",
                    "perception_mmmu_pro"]
                   + [c for k in Y_METRICS for c in (k + "_mean", k + "_sd")])
        for (size, arm), d in sorted(data.items()):
            fx, _ = mean_sd(d["flops"])
            row = [size, SIZES[size]["family"], arm, len(d["flops"]),
                   SIZES[size]["n_eff"], "{:.4e}".format(fx),
                   perception.get(size, "")]
            for k in Y_METRICS:
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
                fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=200)
                fig.patch.set_facecolor(INK["surface"])
                ax.set_facecolor(INK["surface"])
                n = draw_panel(ax, data, FAMILIES[fam], y, x, perception,
                               label_points=not args.no_point_labels)
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
                                     figsize=(5.0 * len(xs), 3.6 * len(ys)),
                                     dpi=200, squeeze=False)
            fig.patch.set_facecolor(INK["surface"])
            any_drawn = False
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    ax = axes[i][j]
                    ax.set_facecolor(INK["surface"])
                    if draw_panel(ax, data, FAMILIES[fam], y, x, perception,
                                  label_points=not args.no_point_labels):
                        any_drawn = True
            if any_drawn:
                fig.tight_layout()
                save(fig, args.out_dir / "pareto_{}_grid".format(fam))
                n_written += 1
            plt.close(fig)
    print("\nwrote {} figures (png+pdf) + points.csv to {}".format(
        n_written, args.out_dir))


if __name__ == "__main__":
    main()
