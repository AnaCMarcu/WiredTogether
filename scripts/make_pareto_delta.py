"""Hebbian benefit vs perception: the figure for the RQ1 mechanism claim.

One point per model: x = the BASE arm's strict perception grounding rate,
y = the Hebbian delta (hebbian - base) on task milestones (left panel) and
task return (right panel), with a zero reference line. Downward trend =
"the Hebbian benefit concentrates where perception is weak and disappears
(then reverses) as the backbone's grounded perception improves".

Direction check (2026-08-31, from the same collect() the grid uses):

    model      base strict   d milestones   d return   d coop
    Gemma-2B      0.056         +0.4 pp        +30      -0.7
    Gemma-4B      0.103         +5.6 pp        +49      +3.6
    Qwen-2B       0.133         -2.4 pp         +5      -2.3
    Gemma-12B     0.232         -1.8 pp        +21      -2.3
    Qwen-9B       0.415         -3.8 pp        -49      -3.6

so the supportable claim is "models where Hebbian DOES help have LOWER
perception" (every model with strict grounding >= 0.133 loses milestones),
NOT the converse. Both axes are measured outcomes of the same runs -- a
relationship, not a controlled sweep.

Usage (from the repo root):
    python scripts/make_pareto_delta.py runs_from_daic/pareto_gemma4 \
        runs_from_daic/new_exp_0_gemma runs_from_daic/medium_runs \
        --out-dir paper_assets_pareto/grid_perception_x_qwen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from make_pareto_fig import SIZES, mean_sd            # noqa: E402
from make_pareto_grid import (                        # noqa: E402
    collect, load_beliefs, run_flops, SHORT_LABEL, SERIES_COLOR, FAMILY_OPEN,
)
import make_pareto_grid as G                          # noqa: E402

PANELS = [("milestone_pct", "$\\Delta$ Task milestones  [pp]"),
          ("reward",        "$\\Delta$ Task return")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("paper_assets_pareto/grid_perception_x_qwen"))
    ap.add_argument("--beliefs", type=Path, nargs="+",
                    default=[Path("analysis_qualitative/out_gemma/tables/beliefs.csv"),
                             Path("analysis_qualitative/out_pareto/tables/beliefs.csv"),
                             Path("analysis_qualitative/out/tables/beliefs.csv")])
    ap.add_argument("--sizes", default="e2b,e4b,qwen2b,12b,qwen9b")
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    args = ap.parse_args()

    import json
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "flops_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    flops_args = dict(image_tokens=args.image_tokens,
                      overhead_tokens=args.overhead_tokens,
                      chars_per_token=args.chars_per_token)
    data = collect(args.roots, flops_args, cache)
    cache_path.write_text(json.dumps(cache, indent=1))
    beliefs = load_beliefs(args.beliefs)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]

    rows = []
    for size in sizes:
        b, h = data.get((size, "base")), data.get((size, "hebbian"))
        gx = beliefs.get((size, "base"), {}).get("grounding_strict") or []
        if not b or not h or not gx:
            print("  (skipping {}: incomplete data)".format(size))
            continue
        row = {"size": size, "x": mean_sd(gx)[0],
               "open": FAMILY_OPEN.get(SIZES[size]["family"], True)}
        for metric, _ in PANELS:
            if b.get(metric) and h.get(metric):
                row[metric] = mean_sd(h[metric])[0] - mean_sd(b[metric])[0]
        rows.append(row)
    rows.sort(key=lambda r: r["x"])
    for r in rows:
        print("  {:<9} x={:.3f}  d_milestones={:+.1f}  d_return={:+.0f}".format(
            r["size"], r["x"], r.get("milestone_pct", float("nan")),
            r.get("reward", float("nan"))))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), dpi=200)
    for ax, (metric, ylabel) in zip(axes, PANELS):
        ax.axhline(0.0, color="#999999", ls="--", lw=1.0, zorder=1)
        xs = [r["x"] for r in rows]
        ys = [r[metric] for r in rows]
        ax.plot(xs, ys, ls="-", lw=1.4, color=SERIES_COLOR, zorder=2)
        for r in rows:
            ax.plot([r["x"]], [r[metric]], ls="none", marker="o", ms=7,
                    mfc="none" if r["open"] else SERIES_COLOR,
                    mec=SERIES_COLOR, mew=1.7, zorder=3)
            above = r[metric] >= 0
            ax.annotate(SHORT_LABEL.get(r["size"], r["size"]),
                        (r["x"], r[metric]), textcoords="offset points",
                        xytext=(0, 10 if above else -10), ha="center",
                        va="bottom" if above else "top",
                        fontsize=8.5, color="#555555", zorder=4)
        ax.set_xlabel("Strict perception grounding rate (base)")
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.margins(x=0.12, y=0.25)
    fig.tight_layout(w_pad=2.0)
    out = args.out_dir / "pareto_hebbian_delta_vs_grounding.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
