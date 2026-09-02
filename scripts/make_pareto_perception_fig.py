"""Paper figure: milestone completion vs perception grounding rate, five models.

The perception-x Pareto in the SAME layout as the social-interval frontier
(make_pareto_social_fig.py single-panel figure): 5.0 x 3.5 in, default
fonts, grey 8.5 pt point labels with the crowd-flip rule, framed legend in
the lower right, the social figure's y-headroom rule. Axis names follow the
paper: "Perception grounding rate" (the strict rate -- nothing impossible
AND at least one chamber-diagnostic object; the word "strict" is dropped in
the paper, where the definition is given in the text) and
"Milestone completion [%]".

One line per arm through every model (Gemma 4 E2B/E4B/12B, Qwen3.5 2B/9B),
family on the marker fill (open Gemma, filled Qwen). Also writes the LaTeX
rows for the per-model analysis table (tab:pareto_perception) so the paper's
numbers come from the same collect()/load_beliefs() the figure uses.

Usage (from the repo root):
    python scripts/make_pareto_perception_fig.py runs_from_daic/pareto_gemma4 \
        runs_from_daic/new_exp_0_gemma runs_from_daic/medium_runs \
        --out-dir paper_assets_pareto/paper \
        --copy-to "C:/.../paper/figures"      # optional: drop the PNG there
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from make_pareto_fig import SIZES, mean_sd                     # noqa: E402
from make_pareto_grid import (                                 # noqa: E402
    collect, load_beliefs, ARM_STYLE, FAMILY_LABEL, FAMILY_OPEN, SHORT_LABEL,
)

Y_KEY = "milestone_pct"
AXES = {  # --x choice: (beliefs column key, axis label, output filename)
    "grounding":   ("grounding_strict", "Perception grounding rate",
                    "pareto_perception.png"),
    "partner_loc": ("partner_loc", "Partner-location accuracy",
                    "pareto_partner.png"),
}
YLABEL = "Milestone completion [%]"
ARM_NAME = {"base": "base", "hebbian": "+Hebbian"}
LABEL_BELOW = {"e2b"}   # models whose label hangs below the lower arm


def gather(data, beliefs, sizes, x_key):
    """{size: {"x": (m, sd), "hal": (m, sd), arm: {"y": (m, sd), "n": n}}}."""
    out = {}
    for size in sizes:
        row = {"family": SIZES[size]["family"]}
        ok = True
        for arm in ("base", "hebbian"):
            d = data.get((size, arm))
            b = beliefs.get((size, arm), {})
            gx, gp = b.get(x_key) or [], b.get("grounding") or []
            if not d or not d.get(Y_KEY) or not gx:
                ok = False
                break
            row[arm] = {"x": mean_sd(gx), "y": mean_sd(d[Y_KEY]),
                        "hal": mean_sd([1.0 - v for v in gp]) if gp else None,
                        "n": len(gx)}
        if ok:
            out[size] = row
        else:
            print("  (skipping {}: incomplete data)".format(size))
    return out


def draw(rows, out_png: Path, xlabel: str, errorbars: str = "none",
         label_below=frozenset(), legend_loc="lower right"):
    """errorbars: "none", "y" (milestone std over pooled episodes) or "xy"
    (also the grounding std over seeds). Thin bars in the arm colour behind
    the markers, no legend entry; labels clear the bar tops."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()          # the social figure runs on the defaults
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    all_y = []
    # One line per arm through every model, sorted by x; family on the fill.
    for arm in ("base", "hebbian"):
        st = ARM_STYLE[arm]
        # BOTH arms sit at the BASE arm's x: the axis is a backbone property
        # (couplings shift it by <= 2 pp), and per-arm x scattered a model's
        # two markers so far apart on the partner axis that neighbouring
        # models interleaved and the point labels became unattributable.
        # Each model is one vertical base/+Hebbian pair under its label.
        pts = sorted(((r["base"]["x"][0], r[arm]["y"][0], r["family"],
                       r["base"]["x"][1], r[arm]["y"][1])
                      for r in rows.values()), key=lambda t: t[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st["color"],
                ls="-", lw=1.6, zorder=2)
        if errorbars != "none":
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                        yerr=[p[4] for p in pts],
                        xerr=[p[3] for p in pts] if errorbars == "xy" else None,
                        fmt="none", ecolor=st["color"], elinewidth=0.9,
                        capsize=2.5, alpha=0.6, zorder=1)
            all_y += [p[1] + p[4] for p in pts] + [p[1] - p[4] for p in pts]
        for family in dict.fromkeys(p[2] for p in pts):
            fp = [p for p in pts if p[2] == family]
            open_marker = FAMILY_OPEN.get(family, True)
            ax.plot([p[0] for p in fp], [p[1] for p in fp], color=st["color"],
                    ls="none", marker=st["marker"], ms=st["ms"],
                    mfc="none" if open_marker else st["color"],
                    mec=st["color"], mew=st["mew"], zorder=3)
        all_y += [p[1] for p in pts]

    # Point labels: one per model above its higher arm; when a model sits
    # within 10 % of the x-span of its left neighbour the label flips to
    # below-right of the LOWER arm (the social figure's crowd rule).
    order = sorted(rows, key=lambda s: rows[s]["base"]["x"][0])
    xs_all = [rows[s]["base"]["x"][0] for s in order]
    span = (max(xs_all) - min(xs_all)) or 1.0
    prev_x = None
    for size in order:
        r = rows[size]
        xv = r["base"]["x"][0]          # both arms share the model's x scale
        hi_arm = max(("base", "hebbian"), key=lambda a: r[a]["y"][0])
        lo_arm = min(("base", "hebbian"), key=lambda a: r[a]["y"][0])
        e = 1.0 if errorbars != "none" else 0.0
        y_hi = r[hi_arm]["y"][0] + e * r[hi_arm]["y"][1]
        y_lo = r[lo_arm]["y"][0] - e * r[lo_arm]["y"][1]
        crowded = prev_x is not None and (xv - prev_x) < 0.10 * span
        if size in label_below:
            # The leftmost model's label would sit on the segment climbing
            # to its right neighbour; hang it under the lower arm instead.
            ax.annotate(SHORT_LABEL.get(size, size),
                        (xv, y_lo),
                        textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=8.5, color="#555555", zorder=4)
        elif crowded:
            ax.annotate(SHORT_LABEL.get(size, size),
                        (xv, y_lo),
                        textcoords="offset points", xytext=(7, -16),
                        ha="left", fontsize=8.5, color="#555555", zorder=4)
        else:
            ax.annotate(SHORT_LABEL.get(size, size),
                        (xv, y_hi),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color="#555555", zorder=4)
        prev_x = xv

    lo, hi = min(all_y), max(all_y)
    pad = max(0.02 * (abs(hi) or 1.0), 0.25 * (hi - lo))
    ax.set_ylim(lo - pad, hi + 1.8 * pad)
    ax.margins(x=0.10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(YLABEL)
    # Two-entry legend in the social-interval figure's style: the arms only
    # (open markers, as there). Family lives on the marker FILL (open Gemma,
    # filled Qwen) and is stated in the caption -- a four-entry box was large
    # enough to collide with the data on the partner axis.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", color=ARM_STYLE[a]["color"],
                      marker=ARM_STYLE[a]["marker"], ms=ARM_STYLE[a]["ms"],
                      mfc="none", mew=ARM_STYLE[a]["mew"], label=ARM_STYLE[a]["label"])
               for a in ("base", "hebbian")]
    ax.legend(handles=handles, loc=legend_loc, fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("wrote {}".format(out_png))


def tex_rows(rows) -> str:
    """Rows for tab:pareto_perception: model, grounding, hallucination,
    milestones base / +Heb, delta. Percent with one decimal, sd after \\pmm."""
    order = sorted(rows, key=lambda s: rows[s]["base"]["x"][0])
    lines = []
    for size in order:
        r = rows[size]
        b, h = r["base"], r["hebbian"]
        d = h["y"][0] - b["y"][0]
        lines.append(
            "{name} & ${gx:.1f}$ \\pmm{{{gs:.1f}}} & ${hx:.1f}$ \\pmm{{{hs:.1f}}}"
            " & ${by:.1f}$ \\pmm{{{bs:.1f}}} & ${hy:.1f}$ \\pmm{{{hhs:.1f}}}"
            " & ${d:+.1f}$ \\\\".format(
                name=SHORT_LABEL.get(size, size),
                gx=100 * b["x"][0], gs=100 * b["x"][1],
                hx=100 * b["hal"][0], hs=100 * b["hal"][1],
                by=b["y"][0], bs=b["y"][1], hy=h["y"][0], hhs=h["y"][1], d=d))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets_pareto/paper"))
    ap.add_argument("--beliefs", type=Path, nargs="+",
                    default=[Path("analysis_qualitative/out_gemma/tables/beliefs.csv"),
                             Path("analysis_qualitative/out_pareto/tables/beliefs.csv"),
                             Path("analysis_qualitative/out/tables/beliefs.csv")])
    ap.add_argument("--sizes", default="e2b,e4b,qwen2b,12b,qwen9b")
    ap.add_argument("--x", choices=tuple(AXES), default="grounding",
                    help="x-axis: strict grounding (paper) or partner-location accuracy")
    ap.add_argument("--legend-loc", default="lower right",
                    help="matplotlib legend location (partner axis: lower left)")
    ap.add_argument("--errorbars", choices=("none", "y", "xy"), default="none",
                    help="draw +-1 std: y = milestones over pooled episodes, "
                         "xy = also grounding over seeds")
    ap.add_argument("--copy-to", type=Path, default=None,
                    help="also copy the PNG into this directory (paper/figures)")
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "flops_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    data = collect(args.roots, dict(image_tokens=args.image_tokens,
                                    overhead_tokens=args.overhead_tokens,
                                    chars_per_token=args.chars_per_token), cache)
    cache_path.write_text(json.dumps(cache, indent=1))
    beliefs = load_beliefs(args.beliefs)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    x_key, xlabel, png_name = AXES[args.x]
    rows = gather(data, beliefs, sizes, x_key)

    print("  {:<10} {:>7} {:>7} {:>12} {:>12} {:>7}".format(
        "model", "ground", "halluc", "milest base", "milest +Heb", "delta"))
    for size in sorted(rows, key=lambda s: rows[s]["base"]["x"][0]):
        r = rows[size]
        print("  {:<10} {:7.3f} {:7.3f} {:6.1f}±{:<5.1f} {:6.1f}±{:<5.1f} {:+7.1f}".format(
            SHORT_LABEL.get(size, size), r["base"]["x"][0], r["base"]["hal"][0],
            r["base"]["y"][0], r["base"]["y"][1], r["hebbian"]["y"][0],
            r["hebbian"]["y"][1], r["hebbian"]["y"][0] - r["base"]["y"][0]))

    png = args.out_dir / png_name
    draw(rows, png, xlabel, args.errorbars,
         label_below=LABEL_BELOW if args.x == "grounding" else frozenset(),
         legend_loc=args.legend_loc)
    if args.x == "grounding":
        tex = args.out_dir / "pareto_perception_rows.tex"
        tex.write_text(tex_rows(rows), encoding="utf-8")
        print("wrote {}".format(tex))
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, args.copy_to / png.name)
        print("copied to {}".format(args.copy_to / png.name))


if __name__ == "__main__":
    main()
