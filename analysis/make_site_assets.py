#!/usr/bin/env python
"""Copy the project-page figures into `site/assets/figures/`, downscaled for the web.

The page shows figures the paper already uses. Their sources live in three
places — `figures/` (the release copies), `paper_assets/` (generated, git-ignored)
and the LaTeX tree — so this collects them under stable names, caps their width
and strips the colour profiles that make a 4000 px PNG a 7 MB download.

The videos are cut by `analysis/make_media_clips.py`; this handles stills only.

Usage:
  python analysis/make_site_assets.py            # copy and downscale
  python analysis/make_site_assets.py --check     # report what is missing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from paths import ASSETS, REPO  # noqa: E402  (also puts siblings on sys.path)

FIGURES = REPO / "figures"
OUT = REPO / "site" / "assets" / "figures"

#: Published name -> (source, max width in px). First existing source wins.
WANTED: dict[str, tuple[tuple[Path, ...], int]] = {
    # environment
    "wire.png": ((FIGURES / "WIRE_FINAL.png",), 1916),
    "first_person.png": ((FIGURES / "wire_first_person_views.png",
                          ASSETS / "chamber_gallery" / "chamber_gallery.png"), 1800),
    # method
    "graph_formation.png": ((FIGURES / "fig1_graph_formation.png",), 1800),
    "couplings.png": ((FIGURES / "fig2_couplings.png",), 1800),
    "overview.png": ((FIGURES / "overview_social_plasticity_loop.png",), 1800),
    # results
    "frontier_milestone.png": (
        (ASSETS / "pareto_social_3f" / "social_frontier_milestone.png",), 1200),
    "partner_accuracy.png": ((ASSETS / "perception_3f" / "pareto_partner.png",), 1200),
    "perception.png": ((ASSETS / "perception_3f" / "pareto_perception.png",), 1200),
    "scaling_coop.png": ((FIGURES / "pareto_coop_500.png",), 1200),
    "three_factor_trace.png": (
        (FIGURES / "reward_modulation_replay_seed456_b_eligibility_trace.png",), 1400),
    "counterfactual_plasticity.png": (
        (ASSETS / "timelines" / "counterfactual" / "counterfactual_compact_a.png",), 1400),
    "counterfactual_orchestration.png": (
        (ASSETS / "timelines" / "counterfactual" / "counterfactual_compact_b.png",), 1400),
    "team_tenure.png": ((ASSETS / "timelines" / "comparison" / "team_tenure.png",), 1600),
    "milestones.png": ((FIGURES / "milestone_completion_timeline.png",), 1800),
    "coordination_timeline.png": (
        (FIGURES / "coordination_timeline_qwen9b_hebbian_seed42.png",), 1800),
    "qualitative_hebbian.png": (
        (ASSETS / "timelines" / "for_paper" / "wide_qwen9b_hebbian.png",), 2000),
    "qualitative_orchestrator.png": (
        (ASSETS / "timelines" / "for_paper" / "wide_orchestrator.png",), 2000),
}


def pick(sources: tuple[Path, ...]) -> Path | None:
    return next((s for s in sources if s.exists()), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report sources, copy nothing")
    ap.add_argument("--outdir", type=Path, default=OUT)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, (sources, maxw) in WANTED.items():
        src = pick(sources)
        if src is None:
            missing.append((name, sources))
            print(f"  {name:<30} MISSING  ({', '.join(str(s.relative_to(REPO)) for s in sources)})")
            continue
        if args.check:
            print(f"  {name:<30} <- {src.relative_to(REPO)}")
            continue
        img = Image.open(src)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if img.width > maxw:
            img = img.resize((maxw, round(img.height * maxw / img.width)), Image.LANCZOS)
        dest = args.outdir / name
        img.save(dest, optimize=True)
        print(f"  {name:<30} {img.width}x{img.height}  "
              f"{dest.stat().st_size / 1e6:.2f} MB  <- {src.relative_to(REPO)}")

    if missing:
        print(f"\n{len(missing)} figure(s) missing. Generated figures come from "
              "analysis/ — see analysis/README.md for which script writes which.",
              file=sys.stderr)
        sys.exit(1 if not args.check else 0)


if __name__ == "__main__":
    main()
