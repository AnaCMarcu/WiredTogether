"""scripts/build_results.py — orchestrate the §B aggregation + table +
plot pipeline against a directory of GRPO runs.

Reads ``runs/grpo/<tag>/seed_*/grpo_metrics.jsonl`` (plus
``time_to_first.json``, ``episode_summary.jsonl``, ``hebbian_snapshots.jsonl``
sidecars when present) and produces:

  results/
  ├── per_ablation/
  │     └── <tag>/summary.json     (AblationSummary as JSON)
  ├── cross_ablation/
  │     ├── plots/*.png            (headline, learning_curves, per_chamber_bars, …)
  │     └── comparisons.json       (PairwiseComparison list)
  └── tables/
        ├── T1_headline.{md,tex}
        ├── T2_per_chamber.{md,tex}
        ├── T3_hebbian_axis.{md,tex}
        ├── T4_coop_comm.{md,tex}
        └── T5_sample_efficiency.{md,tex}

Idempotent: rerun on the same ``runs/`` to regenerate everything.

Usage:
    python scripts/build_results.py \\
        --grpo runs/grpo \\
        --out  results \\
        --ablations G2,G2b,G3a,G3b,G4 \\
        --baseline G2 \\
        --bootstrap 10000 \\
        --window 50

See ``docs/grpo_explained.md`` for what each table reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlvr.compare import (  # noqa: E402
    AblationSummary,
    aggregate_seeds,
    compare_ablations,
    generate_cross_ablation_plots,
    load_runs,
    render_tables,
    save_plots,
    save_summary,
    save_tables,
)
from rlvr.legacy_bridge import translate_directory  # noqa: E402

logger = logging.getLogger(__name__)


def _discover_runs(roots: list[Path], tag: str) -> list[Path]:
    """Glob ``<root>/<tag>/seed_*/grpo_metrics.jsonl`` across every root.

    Phase B+ accepts both ``runs/grpo/`` (Phase A native) and
    ``<output>/legacy_translated/`` (Phase B+ translator output) as
    sources. Tags pick exactly one root in practice — GRPO tags (G2,
    G3a, …) live under the GRPO root; legacy tags (M1-5, L1-2) under
    the translator output — but the loop handles both transparently.

    Handles two-layer naming: SLURM-array runs land in
    ``seed_<N>_job<JOBID>/`` (per ``scripts/grpo.sh``) but the simpler
    ``seed_<N>/`` form is also accepted.
    """
    candidates: list[Path] = []
    for root in roots:
        ablation_dir = root / tag
        if not ablation_dir.is_dir():
            continue
        candidates.extend(sorted(ablation_dir.glob("seed_*/grpo_metrics.jsonl")))
    return candidates


def _build_label(grpo_metrics_path: Path) -> str:
    """Derive a per-seed label from the run directory name.

    ``runs/grpo/G4/seed_42_job12345/grpo_metrics.jsonl`` → ``G4-seed_42_job12345``.
    """
    seed_dir = grpo_metrics_path.parent.name
    ablation = grpo_metrics_path.parent.parent.name
    return f"{ablation}-{seed_dir}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo", type=Path, required=False, default=None,
                        help="Root containing <tag>/seed_*/grpo_metrics.jsonl")
    parser.add_argument("--legacy", type=Path, required=False, default=None,
                        help="Root containing legacy run subdirectories "
                             "(each with final_metrics.json). Translated "
                             "via legacy_to_grpo_schema.py before ingestion "
                             "into <out>/legacy_translated/.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output root (creates per_ablation/, cross_ablation/, tables/)")
    parser.add_argument("--ablations", type=str, required=True,
                        help="Comma-separated tags to aggregate "
                             "(e.g. G2,G2b,G3a,G3b,G4 for GRPO-only or "
                             "M1,L1,L2,M2,M3,M4,M5,G2,G2b,G3a,G3b,G4 for "
                             "full cross-stack)")
    parser.add_argument("--baseline", type=str, default="G2",
                        help="Ablation tag used as the baseline for "
                             "compare_ablations + T3 deltas. Default G2.")
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="Bootstrap resamples for CI / paired delta. "
                             "Default 10000.")
    parser.add_argument("--window", type=int, default=50,
                        help="Final-window size (GRPO steps) for the "
                             "median/p10/p90 cells in tables. Default 50.")
    parser.add_argument("--rolling-window", type=int, default=20,
                        help="Rolling window for line plots. Default 20.")
    parser.add_argument("--metric-comparisons", type=str,
                        default="milestone_fire_rate,group_mean_reward",
                        help="Comma-separated metric keys for pairwise "
                             "Wilcoxon / bootstrap-delta tests.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Bootstrap RNG seed.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib output; tables + summaries only.")
    parser.add_argument("--no-latex", action="store_true",
                        help="Skip LaTeX rendering; markdown tables only.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    tags = [t.strip() for t in args.ablations.split(",") if t.strip()]
    if not tags:
        logger.error("--ablations is empty after parsing")
        return 2
    if args.baseline not in tags:
        logger.warning("--baseline %r not in --ablations; pairwise "
                       "comparisons will fail", args.baseline)
    if args.grpo is None and args.legacy is None:
        logger.error("Must pass at least one of --grpo / --legacy")
        return 2

    metrics = tuple(m.strip() for m in args.metric_comparisons.split(",")
                    if m.strip())

    # ─── Phase B+ §3: translate legacy runs if requested ─────────────────

    args.out.mkdir(parents=True, exist_ok=True)
    legacy_translated_root: Path | None = None
    if args.legacy is not None:
        legacy_translated_root = args.out / "legacy_translated"
        logger.info("Translating legacy runs under %s → %s",
                    args.legacy, legacy_translated_root)
        summaries = translate_directory(args.legacy, legacy_translated_root)
        logger.info("Translated %d legacy run(s)", len(summaries))
        for s in summaries:
            logger.info("  tag=%s seed=%s n_steps=%d n_episodes=%d",
                        s["tag"], s["seed"], s["n_steps"], s["n_episodes"])

    # ─── load per-ablation runs (search both roots) ─────────────────────

    roots: list[Path] = []
    if args.grpo is not None:
        roots.append(args.grpo)
    if legacy_translated_root is not None:
        roots.append(legacy_translated_root)

    ablations: dict[str, list] = {}
    for tag in tags:
        paths = _discover_runs(roots, tag)
        if not paths:
            logger.warning("No runs found for ablation %s under any of %s",
                           tag, roots)
            continue
        labels = [_build_label(p) for p in paths]
        ablations[tag] = load_runs(paths, labels)
        logger.info("Loaded %d seed(s) for %s", len(ablations[tag]), tag)

    if not ablations:
        logger.error("No ablations had any seeds — aborting")
        return 1

    # ─── per-ablation aggregation ───────────────────────────────────────

    summaries: dict[str, AblationSummary] = {}
    per_ab_dir = args.out / "per_ablation"
    per_ab_dir.mkdir(exist_ok=True)
    for tag, runs in ablations.items():
        summary = aggregate_seeds(
            runs, label=tag,
            window=args.window, bootstrap=args.bootstrap, seed=args.seed,
        )
        summaries[tag] = summary
        tag_dir = per_ab_dir / tag
        tag_dir.mkdir(exist_ok=True)
        save_summary(summary.as_dict(), tag_dir / "summary.json")
        logger.info("Wrote per-ablation summary: %s/summary.json", tag_dir)

    # ─── pairwise comparisons (if baseline present) ─────────────────────

    cross_dir = args.out / "cross_ablation"
    cross_dir.mkdir(exist_ok=True)
    comparisons: list = []
    if args.baseline in ablations:
        comparisons = compare_ablations(
            ablations, baseline=args.baseline,
            metrics=metrics,
            window=args.window, bootstrap=args.bootstrap, seed=args.seed,
        )
        save_summary([c.as_dict() for c in comparisons],
                     cross_dir / "comparisons.json")
        logger.info("Wrote %d pairwise comparisons to %s/comparisons.json",
                    len(comparisons), cross_dir)
    else:
        logger.warning("Baseline %r missing — skipping pairwise comparisons",
                       args.baseline)

    # ─── tables (markdown always, LaTeX optional) ───────────────────────

    tables_dir = args.out / "tables"
    md_tables = render_tables(summaries, comparisons, fmt="markdown")
    save_tables(md_tables, tables_dir, extension="md")
    logger.info("Wrote %d markdown tables to %s", len(md_tables), tables_dir)

    if not args.no_latex:
        tex_tables = render_tables(summaries, comparisons, fmt="latex")
        save_tables(tex_tables, tables_dir, extension="tex")
        logger.info("Wrote %d LaTeX tables to %s", len(tex_tables), tables_dir)

    # ─── cross-ablation plots ───────────────────────────────────────────

    if not args.no_plots:
        plots = generate_cross_ablation_plots(
            ablations, summaries, window=args.rolling_window,
        )
        save_plots(plots, cross_dir / "plots")
        logger.info("Wrote %d cross-ablation plots to %s/plots",
                    len(plots), cross_dir)

    logger.info("Done. Results under %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
