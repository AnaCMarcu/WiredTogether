"""Translate legacy ``final_metrics.json`` runs into Phase A sidecar
schema so ``build_results.py`` can ingest them alongside GRPO runs.

Reads each subdirectory under ``--input`` that contains a
``final_metrics.json``, classifies it by the CLI args recorded in
``config.cli_args``, and writes the four GRPO-shaped sidecars to
``--output/<tag>/seed_<N>/``.

Usage:

    python scripts/legacy_to_grpo_schema.py \\
        --input  runs/legacy_E5_hebbian \\
        --output runs/legacy_translated

The output directory layout matches what ``build_results.py --grpo
<dir>`` already consumes — so the same orchestrator handles legacy and
GRPO runs without code duplication.

Single-run mode: pass ``--input`` pointing directly at one legacy run
directory (containing ``final_metrics.json``) and the script translates
just that run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlvr.legacy_bridge import translate_directory, translate_run  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Either a directory containing legacy run subdirectories "
             "(each with a final_metrics.json), or a single legacy run "
             "directory itself.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output root. Sidecars land at <output>/<tag>/seed_<N>/.",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Override auto-tagging (single-run mode only). "
             "Must be one of M1, L1, L2, M2, M3, M4, M5.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override seed inference (single-run mode only).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if (args.input / "final_metrics.json").exists():
        # Single-run mode.
        summary = translate_run(
            args.input, args.output, tag=args.tag, seed=args.seed,
        )
        logger.info(
            "translated 1 run: tag=%s seed=%s n_steps=%d n_episodes=%d "
            "n_milestones_fired=%d",
            summary["tag"], summary["seed"], summary["n_steps"],
            summary["n_episodes"], summary["n_milestones_fired"],
        )
    else:
        # Directory-of-runs mode.
        if args.tag is not None or args.seed is not None:
            logger.warning("--tag / --seed ignored in directory mode "
                           "(auto-classification per run).")
        summaries = translate_directory(args.input, args.output)
        logger.info("translated %d run(s) under %s", len(summaries), args.input)
        for s in summaries:
            logger.info("  tag=%s seed=%s n_steps=%d n_episodes=%d "
                        "n_milestones_fired=%d",
                        s["tag"], s["seed"], s["n_steps"],
                        s["n_episodes"], s["n_milestones_fired"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
