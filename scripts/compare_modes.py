"""scripts/compare_modes.py — side-by-side GRPO ablation plots.

Loads one or more ``grpo_metrics.jsonl`` files (one per run) and writes a
folder of comparison plots + a summary JSON.

Usage:

    python scripts/compare_modes.py \\
        --grpo-metrics runs/grpo_only/grpo_metrics.jsonl \\
                       runs/grpo_hebbian_diffusion/grpo_metrics.jsonl \\
                       runs/grpo_hebbian_composition/grpo_metrics.jsonl \\
                       runs/grpo_hebbian_full/grpo_metrics.jsonl \\
        --labels base hebbian-4a hebbian-4b hebbian-full \\
        --output-dir reports/grpo_ablation \\
        --window 20

The MAPPO baseline integration is currently a placeholder — the legacy
stack writes metrics in a different schema. Bridging it is its own work
item; for now, run MAPPO eval externally and add its summary as a
horizontal line on the bar chart if desired.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# PYTHONPATH=src is the project convention; this script also tolerates the
# common alternative of running from a wrapper that's already set it.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlvr.compare import generate_plots, load_runs, save_summary, summarize_runs

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grpo-metrics", type=Path, nargs="+", required=True,
        help="One or more grpo_metrics.jsonl files",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+",
        help="Run labels (defaults to filename stems)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/comparison"),
        help="Where to write the plots and summary",
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="Rolling window for the line plots",
    )
    parser.add_argument(
        "--final-window", type=int, default=50,
        help="Trailing window for the 'final' summary stats and bar chart",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib output; write only summary.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    runs = load_runs(args.grpo_metrics, args.labels)
    logger.info("Loaded %d runs: %s", len(runs), [r.label for r in runs])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_runs(runs, window=args.final_window)
    save_summary(summary, args.output_dir / "summary.json")
    logger.info("Wrote summary to %s/summary.json", args.output_dir)

    if not args.no_plots:
        figures = generate_plots(runs, window=args.window)
        for name, fig in figures.items():
            path = args.output_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
