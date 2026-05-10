"""Plot per-pair asymmetric Hebbian bond evolution from a saved metrics.json.

Standalone replot tool. Use when you want to regenerate the
graph_bond_asymmetry.png figure for an existing run without re-running
the full trainer.

Usage:
    python scripts/analysis/plot_bond_asymmetry.py <metrics.json> [output.png]

Examples:
    python scripts/analysis/plot_bond_asymmetry.py runs/hebbian_long_*/metrics.json
    python scripts/analysis/plot_bond_asymmetry.py runs/foo/metrics.json out/figs/bonds.png

The script requires the FULL W matrix in each graph snapshot. Runs from
before the get_graph_metrics() fix only stored top_3_pairs, which can't
be plotted as per-pair time-series (pairs that fall out of top-3 mid-
episode show as 0). Re-run training with the patched code to generate
compatible data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot_asymmetric_bonds(metrics_path: Path, output_path: Path) -> bool:
    """Render the per-pair asymmetric bond figure. Returns True on success."""
    with open(metrics_path) as f:
        data = json.load(f)

    snaps = data.get("graph_snapshots", [])
    if not snaps:
        print(f"ERROR: no graph_snapshots in {metrics_path}")
        return False

    first_W = snaps[0].get("W")
    if not first_W:
        print(
            f"ERROR: snapshots in {metrics_path} don't include the full W "
            f"matrix. Re-run training with the patched get_graph_metrics() "
            f"in src/hebbian/graph.py to produce compatible snapshots."
        )
        return False

    N = len(first_W)
    steps = [s["step"] for s in snaps]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if not pairs:
        print(f"ERROR: only {N} agent(s); need at least 2 for pairs.")
        return False

    fig, axes = plt.subplots(
        len(pairs), 1, figsize=(8, 2.4 * len(pairs)),
        sharex=True, constrained_layout=True,
    )
    if len(pairs) == 1:
        axes = [axes]

    c_forward = "#2c7fb8"
    c_reverse = "#d95f0e"
    c_asym    = "#888888"

    for ax, (i, j) in zip(axes, pairs):
        w_ij = [s["W"][i][j] for s in snaps]
        w_ji = [s["W"][j][i] for s in snaps]

        ax.plot(steps, w_ij, color=c_forward, linewidth=2,
                label=f"agent_{i} → agent_{j}")
        ax.plot(steps, w_ji, color=c_reverse, linewidth=2,
                label=f"agent_{j} → agent_{i}")

        lo = [min(a, b) for a, b in zip(w_ij, w_ji)]
        hi = [max(a, b) for a, b in zip(w_ij, w_ji)]
        ax.fill_between(steps, lo, hi, color=c_asym, alpha=0.18,
                        label="|W_ij − W_ji|")

        # End-of-episode values annotated on the right edge.
        ax.text(steps[-1], w_ij[-1], f" {w_ij[-1]:.2f}",
                color=c_forward, fontsize=9, va="center")
        ax.text(steps[-1], w_ji[-1], f" {w_ji[-1]:.2f}",
                color=c_reverse, fontsize=9, va="center")

        ax.set_ylabel("Bond W")
        ax.set_ylim(0, 1)
        ax.set_title(f"Pair (agent_{i}, agent_{j})", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Asymmetric Hebbian Bonds Over One Episode",
                 fontsize=13, y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    metrics_path = Path(sys.argv[1])
    if not metrics_path.exists():
        print(f"ERROR: file not found: {metrics_path}")
        return 1
    output_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else metrics_path.parent / "graph_bond_asymmetry.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return 0 if plot_asymmetric_bonds(metrics_path, output_path) else 1


if __name__ == "__main__":
    sys.exit(main())
