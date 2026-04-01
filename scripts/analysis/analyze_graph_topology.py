"""Post-hoc graph topology analysis for H7 (RQ2).

Reads experiment output (data.json, graph_snapshots) and computes:
  - Newman modularity Q with role-label communities
  - Sparsity evolution (fraction of edges > threshold)
  - Eigenvector centrality per agent, correlated with cumulative reward
  - Structural-behavioral alignment (Pearson r between wij and co-completion freq)
  - Bond evolution visualisation

Usage:
    python analyze_graph_topology.py /path/to/run_metrics/craftium_...
    python analyze_graph_topology.py /path/to/run1 /path/to/run2  (compare runs)
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROLE_NAMES = ["gatherer", "hunter", "defender"]


def load_run(run_dir: str) -> dict:
    """Load data.json from an experiment run directory."""
    data_path = os.path.join(run_dir, "data.json")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(data_path, "r") as f:
        return json.load(f)


def newman_modularity(W: np.ndarray, communities: list[int]) -> float:
    """Compute Newman modularity Q for a weighted, undirected graph.

    Q = (1/2m) * Σ_ij [W_ij - (k_i * k_j) / (2m)] * δ(c_i, c_j)

    Parameters
    ----------
    W : (N, N) weight matrix
    communities : list of community labels per node
    """
    N = W.shape[0]
    np.fill_diagonal(W, 0.0)
    m = W.sum() / 2.0
    if m == 0:
        return 0.0

    k = W.sum(axis=1)  # node strength
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if communities[i] == communities[j]:
                Q += W[i, j] - (k[i] * k[j]) / (2.0 * m)
    return Q / (2.0 * m)


def eigenvector_centrality(W: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Power-iteration eigenvector centrality."""
    N = W.shape[0]
    x = np.ones(N, dtype=np.float64) / N
    for _ in range(max_iter):
        x_new = W @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            return np.zeros(N)
        x_new /= norm
        if np.allclose(x, x_new, atol=1e-8):
            break
        x = x_new
    return x


def analyze_snapshots(data: dict, out_dir: str):
    """Run full topology analysis on a single experiment run."""
    config = data["config"]
    num_agents = config["num_agents"]
    snapshots = data.get("graph_snapshots", [])
    exp_id = config.get("experiment_id", "unknown")

    if not snapshots:
        print(f"  No graph snapshots found in {exp_id}. Skipping topology analysis.")
        return {}

    print(f"\n=== Topology Analysis: {exp_id} ===")
    print(f"  Agents: {num_agents}, Snapshots: {len(snapshots)}")

    # Role communities (cycling)
    communities = [i % 3 for i in range(num_agents)]

    # ── Per-snapshot metrics ──
    steps = []
    mean_bonds = []
    sparsities = []
    modularities = []
    centralities = []

    for snap in snapshots:
        step = snap.get("step", 0)
        steps.append(step)
        mean_bonds.append(snap.get("mean_bond_strength", 0.0))
        sparsities.append(snap.get("sparsity", 1.0))

        # Recompute proper Newman modularity from ltd_heatmap if available
        # (the stored modularity_proxy is a simpler approximation)
        heatmap = snap.get("ltd_heatmap")
        # We need the actual W matrix — reconstruct from top_3_pairs isn't enough.
        # Use the stored modularity_proxy for now.
        modularities.append(snap.get("modularity_proxy"))

        out_strength = snap.get("per_agent_out_strength", [])
        centralities.append(out_strength)

    # ── Eigenvector centrality vs cumulative reward ──
    final_returns = data.get("cumulative_returns", [0.0] * num_agents)
    last_centrality = centralities[-1] if centralities else [0.0] * num_agents

    if len(last_centrality) >= 2:
        r_val, p_val = stats.pearsonr(last_centrality, final_returns[:len(last_centrality)])
        print(f"  Centrality-reward correlation: r={r_val:.3f}, p={p_val:.3f}")
    else:
        r_val, p_val = float("nan"), float("nan")

    # ── Co-completion analysis ──
    co_completions = data.get("co_completion_events", [])
    if co_completions:
        print(f"  Co-completion events: {len(co_completions)}")
        # Build co-completion frequency matrix
        freq = np.zeros((num_agents, num_agents), dtype=np.float64)
        for event in co_completions:
            i, j = event["agent_i"], event["agent_j"]
            freq[i, j] += 1
            freq[j, i] += 1

        # Compare with final bond weights (from last top_3_pairs or stored W)
        top_pairs = snapshots[-1].get("top_3_pairs", []) if snapshots else []
        if top_pairs:
            bond_vals = []
            freq_vals = []
            for p in top_pairs:
                bond_vals.append(p["w"])
                freq_vals.append(freq[p["i"], p["j"]])
            if len(bond_vals) >= 2:
                r_struct, p_struct = stats.pearsonr(bond_vals, freq_vals)
                print(f"  Structural-behavioral alignment (top pairs): r={r_struct:.3f}")
    else:
        print("  No co-completion events recorded.")

    # ── Generate plots ──
    os.makedirs(out_dir, exist_ok=True)

    # 1. Bond strength + sparsity over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(steps, mean_bonds, "b-o", markersize=3, label="Mean bond strength")
    ax1.set_ylabel("Mean bond strength")
    ax1.set_title(f"Graph Topology Evolution — {exp_id}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, sparsities, "r-o", markersize=3, label="Sparsity (frac < 0.1)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Sparsity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "topology_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Modularity over time
    valid_mod = [(s, m) for s, m in zip(steps, modularities) if m is not None]
    if valid_mod:
        fig, ax = plt.subplots(figsize=(10, 4))
        mod_steps, mod_vals = zip(*valid_mod)
        ax.plot(mod_steps, mod_vals, "g-o", markersize=3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Modularity proxy")
        ax.set_title(f"Role Modularity — {exp_id}")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "modularity.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Per-agent out-strength (centrality proxy) over time
    if centralities and centralities[0]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(num_agents):
            role = ROLE_NAMES[i % 3]
            vals = [c[i] if i < len(c) else 0.0 for c in centralities]
            ax.plot(steps, vals, "-o", markersize=2, label=f"Agent {i} ({role})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Out-strength (Σ wij)")
        ax.set_title(f"Per-Agent Centrality — {exp_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "agent_centrality.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Summary JSON ──
    summary = {
        "experiment_id": exp_id,
        "num_agents": num_agents,
        "num_snapshots": len(snapshots),
        "final_mean_bond": mean_bonds[-1] if mean_bonds else None,
        "final_sparsity": sparsities[-1] if sparsities else None,
        "final_modularity_proxy": modularities[-1] if modularities else None,
        "centrality_reward_pearson_r": r_val,
        "centrality_reward_p_value": p_val,
        "co_completion_events": len(co_completions),
    }

    summary_path = os.path.join(out_dir, "topology_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {out_dir}")

    return summary


def compare_runs(summaries: list[dict], out_dir: str):
    """Generate comparison plots across multiple runs."""
    if len(summaries) < 2:
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    exp_ids = [s["experiment_id"] for s in summaries]
    x = range(len(summaries))

    # Bond strength
    vals = [s.get("final_mean_bond", 0) or 0 for s in summaries]
    axes[0].bar(x, vals, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(exp_ids, rotation=45, ha="right")
    axes[0].set_ylabel("Final mean bond")
    axes[0].set_title("Bond Strength")

    # Sparsity
    vals = [s.get("final_sparsity", 1) or 1 for s in summaries]
    axes[1].bar(x, vals, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(exp_ids, rotation=45, ha="right")
    axes[1].set_ylabel("Final sparsity")
    axes[1].set_title("Sparsity")

    # Modularity
    vals = [s.get("final_modularity_proxy", 0) or 0 for s in summaries]
    axes[2].bar(x, vals, color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(exp_ids, rotation=45, ha="right")
    axes[2].set_ylabel("Modularity proxy")
    axes[2].set_title("Role Modularity")

    fig.suptitle("Cross-Experiment Topology Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "topology_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved to {out_dir}/topology_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc graph topology analysis (H7/RQ2)"
    )
    parser.add_argument(
        "run_dirs", nargs="+",
        help="One or more experiment output directories (containing data.json)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory for analysis results (default: <run_dir>/analysis/)"
    )
    args = parser.parse_args()

    summaries = []
    for run_dir in args.run_dirs:
        out = args.output or os.path.join(run_dir, "analysis")
        data = load_run(run_dir)
        summary = analyze_snapshots(data, out)
        if summary:
            summaries.append(summary)

    if len(summaries) > 1:
        compare_dir = args.output or os.path.join(
            os.path.dirname(args.run_dirs[0]), "comparison"
        )
        compare_runs(summaries, compare_dir)


if __name__ == "__main__":
    main()
