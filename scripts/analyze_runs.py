"""Post-hoc analysis of Five Chambers training runs.

Reads run_metrics/{run_id}/ and produces summary plots and stats.

Usage:
    python scripts/analyze_runs.py --run-dir run_metrics/my_run_id
    python scripts/analyze_runs.py --run-dir run_metrics/my_run_id --out-dir analysis_out/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_hebbian_snapshots(run_dir: Path):
    path = run_dir / "hebbian_snapshots.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_episode_summaries(run_dir: Path):
    ep_dir = run_dir / "episodes"
    if not ep_dir.exists():
        return []
    summaries = []
    for ep_path in sorted(ep_dir.iterdir()):
        summary_file = ep_path / "episode_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summaries.append(json.loads(f.read()))
    return summaries


def plot_W_evolution(snapshots, out_dir: Path):
    """Line plot of each W_ij over episodes."""
    if not snapshots:
        return
    episodes = [s.get("episode", i + 1) for i, s in enumerate(snapshots)]
    W_series = [s["W"] for s in snapshots if s.get("W") is not None]
    if not W_series:
        return

    n = len(W_series[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n):
        for j in range(n):
            if i != j:
                vals = [W[i][j] for W in W_series]
                ax.plot(episodes[:len(vals)], vals, label=f"W[{i}→{j}]", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Hebbian Weight")
    ax.set_title("Hebbian Weight Matrix Evolution")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.savefig(out_dir / "W_evolution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved W_evolution.png")


def plot_cooperation_vs_reward(summaries, out_dir: Path):
    """Scatter of cooperation_score vs total reward."""
    if not summaries:
        return
    coop_scores = []
    total_rewards = []
    for s in summaries:
        cm = s.get("cooperation_metrics", {})
        score = cm.get("cooperation_score")
        rewards = s.get("total_reward_per_agent", {})
        total = sum(rewards.values()) if rewards else None
        if score is not None and total is not None:
            coop_scores.append(score)
            total_rewards.append(total)

    if not coop_scores:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coop_scores, total_rewards, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("Cooperation Score")
    ax.set_ylabel("Total Reward (sum over agents)")
    ax.set_title("Cooperation Score vs. Total Reward")
    fig.savefig(out_dir / "cooperation_vs_reward.png", dpi=150)
    plt.close(fig)
    print(f"  Saved cooperation_vs_reward.png")


def plot_chamber_timings(summaries, out_dir: Path):
    """Distribution of chamber completion times across episodes."""
    if not summaries:
        return
    chambers = ["ch1", "ch2", "ch3", "ch4", "ch5"]
    chamber_steps = {ch: [] for ch in chambers}
    for s in summaries:
        entry_steps = s.get("cooperation_metrics", {}).get("chamber_entry_steps", {})
        for ch in chambers:
            if ch in entry_steps:
                chamber_steps[ch].append(entry_steps[ch])

    data = [chamber_steps[ch] for ch in chambers]
    non_empty = [(ch, d) for ch, d in zip(chambers, data) if d]
    if not non_empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels, vals = zip(*non_empty)
    ax.boxplot(vals, labels=labels)
    ax.set_xlabel("Chamber")
    ax.set_ylabel("Step of first entry")
    ax.set_title("Chamber Entry Timing Distribution")
    fig.savefig(out_dir / "chamber_timings.png", dpi=150)
    plt.close(fig)
    print(f"  Saved chamber_timings.png")


def plot_milestone_rate(summaries, out_dir: Path):
    """How the total milestone count per episode evolves during training."""
    if not summaries:
        return
    episodes = [s.get("episode", i + 1) for i, s in enumerate(summaries)]
    milestone_counts = [
        len(s.get("cooperation_metrics", {}).get("milestone_log", []))
        for s in summaries
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, milestone_counts, marker="o", markersize=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Milestones Fired")
    ax.set_title("Milestone Rate by Episode")
    fig.savefig(out_dir / "milestone_rate_by_episode.png", dpi=150)
    plt.close(fig)
    print(f"  Saved milestone_rate_by_episode.png")


def plot_comm_efficacy_vs_coop(summaries, out_dir: Path):
    """Did communication become more predictive of cooperation over training?"""
    if not summaries:
        return
    comm_eff = []
    coop_scores = []
    for s in summaries:
        cm = s.get("cooperation_metrics", {})
        ce = cm.get("communication_efficacy")
        cs = cm.get("cooperation_score")
        if ce is not None and cs is not None:
            comm_eff.append(ce)
            coop_scores.append(cs)

    if not comm_eff:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(comm_eff, coop_scores, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("Communication Efficacy")
    ax.set_ylabel("Cooperation Score")
    ax.set_title("Comm Efficacy vs. Cooperation Score")
    fig.savefig(out_dir / "comm_efficacy_vs_coop.png", dpi=150)
    plt.close(fig)
    print(f"  Saved comm_efficacy_vs_coop.png")


def write_summary_csv(summaries, snapshots, out_dir: Path):
    """One row per episode with all key metrics."""
    if not summaries:
        return

    # Build W snapshot lookup by episode
    snap_by_ep = {s.get("episode", i + 1): s for i, s in enumerate(snapshots)}

    fieldnames = [
        "episode", "final_step", "cooperation_score", "communication_efficacy",
        "carry_imbalance", "joint_dig_events", "proximity_events",
        "ch4_damage_gini", "ch5_damage_gini", "reward_total",
        "hebbian_mean_weight",
    ]

    rows = []
    for s in summaries:
        ep = s.get("episode", "")
        cm = s.get("cooperation_metrics", {})
        rewards = s.get("total_reward_per_agent", {})
        reward_total = sum(rewards.values()) if rewards else 0.0
        snap = snap_by_ep.get(ep)
        W = snap.get("W") if snap else None
        if W:
            flat = [W[i][j] for i in range(len(W)) for j in range(len(W[0])) if i != j]
            mean_w = float(np.mean(flat)) if flat else 0.0
        else:
            mean_w = 0.0
        rows.append({
            "episode": ep,
            "final_step": s.get("final_step", ""),
            "cooperation_score": cm.get("cooperation_score", ""),
            "communication_efficacy": cm.get("communication_efficacy", ""),
            "carry_imbalance": cm.get("carry_imbalance", ""),
            "joint_dig_events": cm.get("joint_dig_events", ""),
            "proximity_events": cm.get("proximity_events", ""),
            "ch4_damage_gini": cm.get("ch4_damage_gini", ""),
            "ch5_damage_gini": cm.get("ch5_damage_gini", ""),
            "reward_total": reward_total,
            "hebbian_mean_weight": mean_w,
        })

    with open(out_dir / "summary_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved summary_stats.csv ({len(rows)} episodes)")


def main():
    parser = argparse.ArgumentParser(description="Analyze Five Chambers training runs")
    parser.add_argument("--run-dir", required=True,
                        help="Path to run_metrics/{run_id}/ directory")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for plots (default: run-dir/analysis/)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {run_dir}")
    print(f"Output to: {out_dir}")

    snapshots = load_hebbian_snapshots(run_dir)
    summaries = load_episode_summaries(run_dir)

    print(f"  Episodes with summaries: {len(summaries)}")
    print(f"  Hebbian snapshots:       {len(snapshots)}")

    plot_W_evolution(snapshots, out_dir)
    plot_cooperation_vs_reward(summaries, out_dir)
    plot_chamber_timings(summaries, out_dir)
    plot_milestone_rate(summaries, out_dir)
    plot_comm_efficacy_vs_coop(summaries, out_dir)
    write_summary_csv(summaries, snapshots, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
