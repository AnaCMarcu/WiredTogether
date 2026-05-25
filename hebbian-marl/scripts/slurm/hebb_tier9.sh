#!/bin/bash
#SBATCH --job-name=hebb-tier9
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier9-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier9-%A_%a.err
#
# Tier 9 — headline 3-variant × 5-seed comparison.
#
#   Indices 0-4  : mappo_hebbian              (baseline, no diffusion)
#   Indices 5-9  : mappo_hebbian_r            (Hebbian-weighted diffusion)
#   Indices 10-14: mappo_hebbian_uniform_r    (uniform diffusion control)
#
# All three on Foraging-Comm-10x10-3p-3f-v3, T=50, 3M env steps.
# Same seeds [0..4] across all three variants → paired Wilcoxon
# applies on the final-window means.
#
# Pre-registered tests (run by scripts/analysis/plot_results.py or
# equivalent after the sweep lands):
#   A) mappo_hebbian_r > mappo_hebbian          (does diffusion help?)
#   B) mappo_hebbian_r > mappo_hebbian_uniform_r (Hebbian beats uniform?)
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 9 --list
#
# Submit all 15 in parallel:
#   sbatch --array=0-14 scripts/slurm/hebb_tier9.sh
#
# Submit only the baseline variant (indices 0-4):
#   sbatch --array=0-4 scripts/slurm/hebb_tier9.sh
#
# Wall-time estimate: MAPPO's parallel runner with batch_size_run=10
# reaches 1M steps in ~17 min on compute-p1, so 3M ≈ 50–60 min per task.
# 6h SLURM budget gives ample headroom for slow nodes, sacred/init, and
# the Hebbian update overhead (single shared graph, ~negligible vs.
# rollout cost).
#
# Resumability: re-submitting an array task whose run already finished
# successfully is a fast no-op (the launcher checks runs.jsonl).

set -euo pipefail

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 9 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier9 task ${SLURM_ARRAY_TASK_ID} done"
