#!/bin/bash
#SBATCH --job-name=hebb-tier10
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier10-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier10-%A_%a.err
#
# Tier 10 — 3-variant × 5-seed comparison on Foraging-Comm-15x15-3p-5f-v3.
# The asymmetric-task test of the refined thesis claim post-tier-9.
#
# Indices 0-4  : mappo_hebbian              (baseline)
# Indices 5-9  : mappo_hebbian_r            (Hebbian-weighted diffusion)
# Indices 10-14: mappo_hebbian_uniform_r    (uniform diffusion control)
#
# Pre-registered tests (paired Wilcoxon on final-window means):
#   A) mappo_hebbian_r > mappo_hebbian
#   B) mappo_hebbian_r > mappo_hebbian_uniform_r
#
# Plus bond-structure descriptive check vs. tier-9 trajectories:
#   - Do final W matrices show non-uniform off-diagonal entries?
#   - Is asymmetry above the tier-9 floor (~0.001 frob)?
#   - Does sparsity stay below 1.0?
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 10 --list
#
# Submit all 15 in parallel:
#   sbatch --array=0-14 scripts/slurm/hebb_tier10.sh
#
# Wall-time estimate: 15x15 is ~2.25x the area of 10x10 with 5 vs. 3
# foods → episodes likely take longer to find rewards, eval rollouts
# are more expensive. Budget 8h vs. tier-9's 6h to be safe. Empirical
# tier-9 wall-time on the smaller map was ~50min for 3M steps; tier-10
# may run 90-150min. 8h gives ~3x headroom.
#
# Note: 3M may not be sufficient for the canonical EPyMARL benchmark
# number on this map (paper uses ~20M). 3M is a first-look budget for
# the bond-differentiation hypothesis. If bonds DO differentiate at 3M
# but policy hasn't fully converged, extend the budget.
#
# Resumability: re-submitting an array task whose run already finished
# successfully is a fast no-op (the launcher checks runs.jsonl).

set -euo pipefail

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 10 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier10 task ${SLURM_ARRAY_TASK_ID} done"
