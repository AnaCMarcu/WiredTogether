#!/bin/bash
#SBATCH --job-name=hebb-tier11
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier11-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier11-%A_%a.err
#
# Tier 11 — same as tier 9 but with force_coop=True. 15 runs total.
# Tests whether forced full-team cooperation surfaces bond
# differentiation that the mixed-level tier 9 env failed to.
#
# Indices 0-4  : mappo_hebbian              (baseline)
# Indices 5-9  : mappo_hebbian_r            (Hebbian-weighted diffusion)
# Indices 10-14: mappo_hebbian_uniform_r    (uniform diffusion control)
#
# Env: Foraging-Comm-10x10-3p-3f-coop-v3 (force_coop=True; level-3 foods
# only; cooperation strongly necessary on every reward event). The
# `-coop-` id is enabled by the wrapper extension landed in this commit.
#
# Pre-registered tests (paired Wilcoxon on final-window means):
#   A) mappo_hebbian_r > mappo_hebbian
#   B) mappo_hebbian_r > mappo_hebbian_uniform_r
#
# Plus bond-structure descriptive check vs. tier-9 trajectories.
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 11 --list
#
# Submit all 15 in parallel:
#   sbatch --array=0-14 scripts/slurm/hebb_tier11.sh
#
# Wall-time estimate: tier 9 hit ~50 min for 3M steps on the same env
# size + agent count. Forced cooperation should not materially change
# per-step cost (same env mechanics). 6h budget matches tier 9; ample
# headroom for slow nodes.
#
# Resumability: re-submitting an array task whose run already finished
# successfully is a fast no-op (the launcher checks runs.jsonl).

set -euo pipefail

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 11 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier11 task ${SLURM_ARRAY_TASK_ID} done"
