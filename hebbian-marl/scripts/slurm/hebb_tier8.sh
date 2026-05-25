#!/bin/bash
#SBATCH --job-name=hebb-tier8
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier8-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier8-%A_%a.err
#
# Tier 8 — MAPPO + HebbianParallelRunner smoke. Single seed, 1M steps.
#
# Verifies that the new `HebbianParallelRunner` (mappo_hebbian config)
# is a strict no-op wrt canonical MAPPO when all hebbian.* flags are off.
# Must reach a trajectory comparable to sanity_mappo_plain_hard (which
# hit 0.74 on the same map). Any collapse to ~0.01 means the runner
# port has a bug.
#
# Indices:
#   0  sanity_mappo_hebbian_offflags   no-op runner check (hebbian.enabled=False)
#   1  sanity_mappo_hebbian_r          headline mechanism (reward_diffusion=True)
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 8 --list
#
# Submit both:
#   sbatch --array=0-1 scripts/slurm/hebb_tier8.sh
#
# Submit only the no-op sanity (recommended first, before turning on diffusion):
#   sbatch --array=0 scripts/slurm/hebb_tier8.sh
#
# Wall-time estimate: ~20-30 min (same as tier-7 MAPPO runs; parallel
# runner with batch_size_run=10). 4h budget gives ample headroom.
#
# Interpretation:
#   - test_return_mean reaches >=0.4 by 1M → HebbianParallelRunner works;
#     proceed to writing the Hebbian-on variant configs (s, r, rs, ...).
#   - test_return_mean stays <0.05 → bug in the runner port (likely
#     reward routing or info-key stripping). Compare hebbian/* logs and
#     bonds.jsonl trajectory against expectations.

set -euo pipefail

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 8 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier8 task ${SLURM_ARRAY_TASK_ID} done"
