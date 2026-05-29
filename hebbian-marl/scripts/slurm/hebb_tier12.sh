#!/bin/bash
#SBATCH --job-name=hebb-tier12
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier12-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier12-%A_%a.err
#
# Tier 12 — smallest 3-agent forced-coop LBF variant.
# 15 runs (3 variants × 5 seeds) on Foraging-Comm-8x8-3p-2f-coop-v3.
#
# Tier 11 confirmed 10x10-3p-3f-coop is unlearnable at 3M (15/15 → 0).
# Tier 12 tests whether dropping to 8x8 with only 2 foods is small
# enough for cooperative-loading events to occur from random
# exploration, bootstrapping learning.
#
# Indices 0-4  : mappo_hebbian              (baseline)
# Indices 5-9  : mappo_hebbian_r            (Hebbian-weighted diffusion)
# Indices 10-14: mappo_hebbian_uniform_r    (uniform diffusion control)
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 12 --list
#
# Submit all 15 in parallel:
#   sbatch --array=0-14 scripts/slurm/hebb_tier12.sh
#
# Wall-time estimate: 8x8 is much smaller than 10x10. Tier 11 hit
# ~1h for 3M steps. Tier 12 should be similar or faster. 4h budget
# gives ~4x headroom.

set -euo pipefail

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 12 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier12 task ${SLURM_ARRAY_TASK_ID} done"
