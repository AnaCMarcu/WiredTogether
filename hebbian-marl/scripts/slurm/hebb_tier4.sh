#!/bin/bash
#SBATCH --job-name=hebb-tier4
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier4-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier4-%A_%a.err
#
# Tier 4 — failure-grace mechanism, full 5-seed grid.
# 10 runs total: hebb_s_grace × 5 seeds + hebb_rs_grace × 5 seeds, each 5M steps.
# Designed to run in parallel with tier 1 (different variant names → no
# sacred/bonds collision).
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 4 --list
#
# Submit:
#   sbatch --array=0-9 scripts/slurm/hebb_tier4.sh
#
# Wall-time estimate: 3M steps at compute-p1's observed ~48 steps/sec
# is ~17.4h per task. 24h budget gives ~6h headroom. Reduced from the
# original 5M plan to match tier 1.

set -euo pipefail   # fail fast: source / env / cd errors halt the job instead of silently continuing into a broken python invocation.

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 4 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier4 task ${SLURM_ARRAY_TASK_ID} done"
