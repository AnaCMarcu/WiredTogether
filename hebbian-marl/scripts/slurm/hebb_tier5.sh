#!/bin/bash
#SBATCH --job-name=hebb-tier5
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier5-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier5-%A_%a.err
#
# Tier 5 — preview pass. 5 single-seed runs at 1M env steps each.
# Variants: hebb_s, hebb_r, hebb_rs, seac, ippo_baseline.
# Use this to establish relative ordering BEFORE committing tier 1.
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 5 --list
#
# Submit:
#   sbatch --array=0-4 scripts/slurm/hebb_tier5.sh
#
# Wall-time estimate: ~25 min per task (1M steps).

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 5 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier5 task ${SLURM_ARRAY_TASK_ID} done"
