#!/bin/bash
#SBATCH --job-name=hebb-tier0
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier0-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier0-%A_%a.err
#
# Tier 0 — diagnostic runs. 5 single-seed 500k-step probes that test the
# three competing interpretations of bond dynamics on LBF (HEBBIAN_MARL_PLAN
# section 1.3). Useful on HPC only if you haven't already run them locally.
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 0 --list
#
# Submit:
#   sbatch --array=0-4 scripts/slurm/hebb_tier0.sh
#
# Wall-time estimate: ~15 min per task (500k steps).

source "$(dirname "$0")/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 0 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier0 task ${SLURM_ARRAY_TASK_ID} done"
