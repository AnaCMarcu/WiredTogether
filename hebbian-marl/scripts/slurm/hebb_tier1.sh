#!/bin/bash
#SBATCH --job-name=hebb-tier1
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier1-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier1-%A_%a.err
#
# Tier 1 — headline grid. 15 single-seed runs at 5M env steps each.
# Pre-registered Wilcoxon comparisons:
#   - hebb_s vs seac                  (Hebbian weights beat uniform sharing?)
#   - hebb_s vs hebb_s_nocomm         (does the comm term carry signal?)
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 1 --list
#
# Submit:
#   sbatch --array=0-14 scripts/slurm/hebb_tier1.sh
#
# Submit a subset (e.g., only the hebb_s seeds):
#   sbatch --array=0-4 scripts/slurm/hebb_tier1.sh
#
# Wall-time estimate: ~2h per task (5M steps, single-threaded CPU).
# 4h budget leaves headroom for slow nodes + sacred/init overhead.
#
# Resumability: re-submitting an array task whose run already finished
# successfully is a fast no-op (the launcher checks runs.jsonl).

set -euo pipefail   # fail fast: source / env / cd errors halt the job instead of silently continuing into a broken python invocation.

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 1 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier1 task ${SLURM_ARRAY_TASK_ID} done"
