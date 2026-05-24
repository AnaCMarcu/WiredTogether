#!/bin/bash
#SBATCH --job-name=hebb-tier6
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier6-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier6-%A_%a.err
#
# Tier 6 — sanity / pipeline diagnostic. 3 single-seed runs at 1M env
# steps each. Goal: localise WHY tier 1 collapsed before re-launching
# any multi-seed grid.
#
# Indices (see scripts/experiments.yaml tier 6 block for rationale):
#   0  sanity_plain_easy     plain lbforaging, no comm wrapper, easy map
#   1  sanity_comm_easy      lbf_comm wrapper on the easy map
#   2  sanity_horizon        original failed env with time_limit=100
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 6 --list
#
# Submit all three:
#   sbatch --array=0-2 scripts/slurm/hebb_tier6.sh
#
# Submit a single index (e.g. just the plain-LBF reference):
#   sbatch --array=0 scripts/slurm/hebb_tier6.sh
#
# Wall-time estimate: 1M steps at ~48 steps/sec ≈ 5.8h per task.
# 8h budget gives ~2h headroom for slow nodes and sacred/init overhead.
#
# Resumability: re-submitting a task whose run already finished
# successfully is a fast no-op (the launcher checks runs.jsonl).
#
# Interpretation guide (read AFTER all three finish):
#   - test_total_return_mean trend in
#     /scratch/$USER/WiredTogether/runs/hebbian-marl/logs/sanity_*.log
#   - sanity_plain_easy fails  -> pipeline-level bug (sacred / torch / wrapper).
#                                 Do not re-run tier 1 until fixed.
#   - sanity_plain_easy OK, sanity_comm_easy fails  -> comm wrapper bug
#                                 (likely the `<class 'list'>` reward warning).
#   - both pass, sanity_horizon fails  -> 10x10-3p-3f-v3 needs >50-step
#                                 horizon and/or shaping. Bump time_limit
#                                 in defaults before re-running tier 1.
#   - all three reach >0.3 final-window return  -> tier 1 config simply
#                                 underspecified; longer t_max or different
#                                 LBF map needed.

set -euo pipefail

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 6 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier6 task ${SLURM_ARRAY_TASK_ID} done"
