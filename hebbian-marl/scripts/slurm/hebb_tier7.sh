#!/bin/bash
#SBATCH --job-name=hebb-tier7
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/hebb-tier7-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/hebb-tier7-%A_%a.err
#
# Tier 7 — canonical EPyMARL MAPPO sanity. 3 single-seed runs at 1M env
# steps each. Same code path as the EPyMARL paper / verify_baseline.sh,
# just at a real training budget.
#
# Indices (see scripts/experiments.yaml tier 7 block for rationale):
#   0  sanity_mappo_plain_easy   canonical reference, 8x8-2p-2f-coop
#   1  sanity_mappo_plain_hard   tier-1 map without comm wrapper
#   2  sanity_mappo_comm         tier-1 map with comm wrapper
#
# Verify indices first:
#   python scripts/run_experiments.py --tier 7 --list
#
# Submit all three:
#   sbatch --array=0-2 scripts/slurm/hebb_tier7.sh
#
# Submit only the canonical reference:
#   sbatch --array=0 scripts/slurm/hebb_tier7.sh
#
# Wall-time estimate: mappo uses the `parallel` runner with batch_size_run=10,
# so 1M env steps should take ~1-2h on compute-p1 (vs. ~6h for ippo_baseline
# which uses HebbianRunner with batch_size_run=1). 4h budget gives ample
# headroom for parallel-env overhead and sacred/init.
#
# CPU count bumped from 2 -> 4 because the parallel runner spawns 10 env
# workers; 4 cores keeps OMP/MKL pinned and avoids oversubscription.
#
# Resumability: re-submitting a task whose run already finished
# successfully is a fast no-op (the launcher checks runs.jsonl).
#
# Interpretation guide (read AFTER all three finish):
#   - test_return_mean trend in
#     /scratch/$USER/WiredTogether/runs/hebbian-marl/logs/sanity_mappo_*.log
#   - sanity_mappo_plain_easy fails  -> this fork's epymarl is broken at
#                                       the framework level. Investigate
#                                       before running any other tier.
#   - 7.0 OK, 7.1 fails              -> 10x10-3p-3f-v3 needs more steps
#                                       or shaping; not a code issue.
#   - 7.0+7.1 OK, 7.2 fails          -> the lbf_comm wrapper (or its
#                                       common_reward=False setup) is
#                                       what kills tier-1 runs. Investigate
#                                       reward routing in the wrapper.
#   - all three reach >0.3 final     -> the bug is in ippo_baseline.yaml's
#                                       runner: hebbian codepath; tier 1
#                                       needs to be re-run with mappo or
#                                       a fixed HebbianRunner.

set -euo pipefail

source "/scratch/acmarcu/WiredTogether/hebbian-marl/scripts/slurm/_common_hebbmarl.sh"

python scripts/run_experiments.py \
    --tier 7 \
    --single-index "${SLURM_ARRAY_TASK_ID}" \
    --parallel 1

echo "hebb-tier7 task ${SLURM_ARRAY_TASK_ID} done"
