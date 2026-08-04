#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_seed_extension.sh — 3-seed extension of the MEDIUM suite.
#
# Pre-registered 2026-07-07: seeds 789, 1011, 1213 added to all 11 medium
# conditions, plus a backfill of the one lost run (exp07 / seed_42). Every
# run that completes goes into the paper's pooled analysis alongside the
# original seeds (42, 123, 456) — no seed selection.
#
# The env overrides below reproduce the medium-suite configuration exactly
# (the sbatch files default to the FINAL suite: 5 eps × 2500 steps,
# runs/final/). Do not change them, or the new runs cannot be pooled with
# the existing medium data:
#   RUN_GROUP=legacy            → results in runs/legacy/<exp>/seed_<N>/
#   EPISODES=3, MAX_STEPS=1000  → 3 episodes × ≤1000 steps
#   WANDB_PROJECT=medium_wired_together
#
# Idempotent: skips any exp/seed whose runs/legacy/.../final_metrics.json
# already exists on PRB, so re-running the script only submits what is
# missing (this is also what makes the exp07/seed_42 backfill automatic —
# seed 42 is in the loop and only its missing run gets queued).
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   bash submit_seed_extension.sh          # submit everything missing
#   DRY_RUN=1 bash submit_seed_extension.sh  # print what would be submitted
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

# Keep jobs off GPUs too small to hold the model — the failure mode that
# produced six 99.8%-NoOp RL runs (see bad_gpu_nodes.txt). Adds --exclude for
# every known-bad node, plus --constraint when GPU_CONSTRAINT is set.
source "$(dirname "$0")/gpu_filter.sh"
echo "GPU filter: --exclude=$GPU_EXCLUDE${GPU_CONSTRAINT:+ --constraint=$GPU_CONSTRAINT}"

REPO=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether

export RUN_GROUP=legacy
export EPISODES=3
export MAX_STEPS=1000
export WANDB_PROJECT=medium_wired_together

EXPS=(
    exp01_llm_2b
    exp02_llm_9b
    exp03_mappo
    exp04_ippo
    exp05_mappo_hebbian
    exp06_ippo_hebbian
    exp07_llm_2b_social_prompt
    exp08_llm_9b_social_prompt
    exp09_llm_9b_allied_all
    exp10_llm_9b_allied_pair
    exp11_llm_9b_allied_none
)
SEEDS=(42 123 456 789 1011 1213)

n_queued=0
n_skipped=0
for exp in "${EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ -f "$REPO/runs/legacy/$exp/seed_$seed/final_metrics.json" ]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $exp  seed_$seed  (${GPU_FILTER_FLAGS[*]})"
        else
            SEED=$seed sbatch \
                ${GPU_FILTER_FLAGS[@]:+"${GPU_FILTER_FLAGS[@]}"} \
                "$exp.sbatch"
            echo "queued  $exp  seed_$seed"
        fi
        n_queued=$((n_queued + 1))
    done
done
echo "── done: $n_queued submitted, $n_skipped already complete ──"
