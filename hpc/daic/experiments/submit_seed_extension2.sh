#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_seed_extension2.sh — second 6-seed extension of the MAIN-TABLE suite.
#
# Pre-registered 2026-08-18: seeds 1415, 1617, 1819, 2021, 2223, 2425 added to
# the ten tab:final_comparison conditions — the eight Qwen medium conditions
# (exp01-exp08) and both Gemma arms (new_exp_0_gemma, HEBBIAN=0/1). Every run
# that completes pools with the existing seeds (42..1213) — no seed selection.
# The topology arms (exp09-11) are NOT part of the main table and are not
# extended here.
#
# Qwen arms reproduce the medium-suite configuration exactly (cf.
# submit_seed_extension.sh, which this extends):
#   RUN_GROUP=legacy, EPISODES=3, MAX_STEPS=1000,
#   WANDB_PROJECT=medium_wired_together, MODEL_2B/9B pinned to Qwen3.5
#   (the _common.sh default is now Gemma 4, which the medium runs never
#   used; the old PEFT-vs-Gemma4ClippableLinear crash on RL arms is fixed
#   by rl_layer/peft_compat.py, so the pin is purely for reproducibility).
#
# Gemma arms go through new_exp_0_gemma.sbatch, whose own defaults already
# match its suite (runs/new_exp_0_gemma/, 3x1000, Gemma image, vision on).
# NOTE: that file requests --qos=medium/36h; slow seeds can time out (seed_123
# needed ~50h). Resubmit stragglers with:
#     SEED=<s> HEBBIAN=<h> sbatch --qos=long --time=168:00:00 new_exp_0_gemma.sbatch
#
# Idempotent: skips any exp/seed whose final_metrics.json already exists on
# PRB, so re-running only submits what is missing.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   DRY_RUN=1 bash submit_seed_extension2.sh   # print what would be submitted
#   bash submit_seed_extension2.sh             # submit everything missing
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

# Keep jobs off GPUs too small to hold the model (belt); _common.sh's
# MIN_GPU_MEM_MIB preflight aborts at startup if one slips through (braces).
source "$(dirname "$0")/gpu_filter.sh"
echo "GPU filter: --exclude=$GPU_EXCLUDE${GPU_CONSTRAINT:+ --constraint=$GPU_CONSTRAINT}"

REPO=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether
WORKSPACE_MODELS=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/models

SEEDS=(1415 1617 1819 2021 2223 2425)

n_queued=0
n_skipped=0

# ── Qwen medium conditions (exp01-exp08) ──────────────────────────────────
QWEN_EXPS=(
    exp01_llm_2b
    exp02_llm_9b
    exp03_mappo
    exp04_ippo
    exp05_mappo_hebbian
    exp06_ippo_hebbian
    exp07_llm_2b_social_prompt
    exp08_llm_9b_social_prompt
)

for exp in "${QWEN_EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ -f "$REPO/runs/legacy/$exp/seed_$seed/final_metrics.json" ]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $exp  seed_$seed"
        else
            SEED=$seed \
            RUN_GROUP=legacy EPISODES=3 MAX_STEPS=1000 \
            WANDB_PROJECT=medium_wired_together \
            MODEL_2B="$WORKSPACE_MODELS/Qwen3.5-2B" \
            MODEL_9B="$WORKSPACE_MODELS/Qwen3.5-9B" \
            sbatch ${GPU_FILTER_FLAGS[@]:+"${GPU_FILTER_FLAGS[@]}"} \
                "$exp.sbatch"
            echo "queued  $exp  seed_$seed"
        fi
        n_queued=$((n_queued + 1))
    done
done

# ── Gemma arms (new_exp_0_gemma, both HEBBIAN values) ─────────────────────
for h in 0 1; do
    arm=$([ "$h" = "1" ] && echo new_exp_0_gemma_hebbian || echo new_exp_0_gemma_base)
    for seed in "${SEEDS[@]}"; do
        if [ -f "$REPO/runs/new_exp_0_gemma/$arm/seed_$seed/final_metrics.json" ]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $arm  seed_$seed"
        else
            SEED=$seed HEBBIAN=$h \
            sbatch ${GPU_FILTER_FLAGS[@]:+"${GPU_FILTER_FLAGS[@]}"} \
                new_exp_0_gemma.sbatch
            echo "queued  $arm  seed_$seed"
        fi
        n_queued=$((n_queued + 1))
    done
done

echo "── done: $n_queued submitted, $n_skipped already complete ──"
echo "Track with:   squeue -u \$USER"
echo "Cancel all:   scancel -u \$USER"
