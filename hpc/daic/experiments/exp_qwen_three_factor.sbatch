#!/bin/bash
#SBATCH --job-name=qwen-three-factor
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/qwen_three_factor_%j.out
#SBATCH --error=slurm_logs/qwen_three_factor_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# exp_qwen_three_factor — Qwen3.5 (2B or 9B) + the three-factor Hebbian rule.
#
# ONE file, TWO backbones, selected by QWEN:
#   QWEN=2b → exp32_llm_2b_three_factor   (Qwen3.5-2B)
#   QWEN=9b → exp33_llm_9b_three_factor   (Qwen3.5-9B)
#
# Same medium-suite shape as exp07/exp08 (3 eps × 1000 steps, vision auto,
# prompt social module, RUN_GROUP=medium_runs so the runs pool with the Qwen
# lane), differing from exp07/exp08 ONLY in the update rule — the three-factor
# parameters mirror the Gemma arm (new_exp_0_gemma_three_factor.sbatch):
#   η0 0.001, λ 0.001, ρ 0.9, coact floor 0.25, R 50, η+ 0.05 (default).
# Original Qwen image (default WT_IMAGE), Qwen model pins set explicitly
# because _common.sh's MODEL_LLM now defaults to Gemma-4-E4B.
#
# qos=long/72 h: the 9B vision runs are the slowest in the suite; medium's
# 36 h wall would risk a TIMEOUT kill with no resumable state.
#
# Both backbones, three seeds (6 jobs):
#   cd $REPO/hpc/daic/experiments
#   for s in 42 123 456; do for q in 2b 9b; do
#       SEED=$s QWEN=$q sbatch exp_qwen_three_factor.sbatch
#   done; done
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

: "${RUN_GROUP:=medium_runs}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=qwen_three_factor}"
export RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

QWEN="${QWEN:-9b}"
if [ "$QWEN" = "2b" ]; then
    EXP_NAME="exp32_llm_2b_three_factor"
    MODEL="${MODEL_2B_PIN:-$WORKSPACE/models/Qwen3.5-2B}"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,2b,hebbian,three_factor,prompt,medium}"
else
    EXP_NAME="exp33_llm_9b_three_factor"
    MODEL="${MODEL_9B_PIN:-$WORKSPACE/models/Qwen3.5-9B}"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,three_factor,prompt,medium}"
fi

if [ ! -d "$MODEL" ]; then
    echo "FATAL: model path not found: $MODEL" >&2
    echo "       set MODEL_2B_PIN / MODEL_9B_PIN to the correct Qwen dir" >&2
    exit 1
fi

echo "── exp_qwen_three_factor: QWEN=$QWEN → $EXP_NAME ($MODEL, seed=${SEED:-unset}) ──"

run_exp "$EXP_NAME" "$MODEL" \
    --hebbian \
    --hebbian-mode three_factor \
    --hebbian-eta-0 0.001 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.001 \
    --hebbian-eligibility-rho 0.9 \
    --hebbian-coact-floor 0.25 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --simultaneous
