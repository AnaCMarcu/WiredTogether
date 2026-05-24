#!/bin/bash
#SBATCH --job-name=exp11-llm-9b-social-prompt
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp11_llm_9b_social_prompt-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp11_llm_9b_social_prompt-%j.out
# Submit:  sbatch scripts/experiments/exp11_llm_9b_social_prompt.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# 9B LLM + Hebbian + SocialModule (prompt coupling).
# Companion to exp9 at the 9B scale — tests whether a larger LLM honors
# the deliberated directive in text-only form, or whether it ignores soft
# prompt hints the way the 2B model might.

EXP_NAME="exp11_llm_9b_social_prompt"
SEED=42
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_9B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 1500 \
    --warmup-time 300 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"