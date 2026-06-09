#!/bin/bash
#SBATCH --job-name=exp10-llm-2b-social-bias
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp10_llm_2b_social_bias-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp10_llm_2b_social_bias-%j.out
# Submit:  sbatch hpc/delft_blue/experiments/exp10_llm_2b_social_bias.sh

source "/scratch/acmarcu/WiredTogether/hpc/delft_blue/experiments/_common.sh"

# 2B LLM + Hebbian + SocialModule (bias coupling).
# The social module deliberates each step AND its ask_target field
# overwrites the action LLM's communication_target at the routing site —
# the bond-weighted social pick wins over whatever the action LLM chose.
# Stronger coupling than exp9; tests whether forcing routing matters.

EXP_NAME="exp10_llm_2b_social_bias"
SEED=42
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_2B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 1000 \
    --warmup-time 300 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module bias \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    --simultaneous \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"