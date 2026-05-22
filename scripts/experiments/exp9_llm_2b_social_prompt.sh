#!/bin/bash
#SBATCH --job-name=exp9-llm-2b-social-prompt
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp9_llm_2b_social_prompt-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp9_llm_2b_social_prompt-%j.out
# Submit:  sbatch scripts/experiments/exp9_llm_2b_social_prompt.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# 2B LLM + Hebbian + SocialModule (prompt coupling).
# The social module deliberates each step over (bond weights, bond deltas,
# incoming messages) and emits a directive into the action prompt as TEXT
# only — the action LLM may still ignore it. Weakest coupling rung of the
# social-module ablation; isolates the effect of having an explicit
# deliberation step independent of any forced routing.

EXP_NAME="exp9_llm_2b_social_prompt"
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
    --social-module prompt \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"