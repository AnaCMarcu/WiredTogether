#!/bin/bash
#SBATCH --job-name=exp12-llm-9b-social-bias
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp12_llm_9b_social_bias-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp12_llm_9b_social_bias-%j.out
# Submit:  sbatch scripts/experiments/exp12_llm_9b_social_bias.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# 9B LLM + Hebbian + SocialModule (bias coupling).
# Top of the social-module ablation grid at the 9B scale. The social
# module's ask_target literally overrides the action LLM's
# communication_target — strongest causal claim that bond weights drive
# behavior. routing_source = "social_bias" in messages.jsonl tags each
# routing decision the module overrode for post-hoc counting.

EXP_NAME="exp12_llm_9b_social_bias"
SEED=42
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_9B"

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