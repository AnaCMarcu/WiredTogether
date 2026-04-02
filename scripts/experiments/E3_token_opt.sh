#!/bin/bash
#SBATCH --job-name=E3-tokenopt
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E3-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E3-%A_%a.err
# Submit: sbatch --array=0-2 E3_token_opt.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# H3: Token-level optimization only (no action head)
# Compared against: E1b baseline
# Model: Qwen3.5-2B  |  RQ: Baseline
# NOTE: --rl enables the RL layer infrastructure needed for token-opt.
# The action head still runs but token-opt is the variable under test.

export LLM_MODEL_PATH="$MODEL_2B"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --warmup-time 300 \
    --rl \
    --rl-mode token \
    --rl-model-path "$MODEL_2B" \
    --rl-auto-token-opt \
    --seed "$SEED" \
    --experiment-id "E3"

echo "E3 done (seed=$SEED)"
