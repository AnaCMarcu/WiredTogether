#!/bin/bash
#SBATCH --job-name=E10c-6hebb
#SBATCH --partition=gpu-a100
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E10c-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E10c-%A_%a.err
# Submit: sbatch --array=0-2 E10c_6agents_hebbian.sh

source "$(dirname "$0")/_common.sh"

# H10: Scaling — 6 agents WITH Hebbian
# Compared against: E10d (6 agents, no Hebbian)
# Model: Qwen3.5-2B  |  RQ: RQ1+RQ2

export LLM_MODEL_PATH="$MODEL_2B"
cd src/mindforge

python multi_agent_craftium.py \
    --num-agents 6 \
    --episodes 3 \
    --max-steps 200 \
    --warmup-time 300 \
    --rl \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-auto-token-opt \
    --hebbian \
    --seed "$SEED" \
    --experiment-id "E10c"

echo "E10c done (seed=$SEED)"
