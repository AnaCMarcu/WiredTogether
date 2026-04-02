#!/bin/bash
#SBATCH --job-name=E10d-6nohebb
#SBATCH --partition=gpu-a100
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E10d-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E10d-%A_%a.err
# Submit: sbatch --array=0-2 E10d_6agents_no_hebbian.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# H10: Scaling — 6 agents WITHOUT Hebbian
# Compared against: E10c (6 agents, Hebbian)
# Model: Qwen3.5-2B  |  RQ: RQ1+RQ2

export LLM_MODEL_PATH="$MODEL_2B"

python multi_agent_craftium.py \
    --num-agents 6 \
    --episodes 3 \
    --max-steps 200 \
    --warmup-time 300 \
    --rl \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-auto-token-opt \
    --seed "$SEED" \
    --experiment-id "E10d"

echo "E10d done (seed=$SEED)"
