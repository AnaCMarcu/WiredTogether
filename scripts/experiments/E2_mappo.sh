#!/bin/bash
#SBATCH --job-name=E2-mappo
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E2-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E2-%A_%a.err
# Submit: sbatch --array=0-2 E2_mappo.sh

source "$(dirname "$0")/_common.sh"

# H2: Action-level RL (MAPPO)
# Compared against: E1b baseline
# Model: Qwen3.5-2B  |  RQ: Baseline

export LLM_MODEL_PATH="$MODEL_2B"
cd src/mindforge

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --warmup-time 300 \
    --rl \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --seed "$SEED" \
    --experiment-id "E2"

echo "E2 done (seed=$SEED)"
