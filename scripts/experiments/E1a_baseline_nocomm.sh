#!/bin/bash
#SBATCH --job-name=E1a-nocomm
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E1a-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E1a-%A_%a.err
# Submit: sbatch --array=0-2 E1a_baseline_nocomm.sh

source "$(dirname "$0")/_common.sh"

# H1: Baseline — MindForge agents WITHOUT communication
# Compared against: E1b (comm=on)
# Model: Qwen3.5-2B  |  RQ: Baseline

export LLM_MODEL_PATH="$MODEL_2B"
cd src/mindforge

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --no-communication \
    --warmup-time 300 \
    --seed "$SEED" \
    --experiment-id "E1a"

echo "E1a done (seed=$SEED)"
