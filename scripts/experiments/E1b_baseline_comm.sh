#!/bin/bash
#SBATCH --job-name=E1b-comm
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E1b-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E1b-%A_%a.err
# Submit: sbatch --array=0-2 E1b_baseline_comm.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# H1: Baseline — MindForge agents WITH communication
# Compared against: E1a (comm=off)
# Model: Qwen3.5-2B  |  RQ: Baseline

export LLM_MODEL_PATH="$MODEL_2B"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --warmup-time 300 \
    --seed "$SEED" \
    --experiment-id "E1b"

echo "E1b done (seed=$SEED)"
