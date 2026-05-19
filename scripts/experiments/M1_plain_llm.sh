#!/bin/bash
#SBATCH --job-name=M1-plain-llm
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/M1-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/M1-%A_%a.err
# Submit: sbatch --array=0-4 M1_plain_llm.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# M1 — plain LLM baseline. No RL training, no Hebbian.
# This is the Phase B+ thesis comparison's FLOOR — answers "what does
# the base LLM achieve with zero training and zero social context?"
# Compared against: M2 (MAPPO), L1 (LLM + Hebbian-in-prompt), M3 (MAPPO + Hebbian).
# Model: Qwen3.5-2B  |  RQ: Phase B+ headline floor

export LLM_MODEL_PATH="$MODEL_2B"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --warmup-time 300 \
    --seed "$SEED" \
    --experiment-id "M1" \
    --tag "M1"

echo "M1 done (seed=$SEED)"
