#!/bin/bash
#SBATCH --job-name=E12-nocomm
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E12-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E12-%A_%a.err
# Submit: sbatch --array=0-2 E12_hebbian_nocomm.sh

source "$(dirname "$0")/_common.sh"

# H12: Hebbian with NO communication (RQ4)
# Compared against: E1a (no comm, no Hebbian)
# "Firing together" = spatial co-activity only (δ_comm is irrelevant since
#  no communication events are generated with --no-communication)
# Model: Qwen3.5-2B  |  RQ: RQ4

export LLM_MODEL_PATH="$MODEL_2B"
cd src/mindforge

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 200 \
    --warmup-time 300 \
    --no-communication \
    --rl \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-auto-token-opt \
    --hebbian \
    --hebbian-no-comm-bond \
    --seed "$SEED" \
    --experiment-id "E12"

echo "E12 done (seed=$SEED)"
