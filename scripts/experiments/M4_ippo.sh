#!/bin/bash
#SBATCH --job-name=M4-ippo
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/M4-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/M4-%A_%a.err
# Submit: sbatch --array=0-4 M4_ippo.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# M4 — IPPO baseline (independent per-agent critic, no Hebbian).
# Compared against: M2 (MAPPO with shared critic), M5 (IPPO + Hebbian).
# Critic flavor distinction lives in --rl-critic-mode. Default is
# 'centralized' (= MAPPO); 'independent' selects per-agent value heads.
# Model: Qwen3.5-2B  |  RQ: Does the shared critic help cooperation?

export LLM_MODEL_PATH="$MODEL_2B"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --warmup-time 300 \
    --rl \
    --rl-critic-mode independent \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --seed "$SEED" \
    --experiment-id "M4" \
    --tag "M4"

echo "M4 done (seed=$SEED)"
