#!/bin/bash
#SBATCH --job-name=L1-llm-heb-prompt
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/L1-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/L1-%A_%a.err
# Submit: sbatch --array=0-4 L1_llm_hebbian_prompt.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# L1 — LLM with Hebbian weights in the prompt, no RL.
# The Hebbian graph updates per env step from co-activity + comm events,
# and the current bond weights surface in instruction_prompt_p2.txt
# (existing ``{social_bonds}`` placeholder, already wired pre-Phase-B+).
# Phase B+ adds interpretability.jsonl per-step emission by default.
#
# Compared against: M1 (no Hebbian), L2 (L1 + reward propagation).
# Model: Qwen3.5-2B  |  RQ: Does Hebbian-in-prompt alone help an
#                          untrained LLM coordinate?

export LLM_MODEL_PATH="$MODEL_2B"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --warmup-time 300 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --seed "$SEED" \
    --experiment-id "L1" \
    --tag "L1"

echo "L1 done (seed=$SEED)"
