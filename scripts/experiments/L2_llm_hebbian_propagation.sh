#!/bin/bash
#SBATCH --job-name=L2-llm-heb-prop
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/L2-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/L2-%A_%a.err
# Submit: sbatch --array=0-4 L2_llm_hebbian_propagation.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# L2 — L1 + per-step reward propagation in the prompt.
# After each env step, the agent sees a line like
#   "Propagated rewards this step: +2.50 from agent_1 (m17_switch_pressed)"
# in the action-selection prompt. This is the Phase B+ NEW feature: an
# inference-time test of whether the LLM can adapt to teammate-driven
# rewards via in-context learning, no RL gradient.
#
# Requires --hebbian (the propagation lines decompose the diffused-reward
# signal produced by the Hebbian graph).
#
# Compared against: L1 (Hebbian-in-prompt only, no propagation).
# Model: Qwen3.5-2B  |  RQ: Does per-step propagated-reward attribution
#                          improve LLM-as-policy coordination?

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
    --reward-propagation \
    --seed "$SEED" \
    --experiment-id "L2" \
    --tag "L2"

echo "L2 done (seed=$SEED)"
