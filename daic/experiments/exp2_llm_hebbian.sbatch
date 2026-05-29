#!/bin/bash
#SBATCH --job-name=exp2-llm-hebbian
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp2_llm_hebbian_%j.out
#SBATCH --error=exp2_llm_hebbian_%j.err

source "$(dirname "$0")/_common.sh"

# LLM + Hebbian-in-prompt. Hebbian graph updates per env step from co-activity
# + comm events; bond weights are injected into the agent's prompt via the
# {social_bonds} placeholder. No RL training.
run_exp "exp2_llm_hebbian" "$MODEL_2B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2
