#!/bin/bash
#SBATCH --job-name=exp8-llm-9b-hebbian
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp8_llm_9b_hebbian_%j.out
#SBATCH --error=exp8_llm_9b_hebbian_%j.err

source "$(dirname "$0")/_common.sh"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian}"

# LLM + Hebbian-in-prompt at 9B scale. Model-size ablation companion to exp2.
run_exp "exp8_llm_9b_hebbian" "$MODEL_9B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2
