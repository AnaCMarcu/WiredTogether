#!/bin/bash
#SBATCH --job-name=exp1-llm
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp1_llm_%j.out
#SBATCH --error=exp1_llm_%j.err

source "$(dirname "$0")/_common.sh"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,2b,baseline}"

# Plain LLM agents — no RL training, no Hebbian. Floor of the comparison.
run_exp "exp1_llm" "$MODEL_2B"
