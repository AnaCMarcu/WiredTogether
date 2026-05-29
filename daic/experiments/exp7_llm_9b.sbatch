#!/bin/bash
#SBATCH --job-name=exp7-llm-9b
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp7_llm_9b_%j.out
#SBATCH --error=exp7_llm_9b_%j.err

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b}"

# Plain LLM agents at 9B scale. Model-size ablation companion to exp1 (2B).
run_exp "exp7_llm_9b" "$MODEL_9B"
