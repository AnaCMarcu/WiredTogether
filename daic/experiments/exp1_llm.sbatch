#!/bin/bash
#SBATCH --job-name=exp1-llm
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp1_llm_%j.out
#SBATCH --error=slurm_logs/exp1_llm_%j.err

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,2b,baseline}"

# Plain LLM agents — no RL training, no Hebbian. Floor of the comparison.
run_exp "exp1_llm" "$MODEL_2B"
