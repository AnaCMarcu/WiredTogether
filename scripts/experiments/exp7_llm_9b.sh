#!/bin/bash
#SBATCH --job-name=exp7-llm-9b
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp7_llm_9b-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp7_llm_9b-%j.out

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b}"

# Plain LLM agents at 9B scale. Model-size ablation companion to exp1 (2B).
run_exp "exp7_llm_9b" "$MODEL_9B" --simultaneous
