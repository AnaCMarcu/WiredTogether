#!/bin/bash
#SBATCH --job-name=exp1-llm
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp1_llm-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp1_llm-%j.out

source "/scratch/acmarcu/WiredTogether/hpc/delft_blue/experiments/_common.sh"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,2b,baseline}"

# Plain LLM agents — no RL training, no Hebbian. Floor of the comparison.
run_exp "exp1_llm" "$MODEL_2B" --simultaneous
