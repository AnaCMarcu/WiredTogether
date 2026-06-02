#!/bin/bash
#SBATCH --job-name=exp11-llm-9b-social-prompt
#SBATCH --partition=gpu-a100
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp11_llm_9b_social_prompt-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp11_llm_9b_social_prompt-%j.out

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# 9B LLM + Hebbian + SocialModule (prompt coupling). 9B companion to exp9.
run_exp "exp11_llm_9b_social_prompt" "$MODEL_9B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous
