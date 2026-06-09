#!/bin/bash
#SBATCH --job-name=exp9-llm-2b-social-prompt
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp9_llm_2b_social_prompt-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp9_llm_2b_social_prompt-%j.out

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# 2B LLM + Hebbian + SocialModule (prompt coupling). Weakest coupling rung
# of the social-module ablation — text-only directive into the action prompt.
run_exp "exp9_llm_2b_social_prompt" "$MODEL_2B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous
