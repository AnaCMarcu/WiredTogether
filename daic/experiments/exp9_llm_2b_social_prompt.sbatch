#!/bin/bash
#SBATCH --job-name=exp9-llm-2b-social-prompt
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp9_llm_2b_social_prompt_%j.out
#SBATCH --error=exp9_llm_2b_social_prompt_%j.err

source "$(dirname "$0")/_common.sh"

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
    --social-module prompt
