#!/bin/bash
#SBATCH --job-name=exp11-llm-9b-social-prompt
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp11_llm_9b_social_prompt_%j.out
#SBATCH --error=exp11_llm_9b_social_prompt_%j.err

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

# 9B LLM + Hebbian + SocialModule (prompt coupling). 9B companion to exp9.
run_exp "exp11_llm_9b_social_prompt" "$MODEL_9B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module prompt
