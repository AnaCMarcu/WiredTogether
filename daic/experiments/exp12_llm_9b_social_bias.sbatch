#!/bin/bash
#SBATCH --job-name=exp12-llm-9b-social-bias
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp12_llm_9b_social_bias_%j.out
#SBATCH --error=exp12_llm_9b_social_bias_%j.err

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

# 9B LLM + Hebbian + SocialModule (bias coupling). Top of the social-module
# ablation grid at 9B scale.
run_exp "exp12_llm_9b_social_bias" "$MODEL_9B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module bias
