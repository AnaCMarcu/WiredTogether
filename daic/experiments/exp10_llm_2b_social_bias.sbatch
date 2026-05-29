#!/bin/bash
#SBATCH --job-name=exp10-llm-2b-social-bias
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp10_llm_2b_social_bias_%j.out
#SBATCH --error=exp10_llm_2b_social_bias_%j.err

source "$(dirname "$0")/_common.sh"

# 2B LLM + Hebbian + SocialModule (bias coupling). Stronger than exp9: the
# module's ask_target overwrites the action LLM's communication_target at
# the routing site.
run_exp "exp10_llm_2b_social_bias" "$MODEL_2B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module bias
