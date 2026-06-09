#!/bin/bash
#SBATCH --job-name=exp07-llm-2b-social-prompt
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp07_llm_2b_social_prompt_%j.out
#SBATCH --error=slurm_logs/exp07_llm_2b_social_prompt_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP=final
EPISODES=5
MAX_STEPS=2500
WANDB_PROJECT=final_wired_together

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,2b,hebbian,social_prompt,final}"

# 2B LLM + Hebbian + SocialModule (prompt coupling). Weakest coupling rung
# of the social-module ablation — text-only directive into the action prompt.
run_exp "exp07_llm_2b_social_prompt" "$MODEL_2B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous
