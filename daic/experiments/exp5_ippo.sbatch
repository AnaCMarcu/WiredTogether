#!/bin/bash
#SBATCH --job-name=exp5-ippo
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp5_ippo_%j.out
#SBATCH --error=exp5_ippo_%j.err

source "$(dirname "$0")/_common.sh"

# IPPO — action-level PPO with PER-AGENT independent value heads. No shared
# critic, no Hebbian. The "no centralized info" baseline.
run_exp "exp5_ippo" "$MODEL_2B" \
    --rl \
    --rl-critic-mode independent \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4
