#!/bin/bash
#SBATCH --job-name=exp6-ippo-hebbian
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp6_ippo_hebbian_%j.out
#SBATCH --error=exp6_ippo_hebbian_%j.err

source "$(dirname "$0")/_common.sh"

# IPPO + Hebbian — "Hebbian without a shared critic" cell of the 2x3
# comparison matrix.
run_exp "exp6_ippo_hebbian" "$MODEL_2B" \
    --rl \
    --rl-critic-mode independent \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2
