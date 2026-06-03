#!/bin/bash
#SBATCH --job-name=exp4-mappo-hebbian
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp4_mappo_hebbian_%j.out
#SBATCH --error=slurm_logs/exp4_mappo_hebbian_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

# Per-experiment wandb tags layered on top of the auto tags (exp_*, seed_*).
# Useful for filtering "all hebbian runs" or "all mappo runs" in the UI.
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-mappo,hebbian,centralized_critic}"

# MAPPO + Hebbian — the headline thesis claim. Shared centralized critic plus
# a Hebbian social-plasticity graph that diffuses rewards across co-bonded
# teammates and seeds the in-prompt {social_bonds} block.
run_exp "exp4_mappo_hebbian_rl" "$MODEL_2B" \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 --simultaneous
