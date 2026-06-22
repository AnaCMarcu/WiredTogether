#!/bin/bash
#SBATCH --job-name=exp05-mappo-hebbian
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp05_mappo_hebbian_%j.out
#SBATCH --error=slurm_logs/exp05_mappo_hebbian_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP="${RUN_GROUP:-final}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-2500}"
WANDB_PROJECT="${WANDB_PROJECT:-final_wired_together}"

# Per-experiment wandb tags layered on top of the auto tags (exp_*, seed_*).
# Useful for filtering "all hebbian runs" or "all mappo runs" in the UI.
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-mappo,hebbian,centralized_critic,final}"

# MAPPO + Hebbian — the headline thesis claim. Shared centralized critic plus
# a Hebbian social-plasticity graph that diffuses rewards across co-bonded
# teammates and seeds the in-prompt {social_bonds} block.
run_exp "exp05_mappo_hebbian" "$MODEL_2B" \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 --simultaneous
