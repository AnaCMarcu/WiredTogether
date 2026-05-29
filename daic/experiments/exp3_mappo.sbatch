#!/bin/bash
#SBATCH --job-name=exp3-mappo
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp3_mappo_%j.out
#SBATCH --error=exp3_mappo_%j.err

source "$(dirname "$0")/_common.sh"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-mappo,centralized_critic}"

# MAPPO — action-level PPO with a SHARED centralized critic V(joint_state).
# No Hebbian. The classical multi-agent baseline.
run_exp "exp3_mappo" "$MODEL_2B" \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4
