#!/bin/bash
#SBATCH --job-name=exp04-ippo
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp04_ippo_%j.out
#SBATCH --error=slurm_logs/exp04_ippo_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP="${RUN_GROUP:-final}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-2500}"
WANDB_PROJECT="${WANDB_PROJECT:-final_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-ippo,independent_critic,final}"

# IPPO — action-level PPO with PER-AGENT independent value heads. No shared
# critic, no Hebbian. The "no centralized info" baseline.
run_exp "exp04_ippo" "$MODEL_2B" \
    --rl \
    --rl-critic-mode independent \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 --simultaneous
