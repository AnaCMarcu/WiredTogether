#!/bin/bash
#SBATCH --job-name=exp03-mappo
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp03_mappo_%j.out
#SBATCH --error=slurm_logs/exp03_mappo_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP=final
EPISODES=5
MAX_STEPS=2500
WANDB_PROJECT=final_wired_together

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-mappo,centralized_critic,final}"

# MAPPO — action-level PPO with a SHARED centralized critic V(joint_state).
# No Hebbian. The classical multi-agent baseline.
run_exp "exp03_mappo" "$MODEL_2B" \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 --simultaneous
