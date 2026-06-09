#!/bin/bash
#SBATCH --job-name=exp3-mappo
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp3_mappo-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp3_mappo-%j.out

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-mappo,centralized_critic}"

# MAPPO — action-level PPO with a SHARED centralized critic V(joint_state).
# No Hebbian. The classical multi-agent baseline.
run_exp "exp3_mappo" "$MODEL_2B" \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 --simultaneous
