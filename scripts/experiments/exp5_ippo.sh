#!/bin/bash
#SBATCH --job-name=exp5-ippo
#SBATCH --partition=gpu-a100
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp5_ippo-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp5_ippo-%j.out

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# IPPO — action-level PPO with PER-AGENT independent value heads. No shared
# critic, no Hebbian. The "no centralized info" baseline.
run_exp "exp5_ippo" "$MODEL_2B" \
    --rl \
    --rl-critic-mode independent \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 --simultaneous
