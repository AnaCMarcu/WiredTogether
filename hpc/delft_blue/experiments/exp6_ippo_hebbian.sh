#!/bin/bash
#SBATCH --job-name=exp6-ippo-hebbian
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp6_ippo_hebbian-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp6_ippo_hebbian-%j.out

source "/scratch/acmarcu/WiredTogether/hpc/delft_blue/experiments/_common.sh"

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
    --hebbian-gamma 0.2 --simultaneous
