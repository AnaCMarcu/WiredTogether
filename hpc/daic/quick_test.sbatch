#!/bin/bash
#SBATCH --job-name=fast_rl
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/fast%j.out
#SBATCH --error=slurm_logs/fast%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

run_exp "opt_test_7" "$MODEL_2B" \
    --episodes 2 \
    --max-steps 50 \
    --simultaneous \
    # --rl \
    # --rl-critic-mode centralized \
    # --rl-model-path "$MODEL_2B" \
    # --rl-update-interval 16 \
    # --rl-lr 3e-4 \
    # --hebbian \
    # --hebbian-ltp 0.01 \
    # --hebbian-ltd 0.005 \
    # --hebbian-decay 0.005 \
    # --hebbian-beta 1.0 \
    # --hebbian-rho 0.3 \
    # --hebbian-gamma 0.2
