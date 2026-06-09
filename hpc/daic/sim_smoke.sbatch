#!/bin/bash
#SBATCH --job-name=sim_smoke
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=cor1
#SBATCH --output=slurm_logs/sim_smoke_%j.out
#SBATCH --error=slurm_logs/sim_smoke_%j.err

# Stage-1 --simultaneous smoke: LLM agents (NO --rl), 1 episode, 40 steps.
# Confirms the simultaneous-move loop (concurrent cognition -> step_all once
# -> shared comm/Hebbian/metric bookkeeping) runs end-to-end without crashing
# on the real serving model + Luanti. Does NOT test RL (Stage 2) or macros
# (Stage 3). Uses the same apptainer/wandb/env wiring as the real experiments
# via run_exp; --simultaneous + the small budget are appended as extra args.

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# wandb off for a throwaway smoke (override with WANDB=1 sbatch ... if wanted).
WANDB="${WANDB:-0}" run_exp "opt_test_rl_4" "$MODEL_2B" \
    --episodes 2 \
    --max-steps 20 \
    # --simultaneous \
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
