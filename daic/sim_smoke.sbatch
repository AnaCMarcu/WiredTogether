#!/bin/bash
#SBATCH --job-name=sim_smoke
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/sim_smoke_%j.out
#SBATCH --error=slurm_logs/sim_smoke_%j.err

# Stage-1 --simultaneous smoke: LLM agents (NO --rl), 1 episode, 40 steps.
# Confirms the simultaneous-move loop (concurrent cognition -> step_all once
# -> shared comm/Hebbian/metric bookkeeping) runs end-to-end without crashing
# on the real serving model + Luanti. Does NOT test RL (Stage 2) or macros
# (Stage 3). Uses the same apptainer/wandb/env wiring as the real experiments
# via run_exp; --simultaneous + the small budget are appended as extra args.

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

# wandb off for a throwaway smoke (override with WANDB=1 sbatch ... if wanted).
WANDB="${WANDB:-0}" run_exp "sim_smoke" "$MODEL_9B" \
    --episodes 1 \
    --max-steps 40 \
    --simultaneous
