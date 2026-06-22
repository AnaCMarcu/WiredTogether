#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# QUICK DIAGNOSTIC — not a real experiment.
#
# Purpose: figure out why the final suite (exp01..exp11) died instantly on
# 2026-06-19 (all FAILED 1:0 in 0-4s, no logs, runs/final/ never created).
#
# Differences from the real sbatch files, ALL chosen to make this cheap+fast:
#   * short --time (20 min) and NO --qos=long  -> schedules in seconds, and if
#     the long-QOS / 168h combo is itself the cause, this job will SUCCEED and
#     prove it.
#   * tiny workload (1 episode, 20 steps, no wandb) -> a healthy startup finishes
#     in a few minutes instead of hours.
#   * loud echo markers BEFORE source/run_exp -> if the output file contains
#     "STAGE 1" the job at least started; the last STAGE printed tells us
#     exactly how far it got before dying.
#
# Submit it forcing the log to $HOME so we can read it no matter what the
# slurm_logs/ path does:
#   cd /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether
#   sbatch --output=$HOME/diag_%j.out --error=$HOME/diag_%j.err \
#          hpc/daic/experiments/diag_quick.sbatch
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=diag-quick
#SBATCH --partition=general
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=diag_quick_%j.out
#SBATCH --error=diag_quick_%j.err

echo "STAGE 1: job started   host=$(hostname) date=$(date)"
echo "STAGE 1: pwd=$(pwd)"

REPO=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether
echo "STAGE 2: repo readable? $([ -d "$REPO" ] && echo yes || echo NO)"
echo "STAGE 2: _common.sh present? $([ -f "$REPO/hpc/daic/experiments/_common.sh" ] && echo yes || echo NO)"

echo "STAGE 3: sourcing _common.sh ..."
source "$REPO/hpc/daic/experiments/_common.sh"
echo "STAGE 3: sourced OK. IMG=$IMG"
echo "STAGE 3: image present? $([ -f "$IMG" ] && echo yes || echo NO)"
echo "STAGE 3: model present? $([ -d "$MODEL_2B" ] && echo yes || echo NO)"

# Tiny workload so a healthy run finishes fast.
EPISODES=1
MAX_STEPS=20
WANDB=0
RUN_GROUP=diag

echo "STAGE 4: calling run_exp (tiny: 1 episode, 20 steps, no wandb) ..."
run_exp "diag_quick" "$MODEL_2B" --simultaneous
echo "STAGE 5: run_exp returned exit=$?  (if you see this, startup works)"
