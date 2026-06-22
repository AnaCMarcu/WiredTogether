#!/bin/bash
#SBATCH --job-name=exp10-llm-9b-allied-pair
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp10_llm_9b_allied_pair_%j.out
#SBATCH --error=slurm_logs/exp10_llm_9b_allied_pair_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP="${RUN_GROUP:-final}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-2500}"
WANDB_PROJECT="${WANDB_PROJECT:-final_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,frozen,allied_pair,social_bias,final}"

# ───────────────────────────────────────────────────────────────────────────
# 9B LLM-only + HARDCODED (frozen) Hebbian graph + SocialModule (bias).
# CONDITION B of 2:  TWO AGENTS ALLIED, ONE OUTSIDER  ("pair" preset).
#
#   W:  agent 0 ↔ agent 1 bonded at 0.8;  agent 2 has NO bonds (0.0) to anyone
#       and nobody bonds to it. So 0 & 1 coordinate; 2 is socially excluded.
#       (Set WEAK=0.1 for a soft version where 2 is weakly, not zero, bonded.)
#
# HYPOTHESIS (H1, paired with exp09): the team completes FEWER cooperative
# milestones here than in the all-allied condition, because the excluded
# agent 2 isn't coordinated with — and the milestones needing all three
# (m19_all_in_communal, m22_all_mobs_killed, m23_all_alive, m28_all_alive)
# become hard or impossible. Expect agent 2 to lag / be left behind.
#
# Frozen graph ⇒ social MECHANISM, not learning. No --rl; --hebbian-gamma 0.
# Primary metric: # distinct Ch2–Ch5 milestones the team completes (per seed).
# ───────────────────────────────────────────────────────────────────────────

WEAK="${WEAK:-0.0}"   # 0.0 = agent 2 fully excluded; 0.1 = weakly bonded

run_exp "exp10_llm_9b_allied_pair" "$MODEL_9B" \
    --hebbian \
    --hebbian-freeze \
    --hebbian-preset pair \
    --hebbian-bond-strong 0.8 \
    --hebbian-bond-weak "$WEAK" \
    --hebbian-gamma 0.0 \
    --social-module bias \
    --simultaneous
