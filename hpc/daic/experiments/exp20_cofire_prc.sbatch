#!/bin/bash
#SBATCH --job-name=exp20-cofire-prc
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp20_cofire_prc_%j.out
#SBATCH --error=slurm_logs/exp20_cofire_prc_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "prc".
#
# PURPOSE: communication as a CHOICE (vs. exp27's enforced legacy comm).
# The only social act in the menu is "communicate"; the agent may also pick
# "none". Co-firing credits comm only: cofiring = prox & reward & comm.
#
# HYPOTHESIS (pre-registered): with the choice restored, co-firing rate
# drops well below the enforced anchor (exp27) and bonds differentiate more
# (off-diagonal W spread) than under mandatory chatter. Primary wiring
# metric: bond differentiation + per-channel growth share (all comm here by
# construction). Primary task metric: distinct Ch2-Ch5 team milestones per
# episode; secondary: comm-independent subset (Ch2+Ch4).
#
# Seeds 42/123/456 x 3 episodes x <=1000 steps; pooling rule binding —
# dropping seeds post hoc is seed-shopping.

# Model must be pinned EXPLICITLY (decision deferred): _common.sh would
# otherwise silently default MODEL_LLM to gemma-4-E4B-it, which is NOT the
# model behind the Experiment-1 numbers. Use submit_cofiring.sh.
if [ -z "${MODEL_LLM:-}" ]; then
    echo "ERROR: MODEL_LLM is not set — refuse to guess the LLM backbone." >&2
    echo "       Launch via submit_cofiring.sh, which requires an explicit pin." >&2
    exit 1
fi

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

RUN_GROUP="${RUN_GROUP:-cofiring_final}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-cofiring_final_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,prc}"

run_exp "exp20_cofire_prc" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts comm \
    --cofiring-channels comm \
    --simultaneous
