#!/bin/bash
#SBATCH --job-name=exp24-cofire-prc-credit
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp24_cofire_prc_credit_%j.out
#SBATCH --error=slurm_logs/exp24_cofire_prc_credit_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "prc*"
# (credit family — the GOAL-1 PRIMARY contrast, with exp25/exp26).
#
# PURPOSE: full menu {communicate, observe, imitate, none} — identical
# policy space and prompts to exp23/25/26 — but co-firing credits COMM
# ONLY. Cross-arm contrast within the credit family isolates "which channel
# WIRES them" from "which affordance HELPS them" (the affordance family
# exp20-22 manipulates both at once).
#
# HYPOTHESIS (pre-registered): if comm is the dominant wiring channel, this
# arm's graphs should match exp23's; if obs/imit carry the wiring, this
# arm's graphs should be flatter/sparser than exp25/exp26's despite
# identical behavior distributions at t=0. (Caveat: W feeds back into
# prompts, so behavior diverges once the graphs do.)

# Model must be pinned EXPLICITLY (decision deferred) — see exp20 header.
if [ -z "${MODEL_LLM:-}" ]; then
    echo "ERROR: MODEL_LLM is not set — refuse to guess the LLM backbone." >&2
    echo "       Launch via submit_cofiring.sh, which requires an explicit pin." >&2
    exit 1
fi

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

RUN_GROUP="${RUN_GROUP:-cofiring}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-cofiring_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,prc_credit,credit_family}"

run_exp "exp24_cofire_prc_credit" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts comm,obs,imit \
    --cofiring-channels comm \
    --simultaneous
