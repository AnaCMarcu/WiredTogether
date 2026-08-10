#!/bin/bash
#SBATCH --job-name=exp23-cofire-prcoi
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp23_cofire_prcoi_%j.out
#SBATCH --error=slurm_logs/exp23_cofire_prcoi_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "prcoi".
#
# PURPOSE: the full menu {communicate, observe, imitate, none} with ALL
# three channels credited: cofiring = prox & reward & (comm|obs|imit).
# This is the GOAL-1 PRIMARY arm (the credit-family arms were removed
# 2026-08-05; prcoi keeps every co-firing signal): cofiring_events.jsonl +
# the graph's per-channel growth accumulator rank the channels within the
# run, with zero cross-arm confound.
#
# CAVEAT (pre-registered): within-run attribution mixes channel efficacy
# with the LLM's act PREFERENCE (if it picks comm 70% of the time, comm
# dominates raw growth share). Mitigation: attribution is reported raw AND
# per-act-normalised, next to the act-choice distribution, and cross-checked
# against the single-channel affordance arms exp20-22.
#
# SMOKE GATES run on this arm before the sweep: act mix non-degenerate
# (no act >85%), all three channels fire in cofiring_events.jsonl, W in
# the differentiated 0.2-0.8 band.

# Model must be pinned EXPLICITLY (decision deferred) — see exp20 header.
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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,prcoi}"

run_exp "exp23_cofire_prcoi" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts comm,obs,imit \
    --cofiring-channels comm,obs,imit \
    --simultaneous ${EXTRA_ARGS:-}
