#!/bin/bash
#SBATCH --job-name=exp29-cofire-prco
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp29_cofire_prco_%j.out
#SBATCH --error=slurm_logs/exp29_cofire_prco_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "prco".
#
# PURPOSE: communication + observation, WITHOUT imitation. Menu
# {communicate, observe, none}; both channels credited:
# cofiring = prox & reward & (comm|obs).
#
# This is the middle rung of the channel ladder prc -> prco -> prcoi:
#   prco vs prc    isolates what OBSERVATION adds on top of talking
#                  (information ABOUT teammates vs information FROM them);
#   prcoi vs prco  isolates what IMITATION adds on top of both
#                  (behavior transfer vs pure information exchange).
#
# HYPOTHESIS (pre-registered): if observation's directed, costly-to-fake
# attention is the socially informative signal, prco's graph should be more
# differentiated/asymmetric than prc's at similar task performance; if obs
# is redundant with what agents already say, prco collapses onto prc.

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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,prco}"

run_exp "exp29_cofire_prco" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts comm,obs \
    --cofiring-channels comm,obs \
    --simultaneous ${EXTRA_ARGS:-}
