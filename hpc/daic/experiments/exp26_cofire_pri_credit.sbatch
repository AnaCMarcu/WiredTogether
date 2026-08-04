#!/bin/bash
#SBATCH --job-name=exp26-cofire-pri-credit
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp26_cofire_pri_credit_%j.out
#SBATCH --error=slurm_logs/exp26_cofire_pri_credit_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "pri*"
# (credit family — GOAL-1 PRIMARY contrast, with exp24/exp25).
#
# PURPOSE: full menu {communicate, observe, imitate, none}, co-firing
# credits IMIT ONLY — and imitation credit fires per committed replayed
# step (directed, imitator→target), never on gate-failed informational
# reports. Agents can still talk and observe; neither wires anything.
#
# HYPOTHESIS (pre-registered): if imitation is a rare/hard act (proximity +
# same-chamber gate), this arm's graph is the sparsest of the credit
# family; where it does grow, bonds should encode "who is worth copying" —
# check rho against joint_dig/proximity planes and the imitation_payoff
# metric (reward during replay vs own baseline).

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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,pri_credit,credit_family}"

run_exp "exp26_cofire_pri_credit" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts comm,obs,imit \
    --cofiring-channels imit \
    --simultaneous
