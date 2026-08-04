#!/bin/bash
#SBATCH --job-name=exp25-cofire-pro-credit
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp25_cofire_pro_credit_%j.out
#SBATCH --error=slurm_logs/exp25_cofire_pro_credit_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "pro*"
# (credit family — GOAL-1 PRIMARY contrast, with exp24/exp26).
#
# PURPOSE: full menu {communicate, observe, imitate, none}, co-firing
# credits OBS ONLY. Agents can still talk (Ch3 stays solvable — unlike the
# mute exp21), but talking wires nothing; only deliberate observation does.
# Directed credit: W[observer, target] grows, W[target, observer] does not.
#
# HYPOTHESIS (pre-registered): obs-credited graphs are sparser but more
# behaviorally predictive than comm-credited ones (exp24), and more
# asymmetric (mechanism check H3) — attention is directed, chatter is
# reciprocal.

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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,pro_credit,credit_family}"

run_exp "exp25_cofire_pro_credit" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts comm,obs,imit \
    --cofiring-channels obs \
    --simultaneous
