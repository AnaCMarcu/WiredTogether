#!/bin/bash
#SBATCH --job-name=exp21-cofire-pro
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp21_cofire_pro_%j.out
#SBATCH --error=slurm_logs/exp21_cofire_pro_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "pro".
#
# PURPOSE: observation as the ONLY social channel. The menu is
# {observe, none} — this arm is MUTE by construction (Ch3's comm-gated door
# is unsolvable for it; accepted design choice, the analysis reports the
# comm-independent milestone subset alongside). Observe is directed and
# silent: the observer receives the target's position/hp/inventory/beliefs
# next step; the target is never notified and gets no co-firing credit.
# cofiring = prox & reward & obs.
#
# HYPOTHESIS (pre-registered): obs produces a sparser but more behaviorally
# predictive graph than comm (higher rho in bonds_behavior) — deliberate,
# costly-to-fake attention beats mandatory chatter. Directed credit should
# also make the graph measurably more ASYMMETRIC than comm arms (mechanism
# check H3).

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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,pro,mute}"

run_exp "exp21_cofire_pro" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts obs \
    --cofiring-channels obs \
    --simultaneous ${EXTRA_ARGS:-}
