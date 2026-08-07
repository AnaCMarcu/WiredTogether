#!/bin/bash
#SBATCH --job-name=exp22-cofire-pri
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp22_cofire_pri_%j.out
#SBATCH --error=slurm_logs/exp22_cofire_pri_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "pri".
#
# PURPOSE: imitation as the ONLY social channel. Menu {imitate, none} —
# MUTE arm (Ch3 unsolvable by construction; comm-independent subset is the
# fallback metric). Imitation is GUIDED (2026-08-07 redesign): imitate(j,
# h<=5) delivers j's last h actions PLUS j's state (task, position) with
# instructions to judge whether the agent's own situation matches before
# re-enacting — the agent stays in control of every action, nothing is
# auto-replayed. The imit channel is credited on ADOPTION: each step the
# agent's own chosen action matches the next element of the delivered
# sequence (directed, imitator only). cofiring = prox & reward & imit.
#
# HYPOTHESIS (pre-registered): imitation-driven wiring requires a teammate
# whose situation transfers, so bonds should form toward agents whose
# behavior is WORTH copying — check rho_joint_dig / rho_proximity and the
# imit_adoption_rate + adopted-vs-baseline reward payoff.
#
# PRE-STATED expected risk: zero adoption (sequences delivered but never
# re-enacted) means the arm measures nothing — the smoke gate checks this
# before the full sweep.

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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,pri,mute}"

run_exp "exp22_cofire_pri" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts imit \
    --cofiring-channels imit \
    --simultaneous
