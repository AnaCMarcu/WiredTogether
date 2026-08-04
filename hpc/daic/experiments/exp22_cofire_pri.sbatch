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
# fallback metric). Imitation is COMMITTED REPLAY: imitate(j, h<=5) within
# interaction_radius AND the same chamber replays j's last h actions instead
# of the agent's own choices (abort on chamber change / damage / reset);
# outside the gate it degrades to an informational report with NO credit.
# Credit is directed (imitator only) and fires per replayed step.
# cofiring = prox & reward & imit.
#
# HYPOTHESIS (pre-registered): imitation-driven wiring REQUIRES co-location,
# so if agents drift apart the graph stays near-empty — that is a finding,
# not a failure. Where the gate does open, bonds should track behavioral
# copying tightly (high rho_joint_dig / rho_proximity).
#
# PRE-STATED expected risk: imitation_abort_rate ~1.0 or gate never passing
# means the arm is measuring proximity, not imitation — the smoke gate
# checks this before the full sweep.

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
