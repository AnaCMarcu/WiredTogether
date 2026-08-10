#!/bin/bash
#SBATCH --job-name=exp28-cofire-null
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp28_cofire_null_%j.out
#SBATCH --error=slurm_logs/exp28_cofire_null_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "null" (pr).
#
# PURPOSE: the FLOOR — no social acts in the menu, no channel credited:
# cofiring = prox & reward only. Choice mode with an empty menu, so the
# agent stack, prompts, and code path match the other choice arms (the
# structural analogue of exp11's strict-null role in the topology
# ablation); bonds can only grow when two agents are co-located AND both
# carry engagement from reward alone. MUTE arm — comm-independent subset
# is the fallback task metric.
#
# H0 THAT MUST BE REPORTABLE (pre-registered): if this arm matches the
# best channel arm on both wiring differentiation and task performance,
# the social term adds nothing over proximity+reward — and that is the
# finding, not a failure of the experiment.

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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,choice,null,mute}"

run_exp "exp28_cofire_null" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --social-act-mode choice \
    --social-acts none \
    --cofiring-channels none \
    --simultaneous ${EXTRA_ARGS:-}
