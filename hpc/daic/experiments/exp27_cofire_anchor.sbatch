#!/bin/bash
#SBATCH --job-name=exp27-cofire-anchor
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp27_cofire_anchor_%j.out
#SBATCH --error=slurm_logs/exp27_cofire_anchor_%j.err
#SBATCH --exclude=cor1

# Experiment 2 (co-firing without enforced communication), arm "anchor".
#
# PURPOSE: the ENFORCED-communication reference, run on the NEW code in
# --social-act-mode legacy (the default — no choice flags below). This is
# exp08's configuration re-run inside Experiment 2's suite, serving two
# roles: (1) the enforcement contrast for exp20 (anchor vs prc = mandatory
# vs chosen communication), and (2) a live end-to-end non-regression proof
# that legacy mode on the new code reproduces the historical behavior. If
# the model decision lands on a backbone other than Experiment 1's, this
# arm is the ONLY within-experiment reference — no Exp-1 number is then
# comparable.
#
# HYPOTHESIS (pre-registered): co-firing rate here is the ceiling of the
# suite (comm indicator ~always 1 → c_ij ~ proximity), and bond matrices
# are the flattest (least differentiated) — the "enforced co-firing"
# regime Experiment 2 removes.

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

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,hebbian,cofiring,legacy,anchor}"

# NOTE: no --social-act-mode flag — legacy is the default, and that IS the
# condition (mandatory per-step communication, historical co-firing rule).
run_exp "exp27_cofire_anchor" "$MODEL_LLM" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt \
    --simultaneous
