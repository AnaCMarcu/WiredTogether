#!/bin/bash
#SBATCH --job-name=expB-neither-3f
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expB_neither_3f_%j.out
#SBATCH --error=slurm_logs/expB_neither_3f_%j.err
# Small-GPU nodes: see gpu_filter.sh / bad_gpu_nodes.txt (submit via
# submit_transplant_2x2.sh). Header stays cor1-only — a command-line
# --exclude replaces it, and gpu_filter.sh re-adds cor1 as BASE_EXCLUDE.
#SBATCH --exclude=cor1,influ3

# ── Memory x bond design, cell D: memory RESET, bond RESET ──
#
#   cell   memory      bond        arm
#   A      retained    retained    expB_merged_transplant  (+ _shuffled control)
#   B      retained    reset       expB_memory_only
#   C      reset       retained    expB_bond_only
#   D      reset       reset       THIS FILE
#
# Six fresh agents, flat W at the same mean bond mass as every other cell.
# Nothing distinguishes any dyad from any other at step 0.
#
# WHY THIS ARM EXISTS. Seat pairs are index-adjacent, and WIRE makes
# index-adjacent agents task partners: switch i opens the door of agent
# (i+1) mod N (mods/wire/switches.lua), and the Ch4/Ch5 rescue spawns lay
# agents along one row in seat order. So the nominal 1/(N-1) = 0.20 chance
# line badly understates what a team with no relationships at all will score
# on seatmate preference. D measures that floor directly, and every other
# cell is read against it rather than against 0.20.
#
# It is also the cheapest arm: no merge inputs beyond the flat matrix.
#
# Submit with:
#   UNIFORM_DIR=$REPO/runs/pair_bonding/merged/uniform \
#       sbatch hpc/daic/experiments/expB_neither.sbatch

# ── THE RULE: three_factor (v2.2), NOT the reward_modulated default ──
# Same parameters as submit_cofiring_3f.sh and exp36: eligibility trace
# e <- 0.9*e + c, eta_0 and decay dropped to 0.001, co-activity floor 0.25,
# signed death LTD on.
#
# Phase A is NOT re-run. The transplanted W and memories still come from the
# reward_modulated pair runs under runs/pair_bonding/merged/, which is
# deliberate: Phase A is the relationship-FORMATION stage, and this suite asks
# how the new rule treats a relationship that already exists.
#
# Output lands in runs/pair_bonding_3f/ so it never mixes with the
# reward_modulated cells. All five arms of the 2x2 move together — comparing a
# three_factor cell against a reward_modulated one would measure the rule, not
# the factor under test.

SEED="${SEED:-42}"

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

if [ -z "${UNIFORM_DIR:-}" ] || [ ! -f "$UNIFORM_DIR/uniform_W.json" ]; then
    echo "FATAL: UNIFORM_DIR must point at a merge_pair_runs.py uniform-w" >&2
    echo "       output dir (uniform_W.json). Got: '${UNIFORM_DIR:-}'" >&2
    exit 1
fi

NUM_AGENTS=6
RUN_GROUP="${RUN_GROUP:-pair_bonding_3f}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseB,neither,cellD,floor,three_factor}"

run_exp "expB_neither_3f" "$MODEL_9B" \
    --hebbian \
    --hebbian-mode three_factor \
    --hebbian-eta-0 0.001 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.001 \
    --hebbian-eligibility-rho 0.9 \
    --hebbian-coact-floor 0.25 \
    --hebbian-death-ltd 0.05 \
    --hebbian-death-cap 10 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous \
    --start-chamber 3 \
    --hebbian-init-file "$UNIFORM_DIR/uniform_W.json"
