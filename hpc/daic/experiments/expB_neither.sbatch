#!/bin/bash
#SBATCH --job-name=expB-neither
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expB_neither_%j.out
#SBATCH --error=slurm_logs/expB_neither_%j.err
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

SEED="${SEED:-42}"

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

if [ -z "${UNIFORM_DIR:-}" ] || [ ! -f "$UNIFORM_DIR/uniform_W.json" ]; then
    echo "FATAL: UNIFORM_DIR must point at a merge_pair_runs.py uniform-w" >&2
    echo "       output dir (uniform_W.json). Got: '${UNIFORM_DIR:-}'" >&2
    exit 1
fi

NUM_AGENTS=6
RUN_GROUP="${RUN_GROUP:-pair_bonding}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseB,neither,cellD,floor}"

run_exp "expB_neither" "$MODEL_9B" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous \
    --start-chamber 3 \
    --hebbian-init-file "$UNIFORM_DIR/uniform_W.json"
