#!/bin/bash
#SBATCH --job-name=expB-memory-only-3f
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expB_memory_only_3f_%j.out
#SBATCH --error=slurm_logs/expB_memory_only_3f_%j.err
# Small-GPU nodes: see gpu_filter.sh / bad_gpu_nodes.txt (submit via
# submit_transplant_2x2.sh). Header stays cor1-only — a command-line
# --exclude replaces it, and gpu_filter.sh re-adds cor1 as BASE_EXCLUDE.
#SBATCH --exclude=cor1,influ3

# ── Memory x bond design, cell B: memory RETAINED, bond RESET ──
#
#   cell   memory      bond        arm
#   A      retained    retained    expB_merged_transplant  (+ _shuffled control)
#   B      retained    reset       THIS FILE
#   C      reset       retained    expB_bond_only
#   D      reset       reset       expB_neither
#
# Agents carry their real Phase A partner's memories into the same seats as
# the transplant arm, but the learned block topology is replaced by a flat W
# whose mean off-diagonal is IDENTICAL to the merged matrix (0.133 vs the
# merged 0.265-within / 0.100-cross). Total bond mass is therefore held
# constant and only the STRUCTURE is destroyed.
#
# Reads against A: if A > B, the relational graph contributes something the
# episodic memory alone does not. If A == B, partner preference is carried by
# individual memory and the Hebbian structure is decorative — the objection
# this cell exists to settle.
#
# Submit with both merge outputs:
#   MERGED_DIR=$REPO/runs/pair_bonding/merged/transplant \
#   UNIFORM_DIR=$REPO/runs/pair_bonding/merged/uniform \
#       sbatch hpc/daic/experiments/expB_memory_only.sbatch
#
# Smoke greps: "[TRANSPLANT] agent_i:" for all six agents, "[START_CHAMBER]"
# once per episode, and a flat starting W in the "[HEBBIAN] init" dump.

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

if [ -z "${MERGED_DIR:-}" ] || [ ! -f "$MERGED_DIR/merged_manifest.json" ]; then
    echo "FATAL: MERGED_DIR must point at the TRANSPLANT merge dir" >&2
    echo "       (merged_manifest.json). Got: '${MERGED_DIR:-}'" >&2
    exit 1
fi
if [ -z "${UNIFORM_DIR:-}" ] || [ ! -f "$UNIFORM_DIR/uniform_W.json" ]; then
    echo "FATAL: UNIFORM_DIR must point at a merge_pair_runs.py uniform-w" >&2
    echo "       output dir (uniform_W.json). Got: '${UNIFORM_DIR:-}'" >&2
    exit 1
fi

# The memories must be the REAL pairings: seating strangers here would make
# this a second shuffled arm, not the bond-reset cell.
if command -v python3 >/dev/null 2>&1; then
    _cond=$(python3 -c "import json;print(json.load(open('$MERGED_DIR/merged_manifest.json')).get('condition',''))" 2>/dev/null || echo "")
    if [ -n "$_cond" ] && [ "$_cond" != "transplant" ]; then
        echo "FATAL: $MERGED_DIR/merged_manifest.json has condition='$_cond'," >&2
        echo "       expected 'transplant'. Cell B keeps the true pairings." >&2
        exit 1
    fi
fi

NUM_AGENTS=6
RUN_GROUP="${RUN_GROUP:-pair_bonding_3f}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseB,memory_only,cellB,three_factor}"

run_exp "expB_memory_only_3f" "$MODEL_9B" \
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
    --hebbian-init-file "$UNIFORM_DIR/uniform_W.json" \
    --agent-state-init "$MERGED_DIR/merged_manifest.json"
