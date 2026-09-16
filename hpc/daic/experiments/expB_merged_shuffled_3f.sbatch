#!/bin/bash
#SBATCH --job-name=expB-merged-shuffled-3f
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expB_merged_shuffled_3f_%j.out
#SBATCH --error=slurm_logs/expB_merged_shuffled_3f_%j.err
# Small-GPU nodes: see gpu_filter.sh / bad_gpu_nodes.txt (submit via
# submit_transplant.sh). Header stays cor1-only — a command-line --exclude
# replaces it, and gpu_filter.sh re-adds cor1 as BASE_EXCLUDE.
#SBATCH --exclude=cor1,influ3

# ── Phase B of the pair-bonding transplant experiment (S arm: control) ──
# Identical to expB_merged_transplant except MERGED_DIR must point at the
# --shuffled output of merge_pair_runs.py: same W blocks, same memories, but
# partners scrambled so every seat pair holds two agents who never trained
# together (memories claim a shared history with a stranger). If pairs cohere
# in the T arm but dissolve here, the dyad identity — not bond mass or
# memory volume — carries the effect.
#
# Submit with:
#   MERGED_DIR=$REPO/runs/pair_bonding/merged/shuffled \
#       sbatch hpc/daic/experiments/expB_merged_shuffled.sbatch

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

if [ -z "${MERGED_DIR:-}" ] || [ ! -f "$MERGED_DIR/merged_W.json" ] \
        || [ ! -f "$MERGED_DIR/merged_manifest.json" ]; then
    echo "FATAL: MERGED_DIR must point at a merge_pair_runs.py --shuffled" >&2
    echo "       output dir (merged_W.json + merged_manifest.json). Got: '${MERGED_DIR:-}'" >&2
    exit 1
fi

# Guard against submitting the control arm with the real-transplant inputs.
if command -v python3 >/dev/null 2>&1; then
    _cond=$(python3 -c "import json;print(json.load(open('$MERGED_DIR/merged_manifest.json')).get('condition',''))" 2>/dev/null || echo "")
    if [ -n "$_cond" ] && [ "$_cond" != "shuffled" ]; then
        echo "FATAL: $MERGED_DIR/merged_manifest.json has condition='$_cond'," >&2
        echo "       expected 'shuffled'. Re-run merge_pair_runs.py with --shuffled." >&2
        exit 1
    fi
fi

NUM_AGENTS=6
RUN_GROUP="${RUN_GROUP:-pair_bonding_3f}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseB,shuffled,three_factor}"

run_exp "expB_merged_shuffled_3f" "$MODEL_9B" \
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
    --hebbian-init-file "$MERGED_DIR/merged_W.json" \
    --agent-state-init "$MERGED_DIR/merged_manifest.json"
