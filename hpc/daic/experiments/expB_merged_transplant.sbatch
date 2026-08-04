#!/bin/bash
#SBATCH --job-name=expB-merged-transplant
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expB_merged_transplant_%j.out
#SBATCH --error=slurm_logs/expB_merged_transplant_%j.err
# influ3 excluded: 11GB GPU cannot hold gemma-4-E4B -> CPU offload starves
# Luanti's software renderer. Even more critical here (6 clients).
#SBATCH --exclude=cor1,influ3

# ── Phase B of the pair-bonding transplant experiment (T arm) ──
# One 6-agent run starting directly in the Chamber 3 cells (--start-chamber 3;
# remaining timers: Ch3→Ch4 at 33%, Ch4→Ch5 at 67% of --max-steps). The three
# Phase A winners are transplanted as seat pairs (0,1), (2,3), (4,5): merged
# block-diagonal W (--hebbian-init-file) + full cognitive state with names
# remapped (--agent-state-init).
#
# Submit with MERGED_DIR pointing at merge_pair_runs.py output:
#   MERGED_DIR=$REPO/runs/pair_bonding/merged/transplant \
#       sbatch hpc/daic/experiments/expB_merged_transplant.sbatch
#
# Notes:
#  * 64GB/12cpu: 6 Luanti clients + Xvfb + the 9B model (3 clients fit in
#    32GB). Check MaxRSS on the first smoke job; bump to 80GB if tight.
#  * BOSS_HP stays 60 for N=6 — boss-kill speed is NOT comparable to the
#    3-agent suites.
#  * Smoke-check greps: "[TRANSPLANT] agent_i:" for all six agents,
#    "[START_CHAMBER]" once per episode, "[CH2_TIMEOUT] forced teleport"
#    with idx=0..5.

SEED="${SEED:-42}"

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

if [ -z "${MERGED_DIR:-}" ] || [ ! -f "$MERGED_DIR/merged_W.json" ] \
        || [ ! -f "$MERGED_DIR/merged_manifest.json" ]; then
    echo "FATAL: MERGED_DIR must point at a merge_pair_runs.py output dir" >&2
    echo "       (merged_W.json + merged_manifest.json). Got: '${MERGED_DIR:-}'" >&2
    exit 1
fi

NUM_AGENTS=6
RUN_GROUP="${RUN_GROUP:-pair_bonding}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseB,transplant}"

run_exp "expB_merged_transplant" "$MODEL_9B" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous \
    --start-chamber 3 \
    --hebbian-init-file "$MERGED_DIR/merged_W.json" \
    --agent-state-init "$MERGED_DIR/merged_manifest.json"
