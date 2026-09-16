#!/bin/bash
#SBATCH --job-name=expB-merged-transplant-3f
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expB_merged_transplant_3f_%j.out
#SBATCH --error=slurm_logs/expB_merged_transplant_3f_%j.err
# Small-GPU nodes: see gpu_filter.sh / bad_gpu_nodes.txt (submit via
# submit_transplant.sh). Header stays cor1-only — a command-line --exclude
# replaces it, and gpu_filter.sh re-adds cor1 as BASE_EXCLUDE.
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
    echo "FATAL: MERGED_DIR must point at a merge_pair_runs.py output dir" >&2
    echo "       (merged_W.json + merged_manifest.json). Got: '${MERGED_DIR:-}'" >&2
    exit 1
fi

NUM_AGENTS=6
RUN_GROUP="${RUN_GROUP:-pair_bonding_3f}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseB,transplant,three_factor}"

run_exp "expB_merged_transplant_3f" "$MODEL_9B" \
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
