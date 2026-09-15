#!/bin/bash
#SBATCH --job-name=expB-bond-only
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expB_bond_only_%j.out
#SBATCH --error=slurm_logs/expB_bond_only_%j.err
# Small-GPU nodes: see gpu_filter.sh / bad_gpu_nodes.txt (submit via
# submit_transplant_2x2.sh). Header stays cor1-only — a command-line
# --exclude replaces it, and gpu_filter.sh re-adds cor1 as BASE_EXCLUDE.
#SBATCH --exclude=cor1,influ3

# ── Memory x bond design, cell C: memory RESET, bond RETAINED ──
#
#   cell   memory      bond        arm
#   A      retained    retained    expB_merged_transplant  (+ _shuffled control)
#   B      retained    reset       expB_memory_only
#   C      reset       retained    THIS FILE
#   D      reset       reset       expB_neither
#
# Six FRESH agents — no transplanted skills, episodes or curriculum — start in
# the merged block topology: within-seat 0.265, cross-seat 0.100. Nobody can
# remember a partner; the only thing carrying the Phase A relationships is the
# graph itself.
#
# This is NOT the frozen-topology ablation (exp09-exp11). Those impose a
# hand-set 0.8/0.0 graph on three Qwen agents with `--social-module bias`,
# which overwrites the message target outright. Here the graph is learned,
# modest, PLASTIC, and coupled only as prompt text — so C asks whether the
# structure survives its own plasticity without memories to anchor it.
#
# Reads against A and D: preference above D means the relationship graph is
# socially informative on its own; preference at D means W without memory
# relaxes to its fixed point and carries nothing.
#
# NOTE: --agent-state-init is deliberately ABSENT. Fresh agents must take the
# normal clear_data() path on their first on_messages call, so do not set
# agent._initialized here (see agent_modules/agent_state_io.py).
#
# Submit with:
#   MERGED_DIR=$REPO/runs/pair_bonding/merged/transplant \
#       sbatch hpc/daic/experiments/expB_bond_only.sbatch

SEED="${SEED:-42}"

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

if [ -z "${MERGED_DIR:-}" ] || [ ! -f "$MERGED_DIR/merged_W.json" ]; then
    echo "FATAL: MERGED_DIR must point at a merge_pair_runs.py output dir" >&2
    echo "       (merged_W.json). Got: '${MERGED_DIR:-}'" >&2
    exit 1
fi

NUM_AGENTS=6
RUN_GROUP="${RUN_GROUP:-pair_bonding}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseB,bond_only,cellC}"

run_exp "expB_bond_only" "$MODEL_9B" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous \
    --start-chamber 3 \
    --hebbian-init-file "$MERGED_DIR/merged_W.json"
