#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_transplant.sh — pair-bonding transplant experiment (Phase A / B).
#
# Phase A: 9 independent 2-agent runs capped at Chamber 2 ("do these two
#          cofire?"). Rank them, merge the best three.
# Phase B: one 6-agent run starting in Chamber 3, seeded with the merged
#          bonds + transplanted memories, plus its shuffled control.
#
# Routed through gpu_filter.sh so jobs never land on a GPU too small to hold
# the model — the failure mode that produced six 99.8%-NoOp RL runs, and that
# made the first Phase A smoke (job 12765952, influ3 / 11 GB) take 9h45m per
# env step. _common.sh's MIN_GPU_MEM_MIB preflight is the second line of
# defence and records any new offender in bad_gpu_nodes.txt.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#
#   # smoke first: 2 seeds, 2 episodes x 150 steps, no wandb (~1h on a good GPU)
#   SMOKE=1 bash submit_transplant.sh phaseA
#
#   # the real Phase A: 9 seeds, 3 episodes x 1000 steps
#   bash submit_transplant.sh phaseA
#
#   # after ranking + merging (see merge_pair_runs.py):
#   MERGED_DIR=$REPO/runs/pair_bonding/merged bash submit_transplant.sh phaseB
#
#   DRY_RUN=1 ... to print the sbatch calls without submitting.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

source "$(dirname "$0")/gpu_filter.sh"
echo "GPU filter: --exclude=$GPU_EXCLUDE${GPU_CONSTRAINT:+ --constraint=$GPU_CONSTRAINT}"

REPO=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether
PHASE="${1:-phaseA}"

export RUN_GROUP="${RUN_GROUP:-pair_bonding}"
export WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

# SMOKE=1 shrinks everything to a pipeline check: 2 seeds, short episodes, no
# wandb. Enough to prove the flags, the Lua agent-count sync, the agent_state
# export and (for phaseB) the transplant import — without a real training run.
if [ "${SMOKE:-0}" = "1" ]; then
    export EPISODES="${EPISODES:-2}"
    export MAX_STEPS="${MAX_STEPS:-150}"
    export WANDB="${WANDB:-0}"
    ARRAY_SPEC="${ARRAY_SPEC:-0-1}"
    echo "SMOKE mode: EPISODES=$EPISODES MAX_STEPS=$MAX_STEPS WANDB=$WANDB array=$ARRAY_SPEC"
else
    export EPISODES="${EPISODES:-3}"
    export MAX_STEPS="${MAX_STEPS:-1000}"
    ARRAY_SPEC="${ARRAY_SPEC:-0-8}"
fi

_submit() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "would queue: sbatch ${GPU_FILTER_FLAGS[*]} $*"
    else
        # shellcheck disable=SC2068
        sbatch ${GPU_FILTER_FLAGS[@]:+"${GPU_FILTER_FLAGS[@]}"} "$@"
    fi
}

case "$PHASE" in
    phaseA)
        echo "── Phase A: pair bonding (2 agents, Ch1-2), array $ARRAY_SPEC ──"
        _submit --array="$ARRAY_SPEC" expA_pair_bonding.sbatch
        echo
        echo "When these finish, rank and merge:"
        echo "  PYTHONPATH=\$REPO/src python3 \$REPO/src/mindforge/tools/merge_pair_runs.py \\"
        echo "      rank \$REPO/runs/$RUN_GROUP/expA_pair_bonding/seed_*"
        echo "  # then, with the top three run dirs in seat order:"
        echo "  PYTHONPATH=\$REPO/src python3 \$REPO/src/mindforge/tools/merge_pair_runs.py \\"
        echo "      merge --pair-run <top1> --pair-run <top2> --pair-run <top3> \\"
        echo "            --out-dir \$REPO/runs/$RUN_GROUP/merged/transplant"
        echo "  # and again with --shuffled --out-dir .../merged/shuffled"
        ;;
    phaseB)
        if [ -z "${MERGED_DIR:-}" ]; then
            echo "FATAL: phaseB needs MERGED_DIR — the parent holding" >&2
            echo "       transplant/ and shuffled/ from merge_pair_runs.py." >&2
            exit 1
        fi
        for arm in transplant shuffled; do
            d="$MERGED_DIR/$arm"
            if [ ! -f "$d/merged_W.json" ] || [ ! -f "$d/merged_manifest.json" ]; then
                echo "FATAL: $d is missing merged_W.json / merged_manifest.json" >&2
                exit 1
            fi
        done
        # Seeds for the full suite (3 per arm, medium-suite convention).
        # Override to add/backfill without duplicating queued jobs, e.g.:
        #     PHASEB_SEEDS="123 456" MERGED_DIR=... bash submit_transplant.sh phaseB
        PHASEB_SEEDS="${PHASEB_SEEDS:-42 123 456}"
        echo "── Phase B: merged 6-agent runs (Ch3-5), T + S arms, seeds: $PHASEB_SEEDS ──"
        # Subshells: a var-assignment prefix on a shell FUNCTION can persist in
        # bash, which would send the shuffled arm the transplant path.
        for _seed in $PHASEB_SEEDS; do
            ( export SEED="$_seed" MERGED_DIR="$MERGED_DIR/transplant"
              _submit expB_merged_transplant.sbatch )
            ( export SEED="$_seed" MERGED_DIR="$MERGED_DIR/shuffled"
              _submit expB_merged_shuffled.sbatch )
        done
        ;;
    *)
        echo "usage: bash submit_transplant.sh {phaseA|phaseB}" >&2
        exit 1
        ;;
esac
