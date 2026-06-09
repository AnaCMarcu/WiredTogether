#!/bin/bash
# ── exp0_long.sh ─────────────────────────────────────────────────────────────
# Login-node launcher for the exp0 baseline sweep on five-chambers.
#
# Submits 3 condition chains × 5 chunks of 8k env-steps each
#   = 15 SLURM jobs, 24 h wall-clock per job, 40k total env-steps per condition.
#
# Conditions:
#   llm_only      LLM-driven actions only (no RL, no Hebbian)
#   llm_rl        LLM + RL (MAPPO central critic, no Hebbian)
#   llm_rl_hebb   LLM + RL + Hebbian routing (full thesis condition)
#
# Each chunk resumes from the previous chunk's checkpoint via
# scripts/exp0_chunk.sh, so the 5 chunks form one continuous 40k-step run.
# RNG state is restored across chunks (see save_checkpoint in
# src/mindforge/multi_agent_craftium.py) so action sampling is consistent.
#
# Usage (login node):
#   bash scripts/experiments/exp0_long.sh
#   SEED=123 bash scripts/experiments/exp0_long.sh
#   bash scripts/experiments/exp0_long.sh --dry-run
#
# Each chunk uses --tag so all 5 chunks of one (condition, seed) collapse
# into ONE run dir at runs/legacy/<EXP_ID>_<COND>/seed_<SEED>/. Episode
# counter is restored on resume so ep0001/, ep0002/, ... form a single
# contiguous sequence across chunks.
#
# Aggregation after all 15 jobs finish:
#   python scripts/aggregate_seeds.py \
#       runs/legacy/exp0_long_*/seed_${SEED}/ --out runs/exp0_long/agg/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR=/scratch/${USER}/WiredTogether
EXP_ID="exp0_long"
SEED="${SEED:-42}"
NUM_CHUNKS="${NUM_CHUNKS:-5}"
WORKER="$PROJECT_DIR/scripts/exp0_chunk.sh"

CONDITIONS=(llm_only llm_rl llm_rl_hebb)

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "ERROR: unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$WORKER" ]]; then
    echo "ERROR: worker not found: $WORKER" >&2
    exit 1
fi

echo "==============================================="
echo "  $EXP_ID launcher"
echo "  seed       : $SEED"
echo "  conditions : ${CONDITIONS[*]}"
echo "  num_chunks : $NUM_CHUNKS  (× 8k env-steps each = $((NUM_CHUNKS * 8))k per condition)"
echo "  worker     : $WORKER"
echo "  dry_run    : $DRY_RUN"
echo "==============================================="

for COND in "${CONDITIONS[@]}"; do
    CKPT_ROOT="$PROJECT_DIR/runs/$EXP_ID/$COND/seed_$SEED/checkpoints"
    mkdir -p "$CKPT_ROOT"
    echo
    echo "── condition: $COND"
    echo "   CKPT_ROOT: $CKPT_ROOT"

    PREV_JID=""
    for K in $(seq 1 "$NUM_CHUNKS"); do
        if [[ "$K" -eq 1 ]]; then
            IS_FIRST=1
        else
            IS_FIRST=0
        fi

        SBATCH_CMD=(
            sbatch --parsable
            --job-name "wt_${EXP_ID}_${COND}_c${K}_s${SEED}"
            --export "ALL,EXP_ID=$EXP_ID,COND=$COND,SEED=$SEED,CKPT_ROOT=$CKPT_ROOT,CHUNK=$K,IS_FIRST=$IS_FIRST"
        )
        if [[ -n "$PREV_JID" ]]; then
            SBATCH_CMD+=( --dependency=afterok:"$PREV_JID" )
        fi
        SBATCH_CMD+=( "$WORKER" )

        printf '   chunk %d  cmd: ' "$K"
        printf '%q ' "${SBATCH_CMD[@]}"
        printf '\n'

        if [[ "$DRY_RUN" -eq 1 ]]; then
            PREV_JID="DRYRUN_${COND}_${K}"
            continue
        fi

        JID=$("${SBATCH_CMD[@]}")
        echo "            submitted: $JID"
        PREV_JID="$JID"
    done
done

echo
echo "All chains submitted."
echo "Monitor with:"
echo "  squeue -u \$USER -o '%.10i %.30j %.2t %.10M %.20R' | grep wt_${EXP_ID}"
echo "Logs:"
echo "  /scratch/\$USER/WiredTogether/slurm_logs/wt_${EXP_ID}_*"
