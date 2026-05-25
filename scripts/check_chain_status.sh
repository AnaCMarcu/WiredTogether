#!/bin/bash
# ── check_chain_status.sh ────────────────────────────────────────────────────
# Status snapshot for the exp0_100k_steps chained runs (3 conditions × 5
# chunks). Tells you, at a glance:
#   - which chunks have finished, are running, are queued, or failed
#   - the current cumulative training progress (episode, global_step)
#   - which jobs are still on the SLURM queue
#
# Usage:
#   bash scripts/check_chain_status.sh
#   SEED=123 bash scripts/check_chain_status.sh
#   EXP_ID=exp0_100k_steps NUM_CHUNKS=5 bash scripts/check_chain_status.sh
#
# Status legend:
#   PENDING    in queue, waiting for dependency / resources
#   RUNNING    currently executing
#   DONE       worker printed its "[chunk N] done" marker — success
#   FAILED     no longer in queue, but log doesn't show the success marker
#   NOT_SUBMITTED  never submitted (or log file deleted)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR=/scratch/${USER}/WiredTogether
EXP_ID="${EXP_ID:-exp0_100k_steps}"
SEED="${SEED:-42}"
NUM_CHUNKS="${NUM_CHUNKS:-5}"
CONDITIONS=(llm_only llm_rl llm_rl_hebb)

# Single squeue snapshot reused across all chunks (avoids hammering the controller).
SQUEUE_OUT=$(squeue -u "$USER" -h -o "%i|%j|%t|%M|%R" 2>/dev/null || true)

# Map SLURM job state for one chunk → human label.
chunk_state() {
    local jname="$1"
    local line
    line=$(printf '%s\n' "$SQUEUE_OUT" | awk -F'|' -v j="$jname" '$2==j {print; exit}')
    if [[ -n "$line" ]]; then
        case "$(printf '%s' "$line" | cut -d'|' -f3)" in
            R)   echo "RUNNING" ;;
            PD)  echo "PENDING" ;;
            CG) echo "COMPLETING" ;;
            *)   printf '%s\n' "$line" | cut -d'|' -f3 ;;
        esac
        return
    fi
    local latest_log
    latest_log=$(ls -t "$PROJECT_DIR/slurm_logs/${jname}-"*.out 2>/dev/null | head -1 || true)
    if [[ -z "$latest_log" ]]; then
        echo "NOT_SUBMITTED"
        return
    fi
    # Worker prints "[chunk N] done" at the very end on success.
    if grep -Eq '\[chunk [0-9]+\] done' "$latest_log" 2>/dev/null; then
        echo "DONE"
    else
        echo "FAILED"
    fi
}

# Read the latest valid checkpoint's episode + global_step + dir basename.
# Output: three space-separated tokens (ep step ckpt_basename), or "- - -" if absent.
ckpt_progress() {
    local ckpt_root="$1"
    local ptr="${ckpt_root}/latest_checkpoint.txt"
    if [[ ! -f "$ptr" ]]; then
        echo "- - -"
        return
    fi
    local pointed
    pointed=$(cat "$ptr")
    if [[ ! -f "${pointed}/run_state.json" ]]; then
        echo "- - -"
        return
    fi
    python -c "
import json, os, sys
d = json.load(open(sys.argv[1]))
print(d.get('episode', '-'), d.get('global_step', '-'), os.path.basename(os.path.dirname(sys.argv[1])))
" "${pointed}/run_state.json" 2>/dev/null || echo "- - -"
}

echo "================================================"
echo "  $EXP_ID  seed=$SEED  chunks_per_condition=$NUM_CHUNKS"
echo "================================================"

for COND in "${CONDITIONS[@]}"; do
    CKPT_ROOT="$PROJECT_DIR/runs/$EXP_ID/$COND/seed_$SEED/checkpoints"
    read -r EP STEP CKPT <<< "$(ckpt_progress "$CKPT_ROOT")"

    DONE_COUNT=0
    STATES=()
    for K in $(seq 1 "$NUM_CHUNKS"); do
        JNAME="wt_${EXP_ID}_${COND}_c${K}_s${SEED}"
        S=$(chunk_state "$JNAME")
        STATES+=("$S")
        [[ "$S" == "DONE" ]] && DONE_COUNT=$((DONE_COUNT + 1))
    done

    printf "\n%-12s : %d/%d DONE   ep=%s  step=%s  ckpt=%s\n" \
        "$COND" "$DONE_COUNT" "$NUM_CHUNKS" "$EP" "$STEP" "$CKPT"
    for K in $(seq 1 "$NUM_CHUNKS"); do
        printf "  chunk %d : %s\n" "$K" "${STATES[$((K - 1))]}"
    done
done

echo
echo "── active jobs (squeue) ──"
ACTIVE_LINES=$(printf '%s\n' "$SQUEUE_OUT" \
    | awk -F'|' -v exp="wt_${EXP_ID}" '$2 ~ exp {
        printf "  %-9s  %-50s  %-2s  %-10s  %s\n", $1, $2, $3, $4, $5
      }')
if [[ -n "$ACTIVE_LINES" ]]; then
    printf '%s\n' "  JOBID      JOB_NAME                                            ST  ELAPSED     REASON/NODE"
    printf '%s\n' "$ACTIVE_LINES"
else
    echo "  (no $EXP_ID jobs currently on the queue)"
fi
