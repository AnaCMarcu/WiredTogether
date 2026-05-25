#!/bin/bash
#SBATCH --job-name=wt_exp0
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/%x-%j.err

# ── exp0_chunk.sh ────────────────────────────────────────────────────────────
# Per-chunk SLURM worker for the exp0_long experiment.
# Runs ONE 8k-step chunk (--episodes 2 --max-steps 4000) of
# multi_agent_craftium.py on five-chambers, then writes
# latest_checkpoint.txt so the next chunk in the chain can resume.
#
# Submitted by scripts/experiments/exp0_long.sh, which chains 5 of
# these via --dependency=afterok so chunk K+1 only starts after chunk K
# succeeds.
#
# Env vars (set by the launcher via sbatch --export):
#   EXP_ID     experiment id, e.g. exp0_long
#   COND       condition tag: llm_only | llm_rl | llm_rl_hebb
#   SEED       seed integer
#   CKPT_ROOT  shared checkpoint dir for this (cond, seed); ep* dirs land here
#   CHUNK      chunk index (1..N)
#   IS_FIRST   1 for chunk 1 (no --resume), 0 otherwise
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

: "${EXP_ID:?EXP_ID not set}"
: "${COND:?COND not set}"
: "${SEED:?SEED not set}"
: "${CKPT_ROOT:?CKPT_ROOT not set}"
: "${CHUNK:?CHUNK not set}"
: "${IS_FIRST:?IS_FIRST not set}"

source "/scratch/${USER}/WiredTogether/scripts/experiments/_common.sh"

# _common.sh derives SEED from SLURM_ARRAY_TASK_ID. We're not array-submitted,
# so re-export the value the launcher set above.
export SEED

# Chunk size: --episodes 2 × --max-steps 4000 = 8 000 env steps.
# Sized to comfortably finish inside the 24 h SLURM wall-clock cap with
# margin: at the rate observed in earlier runs (~333 steps/hr LLM+RL),
# 8k steps fits in ~24 h and the per-episode checkpoint at episode end
# means every chunk hands off a clean ep{N}_end checkpoint (no
# mid-episode resume).
EPISODES=2
MAX_STEPS=4000

# Ch1 fallback teleport: at CH1_TIMEOUT_PCT% of the episode (default 50%)
# Python checks how many agents have advanced past Ch1:
#   ≥ 2 advanced → skip (the anvil-coop pair is already in Ch2)
#   1 advanced  → REGROUP (pull the lone leader + stragglers to fallback spawns)
#   0 advanced  → RESCUE (classic all-teleport)
# Implementation lives at multi_agent_craftium.py (search "[CH1_TIMEOUT]").
# Set CH1_TIMEOUT_PCT=999 to effectively disable the fallback.
CH1_TIMEOUT_PCT="${CH1_TIMEOUT_PCT:-50}"
CH1_TIMEOUT_STEPS=$((MAX_STEPS * CH1_TIMEOUT_PCT / 100))

# ── Per-condition flags ─────────────────────────────────────────────────────
# Defined here (not in the launcher) so the launcher only needs to pass
# scalar env vars — no shell-escaping of flag strings across sbatch --export.
case "$COND" in
    llm_only)
        COND_ARGS=()
        ;;
    llm_rl)
        COND_ARGS=(
            --rl
            --rl-critic-mode centralized
            --rl-model-path "$MODEL_2B"
            --rl-update-interval 64
            --rl-lr 3e-4
        )
        ;;
    llm_rl_hebb)
        COND_ARGS=(
            --rl
            --rl-critic-mode centralized
            --rl-model-path "$MODEL_2B"
            --rl-update-interval 64
            --rl-lr 3e-4
            --hebbian
            --hebbian-gamma 0.2
            --hebbian-ltp 0.05
            --hebbian-ltd 0.005
            --hebbian-decay 0.001
            --hebbian-beta 3.0
            --hebbian-radius 20.0
            --hebbian-init-weight 0.1
            --team-mode homogeneous-agent
        )
        ;;
    *)
        echo "ERROR: unknown COND=$COND (expected llm_only | llm_rl | llm_rl_hebb)" >&2
        exit 2
        ;;
esac

# ── Resume resolution ───────────────────────────────────────────────────────
RESUME_ARGS=()
if [[ "$IS_FIRST" -eq 0 ]]; then
    LATEST_FILE="${CKPT_ROOT}/latest_checkpoint.txt"
    RESUME_DIR=""
    if [[ -f "$LATEST_FILE" ]]; then
        _pointed=$(cat "$LATEST_FILE")
        if [[ -f "${_pointed}/run_state.json" ]]; then
            RESUME_DIR="$_pointed"
        else
            echo "WARNING: latest_checkpoint.txt → ${_pointed} has no run_state.json — scanning..."
        fi
    fi
    if [[ -z "$RESUME_DIR" ]]; then
        for dir in $(ls -td "${CKPT_ROOT}"/ep* 2>/dev/null); do
            if [[ -f "${dir}/run_state.json" ]]; then
                RESUME_DIR="$dir"
                echo "Found valid checkpoint via scan: $RESUME_DIR"
                break
            fi
        done
    fi
    if [[ -z "$RESUME_DIR" ]]; then
        echo "ERROR: chunk ${CHUNK} expected a valid checkpoint in ${CKPT_ROOT} from previous chunk" >&2
        exit 1
    fi
    echo "Resuming chunk ${CHUNK} from ${RESUME_DIR}"
    RESUME_ARGS=( --resume "$RESUME_DIR" --resume-skip-warmup )
fi

echo "================================================================"
echo "  ${EXP_ID} :: chunk ${CHUNK}  cond=${COND}  seed=${SEED}"
echo "  ckpt_root  : ${CKPT_ROOT}"
echo "  episodes   : ${EPISODES} × ${MAX_STEPS} steps = $((EPISODES * MAX_STEPS)) env-steps"
echo "  ch1 rescue : at step ${CH1_TIMEOUT_STEPS} (${CH1_TIMEOUT_PCT}% of episode) — only if no agent past Ch1"
echo "================================================================"

mkdir -p "$CKPT_ROOT"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" \
    --warmup-time 300 \
    --ch1-timeout-steps "$CH1_TIMEOUT_STEPS" \
    --experiment-id "${EXP_ID}_${COND}" \
    --tag "${EXP_ID}_${COND}" \
    --seed "$SEED" \
    --checkpoint-dir "$CKPT_ROOT" \
    --checkpoint-interval 200 \
    "${RESUME_ARGS[@]}" \
    "${COND_ARGS[@]}"

# ── Refresh latest_checkpoint.txt for the next chunk ────────────────────────
LATEST=""
for dir in $(ls -td "${CKPT_ROOT}"/ep* 2>/dev/null); do
    if [[ -f "${dir}/run_state.json" ]]; then
        LATEST="$dir"
        break
    fi
done
if [[ -n "$LATEST" ]]; then
    echo "$LATEST" > "${CKPT_ROOT}/latest_checkpoint.txt"
    echo "[chunk ${CHUNK}] latest_checkpoint.txt → $LATEST"
else
    echo "[chunk ${CHUNK}] WARNING: no valid checkpoint found in ${CKPT_ROOT}" >&2
fi

echo "[chunk ${CHUNK}] done"
