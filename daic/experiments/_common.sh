#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Shared DAIC setup for WiredTogether experiments.
#
# Mirrors scripts/experiments/_common.sh but uses Apptainer instead of
# module load + conda activate.
#
# Source from per-experiment sbatch files:
#     source "$(dirname "$0")/_common.sh"
#     run_exp "exp1_llm" "$MODEL_2B" <extra python args...>
# ────────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
IMG="$WORKSPACE/images/wiredtogether.sif"
REPO="$WORKSPACE/WiredTogether"

MODEL_2B="$WORKSPACE/models/Qwen3.5-2B"
MODEL_9B="$WORKSPACE/models/Qwen3.5-9B"

# Seed resolution: SLURM array index → SEED env var → 42.
SEEDS=(42 123 456)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
elif [ -z "${SEED:-}" ]; then
    SEED=42
fi

# ── Weights & Biases ──────────────────────────────────────────────────────
# Enabled by default. Opt out with `WANDB=0 sbatch ...`.
#
# API key resolution order:
#   1. $WANDB_API_KEY in the launching shell
#   2. $WORKSPACE/secrets/wandb_api_key (one line, no trailing newline)
#   3. ~/.netrc (wandb finds this itself; we just need to mount $HOME)
# If none are found we fall back to offline mode (WANDB_MODE=offline) so the
# run still produces local wandb dirs you can `wandb sync` later. To disable
# wandb entirely, set WANDB=0.
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-wired-together}"
WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-}"  # comma-separated, appended to auto tags

if [ "$WANDB" = "1" ]; then
    if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WORKSPACE/secrets/wandb_api_key" ]; then
        WANDB_API_KEY="$(cat "$WORKSPACE/secrets/wandb_api_key")"
        export WANDB_API_KEY
    fi
    if [ -z "${WANDB_API_KEY:-}" ]; then
        # No key found — fall back to offline mode rather than fail loudly.
        # `wandb sync $WORK_DIR/wandb/offline-run-*` after the job uploads them.
        export WANDB_MODE="${WANDB_MODE:-offline}"
        echo "[wandb] No API key found — falling back to WANDB_MODE=offline."
        echo "[wandb] Run \`wandb sync\` on the offline-run-* dirs to upload later."
    fi
fi

# Usage: run_exp <EXP_NAME> <LLM_MODEL_PATH> [extra python args...]
run_exp() {
    local EXP_NAME="$1"
    local LLM_MODEL="$2"
    shift 2

    local RUN_DIR="$REPO/runs/legacy/${EXP_NAME}/seed_${SEED}"
    local WORK_DIR="/tmp/$USER/${EXP_NAME}_${SLURM_JOB_ID:-nojob}"
    mkdir -p "$RUN_DIR" "$WORK_DIR"

    # Clear stale luanti/minetest processes from previous failed runs on this
    # compute node. Best-effort; only affects this user's procs.
    echo "── pre-flight: clearing stale luanti/minetest procs (best-effort) ──"
    pkill -9 -u "$USER" -f "minetest" 2>/dev/null || true
    pkill -9 -u "$USER" -f "luanti"   2>/dev/null || true
    sleep 5

    # Compose wandb flags. Tags include exp name + seed automatically;
    # WANDB_EXTRA_TAGS can append more (e.g., "ablation_A,prompt_v2").
    local WANDB_FLAGS=()
    if [ "$WANDB" = "1" ]; then
        local tags="exp_${EXP_NAME},seed_${SEED}"
        if [ -n "$WANDB_EXTRA_TAGS" ]; then
            tags="${tags},${WANDB_EXTRA_TAGS}"
        fi
        WANDB_FLAGS=(
            --wandb
            --wandb-project "$WANDB_PROJECT"
            --wandb-tags "$tags"
            --wandb-upload-artifacts
        )
    fi

    echo "== $EXP_NAME =="
    echo "Host:      $(hostname)"
    echo "Image:     $IMG"
    echo "Repo:      $REPO"
    echo "Run dir:   $RUN_DIR"
    echo "Work dir:  $WORK_DIR"
    echo "Model:     $LLM_MODEL"
    echo "Seed:      $SEED"
    if [ "$WANDB" = "1" ]; then
        echo "wandb:     project=$WANDB_PROJECT tags=exp_${EXP_NAME},seed_${SEED}${WANDB_EXTRA_TAGS:+,${WANDB_EXTRA_TAGS}} mode=${WANDB_MODE:-online}"
    else
        echo "wandb:     disabled (set WANDB=1 to enable)"
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo "============================"

    apptainer exec --nv \
        --bind /tmp:/tmp \
        --bind /tudelft.net:/tudelft.net \
        --env PYTHONPATH="$REPO/src" \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONIOENCODING=utf-8 \
        --env LANG=C.UTF-8 \
        --env LC_ALL=C.UTF-8 \
        --env LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/craftium.libs \
        --env LLM_MODEL_PATH="$LLM_MODEL" \
        --env LLM_ENABLE_THINKING=0 \
        --env ST_MODEL_NAME="$WORKSPACE/models/all-MiniLM-L6-v2" \
        --env SENTENCE_TRANSFORMERS_HOME="$WORKSPACE/models" \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        --env CRAFTIUM_ENV_DIR="$REPO/src/marl_craftium/craftium-envs/five-chambers" \
        --env WIREDTOGETHER_RUNS_ROOT="$REPO/runs" \
        --env SDL_VIDEODRIVER=dummy \
        --env SDL_AUDIODRIVER=dummy \
        --env DISPLAY= \
        --env LIBGL_ALWAYS_SOFTWARE=1 \
        --env MESA_GL_VERSION_OVERRIDE=3.3 \
        --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
        --env WANDB_MODE="${WANDB_MODE:-online}" \
        --env WANDB_DIR="$WORK_DIR" \
        --env WANDB_SILENT=true \
        --pwd "$WORK_DIR" \
        "$IMG" \
        python -u "$REPO/src/mindforge/multi_agent_craftium.py" \
            --num-agents 3 \
            --episodes 3 \
            --max-steps 1500 \
            --warmup-time 300 \
            --seed "$SEED" \
            --experiment-id "$EXP_NAME" \
            --tag "$EXP_NAME" \
            "${WANDB_FLAGS[@]}" \
            "$@" \
        2>&1 | tee "$RUN_DIR/run.log"

    local EXIT_CODE=${PIPESTATUS[0]}

    # Salvage craftium's per-run dirs (debug.txt, gifs, etc.) back to PRB.
    # If wandb ran in offline mode, this also captures wandb/offline-run-*
    # which you can later upload with `wandb sync <dir>`.
    echo "── archiving $WORK_DIR -> $RUN_DIR/work_artifacts/ ──"
    mkdir -p "$RUN_DIR/work_artifacts"
    rsync -r --no-perms --no-owner --no-group --no-times \
        "$WORK_DIR/" "$RUN_DIR/work_artifacts/" 2>&1 | tail -5 || true
    if [ "${WANDB_MODE:-online}" = "offline" ]; then
        echo "── wandb offline runs archived. Upload with:"
        echo "       wandb sync $RUN_DIR/work_artifacts/wandb/offline-run-*"
    fi

    echo "── $EXP_NAME (seed=$SEED) python exit: $EXIT_CODE ──"
    return "$EXIT_CODE"
}
