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

    echo "== $EXP_NAME =="
    echo "Host:      $(hostname)"
    echo "Image:     $IMG"
    echo "Repo:      $REPO"
    echo "Run dir:   $RUN_DIR"
    echo "Work dir:  $WORK_DIR"
    echo "Model:     $LLM_MODEL"
    echo "Seed:      $SEED"
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
            "$@" \
        2>&1 | tee "$RUN_DIR/run.log"

    local EXIT_CODE=${PIPESTATUS[0]}

    # Salvage craftium's per-run dirs (debug.txt, gifs, etc.) back to PRB.
    echo "── archiving $WORK_DIR -> $RUN_DIR/work_artifacts/ ──"
    mkdir -p "$RUN_DIR/work_artifacts"
    rsync -r --no-perms --no-owner --no-group --no-times \
        "$WORK_DIR/" "$RUN_DIR/work_artifacts/" 2>&1 | tail -5 || true

    echo "── $EXP_NAME (seed=$SEED) python exit: $EXIT_CODE ──"
    return "$EXIT_CODE"
}
