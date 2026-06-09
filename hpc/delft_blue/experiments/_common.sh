#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Shared DelftBlue setup for WiredTogether experiments.
#
# Mirrors hpc/daic/experiments/_common.sh — same run_exp() contract, same
# wandb wiring, same archive layout — but uses `module load + conda
# activate` instead of apptainer (DelftBlue does not provide apptainer
# images for this project).
#
# Source from per-experiment sbatch files:
#     source "$(dirname "$0")/_common.sh"
#     run_exp "exp1_llm" "$MODEL_2B" <extra python args...>
# ────────────────────────────────────────────────────────────────────────────

PROJECT_DIR=/scratch/acmarcu/WiredTogether
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether

# ── Modules & Conda ──────────────────────────────────────────────────────
module purge
module load 2025
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# ── Headless rendering (no apptainer here, so we set the env vars
#    directly — Mesa's software fallback on DelftBlue compute nodes works
#    without xvfb because Luanti's Irrlicht/EGL can render to a surfaceless
#    EGL context when LIBGL_ALWAYS_SOFTWARE=1 + GALLIUM_DRIVER=llvmpipe
#    are set. If a node ever lacks /dev/dri permissions, the existing
#    DISPLAY="" / SDL_VIDEODRIVER=dummy combo keeps SDL out of the way.) ──
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export DISPLAY=
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export EGL_PLATFORM=surfaceless

# ── Python / model paths ─────────────────────────────────────────────────
export LLM_ENABLE_THINKING=0
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export SCRATCH=/scratch/acmarcu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=/scratch/acmarcu/models/st_cache
export ST_MODEL_NAME=/scratch/acmarcu/models/all-MiniLM-L6-v2
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CRAFTIUM_ENV_DIR="${PROJECT_DIR}/src/marl_craftium/craftium-envs/five-chambers"
export WIREDTOGETHER_RUNS_ROOT="${PROJECT_DIR}/runs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pre-cached model paths under /scratch.
MODEL_2B=/scratch/acmarcu/models/Qwen3.5-2B
MODEL_9B=/scratch/acmarcu/models/Qwen3.5-9B

# Default model for runs that don't pass an explicit one to run_exp.
export LLM_MODEL_PATH="$MODEL_9B"

# ── Seed resolution: SLURM array index → SEED env var → 42. ──────────────
SEEDS=(42 123 456)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
elif [ -z "${SEED:-}" ]; then
    SEED=42
fi

# ── Weights & Biases ─────────────────────────────────────────────────────
# Enabled by default. Opt out with `WANDB=0 sbatch ...`.
#
# Auth lives in ~/.netrc (per-user, chmod 600). DelftBlue runs aren't
# containerised so ~/.netrc is auto-readable from the job; wandb.init()
# reads it without any env-var passing. To set it up once:
#     cat > ~/.netrc <<EOF
#     machine api.wandb.ai
#       login user
#       password <YOUR-KEY>
#     EOF
#     chmod 600 ~/.netrc
#
# If ~/.netrc is missing or the key is invalid, wandb falls back to
# offline mode and writes to $WORK_DIR/wandb/offline-run-*. The salvage
# hook below auto-syncs those after the job finishes (provided ~/.netrc
# is present by then; if not, run `wandb sync <dir>` from the login node
# any time later).
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-wired-together}"
WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-}"  # comma-separated, appended to auto tags
WANDB_MODE="${WANDB_MODE:-online}"        # default online; offline if env-forced

# ── Logging dir ──
mkdir -p "$PROJECT_DIR/slurm_logs"

# Usage: run_exp <EXP_NAME> <LLM_MODEL_PATH> [extra python args...]
run_exp() {
    local EXP_NAME="$1"
    local LLM_MODEL="$2"
    shift 2

    local RUN_DIR="$PROJECT_DIR/runs/legacy/${EXP_NAME}/seed_${SEED}"
    # Heavy artifacts (craftium debug.txt, wandb offline-runs, intermediate
    # per-100-step gifs) live in a PARALLEL tree so the runs/ dir stays
    # small and fast to scp. Mirror the same exp/seed structure for easy
    # cross-referencing — same convention as DAIC.
    local ARTIFACTS_DIR="$PROJECT_DIR/run_artifacts/legacy/${EXP_NAME}/seed_${SEED}"
    local WORK_DIR="/tmp/$USER/${EXP_NAME}_${SLURM_JOB_ID:-nojob}"
    mkdir -p "$RUN_DIR" "$ARTIFACTS_DIR" "$WORK_DIR"

    # Clear stale luanti/minetest processes from previous failed runs on
    # this compute node. Best-effort; only affects this user's procs.
    echo "── pre-flight: clearing stale luanti/minetest procs (best-effort) ──"
    pkill -9 -u "$USER" -f "minetest" 2>/dev/null || true
    pkill -9 -u "$USER" -f "luanti"   2>/dev/null || true
    sleep 5

    # Pin the launching shell into $WORK_DIR so the multiagent env's
    # client-spawn cwd ends up under /tmp (where the per-run debug.txt
    # lands), not under src/mindforge/ (which would scatter outputs).
    cd "$WORK_DIR"

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

    # Force wandb's internal dir under $WORK_DIR so its offline-runs land
    # somewhere the salvage hook will rsync back to $ARTIFACTS_DIR.
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_DIR="$WORK_DIR"
    export WANDB_SILENT=true
    export WIREDTOGETHER_INTERMEDIATE_GIF_DIR="$WORK_DIR/intermediate_gifs"
    export LLM_MODEL_PATH="$LLM_MODEL"

    echo "== $EXP_NAME =="
    echo "Host:      $(hostname)"
    echo "Repo:      $PROJECT_DIR"
    echo "Run dir:   $RUN_DIR"
    echo "Work dir:  $WORK_DIR"
    echo "Model:     $LLM_MODEL"
    echo "Seed:      $SEED"
    if [ "$WANDB" = "1" ]; then
        echo "wandb:     project=$WANDB_PROJECT tags=exp_${EXP_NAME},seed_${SEED}${WANDB_EXTRA_TAGS:+,${WANDB_EXTRA_TAGS}} mode=${WANDB_MODE:-online}"
    else
        echo "wandb:     disabled (set WANDB=1 to enable)"
    fi
    python -c "import torch; print('CUDA:', torch.cuda.is_available())"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo "============================"

    python -u "$PROJECT_DIR/src/mindforge/multi_agent_craftium.py" \
        --num-agents 3 \
        --episodes 3 \
        --max-steps 1000 \
        --warmup-time 300 \
        --seed "$SEED" \
        --experiment-id "$EXP_NAME" \
        --tag "$EXP_NAME" \
        "${WANDB_FLAGS[@]}" \
        "$@" \
        2>&1 | tee "$RUN_DIR/run.log"

    local EXIT_CODE=${PIPESTATUS[0]}

    # Salvage craftium's per-run dirs (debug.txt, gifs, etc.) back to
    # /scratch. If wandb ran in offline mode, this also captures
    # wandb/offline-run-* which the auto-sync hook below uploads.
    # Same split as DAIC:
    #   - $WORK_DIR/intermediate_gifs/ → $ARTIFACTS_DIR/intermediate_gifs/
    #   - everything else (craftium debug.txt, wandb offline-runs, etc.)
    #     → $ARTIFACTS_DIR/work_artifacts/
    if [ -d "$WORK_DIR/intermediate_gifs" ]; then
        echo "── archiving intermediate gifs -> $ARTIFACTS_DIR/intermediate_gifs/ ──"
        mkdir -p "$ARTIFACTS_DIR/intermediate_gifs"
        rsync -r --no-perms --no-owner --no-group --no-times \
            "$WORK_DIR/intermediate_gifs/" "$ARTIFACTS_DIR/intermediate_gifs/" \
            2>&1 | tail -3 || true
    fi
    echo "── archiving $WORK_DIR -> $ARTIFACTS_DIR/work_artifacts/ ──"
    mkdir -p "$ARTIFACTS_DIR/work_artifacts"
    rsync -r --no-perms --no-owner --no-group --no-times \
        --exclude='intermediate_gifs/' \
        "$WORK_DIR/" "$ARTIFACTS_DIR/work_artifacts/" 2>&1 | tail -5 || true

    # Auto-sync any offline-mode wandb runs to wandb.ai. Two cases this
    # catches:
    #   1. The whole job ran in offline mode (e.g. login node had no key,
    #      or network was unreachable when wandb.init fired).
    #   2. The job started online, lost the network mid-run, fell back to
    #      offline for the remainder.
    # In both cases an offline-run-* dir lands under $WORK_DIR/wandb/ and
    # we just rsync'd it into work_artifacts/wandb/. `wandb sync` reads
    # ~/.netrc the same way wandb.init() does, so the auth is already
    # in place — no env-pass needed.
    if [ "$WANDB" = "1" ] && ls "$ARTIFACTS_DIR/work_artifacts/wandb/offline-run-"* >/dev/null 2>&1; then
        echo "── auto-syncing offline wandb runs to wandb.ai ──"
        python -m wandb sync "$ARTIFACTS_DIR/work_artifacts/wandb/offline-run-"* \
            2>&1 | tail -20 || echo "[wandb] auto-sync failed (will need a manual retry)"
    fi

    echo "── $EXP_NAME (seed=$SEED) python exit: $EXIT_CODE ──"
    return "$EXIT_CODE"
}
