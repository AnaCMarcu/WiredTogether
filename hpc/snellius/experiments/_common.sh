#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Shared Snellius (SURF) setup for WiredTogether experiments.
#
# Ports hpc/daic/experiments/_common.sh to Snellius. SAME Apptainer image and
# same run_exp contract; the Snellius-specific deltas are:
#   * scratch → $TMPDIR (per-job node-local NVMe, auto-removed by SLURM at job
#     end) — so the /tmp-orphan problem from DAIC does NOT exist here.
#   * paths under the project space /projects/prjs1879.
#   * wandb OFFLINE by default — Snellius COMPUTE nodes have NO outbound
#     internet, so wandb online would hang. Runs are written to disk and synced
#     LATER from an int/login node (see the printed `wandb sync` hint at the end,
#     or hpc/snellius/sync_wandb.sh).
#   * GPU requested via SLURM `--gpus 1` on `--partition=gpu_a100` (in each
#     per-experiment sbatch, NOT here).
#
# ── Confirmed on Snellius (2026-06) ────────────────────────────────────────
#   * partition gpu_a100, max walltime 5-00:00:00 (120h).
#   * apptainer at /usr/bin/apptainer (no `module load` needed).
#   * $TMPDIR = /scratch-local/<jobid> (node-local NVMe, auto-wiped at job end).
#   * /projects/prjs1879 is a SYMLINK → /gpfs/work2/0/prjs1879. We use the REAL
#     path below so apptainer --bind and in-container paths resolve identically
#     (binding the symlink is unreliable).
# ── VERIFY once: whether jobs need an explicit account. The sbatch headers set
#   `--account=tdsei14435` (from accinfo); if SLURM rejects it, run
#   `sacctmgr show assoc user=$USER format=account` and use the name it prints.
# ────────────────────────────────────────────────────────────────────────────

WORKSPACE=/gpfs/work2/0/prjs1879   # = /projects/prjs1879 (resolved symlink)
IMG="$WORKSPACE/wiredtogether/images/wiredtogether.sif"
REPO="$WORKSPACE/WiredTogether"
MODELS="$WORKSPACE/wiredtogether/models"

MODEL_2B="$MODELS/Qwen3.5-2B"
MODEL_9B="$MODELS/Qwen3.5-9B"

# Seed resolution: SLURM array index → SEED env var → 123 (mirrors DAIC).
SEEDS=(42 123 456)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
elif [ -z "${SEED:-}" ]; then
    SEED=123
fi

# ── Weights & Biases ──────────────────────────────────────────────────────
# OFFLINE on Snellius compute nodes (no internet). Runs land under
# run_artifacts/.../work_artifacts/wandb/offline-run-* and are synced from an
# int node afterward. Auth (~/.netrc) is only needed at sync time, not here.
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-final_wired_together}"
WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-}"
WANDB_MODE="${WANDB_MODE:-offline}"      # ← offline; do NOT set online on compute nodes

# Usage: run_exp <EXP_NAME> <LLM_MODEL_PATH> [extra python args...]
run_exp() {
    local EXP_NAME="$1"
    local LLM_MODEL="$2"
    shift 2

    local RUN_GROUP="${RUN_GROUP:-final}"
    local RUN_DIR="$REPO/runs/${RUN_GROUP}/${EXP_NAME}/seed_${SEED}"
    local ARTIFACTS_DIR="$REPO/run_artifacts/${RUN_GROUP}/${EXP_NAME}/seed_${SEED}"

    # Node-local scratch. $TMPDIR is Snellius' per-job dir on local NVMe and is
    # auto-removed by SLURM at job end, so nothing accumulates in /tmp. The
    # trap below is belt-and-suspenders (and matches the DAIC code path).
    local SCRATCH="${TMPDIR:-/tmp/$USER}"
    local WORK_DIR="$SCRATCH/${EXP_NAME}_${SLURM_JOB_ID:-nojob}"
    export APPTAINER_TMPDIR="$SCRATCH/apptainer_${EXP_NAME}_${SLURM_JOB_ID:-nojob}"
    mkdir -p "$RUN_DIR" "$ARTIFACTS_DIR" "$WORK_DIR" "$APPTAINER_TMPDIR" "$WORK_DIR/empty_dri"
    trap "rm -rf '$WORK_DIR' '$APPTAINER_TMPDIR' 2>/dev/null || true" EXIT INT TERM

    local WANDB_FLAGS=()
    if [ "$WANDB" = "1" ]; then
        local tags="exp_${EXP_NAME},seed_${SEED}"
        [ -n "$WANDB_EXTRA_TAGS" ] && tags="${tags},${WANDB_EXTRA_TAGS}"
        WANDB_FLAGS=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-tags "$tags" --wandb-upload-artifacts)
    fi

    echo "== $EXP_NAME =="
    echo "Host:    $(hostname)"
    echo "Image:   $IMG"
    echo "Run dir: $RUN_DIR"
    echo "Scratch: $WORK_DIR   (TMPDIR=$TMPDIR)"
    echo "Model:   $LLM_MODEL   Seed: $SEED   wandb: $WANDB_MODE"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo "============================"

    # Same headless-rendering strategy as DAIC: mask /dev/dri with an empty dir
    # so Mesa falls back to CPU llvmpipe (compute A100s have no render node), and
    # wrap python in Xvfb so Irrlicht/EGL has a software surface. --pid gives the
    # container its own pid namespace so craftium's Minetest procs are reaped on
    # exit (otherwise apptainer hangs on FUSE teardown — see the DAIC fix).
    apptainer exec --nv \
        --pid \
        --bind "$SCRATCH:$SCRATCH" \
        --bind /gpfs/work2/0/prjs1879:/gpfs/work2/0/prjs1879 \
        --bind "$WORK_DIR/empty_dri:/dev/dri" \
        --env PYTHONPATH="$REPO/src" \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONIOENCODING=utf-8 \
        --env LANG=C.UTF-8 \
        --env LC_ALL=C.UTF-8 \
        --env LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/craftium.libs \
        --env LLM_MODEL_PATH="$LLM_MODEL" \
        --env LLM_ENABLE_THINKING=0 \
        --env ST_MODEL_NAME="$MODELS/all-MiniLM-L6-v2" \
        --env SENTENCE_TRANSFORMERS_HOME="$MODELS" \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        --env CRAFTIUM_ENV_DIR="$REPO/src/marl_craftium/craftium-envs/five-chambers" \
        --env WIREDTOGETHER_RUNS_ROOT="$REPO/runs" \
        --env SDL_VIDEODRIVER=dummy \
        --env SDL_AUDIODRIVER=dummy \
        --env LIBGL_ALWAYS_SOFTWARE=1 \
        --env MESA_GL_VERSION_OVERRIDE=3.3 \
        --env GALLIUM_DRIVER=llvmpipe \
        --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
        --env EGL_PLATFORM=surfaceless \
        --env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
        --env WANDB_MODE="$WANDB_MODE" \
        --env WANDB_DIR="$WORK_DIR" \
        --env WANDB_SILENT=true \
        --env TMPDIR="$WORK_DIR" \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --env WIREDTOGETHER_INTERMEDIATE_GIF_DIR="$WORK_DIR/intermediate_gifs" \
        --pwd "$WORK_DIR" \
        "$IMG" \
        sh -c '
            if command -v xvfb-run >/dev/null 2>&1 && command -v xauth >/dev/null 2>&1; then
                exec xvfb-run -a -s "-screen 0 1024x768x24 -nolisten tcp" "$@"
            elif command -v Xvfb >/dev/null 2>&1; then
                Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
                _xvfb_pid=$!
                trap "kill $_xvfb_pid 2>/dev/null || true" EXIT
                export DISPLAY=:99
                export EGL_PLATFORM=surfaceless
                export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}"
                sleep 1
                exec "$@"
            else
                echo "[WARN] no Xvfb in image — needs real GPU render access" >&2
                unset DISPLAY
                export EGL_PLATFORM=surfaceless
                exec "$@"
            fi
        ' sh \
        python -u "$REPO/src/mindforge/multi_agent_craftium.py" \
            --num-agents 3 \
            --episodes "${EPISODES:-3}" \
            --max-steps "${MAX_STEPS:-1000}" \
            --warmup-time 300 \
            --seed "$SEED" \
            --experiment-id "$EXP_NAME" \
            --tag "$EXP_NAME" \
            "${WANDB_FLAGS[@]}" \
            "$@" \
        2>&1 | tee "$RUN_DIR/run.log"

    local EXIT_CODE=${PIPESTATUS[0]}

    # Salvage heavy artifacts + the OFFLINE wandb dir to the project space BEFORE
    # $TMPDIR is wiped. (Primary results — metrics/episodes/plots/checkpoints —
    # already went straight to $REPO/runs via WIREDTOGETHER_RUNS_ROOT.)
    echo "── archiving $WORK_DIR → $ARTIFACTS_DIR/work_artifacts/ ──"
    mkdir -p "$ARTIFACTS_DIR/work_artifacts"
    rsync -r --no-perms --no-owner --no-group --no-times \
        "$WORK_DIR/" "$ARTIFACTS_DIR/work_artifacts/" 2>&1 | tail -5 || true

    echo "── $EXP_NAME (seed=$SEED) python exit: $EXIT_CODE ──"
    if [ "$WANDB" = "1" ]; then
        echo "── wandb ran OFFLINE. From an int node, sync with:"
        echo "     apptainer exec $IMG python -m wandb sync \\"
        echo "       $ARTIFACTS_DIR/work_artifacts/wandb/offline-run-*"
    fi
    return "$EXIT_CODE"
}
