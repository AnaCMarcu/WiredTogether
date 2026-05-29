#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=quick_test_%j.out
#SBATCH --error=quick_test_%j.err

# ── DAIC port of scripts/experiments/quick_test.sh ──
# 50-step LLM-only smoke test (Qwen3.5-2B for fast inference).
# Original DelftBlue env replaced with Apptainer; /scratch → PRB.

set -uo pipefail   # NOT -e: pkill must be best-effort

# Pre-flight: clear stale luanti processes from previous failed runs on this node.
echo "── pre-flight: clearing stale luanti processes (best-effort) ──"
pkill -9 -u "$USER" -f "minetest" 2>/dev/null || true
pkill -9 -u "$USER" -f "luanti"   2>/dev/null || true
sleep 5
echo "── pre-flight done ──"

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
IMG="$WORKSPACE/images/wiredtogether.sif"
REPO="$WORKSPACE/WiredTogether"

EXP_NAME="quick_test"
SEED="${SEED:-42}"
RUN_DIR="$WORKSPACE/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

# Working directory MUST be on a filesystem that supports chmod, because
# craftium's MTServerOnly copies its bundled luanti binary into a
# `minetest-srv-<uuid>/bin/` directory in cwd via shutil.copy2.
# PRB (/tudelft.net) rejects chmod, so we cd to node-local /tmp.
WORK_DIR="/tmp/$USER/${EXP_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "== quick_test (DAIC) =="
echo "Host:      $(hostname)"
echo "Image:     $IMG"
echo "Repo:      $REPO"
echo "Run dir:   $RUN_DIR"
echo "Work dir:  $WORK_DIR"
echo "Seed:      $SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "======================="

apptainer exec --nv \
    --bind /tmp:/tmp \
    --bind /tudelft.net:/tudelft.net \
    --env PYTHONPATH="$REPO/src" \
    --env PYTHONUNBUFFERED=1 \
    --env LLM_MODEL_PATH="$WORKSPACE/models/Qwen3.5-9B" \
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
    --env GALLIUM_DRIVER=llvmpipe \
    --env EGL_PLATFORM=surfaceless \
    --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
    --env LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/craftium.libs \
    --env PYTHONIOENCODING=utf-8 \
    --env LANG=C.UTF-8 \
    --env LC_ALL=C.UTF-8 \
    --pwd "$WORK_DIR" \
    "$IMG" \
    python -u "$REPO/src/mindforge/multi_agent_craftium.py" \
        --num-agents 3 \
        --episodes 2 \
        --max-steps 200 \
        --warmup-time 300 \
        --seed "$SEED" \
        --experiment-id "$EXP_NAME" \
        --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

EXIT_CODE=${PIPESTATUS[0]}

# Salvage anything craftium / the script dropped in cwd (minetest-srv-* debug
# logs, gifs, etc.) back to PRB so we can inspect it after the job ends.
# rsync without perms/owners avoids PRB chmod failures.
echo "── archiving $WORK_DIR -> $RUN_DIR/work_artifacts/ ──"
mkdir -p "$RUN_DIR/work_artifacts"
rsync -r --no-perms --no-owner --no-group --no-times \
    "$WORK_DIR/" "$RUN_DIR/work_artifacts/" 2>&1 | tail -5 || true

echo "── ${EXP_NAME} python exit code: $EXIT_CODE (0 = success) ──"
exit "$EXIT_CODE"
