#!/bin/bash
#SBATCH --job-name=sanity_stepall
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=slurm_logs/sanity_stepall_%j.out
#SBATCH --error=slurm_logs/sanity_stepall_%j.err

# Real-Luanti sanity check for CraftiumEnvironmentInterface.step_all()
# (simultaneous-move stepping). NO GPU / NO LLM: step_all is pure env and
# Luanti renders on CPU via llvmpipe, so this is cheap (~10-15 min incl.
# media warmup) and just confirms the new method drives real Luanti without
# crashing and produces well-formed, world-advancing output.
#
# Byte-identity (step == step_all at N=1) is proven separately on the
# deterministic mock: `python scripts/parity_test_stepping.py` (runs anywhere).
#
# Sources _common.sh ONLY for $IMG / $REPO / $WORKSPACE / $SEED — it does NOT
# call run_exp(); the apptainer block below mirrors run_exp's headless EGL/Xvfb
# wiring (trimmed: no model, no wandb, no gif dirs).

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

WORK_DIR="/tmp/$USER/sanity_stepall_${SLURM_JOB_ID:-nojob}"
mkdir -p "$WORK_DIR"

echo "── pre-flight: clearing stale luanti/minetest procs (best-effort) ──"
pkill -9 -u "$USER" -f "minetest" 2>/dev/null || true
pkill -9 -u "$USER" -f "luanti"   2>/dev/null || true
sleep 5

echo "== sanity_step_all =="
echo "Host:     $(hostname)"
echo "Image:    $IMG"
echo "Repo:     $REPO"
echo "Work dir: $WORK_DIR"
echo "Seed:     $SEED"
echo "====================="

apptainer exec \
    --bind /tmp:/tmp \
    --bind /tudelft.net:/tudelft.net \
    --bind /dev/dri:/dev/dri \
    --env PYTHONPATH="$REPO/src" \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONIOENCODING=utf-8 \
    --env LANG=C.UTF-8 \
    --env LC_ALL=C.UTF-8 \
    --env LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/craftium.libs \
    --env CRAFTIUM_ENV_DIR="$REPO/src/marl_craftium/craftium-envs/five-chambers" \
    --env SDL_VIDEODRIVER=dummy \
    --env SDL_AUDIODRIVER=dummy \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env MESA_GL_VERSION_OVERRIDE=3.3 \
    --env GALLIUM_DRIVER=llvmpipe \
    --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
    --pwd "$WORK_DIR" \
    "$IMG" \
    sh -c '
        # POSIX sh (no bash in image). Same Xvfb resolution ladder as run_exp.
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
            echo "[WARN] No Xvfb in image — Luanti needs real /dev/dri access on this node." >&2
            unset DISPLAY
            export EGL_PLATFORM=surfaceless
            exec "$@"
        fi
    ' sh \
    python -u "$REPO/scripts/sanity_step_all_realenv.py" \
        --num-agents 3 \
        --seed "$SEED" \
        --warmup-time 120 \
        --steps 20 \
    2>&1 | tee "$WORK_DIR/sanity.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "── sanity_step_all exit: $EXIT_CODE ──"
exit "$EXIT_CODE"
