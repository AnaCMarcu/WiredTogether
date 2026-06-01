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
# Auth lives in ~/.netrc (per-user, chmod-able, auto-mounted into the
# container by apptainer). To set it up once:
#     cat > ~/.netrc <<EOF
#     machine api.wandb.ai
#       login user
#       password <YOUR-KEY>
#     EOF
#     chmod 600 ~/.netrc
#
# wandb.init() reads ~/.netrc automatically — we don't need to pass
# WANDB_API_KEY via env. If ~/.netrc is missing or the key is invalid,
# wandb's own init throws and our python-side wrapper catches it; the
# end-of-job salvage hook below will still capture any offline-mode
# dirs and auto-sync them so the data survives.
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-wired-together}"
WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-}"  # comma-separated, appended to auto tags
WANDB_MODE="${WANDB_MODE:-online}"        # default online; offline if env-forced

# Usage: run_exp <EXP_NAME> <LLM_MODEL_PATH> [extra python args...]
run_exp() {
    local EXP_NAME="$1"
    local LLM_MODEL="$2"
    shift 2

    local RUN_DIR="$REPO/runs/legacy/${EXP_NAME}/seed_${SEED}"
    # Heavy artifacts (craftium debug.txt, wandb offline-runs, intermediate
    # per-100-step gifs) live in a PARALLEL tree so the runs/ dir stays
    # small and fast to scp. Mirror the same exp/seed structure for easy
    # cross-referencing.
    local ARTIFACTS_DIR="$REPO/run_artifacts/legacy/${EXP_NAME}/seed_${SEED}"
    local WORK_DIR="/tmp/$USER/${EXP_NAME}_${SLURM_JOB_ID:-nojob}"
    mkdir -p "$RUN_DIR" "$ARTIFACTS_DIR" "$WORK_DIR"

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

    # Headless-display strategy:
    # Luanti renders frames through Irrlicht → EGL, independently of SDL.
    # SDL_VIDEODRIVER=offscreen silences the SDL side but Irrlicht still
    # tries to grab /dev/dri/renderD* for EGL hardware acceleration.
    # On DAIC nodes where the user lacks the `render` / `video` group on
    # those devices ("Permission denied" + "Not allowed to force software
    # rendering when API explicitly selects a hardware device"), MT client 0
    # exits with code 1 and the run dies with `Server socket listen timeout
    # reached`. Observed pattern: 13/14 experiments in the 2026-05-30
    # batch failed this way; the lone survivor (exp15) landed on a node
    # where the user happened to have render-device permissions.
    #
    # Fix is two layered:
    #   1. --bind /dev/dri so the container sees the host's render nodes
    #      when SLURM grants access.
    #   2. xvfb-run wrapper around python so Irrlicht/EGL has a working
    #      software X surface (llvmpipe) even on nodes where GPU device
    #      access is denied. Xvfb owns its own framebuffer; EGL can render
    #      into it without ever touching /dev/dri.
    apptainer exec --nv \
        --bind /tmp:/tmp \
        --bind /tudelft.net:/tudelft.net \
        --bind /dev/dri:/dev/dri \
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
        --env LIBGL_ALWAYS_SOFTWARE=1 \
        --env MESA_GL_VERSION_OVERRIDE=3.3 \
        --env GALLIUM_DRIVER=llvmpipe \
        --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
        --env WANDB_MODE="${WANDB_MODE:-online}" \
        --env WANDB_DIR="$WORK_DIR" \
        --env WANDB_SILENT=true \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --env WIREDTOGETHER_INTERMEDIATE_GIF_DIR="$WORK_DIR/intermediate_gifs" \
        --pwd "$WORK_DIR" \
        "$IMG" \
        sh -c '
            # Resolve Xvfb invocation. xvfb-run is the convenient wrapper
            # (auto-picks a free display, cleans up on exit) but some
            # apptainer images only ship the raw Xvfb binary. Try both
            # in order; fall back to no-wrapper as last resort so we at
            # least produce a diagnostic in run.log rather than a silent
            # hang on missing-binary.
            # POSIX sh, not bash — the container has no bash on PATH.
            # xvfb-run also requires xauth at runtime; if xauth is missing
            # (the Debian xvfb package does NOT pull it in by default), we
            # MUST skip the xvfb-run branch even though it exists, because
            # it will exit with "xauth command not found" before ever
            # starting Xvfb. The manual-Xvfb branch below has no such
            # dependency and works fine without xauth.
            if command -v xvfb-run >/dev/null 2>&1 && command -v xauth >/dev/null 2>&1; then
                exec xvfb-run -a -s "-screen 0 1024x768x24 -nolisten tcp" "$@"
            elif command -v Xvfb >/dev/null 2>&1; then
                Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
                _xvfb_pid=$!
                trap "kill $_xvfb_pid 2>/dev/null || true" EXIT
                export DISPLAY=:99
                # Belt-and-braces for Luanti builds compiled EGL-only that
                # ignore DISPLAY and try to grab /dev/dri/renderD* directly:
                # force EGL to use the surfaceless platform (CPU-side via
                # llvmpipe) so the renderer succeeds without GPU device
                # permissions even when it bypasses Xvfb.
                export EGL_PLATFORM=surfaceless
                export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}"
                sleep 1
                exec "$@"
            else
                echo "[WARN] Neither xvfb-run+xauth nor Xvfb found in image — Luanti will need real GPU access (/dev/dri permissions) on this node." >&2
                unset DISPLAY
                export EGL_PLATFORM=surfaceless
                exec "$@"
            fi
        ' sh \
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
    # Split salvage:
    #   - $WORK_DIR/intermediate_gifs/ → $ARTIFACTS_DIR/intermediate_gifs/
    #     (separately so they're easy to find / delete)
    #   - everything else (craftium debug.txt, wandb offline-runs, etc.)
    #     → $ARTIFACTS_DIR/work_artifacts/
    # The runs/<exp>/seed_<N>/ tree stays small (just plots, episodes/,
    # gifs/ with FINAL per-episode gifs only, config.json, log.txt).
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
    # we just rsync'd it into work_artifacts/wandb/. python -m wandb sync
    # reads ~/.netrc the same way wandb.init() does, so the auth is
    # already in place — no env-pass needed.
    if [ "$WANDB" = "1" ] && ls "$ARTIFACTS_DIR/work_artifacts/wandb/offline-run-"* >/dev/null 2>&1; then
        echo "── auto-syncing offline wandb runs to wandb.ai ──"
        apptainer exec --nv \
            --bind /tudelft.net:/tudelft.net \
            "$IMG" \
            python -m wandb sync "$ARTIFACTS_DIR/work_artifacts/wandb/offline-run-"* \
            2>&1 | tail -20 || echo "[wandb] auto-sync failed (will need a manual retry)"
    fi

    echo "── $EXP_NAME (seed=$SEED) python exit: $EXIT_CODE ──"
    return "$EXIT_CODE"
}
