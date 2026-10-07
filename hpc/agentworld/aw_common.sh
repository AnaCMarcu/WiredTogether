#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Shared setup for AgentWorld runs (Snellius-first; DAIC works the same way).
#
#     source "$WORKSPACE/WiredTogether/hpc/agentworld/aw_common.sh"
#     run_aw <run name> <args for python -m mindforge.multi_agent_agentworld ...>
#
# (Source it through $WORKSPACE, not $(dirname "$0"): inside a SLURM job $0 is
# the spooled copy of the script, not the file in the repository.)
#
# One job = one vLLM server (same launch as hpc/slurm/experiments/_common.sh,
# plus prefix caching) + one game server, supervised on the host and restarted
# whenever the harness asks (episode reset) + the harness in the Gemma image.
# Ports: vLLM 30000 + JOBID % 16000; game 20000 + 2·(JOBID % 4000), API = +1.
#
# Game-server runtime, first match wins:
#   $AW_NODE_IMAGE                      explicit
#   $WORKSPACE/images/node20            sandbox of docker://node:20-bookworm and
#                                       node_modules installed into $AGENTWORLD_ROOT
#                                       (hpc/agentworld/setup_aw.sh; no fakeroot)
#   $WORKSPACE/images/agentworld.sif    self-contained image (agentworld.def)
# ────────────────────────────────────────────────────────────────────────────
WORKSPACE="${WT_WORKSPACE:-${WORKSPACE:?set WT_WORKSPACE (source hpc/snellius/env.sh)}}"
REPO="$WORKSPACE/WiredTogether"
IMG="${WT_IMAGE:-$WORKSPACE/images/wiredtogether_gemma4.sif}"
AW_ROOT="${AGENTWORLD_ROOT:-$WORKSPACE/agentworld}"
MODEL_LLM="${MODEL_LLM:-$WORKSPACE/models/gemma-4-E4B-it}"
SEEDS=(42 123 456)
SEED="${SEED:-${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}}"

aw_runtime() {
    # Prints "<image> <server command> <AGENTWORLD_HOME inside the container>".
    if [ -n "${AW_NODE_IMAGE:-}" ]; then
        echo "$AW_NODE_IMAGE sh:$REPO/hpc/agentworld/run_world.sh $AW_ROOT"
    elif [ -d "$WORKSPACE/images/node20" ]; then
        echo "$WORKSPACE/images/node20 sh:$REPO/hpc/agentworld/run_world.sh $AW_ROOT"
    elif [ -e "$WORKSPACE/images/agentworld.sif" ]; then
        echo "$WORKSPACE/images/agentworld.sif /usr/local/bin/run_world.sh /opt/agentworld"
    else
        return 1
    fi
}

run_aw() {
    local NAME="$1"; shift
    local RUN_DIR="$REPO/runs_aw/${RUN_GROUP:-agentworld}/$NAME/seed_$SEED"
    local SCRATCH_ROOT="${WT_SCRATCH:-/tmp/$USER}"
    local TMP_ROOT="$SCRATCH_ROOT/aw_${NAME}_${SLURM_JOB_ID:-$$}"
    mkdir -p "$RUN_DIR" "$TMP_ROOT"
    trap "rm -rf '$TMP_ROOT' 2>/dev/null || true" EXIT INT TERM
    echo "Run dir:  $RUN_DIR"
    echo "Seed:     $SEED   Node: $(hostname)   Job: ${SLURM_JOB_ID:-none}"

    [ -f "$AW_ROOT/agents/game_tools.py" ] || {
        echo "!! no AgentWorld checkout at $AW_ROOT: run hpc/agentworld/setup_aw.sh" >&2; return 1; }
    local RT AW_IMG AW_CMD AW_HOME
    RT="$(aw_runtime)" || { echo "!! no game-server runtime: run hpc/agentworld/setup_aw.sh" >&2; return 1; }
    read -r AW_IMG AW_CMD AW_HOME <<<"$RT"
    AW_CMD="${AW_CMD/sh:/sh }"

    # Extra binds, as in _common.sh: the scratch root when not under /tmp,
    # plus every directory in WT_BIND (shared images/models on Snellius).
    local EXTRA_BINDS=() _b
    case "$SCRATCH_ROOT" in /tmp|/tmp/*) ;; *) EXTRA_BINDS+=(--bind "$SCRATCH_ROOT:$SCRATCH_ROOT") ;; esac
    if [ -n "${WT_BIND:-}" ]; then
        for _b in ${WT_BIND//,/ }; do EXTRA_BINDS+=(--bind "$_b:$_b"); done
    fi

    # ── vLLM (default on; LLM_SERVER=api uses LLM_BASE_URL/LLM_MODEL instead) ──
    local VLLM_ENV=() VLLM_PID="" VLLM_TMP=""
    if [ "${LLM_SERVER:-vllm}" = "vllm" ]; then
        local VLLM_IMG="${VLLM_IMAGE:-}"
        if [ -z "$VLLM_IMG" ]; then
            if [ -d "$WORKSPACE/images/vllm" ]; then VLLM_IMG="$WORKSPACE/images/vllm"
            else VLLM_IMG="$WORKSPACE/images/vllm.sif"; fi
        fi
        [ -e "$VLLM_IMG" ] || { echo "!! no vLLM image at $VLLM_IMG" >&2; return 1; }
        local VLLM_PORT=$(( 30000 + ${SLURM_JOB_ID:-$$} % 16000 ))
        local VLLM_KEY
        VLLM_KEY="$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        # Short TMPDIR: vLLM's unix sockets must stay under 107 characters.
        VLLM_TMP="/tmp/wtv_${SLURM_JOB_ID:-$$}"
        mkdir -p "$VLLM_TMP" "$TMP_ROOT/vllm_home"
        echo "── starting vLLM ($VLLM_IMG) on port $VLLM_PORT ──"
        apptainer exec --nv --bind /tmp:/tmp --bind "$WORKSPACE:$WORKSPACE" \
            ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"} \
            --env TMPDIR="$VLLM_TMP" --env HF_HUB_OFFLINE=1 \
            --env HOME="$TMP_ROOT/vllm_home" \
            --env XDG_CACHE_HOME="$TMP_ROOT/vllm_home/cache" \
            --env VLLM_CACHE_ROOT="$TMP_ROOT/vllm_home/vllm_cache" \
            "$VLLM_IMG" vllm serve "$MODEL_LLM" --served-model-name wt \
                --host 127.0.0.1 --port "$VLLM_PORT" --api-key "$VLLM_KEY" \
                --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}" \
                --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" \
                --enable-prefix-caching \
            > "$RUN_DIR/vllm_server.log" 2>&1 &
        VLLM_PID=$!
        trap "kill $VLLM_PID 2>/dev/null; rm -rf '$TMP_ROOT' '$VLLM_TMP' 2>/dev/null || true" EXIT INT TERM
        local _ok=0 _i
        for _i in $(seq 1 120); do      # up to 20 min (first start compiles)
            if curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then _ok=1; break; fi
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 10
        done
        if [ "$_ok" != 1 ]; then
            echo "!! vLLM did not come up — see $RUN_DIR/vllm_server.log" >&2
            tail -30 "$RUN_DIR/vllm_server.log" >&2
            return 1
        fi
        echo "── vLLM healthy after ~$(( _i * 10 )) s ──"
        VLLM_ENV=(--env LLM_BACKEND=vllm
                  --env LLM_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
                  --env LLM_SERVER_KEY="$VLLM_KEY"
                  --env LLM_MODEL_PATH="$MODEL_LLM")
    fi

    # ── Game server, supervised on the host ───────────────────────────────
    # The harness container cannot start another container, so this loop owns
    # the server: it restarts it whenever the harness creates RESTART (and
    # deletes RESTART once the old server is dead — the harness waits for
    # that), and stops when STOP appears or the job exits.
    local GAME_PORT=$(( 20000 + 2 * (${SLURM_JOB_ID:-$$} % 4000) ))
    local API_PORT=$(( GAME_PORT + 1 ))
    local RESTART="$TMP_ROOT/restart_world" STOP="$TMP_ROOT/stop_world"
    echo "── game server: $AW_IMG (ports $GAME_PORT/$API_PORT) ──"
    (
        while [ ! -f "$STOP" ]; do
            apptainer exec --bind "$WORKSPACE:$WORKSPACE" \
                ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"} \
                --env AGENTWORLD_HOME="$AW_HOME" \
                --env PORT="$GAME_PORT" --env API_PORT="$API_PORT" --env API_ENABLED=true \
                --env SKIP_DATABASE=true --env MAX_PLAYERS="${MAX_PLAYERS:-256}" \
                --env HOST=127.0.0.1 --env HUSKY=0 \
                "$AW_IMG" $AW_CMD >> "$RUN_DIR/server.log" 2>&1 &
            SPID=$!
            while kill -0 "$SPID" 2>/dev/null && [ ! -f "$RESTART" ] && [ ! -f "$STOP" ]; do
                sleep 1
            done
            kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
            rm -f "$RESTART"            # the acknowledgement the harness waits for
            echo "── game server stopped $(date +%T) ──" >> "$RUN_DIR/server.log"
            sleep 1
        done
    ) &
    local SUPERVISOR=$!
    trap "touch '$STOP'; kill $SUPERVISOR $VLLM_PID 2>/dev/null; rm -rf '$TMP_ROOT' '$VLLM_TMP' 2>/dev/null || true" EXIT INT TERM

    apptainer exec --nv --bind /tmp:/tmp --bind "$WORKSPACE:$WORKSPACE" \
        ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"} \
        --env PYTHONPATH="$REPO/src" --env PYTHONUNBUFFERED=1 --env PYTHONIOENCODING=utf-8 \
        --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 \
        --env AGENTWORLD_ROOT="$AW_ROOT" \
        --env ST_MODEL_NAME="$WORKSPACE/models/all-MiniLM-L6-v2" \
        --env SENTENCE_TRANSFORMERS_HOME="$WORKSPACE/models" \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env LLM_ENABLE_THINKING=0 \
        ${VLLM_ENV[@]+"${VLLM_ENV[@]}"} \
        --pwd "$TMP_ROOT" \
        "$IMG" python -m mindforge.multi_agent_agentworld \
            --agentworld-root "$AW_ROOT" --seed "$SEED" --out "$RUN_DIR" \
            --server-url "http://127.0.0.1:$API_PORT" --restart-file "$RESTART" \
            "$@" \
        2>&1 | tee "$RUN_DIR/run.log"
    local rc=${PIPESTATUS[0]}
    touch "$STOP"
    echo "python exit: $rc"
    return "$rc"
}
