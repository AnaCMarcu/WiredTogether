#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Chained AgentWorld runs with MINDFORGE agents (chains/run_mf_chain.py).
#
#     source "${WT_REPO:-$WORKSPACE/WiredTogether}/hpc/agentworld/mf_chain_common.sh"
#     run_mf_chain_aw chains/pendant_staff_4.yaml [run_mf_chain.py flags ...]
#
# Same job layout as run_aw in aw_common.sh (sourced unchanged for its settings,
# seeds and game-server runtime): one vLLM server (Gemma, 16k context, as the
# earlier MindForge AgentWorld runs), one game server supervised on the host and
# restarted before every chain step, and the runner in the Gemma image. Only the
# command and the output folder differ from run_aw.
#
# Environment:
#   ARM        base | hebbian | uniform | freq (default hebbian)
#   SEED       default from the array index (42 123 456), as aw_common.sh
#   RUN_GROUP  output: $REPO/runs_aw/$RUN_GROUP/<chain>/mf_<arm>_s<seed>_j<job> (default mf_chains)
#   LLM_SERVER vllm (default) | none (with --plumbing)
#   VLLM_MAX_MODEL_LEN (16384), VLLM_GPU_UTIL (0.85)
# ────────────────────────────────────────────────────────────────────────────
source "${WT_REPO:-${WT_WORKSPACE:-$WORKSPACE}/WiredTogether}/hpc/agentworld/aw_common.sh"

run_mf_chain_aw() {
    local CHAIN="$1"; shift
    local ARM="${ARM:-hebbian}"
    local CHAIN_NAME
    CHAIN_NAME="$(sed -n 's/^chain:[[:space:]]*//p' "$REPO/$CHAIN" | tr -d '\r "'"'")"
    [ -n "$CHAIN_NAME" ] || { echo "!! $REPO/$CHAIN has no 'chain:' name" >&2; return 1; }
    local OUT="$REPO/runs_aw/${RUN_GROUP:-mf_chains}"
    local RUN_ID="${RUN_ID:-mf_${ARM}_s${SEED}_j${SLURM_JOB_ID:-$$}}"
    local RUN_DIR="$OUT/$CHAIN_NAME/$RUN_ID"      # the job's logs go here too
    local SCRATCH_ROOT="${WT_SCRATCH:-/tmp/$USER}"
    local TMP_ROOT="$SCRATCH_ROOT/awmf_${SLURM_JOB_ID:-$$}"
    mkdir -p "$RUN_DIR" "$TMP_ROOT"
    trap "rm -rf '$TMP_ROOT' 2>/dev/null || true" EXIT INT TERM
    echo "Chain:    $CHAIN ($CHAIN_NAME)   harness: MindForge   arm: $ARM   seed: $SEED"
    echo "Run dir:  $RUN_DIR"
    echo "Node:     $(hostname)   Job: ${SLURM_JOB_ID:-none}"

    [ -f "$AW_ROOT/agents/verification.py" ] || {
        echo "!! no AgentWorld checkout at $AW_ROOT: run hpc/agentworld/setup_aw.sh" >&2; return 1; }
    local RT AW_IMG AW_CMD AW_HOME
    RT="$(aw_runtime)" || { echo "!! no game-server runtime: run hpc/agentworld/setup_aw.sh" >&2; return 1; }
    read -r AW_IMG AW_CMD AW_HOME <<<"$RT"
    AW_CMD="${AW_CMD/sh:/sh }"

    local EXTRA_BINDS=() _b
    case "$SCRATCH_ROOT" in /tmp|/tmp/*) ;; *) EXTRA_BINDS+=(--bind "$SCRATCH_ROOT:$SCRATCH_ROOT") ;; esac
    if [ -n "${WT_BIND:-}" ]; then
        for _b in ${WT_BIND//,/ }; do EXTRA_BINDS+=(--bind "$_b:$_b"); done
    fi

    # ── vLLM (as run_aw; LLM_SERVER=none skips it for --plumbing) ─────────────
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

    # ── game server, supervised on the host (as run_aw) ──────────────────────
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
                --env SKIP_DATABASE=true --env MAX_PLAYERS="${MAX_PLAYERS:-64}" \
                --env HOST=127.0.0.1 --env HUSKY=0 \
                "$AW_IMG" $AW_CMD >> "$RUN_DIR/server.log" 2>&1 &
            SPID=$!
            while kill -0 "$SPID" 2>/dev/null && [ ! -f "$RESTART" ] && [ ! -f "$STOP" ]; do
                sleep 1
            done
            kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
            rm -f "$RESTART"            # the acknowledgement the runner waits for
            echo "── game server stopped $(date +%T) ──" >> "$RUN_DIR/server.log"
            sleep 1
        done
    ) &
    local SUPERVISOR=$!
    trap "touch '$STOP'; kill $SUPERVISOR $VLLM_PID 2>/dev/null; rm -rf '$TMP_ROOT' '$VLLM_TMP' 2>/dev/null || true" EXIT INT TERM

    # ── the chain ─────────────────────────────────────────────────────────
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
        "$IMG" python "$REPO/chains/run_mf_chain.py" "$REPO/$CHAIN" \
            --agentworld-root "$AW_ROOT" --arm "$ARM" --seed "$SEED" \
            --output "$OUT" --run-id "$RUN_ID" \
            --server-url "http://127.0.0.1:$API_PORT" --restart-file "$RESTART" \
            "$@" \
        2>&1 | tee "$RUN_DIR/run.log"
    local rc=${PIPESTATUS[0]}
    touch "$STOP"
    echo "python exit: $rc"
    return "$rc"
}
