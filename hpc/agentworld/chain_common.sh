#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Chained AgentWorld runs (chains/run_chain.py) on Snellius / DAIC.
#
#     source "${WT_REPO:-$WORKSPACE/WiredTogether}/hpc/agentworld/chain_common.sh"
#     run_chain_aw chains/metallurgy_4.yaml [run_chain.py flags ...]
#
# One job = one vLLM server (Gemma, as in aw_common.sh, but 32k context: the
# AgentWorld agent keeps its whole conversation and its system prompt alone is
# ~7k tokens) + one game server + chains/run_chain.py in the Gemma image.
# Settings, paths, seeds and the game-server runtime come from aw_common.sh,
# which is sourced unchanged. No restart supervisor: a chain resets characters
# itself (new_character: true), so one server serves every step.
#
# Environment:
#   ARM           reference | memory | hebbian (default hebbian)
#   SEED          default from the array index (42 123 456), as aw_common.sh
#   RUN_GROUP     output: $REPO/runs_aw/$RUN_GROUP/<chain>/<arm>_s<seed>_j<job> (default chains)
#   AGENT_CONFIG  default chains/configs/wired_openai.yaml (reads MODEL_* below)
#   LLM_SERVER    vllm (default) | api (export MODEL_NAME, MODEL_API_KEY, MODEL_BASE_URL
#                 and pick AGENT_CONFIG) | none (with --random-agent)
#   VLLM_MAX_MODEL_LEN (32768), VLLM_GPU_UTIL (0.85), VLLM_EXTRA_ARGS (appended to vllm serve)
# ────────────────────────────────────────────────────────────────────────────
source "${WT_REPO:-${WT_WORKSPACE:-$WORKSPACE}/WiredTogether}/hpc/agentworld/aw_common.sh"

run_chain_aw() {
    local CHAIN="$1"; shift
    local ARM="${ARM:-hebbian}"
    local CHAIN_NAME
    CHAIN_NAME="$(sed -n 's/^chain:[[:space:]]*//p' "$REPO/$CHAIN" | tr -d '\r "'"'")"
    [ -n "$CHAIN_NAME" ] || { echo "!! $REPO/$CHAIN has no 'chain:' name" >&2; return 1; }
    local OUT="$REPO/runs_aw/${RUN_GROUP:-chains}"
    local RUN_ID="${RUN_ID:-${ARM}_s${SEED}_j${SLURM_JOB_ID:-$$}}"
    local RUN_DIR="$OUT/$CHAIN_NAME/$RUN_ID"
    local SCRATCH_ROOT="${WT_SCRATCH:-/tmp/$USER}"
    local TMP_ROOT="$SCRATCH_ROOT/awc_${SLURM_JOB_ID:-$$}"
    mkdir -p "$RUN_DIR" "$TMP_ROOT"
    trap "rm -rf '$TMP_ROOT' 2>/dev/null || true" EXIT INT TERM
    echo "Chain:    $CHAIN ($CHAIN_NAME)   arm: $ARM   seed: $SEED"
    echo "Run dir:  $RUN_DIR"
    echo "Node:     $(hostname)   Job: ${SLURM_JOB_ID:-none}"

    [ -f "$AW_ROOT/agents/run.py" ] || {
        echo "!! no AgentWorld checkout at $AW_ROOT: run hpc/agentworld/setup_aw.sh" >&2; return 1; }
    [ -f "$AW_ROOT/scripts/score_psr.py" ] || echo "!! $AW_ROOT predates 23a07d52: PSR will be empty" >&2
    local RT AW_IMG AW_CMD AW_HOME
    RT="$(aw_runtime)" || { echo "!! no game-server runtime: run hpc/agentworld/setup_aw.sh" >&2; return 1; }
    read -r AW_IMG AW_CMD AW_HOME <<<"$RT"
    AW_CMD="${AW_CMD/sh:/sh }"

    local EXTRA_BINDS=() _b
    case "$SCRATCH_ROOT" in /tmp|/tmp/*) ;; *) EXTRA_BINDS+=(--bind "$SCRATCH_ROOT:$SCRATCH_ROOT") ;; esac
    if [ -n "${WT_BIND:-}" ]; then
        for _b in ${WT_BIND//,/ }; do EXTRA_BINDS+=(--bind "$_b:$_b"); done
    fi

    # ── model ──────────────────────────────────────────────────────────────
    local MODEL_ENV=() VLLM_PID="" VLLM_TMP="" PIDS=""
    case "${LLM_SERVER:-vllm}" in
    vllm)
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
        # shellcheck disable=SC2086
        apptainer exec --nv --bind /tmp:/tmp --bind "$WORKSPACE:$WORKSPACE" \
            ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"} \
            --env TMPDIR="$VLLM_TMP" --env HF_HUB_OFFLINE=1 \
            --env HOME="$TMP_ROOT/vllm_home" \
            --env XDG_CACHE_HOME="$TMP_ROOT/vllm_home/cache" \
            --env VLLM_CACHE_ROOT="$TMP_ROOT/vllm_home/vllm_cache" \
            "$VLLM_IMG" vllm serve "$MODEL_LLM" --served-model-name wt \
                --host 127.0.0.1 --port "$VLLM_PORT" --api-key "$VLLM_KEY" \
                --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
                --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" \
                --enable-prefix-caching ${VLLM_EXTRA_ARGS:-} \
            > "$RUN_DIR/vllm_server.log" 2>&1 &
        VLLM_PID=$!
        PIDS="$VLLM_PID"
        trap "kill $PIDS 2>/dev/null; rm -rf '$TMP_ROOT' '$VLLM_TMP' 2>/dev/null || true" EXIT INT TERM
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
        MODEL_ENV=(--env MODEL_NAME=wt --env MODEL_API_KEY="$VLLM_KEY"
                   --env MODEL_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1")
        ;;
    api)
        : "${MODEL_NAME:?LLM_SERVER=api needs MODEL_NAME}" "${MODEL_BASE_URL:?}" "${MODEL_API_KEY:?}"
        MODEL_ENV=(--env MODEL_NAME="$MODEL_NAME" --env MODEL_API_KEY="$MODEL_API_KEY"
                   --env MODEL_BASE_URL="$MODEL_BASE_URL")
        [ -n "${ANTHROPIC_API_KEY:-}" ] && MODEL_ENV+=(--env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY")
        ;;
    none) ;;
    *) echo "!! LLM_SERVER must be vllm, api or none" >&2; return 1 ;;
    esac

    # ── game server (one for the whole chain) ─────────────────────────────
    local GAME_PORT=$(( 20000 + 2 * (${SLURM_JOB_ID:-$$} % 4000) ))
    local API_PORT=$(( GAME_PORT + 1 ))
    echo "── game server: $AW_IMG (ports $GAME_PORT/$API_PORT) ──"
    # shellcheck disable=SC2086
    apptainer exec --bind "$WORKSPACE:$WORKSPACE" \
        ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"} \
        --env AGENTWORLD_HOME="$AW_HOME" \
        --env PORT="$GAME_PORT" --env API_PORT="$API_PORT" --env API_ENABLED=true \
        --env SKIP_DATABASE=true --env MAX_PLAYERS="${MAX_PLAYERS:-64}" \
        --env HOST=127.0.0.1 --env HUSKY=0 \
        "$AW_IMG" $AW_CMD > "$RUN_DIR/server.log" 2>&1 &
    local SERVER_PID=$!
    PIDS="$PIDS $SERVER_PID"
    trap "kill $PIDS 2>/dev/null; rm -rf '$TMP_ROOT' '$VLLM_TMP' 2>/dev/null || true" EXIT INT TERM
    local _up=0 _j
    for _j in $(seq 1 60); do             # up to 5 min
        if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$API_PORT/"; then _up=1; break; fi
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 5
    done
    if [ "$_up" != 1 ]; then
        echo "!! game server did not come up — see $RUN_DIR/server.log" >&2
        tail -30 "$RUN_DIR/server.log" >&2
        return 1
    fi
    echo "── game server up after ~$(( _j * 5 )) s ──"

    # ── the chain ─────────────────────────────────────────────────────────
    apptainer exec --nv --bind /tmp:/tmp --bind "$WORKSPACE:$WORKSPACE" \
        ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"} \
        --env PYTHONPATH="$REPO/src" --env PYTHONUNBUFFERED=1 --env PYTHONIOENCODING=utf-8 \
        --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 \
        --env AGENTWORLD_ROOT="$AW_ROOT" \
        --env AGENTWORLD_BASE_URL="http://127.0.0.1:$API_PORT" \
        ${MODEL_ENV[@]+"${MODEL_ENV[@]}"} \
        --pwd "$REPO" \
        "$IMG" sh -c 'python -c "import jinja2, yaml, requests, numpy" || {
                          echo "!! the image lacks jinja2/yaml/requests/numpy for AgentWorld" >&2; exit 3; }
                      exec python chains/run_chain.py "$@"' run_chain \
            "$CHAIN" --agent "${AGENT_CONFIG:-chains/configs/wired_openai.yaml}" \
            --arm "$ARM" --seed "$SEED" --output "$OUT" --run-id "$RUN_ID" "$@" \
        2>&1 | tee "$RUN_DIR/run.log"
    local rc=${PIPESTATUS[0]}
    echo "python exit: $rc"
    return "$rc"
}
