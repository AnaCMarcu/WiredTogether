#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# One-time AgentWorld setup on a LOGIN node (compute nodes have no internet).
# Needs no fakeroot and no mksquashfs, so it works on Snellius login nodes.
#
#     source hpc/snellius/env.sh                     # sets WT_WORKSPACE
#     bash hpc/agentworld/setup_aw.sh all            # = fetch node install check
#
# Steps (each can be run alone):
#   fetch    clone AgentWorld at hpc/agentworld/AGENTWORLD_COMMIT into
#            $WT_WORKSPACE/agentworld (or $AGENTWORLD_ROOT)
#   node     unpack docker://node:20-bookworm into $WT_WORKSPACE/images/node20
#            (full image: yarn needs git for the uWebSockets.js dependency)
#   install  yarn install (the yarn release vendored in the checkout) inside
#            that image, writing node_modules into the checkout
#   check    start the game server here for up to 3 minutes, wait for
#            /ai/world-status, stop it; prints the startup time
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WT_WORKSPACE:?set WT_WORKSPACE (source hpc/snellius/env.sh first)}"
AW_ROOT="${AGENTWORLD_ROOT:-$WS/agentworld}"
NODE_IMG="${AW_NODE_IMAGE:-$WS/images/node20}"
BINDS=(--bind "$WS:$WS")
if [ -n "${WT_BIND:-}" ]; then
    for _b in ${WT_BIND//,/ }; do BINDS+=(--bind "$_b:$_b"); done
fi

step_fetch() {
    AGENTWORLD_ROOT="$AW_ROOT" bash "$HERE/fetch_agentworld.sh"
}

step_node() {
    if [ -d "$NODE_IMG" ]; then
        echo "node image already at $NODE_IMG"
        return
    fi
    mkdir -p "$(dirname "$NODE_IMG")"
    APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}" \
        apptainer build --sandbox "$NODE_IMG" docker://node:20-bookworm
    apptainer exec "$NODE_IMG" sh -c 'node --version && git --version'
}

step_install() {
    [ -f "$AW_ROOT/package.json" ] || { echo "!! run: $0 fetch" >&2; exit 1; }
    [ -d "$NODE_IMG" ] || [ -e "$NODE_IMG" ] || { echo "!! run: $0 node" >&2; exit 1; }
    local cache="$WS/.cache/yarn"
    mkdir -p "$cache"
    apptainer exec "${BINDS[@]}" \
        --env HUSKY=0 --env YARN_CACHE_FOLDER="$cache" --env YARN_ENABLE_GLOBAL_CACHE=0 \
        --env HOME="$WS/.cache/home" \
        "$NODE_IMG" sh -c "
            set -e
            mkdir -p \"$WS/.cache/home\"
            cd \"$AW_ROOT\"
            YARN=\$(ls .yarn/releases/yarn-*.cjs | head -1)
            node \"\$YARN\" install
            [ -f .env ] || cp .env.defaults .env
            cd packages/server && node -e \"require('uws'); console.log('uws ok')\"
        "
}

step_check() {
    local port=$(( 20000 + 2 * ($$ % 4000) )) api log
    api=$(( port + 1 ))
    log="$WS/.cache/agentworld_check.log"
    echo "starting the game server for a check (ports $port/$api, log $log) ..."
    apptainer exec "${BINDS[@]}" \
        --env AGENTWORLD_HOME="$AW_ROOT" --env PORT="$port" --env API_PORT="$api" \
        --env API_ENABLED=true --env SKIP_DATABASE=true --env HOST=127.0.0.1 --env HUSKY=0 \
        "$NODE_IMG" sh "$HERE/run_world.sh" > "$log" 2>&1 &
    local pid=$! t0=$SECONDS ok=0
    for _ in $(seq 1 90); do
        if curl -sf "http://127.0.0.1:$api/ai/world-status" >/dev/null 2>&1; then ok=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if [ "$ok" = 1 ]; then
        echo "OK: game server answered /ai/world-status after $(( SECONDS - t0 )) s"
    else
        echo "!! game server did not answer; last log lines:" >&2
        tail -30 "$log" >&2
        exit 1
    fi
}

case "${1:-all}" in
    fetch) step_fetch ;;
    node) step_node ;;
    install) step_install ;;
    check) step_check ;;
    all) step_fetch; step_node; step_install; step_check ;;
    *) echo "usage: $0 [fetch|node|install|check|all]" >&2; exit 2 ;;
esac
