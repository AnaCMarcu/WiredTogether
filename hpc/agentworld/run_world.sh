#!/bin/sh
# Start one Kaetram game server (inside agentworld.sif). Ports and the
# in-memory world come from the environment the harness sets
# (src/agentworld/server.py): PORT, API_PORT, SKIP_DATABASE=true, MAX_PLAYERS.
set -e
cd "${AGENTWORLD_HOME:-/opt/agentworld}/packages/server"
exec npx --no-install tsx --preserve-symlinks ./src/main.ts
