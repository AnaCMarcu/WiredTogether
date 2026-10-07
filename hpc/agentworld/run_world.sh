#!/bin/sh
# Start one Kaetram game server. Ports and the in-memory world come from the
# environment the harness / job script sets: PORT, API_PORT, SKIP_DATABASE,
# MAX_PLAYERS, HOST.
#
# Kaetram builds its config from .env files ONLY (packages/common/config.ts:
# dotenv-extended without includeProcessEnv) and ignores the process
# environment, but it layers ../../.env.$NODE_ENV on top. So each server writes
# its own override file named after its API port and selects it with NODE_ENV;
# concurrent servers sharing one checkout never share ports.
set -e
AW_HOME="${AGENTWORLD_HOME:-/opt/agentworld}"
if [ -n "${API_PORT:-}" ]; then
    NAME="aw${API_PORT}"
    {
        echo "PORT=${PORT:-7030}"
        echo "API_PORT=${API_PORT}"
        echo "API_ENABLED=true"
        echo "SKIP_DATABASE=${SKIP_DATABASE:-true}"
        echo "MAX_PLAYERS=${MAX_PLAYERS:-200}"
        echo "HOST=${HOST:-127.0.0.1}"
    } > "$AW_HOME/.env.$NAME"
    export NODE_ENV="$NAME"
fi
cd "$AW_HOME/packages/server"
exec npx --no-install tsx --preserve-symlinks ./src/main.ts
