#!/bin/bash
# fix_craftium_cluster.sh
#
# Patches the installed craftium's multiagent_env.py on the HPC cluster
# to add server-ready polling (replaces the too-short time.sleep(5)).
#
# Usage:
#   bash scripts/fix_craftium_cluster.sh
#
# This is a BACKUP approach. The primary fix is in
# src/envs/openworld_multi_agents.py which uses _PatchedMarlCraftiumEnv
# to override reset() without touching site-packages.

set -e

# Find installed craftium path
CRAFTIUM_PATH=$(python -c "import craftium; import os; print(os.path.dirname(craftium.__file__))")
TARGET="$CRAFTIUM_PATH/multiagent_env.py"

if [ ! -f "$TARGET" ]; then
    echo "ERROR: Could not find $TARGET"
    exit 1
fi

echo "Found craftium at: $CRAFTIUM_PATH"
echo "Patching: $TARGET"

# Back up original
cp "$TARGET" "${TARGET}.bak"
echo "Backup saved to: ${TARGET}.bak"

# Copy our fixed version over it
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE="$PROJECT_ROOT/craftium/craftium/multiagent_env.py"

if [ ! -f "$SOURCE" ]; then
    echo "ERROR: Local fixed version not found at $SOURCE"
    echo "Make sure the craftium submodule is present in the project."
    exit 1
fi

cp "$SOURCE" "$TARGET"
echo "Done! Patched multiagent_env.py with server polling fix."
echo ""
echo "Verify with:"
echo "  python -c \"from craftium.multiagent_env import MarlCraftiumEnv; import inspect; print('polling' if 'listening on' in inspect.getsource(MarlCraftiumEnv.reset) else 'NOT patched')\""
