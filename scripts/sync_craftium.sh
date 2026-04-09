#!/bin/bash
# Sync modified craftium files to DelftBlue.
# craftium/ is a nested git repo so these files can't be tracked by the parent repo.
# Run this after any change to the files listed below.

HOST="${1:-delftblue}"
REMOTE="/scratch/acmarcu/WiredTogether"

FILES=(
    "craftium/craftium/minetest.py"
    "craftium/craftium-envs/voxel-libre2/clientmods/craftium_env/init.lua"
    "craftium/craftium-envs/voxel-libre2/mods/craftium_env/init.lua"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Syncing craftium files to $HOST:$REMOTE ..."
for f in "${FILES[@]}"; do
    remote_path="$REMOTE/$f"
    remote_dir="$(dirname "$remote_path")"
    ssh "$HOST" "mkdir -p '$remote_dir'"
    scp "$REPO_ROOT/$f" "$HOST:$remote_path"
    echo "  ✓ $f"
done
echo "Done."
