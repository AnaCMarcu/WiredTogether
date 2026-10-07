#!/bin/bash
# Clone AgentWorld at the pinned commit into $WT_WORKSPACE/agentworld and apply
# our patches (none yet). Their code stays OUTSIDE this repository (MPL-2.0);
# the harness imports agents/game_tools.py, task YAMLs and verifiers from here.
#
#     WT_WORKSPACE=/path/to/workspace bash hpc/agentworld/fetch_agentworld.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WT_WORKSPACE:?set WT_WORKSPACE}"
DEST="${AGENTWORLD_ROOT:-$WORKSPACE/agentworld}"
SHA="$(tr -d '[:space:]' < "$HERE/AGENTWORLD_COMMIT")"
if [ ! -d "$DEST/.git" ]; then
    git clone --filter=blob:none https://github.com/openagents-org/agentworld.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$SHA"
for p in "$HERE"/patches/*.patch; do
    [ -e "$p" ] || continue
    git -C "$DEST" apply --check "$p"
    git -C "$DEST" apply "$p"
    echo "applied $(basename "$p")"
done
echo "AgentWorld $SHA at $DEST"
