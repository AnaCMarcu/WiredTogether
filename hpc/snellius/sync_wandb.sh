#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Upload offline W&B runs to wandb.ai. Snellius compute nodes have no
# internet, so jobs log offline (env.sh sets WANDB_MODE=offline); run this on
# a LOGIN node afterwards. Needs ~/.netrc with your W&B key (see README).
#
#   bash hpc/snellius/sync_wandb.sh              # every run group
#   bash hpc/snellius/sync_wandb.sh comm_budget  # one run group
#
# Rerunning is harmless: each run keeps its id, so a re-upload lands on the
# same W&B run.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh" >/dev/null

root="$WT_WORKSPACE/WiredTogether/run_artifacts/${1:-}"
mapfile -t runs < <(find "$root" -type d -name 'offline-run-*' -path '*/work_artifacts/wandb/*' 2>/dev/null | sort)
if [ "${#runs[@]}" -eq 0 ]; then
    echo "[sync] no offline runs under $root"
    exit 0
fi

echo "[sync] ${#runs[@]} offline run(s) under $root"
apptainer exec --bind "$WT_WORKSPACE:$WT_WORKSPACE" "$WT_IMAGE" \
    python -m wandb sync "${runs[@]}"
