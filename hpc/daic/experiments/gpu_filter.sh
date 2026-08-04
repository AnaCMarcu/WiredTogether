#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# gpu_filter.sh — build the sbatch flags that keep jobs off GPUs too small to
# hold the model.
#
# Source it from a submit script, then splice the array into the sbatch call:
#
#     source "$(dirname "$0")/gpu_filter.sh"
#     sbatch ${GPU_FILTER_FLAGS[@]:+"${GPU_FILTER_FLAGS[@]}"} exp04_ippo.sbatch
#
# Produces:
#   --exclude=<cor1 + every node in bad_gpu_nodes.txt>
#   --constraint=$GPU_CONSTRAINT       (only when GPU_CONSTRAINT is set)
#
# The exclude list is the belt; _common.sh's MIN_GPU_MEM_MIB preflight is the
# braces — it aborts the job if a small GPU slips through anyway, and appends
# the offending node here so the next submit excludes it.
#
# GPU_CONSTRAINT is the better long-term filter but needs DAIC's node feature
# names, which vary. Find them with:
#     sinfo -N -h -o "%N %G %f" | sort -u
# then e.g.  GPU_CONSTRAINT=a40 bash submit_seed_extension.sh
# ────────────────────────────────────────────────────────────────────────────

_GPU_FILTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAD_NODE_FILE="${BAD_NODE_FILE:-$_GPU_FILTER_DIR/bad_gpu_nodes.txt}"

# cor1 was already excluded in every exp*.sbatch header for unrelated reasons;
# an --exclude on the command line REPLACES the header's, so keep it here.
BASE_EXCLUDE="${BASE_EXCLUDE:-cor1}"

# NOTE the trailing `|| true`: callers run under `set -euo pipefail`, and an
# all-comments node file makes `grep -v` exit 1, which would abort the submit.
_bad_nodes=""
if [ -r "$BAD_NODE_FILE" ]; then
    _bad_nodes=$( { sed -e 's/#.*//' -e 's/[[:space:]]//g' "$BAD_NODE_FILE" \
                    | grep -v '^$' | sort -u | paste -sd, -; } || true )
fi

if [ -n "$_bad_nodes" ]; then
    GPU_EXCLUDE="${BASE_EXCLUDE},${_bad_nodes}"
else
    GPU_EXCLUDE="$BASE_EXCLUDE"
fi

GPU_FILTER_FLAGS=(--exclude="$GPU_EXCLUDE")
if [ -n "${GPU_CONSTRAINT:-}" ]; then
    GPU_FILTER_FLAGS+=(--constraint="$GPU_CONSTRAINT")
fi

export GPU_EXCLUDE
