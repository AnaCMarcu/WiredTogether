#!/usr/bin/env bash
# verify_baseline.sh — quick sanity check that vanilla EPyMARL MAPPO on
# plain LBF still trains.
#
# Phase 7's "Verification gate" deliverable. Not a published-baseline
# reproduction (the published numbers are at 20M+ steps); this is a
# 50k-step smoke test that exits non-zero if EPyMARL/lbforaging/torch
# integration is broken.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EPYMARL_ROOT="${REPO_ROOT}/epymarl"

if [ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]; then
    PY="${REPO_ROOT}/.venv/Scripts/python.exe"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PY="${REPO_ROOT}/.venv/bin/python"
else
    PY="python"
fi

cd "${EPYMARL_ROOT}"
echo "[verify_baseline] training MAPPO on plain lbforaging:Foraging-10x10-3p-3f-v3 for 50k steps"
"${PY}" src/main.py --config=mappo --env-config=gymma \
    with env_args.time_limit=50 \
         env_args.key="lbforaging:Foraging-10x10-3p-3f-v3" \
         t_max=50000 use_cuda=False

echo "[verify_baseline] OK"
