#!/bin/bash
# ============================================================
# Generic GRPO launcher — picks a config + experiment ID, sets
# the model path and per-seed output dirs, hands off to the
# entry point.
#
# Usage from a SLURM script (after sourcing _common.sh):
#   bash scripts/grpo.sh <config-name> <experiment-id> [extra --set ...]
#
# Example:
#   bash scripts/grpo.sh grpo_hebbian_full.yaml G4 \
#        --set grpo.total_steps=2000 \
#        --set grpo.learning_rate=1e-6
#
# Reads from the env (set by _common.sh):
#   PROJECT_DIR   — repo root
#   LLM_MODEL_PATH — base LLM (defaults to MODEL_2B)
#   SEED          — RNG seed (from SLURM array index)
#   SLURM_JOB_ID  — optional, used in run-dir naming when set
# ============================================================
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <config-name.yaml> <experiment-id> [extra --set ...]" >&2
    echo "  e.g.: $0 grpo_hebbian_full.yaml G4" >&2
    exit 64
fi

CONFIG_NAME="$1"
EXPERIMENT_ID="$2"
shift 2

CONFIG_PATH="${PROJECT_DIR}/configs/rlvr/${CONFIG_NAME}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 65
fi

: "${LLM_MODEL_PATH:?LLM_MODEL_PATH must be set (source _common.sh first)}"
: "${SEED:?SEED must be set (source _common.sh first)}"

# Run dir lives on /scratch so it survives the job.
RUN_DIR="/scratch/${USER}/WiredTogether/runs/grpo/${EXPERIMENT_ID}/seed_${SEED}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    RUN_DIR="${RUN_DIR}_job${SLURM_JOB_ID}"
fi
mkdir -p "$RUN_DIR"

echo "== GRPO launcher =="
echo "Config:     $CONFIG_PATH"
echo "Experiment: $EXPERIMENT_ID"
echo "Seed:       $SEED"
echo "Model:      $LLM_MODEL_PATH"
echo "Run dir:    $RUN_DIR"
echo "Extra:      $*"
echo "==================="

cd "${PROJECT_DIR}/src/mindforge"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

python multi_agent_craftium_grpo.py \
    --config "$CONFIG_PATH" \
    --set "seed=$SEED" \
    --set "llm.base_model_name=$LLM_MODEL_PATH" \
    --set "checkpoint_dir=${RUN_DIR}/grpo_lora" \
    --set "log_dir=${RUN_DIR}" \
    "$@"

echo "Run complete: $RUN_DIR"
