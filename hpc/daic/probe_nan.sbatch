#!/bin/bash
#SBATCH --job-name=probe-nan
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
# Paths are relative to the SUBMISSION directory, and sbatch resolves them
# before the script body runs -- so the dir must already exist or the job
# dies at startup with nowhere to write. hpc/daic/experiments/slurm_logs/ is
# created by submit_pareto.sh / submit_cofiring.sh, so it is always there;
# the repo root has no slurm_logs/ (that cost jobs 12790125-6 on 2026-08-20).
# => SUBMIT THIS FROM THE REPO ROOT.
#SBATCH --output=hpc/daic/experiments/slurm_logs/probe_nan_%j.out
#SBATCH --error=hpc/daic/experiments/slurm_logs/probe_nan_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# probe_nan.sbatch — run scripts/probe_nan_logits.py on a GPU node.
#
# Diagnoses the NaN-logits failure that killed every LLM call in the 26b/31b
# pareto smoke runs, WITHOUT the Craftium/Minetest stack around it (a full
# smoke spends 900s on media warm-up before the first LLM call; this reaches
# it in ~2 min).
#
#   MODEL=gemma-4-31B-it sbatch hpc/daic/probe_nan.sbatch
#   MODEL=gemma-4-12B-it CONFIGS=sdpa_auto,eager_auto sbatch hpc/daic/probe_nan.sbatch
#   MODEL=gemma-4-31B-it CONFIGS=sdpa_single GPUS=1 \
#       sbatch --gres=gpu:1 --constraint=a100 hpc/daic/probe_nan.sbatch
#
# CUDA_LAUNCH_BLOCKING=1 is set by default: the device-side assert is raised
# asynchronously, so without it the Python traceback points at whatever op ran
# next rather than the one that actually produced the NaN. It costs throughput,
# which does not matter for a handful of forward passes.
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"
IMG="${WT_IMAGE:-$WORKSPACE/images/wiredtogether_gemma4.sif}"

MODEL="${MODEL:-gemma-4-31B-it}"
MODEL_PATH="$WORKSPACE/models/${MODEL##*/}"
CONFIGS="${CONFIGS:-sdpa_auto,eager_auto}"

TMP_ROOT="/tmp/$USER/probe_nan_${SLURM_JOB_ID:-nojob}"
export APPTAINER_TMPDIR="$TMP_ROOT/apptainer"
export TMPDIR="$TMP_ROOT"
mkdir -p "$APPTAINER_TMPDIR"
# Same HPC-support requirement as run_exp: leave nothing behind in /tmp.
trap "rm -rf '$TMP_ROOT' 2>/dev/null || true" EXIT INT TERM

echo "== probe_nan =="
echo "  node    : $(hostname)"
echo "  model   : $MODEL_PATH"
echo "  configs : $CONFIGS"
echo "  gpus    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd' | ' -)"
echo "==============="

apptainer exec --nv \
    --pid \
    --bind /tmp:/tmp \
    --bind /tudelft.net:/tudelft.net \
    --env PYTHONPATH="$REPO/src" \
    --env PYTHONUNBUFFERED=1 \
    --env TMPDIR="$TMPDIR" \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMG" \
    python -u "$REPO/scripts/probe_nan_logits.py" \
        --model "$MODEL_PATH" \
        --configs "$CONFIGS"

echo "== probe_nan done: $(date) =="
