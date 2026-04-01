#!/bin/bash
# ============================================================
# Common environment setup for WiredTogether experiments.
# Source this from per-experiment SLURM scripts:
#   source "$(dirname "$0")/_common.sh"
# ============================================================

PROJECT_DIR=/scratch/acmarcu/WiredTogether
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether

# ── Modules & Conda ──
module purge
module load 2025
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# ── Environment variables ──
export LLM_ENABLE_THINKING=0
export SCRATCH=/scratch/acmarcu
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export DISPLAY=
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# Offline mode — compute nodes have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=/scratch/acmarcu/models/st_cache
export ST_MODEL_NAME=/scratch/acmarcu/models/all-MiniLM-L6-v2

# Ensure Craftium/Luanti can find libiconv
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ── Model paths ──
export MODEL_2B=/scratch/acmarcu/models/Qwen3.5-2B
export MODEL_9B=/scratch/acmarcu/models/Qwen3.5-9B

# Default: use 9B local model (override in experiment script if needed)
export LLM_MODEL_PATH="$MODEL_9B"

# ── Multi-seed support via SLURM array ──
# Usage: submit with `sbatch --array=0-2 script.sh`
# Seeds array — index by $SLURM_ARRAY_TASK_ID
SEEDS=(42 123 456)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
else
    SEED=42
fi

# ── Logging ──
mkdir -p /scratch/acmarcu/WiredTogether/slurm_logs

cd "$PROJECT_DIR/src/mindforge"

# Diagnostics
echo "== Experiment setup =="
echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "Python:  $(which python)"
echo "Model:   $LLM_MODEL_PATH"
echo "Seed:    $SEED"
echo "ArrayID: ${SLURM_ARRAY_TASK_ID:-N/A}"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "======================"
