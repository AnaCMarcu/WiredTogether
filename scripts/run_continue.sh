#!/bin/bash
#SBATCH --job-name=wt_cont
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/%x-%j.err

# ── Continuation job: resume from the latest checkpoint ──────────────────────
#
# Reads the checkpoint path written by run_first.sh (or the previous
# run_continue.sh job) from latest_checkpoint.txt, then resumes the run.
#
# Usage (manual):
#   sbatch scripts/run_continue.sh
#
# Usage (automatic chaining):
#   Use chain_jobs.sh — it submits this script with --dependency=afterany.
#
# Environment variables (override via sbatch --export):
#   EXPERIMENT_ID   — must match the first job (default: hebbian_rl_v1)
#   CKPT_ROOT       — checkpoint root directory (default: derived from EXPERIMENT_ID)
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_ID=${EXPERIMENT_ID:-hebbian_rl_v1}
PROJECT_DIR=/scratch/acmarcu/WiredTogether
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether
LOCAL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-9B
RL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-2B
CKPT_ROOT=${CKPT_ROOT:-/scratch/acmarcu/WiredTogether/checkpoints/${EXPERIMENT_ID}}

module purge
module load 2025
module load miniconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export LLM_ENABLE_THINKING=0
export SCRATCH=/scratch/acmarcu
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export DISPLAY=
export LIBGL_ALWAYS_SOFTWARE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MESA_GL_VERSION_OVERRIDE=3.3

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=/scratch/acmarcu/models/st_cache
export ST_MODEL_NAME=/scratch/acmarcu/models/all-MiniLM-L6-v2

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"

which python
python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
nvidia-smi

echo "Using local model: $LOCAL_MODEL_PATH"
export LLM_MODEL_PATH="$LOCAL_MODEL_PATH"

# ── Locate resume checkpoint ──────────────────────────────────────────────────
LATEST_FILE="${CKPT_ROOT}/latest_checkpoint.txt"
if [ ! -f "$LATEST_FILE" ]; then
    echo "ERROR: No latest_checkpoint.txt found in $CKPT_ROOT"
    echo "       Run run_first.sh before run_continue.sh."
    exit 1
fi
RESUME_DIR=$(cat "$LATEST_FILE")
if [ ! -f "${RESUME_DIR}/run_state.json" ]; then
    echo "ERROR: Checkpoint dir $RESUME_DIR has no run_state.json"
    exit 1
fi
echo "Resuming from: $RESUME_DIR"

mkdir -p /scratch/acmarcu/WiredTogether/slurm_logs

cd src/mindforge
python -c "from autogen_agentchat.messages import TextMessage; print('autogen OK')"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --warmup-time 300 \
    --rl \
    --rl-model-path "$RL_MODEL_PATH" \
    --hebbian \
    --hebbian-gamma 0.2 \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-radius 5.0 \
    --hebbian-init-weight 0.1 \
    --targeted-communication \
    --experiment-id "$EXPERIMENT_ID" \
    --checkpoint-dir "$CKPT_ROOT" \
    --checkpoint-interval 200 \
    --resume "$RESUME_DIR" \
    --resume-skip-warmup

# Update the latest checkpoint pointer
LATEST=$(ls -td "${CKPT_ROOT}"/ep*_end 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "$LATEST" > "${CKPT_ROOT}/latest_checkpoint.txt"
    echo "Updated latest checkpoint: $LATEST"
fi

echo "Done"
