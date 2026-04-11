#!/bin/bash
#SBATCH --job-name=rl_survival
#SBATCH --partition=gpu-a100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/%x-%j.err

PROJECT_DIR=/scratch/acmarcu/WiredTogether
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether
LOCAL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-9B
RL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-2B

EXPERIMENT_ID=hebbian_rl_survival_v1
CKPT_ROOT=/scratch/acmarcu/WiredTogether/checkpoints/${EXPERIMENT_ID}

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

mkdir -p "$CKPT_ROOT"
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
    --survival-mode \
    --survival-episode 3 \
    --survival-gradual \
    --survival-gradual-delay 500 \
    --experiment-id "$EXPERIMENT_ID" \
    --checkpoint-dir "$CKPT_ROOT" \
    --checkpoint-interval 200

# Update latest checkpoint pointer for chained continuation jobs
LATEST=$(ls -td "${CKPT_ROOT}"/ep*_end 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "$LATEST" > "${CKPT_ROOT}/latest_checkpoint.txt"
    echo "Latest checkpoint: $LATEST"
fi

echo "Done"
