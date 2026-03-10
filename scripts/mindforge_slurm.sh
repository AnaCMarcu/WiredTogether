#!/bin/bash
#SBATCH --job-name=mindforge
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/scratch/acmarcu/WiredTogether
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether

# --- LLM Configuration ---
# Option A: Shared vLLM server (72B model, ask supervisor for hostname)
VLLM_URL="http://gpu-node:8000/v1"          # TODO: replace gpu-node with actual hostname
VLLM_MODEL="Qwen2.5-VL-72B-Instruct"

# Option B: Local model fallback (2B, loaded in-process)
LOCAL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-2B

module purge
module load 2025
module load miniconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export DISPLAY=

cd "$PROJECT_DIR"

which python
python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
nvidia-smi

# Try vLLM server first, fall back to local model
if curl -sf "${VLLM_URL%/v1}/health" > /dev/null 2>&1; then
  echo "Using shared vLLM server: $VLLM_MODEL at $VLLM_URL"
  export LLM_BASE_URL="$VLLM_URL"
  export LLM_MODEL="$VLLM_MODEL"
  export LLM_API_KEY="no-key-needed"
else
  echo "vLLM server not reachable, using local model: $LOCAL_MODEL_PATH"
  export LLM_MODEL_PATH="$LOCAL_MODEL_PATH"
fi

cd src/mindforge
python -c "from autogen_agentchat.messages import TextMessage; print('autogen OK')"
python multi_agent_craftium.py --num-agents 1 --episodes 10 --max-steps 200

echo "Done"
