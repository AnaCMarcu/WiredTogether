#!/bin/bash
#SBATCH --job-name=mindforge
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
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
VLLM_PORT=8000
VLLM_MODEL_PATH=/scratch/acmarcu/models/Qwen2.5-3B-Instruct
VLLM_MODEL_NAME="Qwen2.5-3B-Instruct"

# Fallback: load model in-process (no server)
LOCAL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-2B

module purge
module load 2025
module load miniconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export DISPLAY=

# Ensure Craftium/Luanti can find libiconv
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"

which python
python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
nvidia-smi

# --- Start vLLM server in background ---
echo "Starting vLLM server with $VLLM_MODEL_NAME on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL_PATH" \
  --served-model-name "$VLLM_MODEL_NAME" \
  --port "$VLLM_PORT" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
  &
VLLM_PID=$!

# Wait for vLLM to be ready (up to 5 minutes)
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "vLLM server ready after ~$((i*5))s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM server died. Falling back to local model."
    VLLM_PID=""
    break
  fi
  sleep 5
done

# Configure LLM endpoint
if [ -n "${VLLM_PID:-}" ] && curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
  echo "Using vLLM server: $VLLM_MODEL_NAME at localhost:$VLLM_PORT"
  export LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
  export LLM_MODEL="$VLLM_MODEL_NAME"
  export LLM_API_KEY="no-key-needed"
else
  echo "vLLM not available, using local model: $LOCAL_MODEL_PATH"
  export LLM_MODEL_PATH="$LOCAL_MODEL_PATH"
fi

cd src/mindforge
python -c "from autogen_agentchat.messages import TextMessage; print('autogen OK')"
python multi_agent_craftium.py --num-agents 1 --episodes 10 --max-steps 200

# Cleanup vLLM server
if [ -n "${VLLM_PID:-}" ]; then
  echo "Stopping vLLM server (PID $VLLM_PID)..."
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
fi

echo "Done"
