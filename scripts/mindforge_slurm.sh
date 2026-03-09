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
MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-2B
SGLANG_PORT=8000
SGLANG_LOG=/scratch/acmarcu/WiredTogether/slurm_logs/sglang-${SLURM_JOB_ID}.log

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

cleanup() {
  if [[ -n "${SGLANG_PID:-}" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
    kill "$SGLANG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting SGLang server with model: $MODEL_PATH"
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$SGLANG_PORT" \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --tool-call-parser qwen3_coder \
  > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

echo "Waiting for SGLang server to start..."
READY=0
for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" > /dev/null; then
    echo "SGLang server is ready (took ${i}s)"
    READY=1
    break
  fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "SGLang server died unexpectedly"
    tail -100 "$SGLANG_LOG" || true
    exit 1
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "SGLang server failed to start after 300s"
  tail -100 "$SGLANG_LOG" || true
  exit 1
fi

curl -s "http://127.0.0.1:${SGLANG_PORT}/v1/models" || true

export LLM_BASE_URL="http://127.0.0.1:${SGLANG_PORT}/v1"
export LLM_MODEL="Qwen3.5-2B"
export LLM_API_KEY="no-key-needed"

cd src/mindforge
python multi_agent_craftium.py --num-agents 3 --episodes 10 --max-steps 200

echo "Done"