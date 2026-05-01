#!/bin/bash
#SBATCH --job-name=rl_gatherers
#SBATCH --partition=gpu-a100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/%x-%j.err

# set -uo pipefail  # disabled: unset vars in module/conda init cause instant exit

PROJECT_DIR=/scratch/acmarcu/WiredTogether
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether

# --- LLM Configuration ---
VLLM_PORT=8000
VLLM_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-9B
VLLM_MODEL_NAME="Qwen3.5-9B"

# Fallback: load model in-process (no server)
LOCAL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-9B

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

# Force offline mode — compute nodes have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# sentence-transformers: pre-download all-MiniLM-L6-v2 on login node:
#   python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); m.save('/scratch/acmarcu/models/all-MiniLM-L6-v2')"
export SENTENCE_TRANSFORMERS_HOME=/scratch/acmarcu/models/st_cache
export ST_MODEL_NAME=/scratch/acmarcu/models/all-MiniLM-L6-v2

# Ensure Craftium/Luanti can find libiconv
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CRAFTIUM_ENV_DIR="${PROJECT_DIR}/src/marl_craftium/craftium-envs/five-chambers"

cd "$PROJECT_DIR"

which python
python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
nvidia-smi

# Use local model directly (no vLLM server)
echo "Using local model: $LOCAL_MODEL_PATH"
export LLM_MODEL_PATH="$LOCAL_MODEL_PATH"
VLLM_PID=""

# Make src/ packages (rl_layer, hebbian, mindforge) importable
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
cd src/mindforge
python -c "from autogen_agentchat.messages import TextMessage; print('autogen OK')"
python multi_agent_craftium.py --num-agents 3 --episodes 5 \
    --warmup-time 300 --rl --rl-model-path /scratch/acmarcu/models/Qwen3.5-2B \
    --team-mode homogeneous-agent \
    --experiment-id rl_gatherers_v1

# Cleanup vLLM server
if [ -n "${VLLM_PID:-}" ]; then
  echo "Stopping vLLM server (PID $VLLM_PID)..."
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
fi

echo "Done"
