#!/bin/bash
#SBATCH --job-name=mindforge-think
#SBATCH --partition=gpu-a100
#SBATCH --time=06:00:00
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

LOCAL_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-9B

module purge
module load 2025
module load miniconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export DISPLAY=
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# Force offline mode — compute nodes have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Ensure Craftium/Luanti can find libiconv
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Enable Qwen3.5 thinking mode
export LLM_ENABLE_THINKING=1

cd "$PROJECT_DIR"

which python
python -c "import sys; print('python:', sys.executable)"
python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
nvidia-smi

# Use local model directly (no vLLM server)
echo "Using local model: $LOCAL_MODEL_PATH (thinking ENABLED)"
export LLM_MODEL_PATH="$LOCAL_MODEL_PATH"

cd src/mindforge
python -c "from autogen_agentchat.messages import TextMessage; print('autogen OK')"
python multi_agent_craftium.py --num-agents 2 --episodes 3 --max-steps 50

echo "Done"
