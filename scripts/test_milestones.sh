#!/bin/bash
#SBATCH --job-name=five_chambers_milestones
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/%x-%j.err

PROJECT_DIR=/scratch/acmarcu/WiredTogether
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether

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
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Point Craftium at the five-chambers environment.
export CRAFTIUM_ENV_DIR="${PROJECT_DIR}/src/craftium/craftium-envs/five-chambers"

cd "$PROJECT_DIR"
echo "Python: $(which python)"
echo "CRAFTIUM_ENV_DIR: $CRAFTIUM_ENV_DIR"

mkdir -p /scratch/acmarcu/WiredTogether/slurm_logs

# Make src/ packages (rl_layer, hebbian, mindforge) importable
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
cd src/mindforge
python test_scripted_agent.py \
    --mode milestones \
    --num-agents 3 \
    --max-steps 600 \
    --warmup-time 120

echo "Done"
