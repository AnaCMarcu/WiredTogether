#!/bin/bash
#SBATCH --job-name=G3a-grpo-diff
#SBATCH --partition=gpu-a100
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/G3a-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/G3a-%A_%a.err
# Submit: sbatch --array=0-2 G3a_grpo_hebbian_diffusion.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# G3a: Stage-4a Hebbian reward diffusion ONLY (composition disabled).
# Isolates the per-joint reward spreading effect from cross-agent borrowing.
# Compared against: G2 (no Hebbian), G3b (composition only), G4 (full).
# Model: Qwen3.5-2B  |  RQ: Does Hebbian reward diffusion alone help GRPO?

export LLM_MODEL_PATH="$MODEL_2B"

bash "$PROJECT_DIR/scripts/grpo.sh" \
    grpo_hebbian_diffusion.yaml \
    G3a \
    --set grpo.total_steps=1000

echo "G3a done (seed=$SEED)"
