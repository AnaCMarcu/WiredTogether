#!/bin/bash
#SBATCH --job-name=G4-grpo-hebbian-full
#SBATCH --partition=gpu-a100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/G4-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/G4-%A_%a.err
# Submit: sbatch --array=0-2 G4_grpo_hebbian_full.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# G4: GRPO + Hebbian (reward diffusion 4a + group composition 4b together).
# THIS IS THE HEADLINE: GRPO+Hebbian vs GRPO-vanilla (G2) vs MAPPO baseline (E5).
# Model: Qwen3.5-2B  |  RQ: Does Hebbian-augmented GRPO beat plain GRPO?

export LLM_MODEL_PATH="$MODEL_2B"

bash "$PROJECT_DIR/scripts/grpo.sh" \
    grpo_hebbian_full.yaml \
    G4 \
    --set grpo.total_steps=1000 \
    --set grpo.hebbian_borrow_fraction=0.25

echo "G4 done (seed=$SEED)"
