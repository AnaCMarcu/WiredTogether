#!/bin/bash
#SBATCH --job-name=G3b-grpo-comp
#SBATCH --partition=gpu-a100
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/G3b-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/G3b-%A_%a.err
# Submit: sbatch --array=0-2 G3b_grpo_hebbian_composition.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# G3b: Stage-4b Hebbian-weighted group composition ONLY (diffusion disabled).
# Isolates the cross-agent trajectory borrowing effect.
# Uses Option 4b-i (clipped off-policy — the GRPO surrogate clip handles
# the off-policy bias on its own). See docs/rlvr_grpo_plan.md §5.4 Stage 4b.
# Compared against: G2 (no Hebbian), G3a (diffusion only), G4 (full).
# Model: Qwen3.5-2B  |  RQ: Does Hebbian group composition alone help GRPO?

export LLM_MODEL_PATH="$MODEL_2B"

bash "$PROJECT_DIR/scripts/grpo.sh" \
    grpo_hebbian_composition.yaml \
    G3b \
    --set grpo.total_steps=1000 \
    --set grpo.hebbian_borrow_fraction=0.25

echo "G3b done (seed=$SEED)"
