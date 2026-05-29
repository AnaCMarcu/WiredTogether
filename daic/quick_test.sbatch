#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=general
#SBATCH --qos=interactive
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=quick_test_%j.out
#SBATCH --error=quick_test_%j.err

# 200-step LLM-only smoke test. Uses the same apptainer + wandb + env
# setup as the real experiments via _common.sh (pre-flight pkill, /tmp
# cwd, EGL software fallback, model paths, wandb wiring, archive rsync).

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

run_exp "quick_test" "$MODEL_9B" \
    --episodes 1 \
    --max-steps 100
