#!/bin/bash
#SBATCH --job-name=fast
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=fast%j.out
#SBATCH --error=fast%j.err

# 200-step LLM-only smoke test. Uses the same apptainer + wandb + env
# setup as the real experiments via _common.sh (pre-flight pkill, /tmp
# cwd, EGL software fallback, model paths, wandb wiring, archive rsync).

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

run_exp "fast" "$MODEL_9B" \
    --episodes 3 \
    --max-steps 100 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --social-module prompt
