#!/bin/bash
#SBATCH --job-name=norl_test
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/norl_test_%j.out
#SBATCH --error=slurm_logs/norl_test_%j.err

# Isolation test for the LLM NaN crash. Turn-based (NO --simultaneous), NO
# --rl, 1 episode, 5 steps. Exercises NONE of the new simultaneous-stepping
# code — only the existing cognitive LLM path.
#
#   - If this ALSO crashes with "probability tensor contains inf/nan" at step 1
#     => the bug is in the cognitive model layer (local_model_client), NOT the
#     stepping change. We then fix the model layer.
#   - If this runs 5 steps cleanly (agents produce real actions, no {} spam, no
#     CUDA assert) => the bug is specific to --simultaneous, and Claude digs in.

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

WANDB=0 run_exp "norl_test" "$MODEL_9B" \
    --episodes 1 \
    --max-steps 5
