#!/bin/bash
#SBATCH --job-name=exp2-llm-hebbian
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp2_llm_hebbian_%j.out
#SBATCH --error=slurm_logs/exp2_llm_hebbian_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,2b,hebbian}"

# LLM + Hebbian-in-prompt. Hebbian graph updates per env step from co-activity
# + comm events; bond weights are injected into the agent's prompt via the
# {social_bonds} placeholder. No RL training.
run_exp "exp2_llm_hebbian" "$MODEL_2B" \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 --simultaneous
