#!/bin/bash
#SBATCH --job-name=exp01-llm-2b
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp01_llm_2b_%j.out
#SBATCH --error=slurm_logs/exp01_llm_2b_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP=final
EPISODES=5
MAX_STEPS=2500
WANDB_PROJECT=final_wired_together

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,2b,baseline,final}"

# Plain LLM agents (2B) — no RL training, no Hebbian. Floor of the comparison.
run_exp "exp01_llm_2b" "$MODEL_2B" --simultaneous
