#!/bin/bash
#SBATCH --job-name=exp02-llm-9b
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp02_llm_9b_%j.out
#SBATCH --error=slurm_logs/exp02_llm_9b_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP="${RUN_GROUP:-final}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-2500}"
WANDB_PROJECT="${WANDB_PROJECT:-final_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,baseline,final}"

# Plain LLM agents at 9B scale. Model-size ablation companion to exp01 (2B).
run_exp "exp02_llm_9b" "$MODEL_9B" --simultaneous
