#!/bin/bash
#SBATCH --job-name=E10f-9nohebb
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E10f-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E10f-%A_%a.err
# Submit: sbatch --array=0-2 E10f_9agents_no_hebbian.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# H10: Scaling — 9 agents WITHOUT Hebbian
# Compared against: E10e (9 agents, Hebbian)
# Model: Qwen3.5-2B  |  RQ: RQ1+RQ2

export LLM_MODEL_PATH="$MODEL_2B"

python multi_agent_craftium.py \
    --num-agents 9 \
    --episodes 3 \
    --max-steps 200 \
    --warmup-time 300 \
    --rl \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-auto-token-opt \
    --seed "$SEED" \
    --experiment-id "E10f"

echo "E10f done (seed=$SEED)"
