#!/bin/bash
#SBATCH --job-name=exp7-llm-9b
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp7_llm_9b-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp7_llm_9b-%j.out
# Submit:  sbatch scripts/experiments/exp7_llm_9b.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# Plain LLM agents at 9B scale — no RL training, no Hebbian.
# Model-size ablation companion to exp1 (which uses 2B).

EXP_NAME="exp7_llm_9b"
SEED=42
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_9B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 1500 \
    --warmup-time 300 \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"
