#!/bin/bash
#SBATCH --job-name=exp9-llm-27b
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp9_llm_27b-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp9_llm_27b-%j.out
# Submit:  sbatch scripts/experiments/exp9_llm_27b.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# Plain LLM agents at 27B (BF16) scale — no RL training, no Hebbian.
# Top-of-line model-size point. ~54GB weights, fits 1× A100 80GB
# (gpu-a100 partition) with room for KV cache.

EXP_NAME="exp9_llm_27b"
SEED=42
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_27B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 1000 \
    --warmup-time 300 \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"
