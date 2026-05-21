#!/bin/bash
#SBATCH --job-name=exp1-llm
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp1_llm-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp1_llm-%j.out
# Submit:  sbatch scripts/experiments/exp1_llm.sh
# Or all 6 at once:  bash scripts/experiments/submit_all.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# Plain LLM agents — no RL training, no Hebbian. The floor of the comparison.
# Tag-driven run dir: runs/legacy/exp1_llm/seed_42/
# Holds:  log.txt  final_metrics.json  episodes/  gifs/  run.log (tee'd)

EXP_NAME="exp1_llm"
SEED=42
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_2B"

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
