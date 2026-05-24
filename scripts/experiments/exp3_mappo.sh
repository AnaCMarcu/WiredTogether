#!/bin/bash
#SBATCH --job-name=exp3-mappo
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp3_mappo-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp3_mappo-%j.out
# Submit:  sbatch scripts/experiments/exp3_mappo.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# MAPPO — action-level PPO with a SHARED centralized critic V(joint_state).
# No Hebbian. The classical multi-agent baseline.
# Tag-driven run dir: runs/legacy/exp3_mappo/seed_42/

EXP_NAME="exp3_mappo"
SEED=42
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_2B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 1500 \
    --warmup-time 300 \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"
