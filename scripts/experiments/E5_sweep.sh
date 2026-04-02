#!/bin/bash
#SBATCH --job-name=E5-sweep
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/E5sweep-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/E5sweep-%A_%a.err
# Submit: sbatch --array=0-8 E5_sweep.sh
# 9 configs: η ∈ {0.01, 0.05, 0.1} × λ ∈ {0.001, 0.005, 0.01}

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# H5 hyperparameter sweep (β fixed at 1.0 for now)
export LLM_MODEL_PATH="$MODEL_2B"

# Parameter grid (9 combos, indexed by SLURM_ARRAY_TASK_ID)
LTP_VALS=(0.01 0.01 0.01 0.05 0.05 0.05 0.1 0.1 0.1)
DECAY_VALS=(0.001 0.005 0.01 0.001 0.005 0.01 0.001 0.005 0.01)

IDX=${SLURM_ARRAY_TASK_ID:-0}
LTP=${LTP_VALS[$IDX]}
DECAY=${DECAY_VALS[$IDX]}

echo "Sweep config $IDX: ltp=$LTP, decay=$DECAY"

python multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 5 \
    --max-steps 200 \
    --warmup-time 300 \
    --rl \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-auto-token-opt \
    --hebbian \
    --hebbian-ltp "$LTP" \
    --hebbian-ltd 0.005 \
    --hebbian-decay "$DECAY" \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --seed 42 \
    --experiment-id "E5_sweep_ltp${LTP}_decay${DECAY}"

echo "E5 sweep done (ltp=$LTP, decay=$DECAY)"
