#!/bin/bash
#SBATCH --job-name=exp2-llm-hebbian
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp2_llm_hebbian-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp2_llm_hebbian-%j.out
# Submit:  sbatch scripts/experiments/exp2_llm_hebbian.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# LLM + Hebbian-in-prompt — no RL training. The Hebbian graph updates per env
# step from co-activity + comm events; the current bond weights are injected
# into the agent's prompt via the {social_bonds} placeholder.
# Tag-driven run dir: runs/legacy/exp2_llm_hebbian/seed_42/

EXP_NAME="exp2_llm_hebbian"
# Honor seed inherited from _common.sh (array index, SEED env var, or default 42).
SEED="${SEED:-42}"
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_2B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 1000 \
    --warmup-time 300 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    --simultaneous \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"
