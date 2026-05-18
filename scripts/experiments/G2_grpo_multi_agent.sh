#!/bin/bash
#SBATCH --job-name=G2-grpo-multi
#SBATCH --partition=gpu-a100
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/G2-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/G2-%A_%a.err
# Submit: sbatch --array=0-2 G2_grpo_multi_agent.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# G2: Multi-agent GRPO (N=3 trained, 3B per-agent reward — the Stage-3 headline).
# No Hebbian. Compared against G1 (single-agent) and G4 (full Hebbian).
# Model: Qwen3.5-2B  |  RQ: GRPO scales to multi-agent

export LLM_MODEL_PATH="$MODEL_2B"

bash "$PROJECT_DIR/scripts/grpo.sh" \
    grpo_multi_agent.yaml \
    G2 \
    --set grpo.total_steps=1000 \
    --set grpo.team_reward=false

echo "G2 done (seed=$SEED)"
