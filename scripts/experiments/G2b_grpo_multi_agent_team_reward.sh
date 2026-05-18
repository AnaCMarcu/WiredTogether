#!/bin/bash
#SBATCH --job-name=G2b-grpo-team
#SBATCH --partition=gpu-a100
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/G2b-%A_%a.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/G2b-%A_%a.err
# Submit: sbatch --array=0-2 G2b_grpo_multi_agent_team_reward.sh

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# G2b: Stage-3 Option 3A — team-shared reward across the joint rollout.
# Same setup as G2 but with team_reward=true. The cooperation-axis ablation.
# Compared against: G2 (3B per-agent).
# Model: Qwen3.5-2B  |  RQ: Team reward vs per-agent reward in GRPO

export LLM_MODEL_PATH="$MODEL_2B"

bash "$PROJECT_DIR/scripts/grpo.sh" \
    grpo_multi_agent.yaml \
    G2b \
    --set grpo.total_steps=1000 \
    --set grpo.team_reward=true

echo "G2b done (seed=$SEED)"
