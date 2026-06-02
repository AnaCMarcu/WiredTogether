#!/bin/bash
#SBATCH --job-name=exp15-llm-roles
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp15_llm_roles_%j.out
#SBATCH --error=slurm_logs/exp15_llm_roles_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

# LLM + ROLE-DIFFERENTIATED agents, NO Hebbian, NO RL. Heterogeneous
# counterpart to exp1_llm; completes the 2x2 ablation (role × hebbian).
run_exp "exp15_llm_roles" "$MODEL_2B" \
    --team-mode heterogeneous \
    --roles hunter,harvester,scouter --simultaneous
