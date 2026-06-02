#!/bin/bash
#SBATCH --job-name=exp15-llm-roles
#SBATCH --partition=gpu-a100
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp15_llm_roles-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp15_llm_roles-%j.out

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# LLM + ROLE-DIFFERENTIATED agents, NO Hebbian, NO RL. Heterogeneous
# counterpart to exp1_llm; completes the 2x2 ablation (role × hebbian).
run_exp "exp15_llm_roles" "$MODEL_2B" \
    --team-mode heterogeneous \
    --roles hunter,harvester,scouter --simultaneous
