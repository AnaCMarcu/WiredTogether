#!/bin/bash
#SBATCH --job-name=exp14-llm-hebbian-roles
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=exp14_llm_hebbian_roles_%j.out
#SBATCH --error=exp14_llm_hebbian_roles_%j.err

source "$(dirname "$0")/_common.sh"

# LLM + Hebbian + ROLE-DIFFERENTIATED agents. Tests whether role asymmetry
# gives the bonds something distinct to encode (vs the symmetric collapse).
# Identical Hebbian hyperparameters to exp2 — diff is --team-mode + --roles.
run_exp "exp14_llm_hebbian_roles" "$MODEL_2B" \
    --team-mode heterogeneous \
    --roles hunter,harvester,scouter \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2
