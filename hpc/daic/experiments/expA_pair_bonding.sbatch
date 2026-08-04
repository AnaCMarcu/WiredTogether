#!/bin/bash
#SBATCH --job-name=expA-pair-bonding
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/expA_pair_bonding_%A_%a.out
#SBATCH --error=slurm_logs/expA_pair_bonding_%A_%a.err
# influ3 excluded: its RTX 2080 Ti (11GB) cannot hold gemma-4-E4B, so
# device_map="auto" spills the model onto the CPU, where it then starves
# Luanti's llvmpipe software renderer of cores — measured 9h45m per env step
# there (2026-08-04 smoke), vs ~3 min of actual LLM time per call.
#SBATCH --exclude=cor1,influ3
#SBATCH --array=0-8

# ── Phase A of the pair-bonding transplant experiment ──
# 9 independent 2-agent runs restricted to Chambers 1-2 (solo skills + anvil
# coop). --max-chamber 2 suppresses the Python force-teleport timers past Ch2
# (the Ch1 timer still fires); organic Ch2→Ch3 progression via the anvils is
# deliberately NOT blocked. Afterwards, rank the runs and merge the top 3:
#
#   python src/mindforge/tools/merge_pair_runs.py rank \
#       runs/pair_bonding/expA_pair_bonding/seed_*
#   python src/mindforge/tools/merge_pair_runs.py merge \
#       --pair-run <top1> --pair-run <top2> --pair-run <top3> \
#       --out-dir runs/pair_bonding/merged/transplant
#   (add --shuffled for the control arm's inputs)
#
# Cognitive state (skills / episodic memory / curriculum) is exported to
# <run_dir>/agent_state/ automatically at every checkpoint and at run end.

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# 9 seeds — set AFTER sourcing: _common.sh's own SEEDS table only has 3
# entries and would resolve array ids 3-8 to an empty SEED.
PAIR_SEEDS=(42 123 456 789 1011 1213 1415 1617 1819)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED=${PAIR_SEEDS[$SLURM_ARRAY_TASK_ID]}
fi

NUM_AGENTS=2
RUN_GROUP="${RUN_GROUP:-pair_bonding}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,social_prompt,pair_bonding,phaseA}"

# Same condition as exp08 (9B + Hebbian + SocialModule prompt coupling),
# just 2 agents capped at Ch2.
run_exp "expA_pair_bonding" "$MODEL_9B" \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --social-module prompt --simultaneous \
    --max-chamber 2
