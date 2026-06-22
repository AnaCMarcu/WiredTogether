#!/bin/bash
#SBATCH --job-name=exp09-llm-9b-allied-all
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp09_llm_9b_allied_all_%j.out
#SBATCH --error=slurm_logs/exp09_llm_9b_allied_all_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP="${RUN_GROUP:-final}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-2500}"
WANDB_PROJECT="${WANDB_PROJECT:-final_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,frozen,allied_all,social_bias,final}"

# ───────────────────────────────────────────────────────────────────────────
# 9B LLM-only + HARDCODED (frozen) Hebbian graph + SocialModule (bias).
# CONDITION A of 2:  ALL THREE AGENTS ALLIED  (uniform strong bonds).
#
#   W = 0.8 off-diagonal everywhere → every agent is strongly bonded to every
#   teammate. The social-bias module keeps all three in cooperate-mode.
#
# HYPOTHESIS (H1): the team completes MORE cooperative milestones (Ch2–Ch5)
# when all three are allied (this run) than when only two are allied
# (exp10_llm_9b_allied_pair) — and more than with no social module (exp02).
# Cooperative milestones m19/m22/m23/m28 require ALL THREE agents, so a fully
# allied team should reach more of them.
#
# Frozen graph (W never changes) ⇒ the effect is the social MECHANISM, not
# graph learning. No --rl ⇒ pure LLM agents; --hebbian-gamma 0 ⇒ no diffusion.
# Primary metric: # distinct Ch2–Ch5 milestones the team completes (per seed).
# ───────────────────────────────────────────────────────────────────────────

run_exp "exp09_llm_9b_allied_all" "$MODEL_9B" \
    --hebbian \
    --hebbian-freeze \
    --hebbian-preset uniform \
    --hebbian-bond-strong 0.8 \
    --hebbian-gamma 0.0 \
    --social-module bias \
    --simultaneous
