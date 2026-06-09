#!/bin/bash
#SBATCH --job-name=exp11-llm-9b-allied-none
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp11_llm_9b_allied_none_%j.out
#SBATCH --error=slurm_logs/exp11_llm_9b_allied_none_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Final experiment suite: 5 episodes × 2500 steps, results in runs/final/,
# logged to the "final_wired_together" wandb project.
RUN_GROUP=final
EPISODES=5
MAX_STEPS=2500
WANDB_PROJECT=final_wired_together

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,9b,hebbian,frozen,allied_none,null_control,social_bias,final}"

# ───────────────────────────────────────────────────────────────────────────
# 9B LLM-only + HARDCODED (frozen) Hebbian graph + SocialModule (bias).
# NULL CONTROL:  NO BONDS AT ALL  (uniform preset, strength 0.0 ⇒ W ≡ 0).
#
#   W = 0 everywhere. The Hebbian machinery and the social-bias module run
#   exactly as in exp09 (allied_all) / exp10 (allied_pair), but there are no
#   bonds for the social signal to act on. Same code path, zero structure.
#
# PURPOSE: isolates the BOND STRUCTURE from the social-module PLUMBING. exp02
# (plain 9B) is a weaker baseline — it runs a DIFFERENT code path (no
# --hebbian, no --social-module). This run is the strict null: if exp09 beats
# THIS, the gain is attributable to the bonds, not to merely switching the
# social module on. Anchors the strength axis (none → exp09 strong) and the
# topology axis (none → pair → full).
#
# Frozen graph ⇒ social MECHANISM, not learning. No --rl; --hebbian-gamma 0
# ⇒ no reward diffusion (and with W≡0 there is nothing to diffuse anyway).
# Primary metric: # distinct Ch2–Ch5 milestones the team completes (per seed).
# ───────────────────────────────────────────────────────────────────────────

run_exp "exp11_llm_9b_allied_none" "$MODEL_9B" \
    --hebbian \
    --hebbian-freeze \
    --hebbian-preset uniform \
    --hebbian-bond-strong 0.0 \
    --hebbian-gamma 0.0 \
    --social-module bias \
    --simultaneous
