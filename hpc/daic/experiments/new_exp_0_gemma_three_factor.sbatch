#!/bin/bash
#SBATCH --job-name=new-exp-0-gemma-3f
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/new_exp_0_gemma_3f_%j.out
#SBATCH --error=slurm_logs/new_exp_0_gemma_3f_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# new_exp_0_gemma_three_factor — Gemma-4-E4B + the three-factor Hebbian rule.
#
# Same shape as new_exp_0_gemma.sbatch's HEBBIAN=1 arm (3 episodes × 1000
# steps, vision, social_module=prompt), differing ONLY in the update rule:
#
#   --hebbian-mode three_factor
#     co-activity leaves an eligibility trace e ← ρ·e + c and the reward
#     converts the ACCUMULATED trace into weight,
#         ΔW = η0·c·(1−W) + η+·(|r_bond|/R)·e·(1−W) − λ·W,
#     with monotone co-activity (co-location counts at the floor even for a
#     silent pair; the comm term is not gated to far-apart pairs).
#
# Parameters follow the open-loop prototype's panel C
# (paper_assets_hebbian_terms/three_factor_prototype_seed456.*):
#     η0 0.001 (vs 0.005 in the reward_modulated arm — reward, not
#               co-presence, is the driver)
#     λ  0.001 (vs 0.005 — reward is sparse so it cannot saturate W,
#               which lets credit persist)
#     ρ  0.9, coact floor 0.25, R 50, η+ 0.05 (default), α/radius unchanged.
#
# Writes to runs/new_exp_0_gemma/new_exp_0_gemma_hebbian3f/seed_<N>/ so it
# sits next to the base and reward_modulated arms for direct comparison.
#
# Three seeds:
#   cd $REPO/hpc/daic/experiments
#   for s in 42 123 456; do SEED=$s sbatch new_exp_0_gemma_three_factor.sbatch; done
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=new_exp_0_gemma}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=new_exp_0_gemma}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"
EXP_NAME="new_exp_0_gemma_hebbian3f"
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,hebbian,three_factor,${SOCIAL_MODULE},new_exp_0}"

HEBBIAN_ARGS=(
    --hebbian
    --hebbian-mode three_factor
    --hebbian-eta-0 0.001
    --hebbian-reward-norm 50
    --hebbian-decay 0.001
    --hebbian-eligibility-rho 0.9
    --hebbian-coact-floor 0.25
    --hebbian-gamma 0.2
    --social-module "$SOCIAL_MODULE"
)

echo "── new_exp_0_gemma_three_factor → $EXP_NAME (seed=${SEED:-unset}) ──"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    "${HEBBIAN_ARGS[@]}" \
    --simultaneous
