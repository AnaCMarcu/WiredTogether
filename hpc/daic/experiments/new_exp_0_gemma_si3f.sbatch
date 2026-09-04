#!/bin/bash
#SBATCH --job-name=new-exp-0-gemma-si3f
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/new_exp_0_gemma_si3f_%j.out
#SBATCH --error=slurm_logs/new_exp_0_gemma_si3f_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# new_exp_0_gemma_si3f — social-interval Pareto sweep on the NEW Hebbian rule
# (three_factor + signed death LTD), replacing the reward_modulated sweep of
# new_exp_0_gemma_si.sbatch.
#
# EXACTLY the exp_qwen_three_factor_fdecay rule on Gemma 4 E4B (vision,
# prompt social module, 3 ep × 1000 steps) with ONE knob exposed:
# SOCIAL_INTERVAL → --social-interval N.
#
# Anchors for the new sweep:
#   x = 0 (no social module) → new_exp_0_gemma_base — rule-independent,
#       reused as-is; never re-run here.
#   interval 8 is a REAL sweep point this time (si3f8): the old hebbian
#   anchor used reward_modulated and new_exp_0_gemma_hebbian3f has no death
#   LTD, so neither anchors the new rule.
#
# One run:
#   SEED=42 SOCIAL_INTERVAL=2 sbatch new_exp_0_gemma_si3f.sbatch
# Whole sweep: use submit_pareto_social_3f.sh (idempotent + in-queue dedup).
#
# Lands in runs/pareto_social_3f/new_exp_0_gemma_si3f<N>/seed_<S>/ + wandb
# project pareto_social_3f. The base anchor stays in runs/new_exp_0_gemma/;
# analyses read both roots.
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project — nothing pools with the old pareto_social
# sweep. _common.sh binds WANDB_PROJECT at source time, so all of these go
# BEFORE the source line.
: "${RUN_GROUP:=pareto_social_3f}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=pareto_social_3f}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

# The one knob. Required, positive integer — refuse to burn 36 GPU-hours on a
# typo'd arm name like new_exp_0_gemma_si3f.
SOCIAL_INTERVAL="${SOCIAL_INTERVAL:-}"
case "$SOCIAL_INTERVAL" in
    ''|*[!0-9]*|0)
        echo "ERROR: SOCIAL_INTERVAL must be a positive integer (got '${SOCIAL_INTERVAL}')." >&2
        echo "       e.g. SEED=42 SOCIAL_INTERVAL=2 sbatch new_exp_0_gemma_si3f.sbatch" >&2
        exit 1
        ;;
esac

SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"
EXP_NAME="new_exp_0_gemma_si3f${SOCIAL_INTERVAL}"
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,hebbian,three_factor,fdecay,${SOCIAL_MODULE},pareto,si${SOCIAL_INTERVAL}}"

# NEW-rule hyper-parameters: byte-identical to exp_qwen_three_factor_fdecay
# (exp34/35) — the interval is the ONLY difference between sweep points.
HEBBIAN_ARGS=(
    --hebbian
    --hebbian-mode three_factor
    --hebbian-eta-0 0.001
    --hebbian-reward-norm 50
    --hebbian-decay 0.001
    --hebbian-eligibility-rho 0.9
    --hebbian-coact-floor 0.25
    --hebbian-death-ltd 0.05
    --hebbian-death-cap 10
    --hebbian-gamma 0.2
    --social-module "$SOCIAL_MODULE"
    --social-interval "$SOCIAL_INTERVAL"
)

echo "── new_exp_0_gemma_si3f: SOCIAL_INTERVAL=$SOCIAL_INTERVAL → $EXP_NAME ──"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    "${HEBBIAN_ARGS[@]}" \
    --simultaneous
