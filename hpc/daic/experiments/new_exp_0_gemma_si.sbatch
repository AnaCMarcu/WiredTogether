#!/bin/bash
#SBATCH --job-name=new-exp-0-gemma-si
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/new_exp_0_gemma_si_%j.out
#SBATCH --error=slurm_logs/new_exp_0_gemma_si_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# new_exp_0_gemma_si — social-interval Pareto sweep (compute vs performance).
#
# EXACTLY the new_exp_0_gemma HEBBIAN arm (Gemma 4, vision, plastic Hebbian
# graph feeding the prompt-coupled social module, 3 ep × 1000 steps), with ONE
# knob exposed: SOCIAL_INTERVAL → --social-interval N (the social module
# deliberates via LLM every N steps, cached in between).
#
# The sweep's two anchor points already exist and are NOT re-run here:
#   interval "none" (x = 0)  → new_exp_0_gemma_base     (no social module)
#   interval 8 (code default)→ new_exp_0_gemma_hebbian  (never passes the flag)
# Submitting SOCIAL_INTERVAL=8 therefore duplicates the hebbian arm under a
# new name — only do that deliberately, as a reproduction.
#
# One run:
#   SEED=42 SOCIAL_INTERVAL=2 sbatch new_exp_0_gemma_si.sbatch
# Whole sweep: use submit_pareto_social.sh (idempotent + in-queue dedup).
#
# Lands in its OWN namespace: runs/pareto_social/new_exp_0_gemma_si<N>/
# seed_<S>/ + wandb project pareto_social_wired_together. The anchors stay in
# runs/new_exp_0_gemma/; the analysis reads both roots, e.g.
#   python scripts/compute_flops.py runs_from_daic/new_exp_0_gemma \
#                                   runs_from_daic/pareto_social
# cli_args.social_interval is recorded in final_metrics.json, and runs made
# with the [LocalModel usage] line get exact prefill token counts.
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project — nothing pools with runs/new_exp_0_gemma/.
# _common.sh binds WANDB_PROJECT at source time, so all of these go BEFORE
# the source line.
: "${RUN_GROUP:=pareto_social}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=pareto_social_wired_together}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

# The one knob. Required, positive integer — refuse to burn 36 GPU-hours on a
# typo'd arm name like new_exp_0_gemma_si.
SOCIAL_INTERVAL="${SOCIAL_INTERVAL:-}"
case "$SOCIAL_INTERVAL" in
    ''|*[!0-9]*|0)
        echo "ERROR: SOCIAL_INTERVAL must be a positive integer (got '${SOCIAL_INTERVAL}')." >&2
        echo "       e.g. SEED=42 SOCIAL_INTERVAL=2 sbatch new_exp_0_gemma_si.sbatch" >&2
        exit 1
        ;;
esac
if [ "$SOCIAL_INTERVAL" = "8" ]; then
    echo "WARNING: interval 8 is the code default — this duplicates the existing" >&2
    echo "         new_exp_0_gemma_hebbian arm under a new name. Proceeding anyway." >&2
fi

SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"
EXP_NAME="new_exp_0_gemma_si${SOCIAL_INTERVAL}"
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,hebbian,${SOCIAL_MODULE},pareto,si${SOCIAL_INTERVAL}}"

# Hebbian hyper-parameters: byte-identical to new_exp_0_gemma.sbatch's
# HEBBIAN=1 arm — the interval is the ONLY difference between sweep points.
HEBBIAN_ARGS=(
    --hebbian
    --hebbian-eta-0 0.005
    --hebbian-reward-norm 50
    --hebbian-decay 0.005
    --hebbian-gamma 0.2
    --social-module "$SOCIAL_MODULE"
    --social-interval "$SOCIAL_INTERVAL"
)

echo "── new_exp_0_gemma_si: SOCIAL_INTERVAL=$SOCIAL_INTERVAL → $EXP_NAME ──"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    "${HEBBIAN_ARGS[@]}" \
    --simultaneous
