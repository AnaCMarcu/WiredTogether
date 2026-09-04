#!/bin/bash
#SBATCH --job-name=scale-gemma-3f
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/scale_gemma_3f_%j.out
#SBATCH --error=slurm_logs/scale_gemma_3f_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# scale_gemma_3f — the agent-count scaling suite's HEBBIAN arm on the NEW
# rule (three_factor + signed death LTD). The base arm does not depend on
# the rule and stays in scale_gemma.sbatch / runs/agent_scaling.
#
# Identical to scale_gemma.sbatch HEBBIAN=1 except the update rule (the
# exp34/35 flag block) and the namespace:
#   runs/agent_scaling_3f/scale_gemma_hebbian_n<N>/seed_<S>/ + wandb
#   agent_scaling_3f. EXP_NAME is kept IDENTICAL to the old arm so
#   make_scaling_fig.py needs no alias — point it at a root that merges
#   agent_scaling_3f (hebbian) with agent_scaling (base) locally.
#
# Environment invariants unchanged: --team-scaling, --ch4-mob-count 3,
# 3 ep × 500 steps.
#
# Normally submitted through submit_agent_scaling_3f.sh, which sweeps
# N ∈ {2,3,4,5,6,9} × seeds and scales --mem/--time/--qos with N. Manual:
#   NUM_AGENTS=6 SEED=42 sbatch --mem=64GB --qos=long --time=48:00:00 \
#       scale_gemma_3f.sbatch
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

if [ -z "${NUM_AGENTS:-}" ]; then
    echo "ERROR: NUM_AGENTS is required (e.g. NUM_AGENTS=6 SEED=42 sbatch scale_gemma_3f.sbatch)" >&2
    exit 1
fi

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project (bound at _common.sh source time).
: "${RUN_GROUP:=agent_scaling_3f}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=500}"
: "${WANDB_PROJECT:=agent_scaling_3f}"
: "${CH4_MOBS:=3}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT NUM_AGENTS

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"

EXP_NAME="scale_gemma_hebbian_n${NUM_AGENTS}"
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,hebbian,three_factor,fdecay,scaling,n${NUM_AGENTS}}"
# NEW-rule hyper-parameters: byte-identical to exp_qwen_three_factor_fdecay.
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
)

echo "── scale_gemma_3f: NUM_AGENTS=$NUM_AGENTS → $EXP_NAME (rule: 3f+fdecay) ──"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    "${HEBBIAN_ARGS[@]}" \
    --team-scaling \
    --ch4-mob-count "$CH4_MOBS" \
    --simultaneous
