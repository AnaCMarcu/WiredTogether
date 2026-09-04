#!/bin/bash
#SBATCH --job-name=new-exp-pareto-3f
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/new_exp_pareto_3f_%j.out
#SBATCH --error=slurm_logs/new_exp_pareto_3f_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# new_exp_pareto_3f — model-size Pareto HEBBIAN arm on the NEW rule
# (three_factor + signed death LTD). Base arms are rule-independent and stay
# in new_exp_pareto.sbatch / runs/pareto_gemma4 — reuse them, don't re-run.
#
# ONE file, model picked by MODEL_SIZE (same choices as new_exp_pareto):
#   MODEL_SIZE=e2b|e4b|12b|26b|31b|qwen2b|qwen9b
#
# EXP_NAME stays pareto_<size>_hebbian (the run GROUP carries the rule), so
# the pareto analysis reads runs/pareto_gemma4_3f exactly like pareto_gemma4.
#
# The #SBATCH headers above fit only the smallest sizes — pass per-size
# overrides on the command line (values from submit_pareto.sh; e2b's is
# tightened to 48 h since hebbian E2B measured 28-33 h):
#   e2b: --gres=gpu:1 --mem=32G --qos=long --time=48:00:00
#   12b: --gres=gpu:1 --mem=48G --qos=long --time=96:00:00
#
# Example (both sizes × 3 seeds = 6 jobs):
#   for s in 42 123 456; do
#     MODEL_SIZE=e2b SEED=$s sbatch --mem=32G --qos=long --time=48:00:00 \
#         new_exp_pareto_3f.sbatch
#     MODEL_SIZE=12b SEED=$s sbatch --mem=48G --qos=long --time=96:00:00 \
#         new_exp_pareto_3f.sbatch
#   done
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

MODEL_SIZE="${MODEL_SIZE:?set MODEL_SIZE to one of: e2b e4b 12b 26b 31b qwen2b qwen9b}"
TOTAL_GPU_MEM_MIB=""
case "$MODEL_SIZE" in
    e2b) MODEL_NAME=gemma-4-E2B-it;     : "${MIN_GPU_MEM_MIB:=15000}" ;;
    e4b) MODEL_NAME=gemma-4-E4B-it;     : "${MIN_GPU_MEM_MIB:=15000}" ;;
    12b) MODEL_NAME=gemma-4-12B-it;     : "${MIN_GPU_MEM_MIB:=40000}" ;;
    26b) MODEL_NAME=gemma-4-26B-A4B-it; : "${MIN_GPU_MEM_MIB:=40000}"
         TOTAL_GPU_MEM_MIB=60000 ;;
    31b) MODEL_NAME=gemma-4-31B-it;     : "${MIN_GPU_MEM_MIB:=40000}"
         TOTAL_GPU_MEM_MIB=75000 ;;
    qwen2b) MODEL_NAME=Qwen3.5-2B;      : "${MIN_GPU_MEM_MIB:=15000}" ;;
    qwen9b) MODEL_NAME=Qwen3.5-9B;      : "${MIN_GPU_MEM_MIB:=25000}" ;;
    *)   echo "ERROR: unknown MODEL_SIZE=$MODEL_SIZE (want: e2b e4b 12b 26b 31b qwen2b qwen9b)" >&2; exit 1 ;;
esac
export MIN_GPU_MEM_MIB
case "$MODEL_SIZE" in
    qwen*) MODEL_FAMILY=qwen ;;
    *)     MODEL_FAMILY=gemma4 ;;
esac
: "${MODEL_LLM:=$WORKSPACE/models/$MODEL_NAME}"
export MODEL_LLM

if [ -n "$TOTAL_GPU_MEM_MIB" ]; then
    TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
                    2>/dev/null | awk '{s+=$1} END {print s+0}')
    if [ -z "$TOTAL_MIB" ] || [ "$TOTAL_MIB" -lt "$TOTAL_GPU_MEM_MIB" ]; then
        echo "!! TOTAL-VRAM PREFLIGHT FAILED on ${SLURMD_NODENAME:-$(hostname -s)}:" >&2
        echo "   ${TOTAL_MIB:-0} MiB summed over visible GPUs < ${TOTAL_GPU_MEM_MIB} MiB" >&2
        echo "   needed to shard $MODEL_NAME. Aborting before a load-time OOM." >&2
        exit 1
    fi
    echo "Total-VRAM check: ${TOTAL_MIB} MiB >= ${TOTAL_GPU_MEM_MIB} MiB — OK"
fi

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh (it defaults WT_IMAGE to the pre-Gemma image and binds
# WANDB_PROJECT at source time).
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project: the group carries the rule.
: "${RUN_GROUP:=pareto_gemma4_3f}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=pareto_gemma4_3f}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"

EXP_NAME="pareto_${MODEL_SIZE}_hebbian"
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,${MODEL_FAMILY},vision,pareto,${MODEL_SIZE},hebbian,three_factor,fdecay,${SOCIAL_MODULE}}"
# NEW-rule hyper-parameters: byte-identical to exp_qwen_three_factor_fdecay —
# across the size sweep the only thing that varies is the backbone.
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

echo "── new_exp_pareto_3f: MODEL_SIZE=$MODEL_SIZE → $EXP_NAME (rule: 3f+fdecay) ──"
echo "   model: $MODEL_LLM"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    "${HEBBIAN_ARGS[@]}" \
    --simultaneous
