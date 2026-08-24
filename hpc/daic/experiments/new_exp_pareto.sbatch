#!/bin/bash
#SBATCH --job-name=new-exp-pareto
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/new_exp_pareto_%j.out
#SBATCH --error=slurm_logs/new_exp_pareto_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# new_exp_pareto — Gemma 4 size sweep for the Pareto plots (perception ×
# performance, points = model size; compute axis via scripts/compute_flops.py).
# Same shape as new_exp_0_gemma (3 episodes × 1000 steps, vision on, base vs
# hebbian arms) so the existing E4B runs slot in as the 4.5B point.
#
# ONE file, model picked by MODEL_SIZE, arm picked by HEBBIAN:
#
#   MODEL_SIZE=e2b|e4b|12b|26b|31b   → gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it
#   HEBBIAN=0 → pareto_<size>_base       plain LLM         (= exp01 shape)
#   HEBBIAN=1 → pareto_<size>_hebbian    + plastic Hebbian  (= exp07 shape)
#
# SUBMIT THROUGH submit_pareto.sh — the #SBATCH headers above are only the
# smallest size's resources; the launcher overrides gres/mem/qos/time per
# size. Direct submission works for one-offs but YOU must pass the overrides:
#   MODEL_SIZE=31b HEBBIAN=1 SEED=42 \
#       sbatch --gres=gpu:2 --mem=64G --qos=long --time=168:00:00 \
#       new_exp_pareto.sbatch
#
# Per-size resource rationale (bf16, vision on; E4B numbers MEASURED on the
# 12 new_exp_0_gemma runs, the rest scaled from raw parameter counts):
#
#   size  raw     resident  wall-clock (3×1000)    gres    min/GPU  total-VRAM
#   e2b   ~5B     ~10 GB    ~10-20 h (½× E4B)      gpu:1   15000    —
#   e4b    8B     15.9 GB   19-38 h  measured      gpu:1   15000    —
#   12b    12B    ~24 GB    ~30-90 h (1.5-2.5×)    gpu:1   40000    —
#   26b    25.2B  ~50 GB    ~20-75 h (MoE: 3.8B    gpu:2   40000    60000
#                            active + shard cost)
#   31b    30.7B  ~62 GB    ~60-165 h (3-5×)       gpu:2   40000    75000
#
# min/GPU feeds _common.sh's MIN_GPU_MEM_MIB preflight (smallest visible
# card). That check alone is blind to "one 46 GiB A40 when the sharded model
# needs 60+", so the 2-GPU sizes get an additional TOTAL-VRAM preflight here:
# device_map=auto shards over every visible GPU, so the SUM is what must fit.
# A single 80 GB A100 also passes it — override with --gres=gpu:1 if DAIC
# has them (check: sinfo -N -h -o "%N %G %f" | sort -u).
#
# 31b sits near the 168 h long-qos cap — run SMOKE=1 through the launcher
# first, read steps/hour from the smoke log, and only then commit 12 jobs.
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
    # Qwen3.5 arms — the SECOND family, for the with-Qwen Pareto. Both are
    # already proven in this pipeline (the medium runs loaded Qwen3.5-9B as a
    # VISION model at 18.82 GB) and both fit one card, so neither touches the
    # broken vision-sharding path that blocks gemma 26b/31b.
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

# Own run group + wandb project: nothing pools with new_exp_0_gemma/ or any
# earlier suite. Results land in runs/pareto_gemma4/<exp>/seed_<N>/.
: "${RUN_GROUP:=pareto_gemma4}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=pareto_gemma4}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

HEBBIAN="${HEBBIAN:-1}"
SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"

# Hebbian hyper-parameters mirror new_exp_0_gemma (and through it exp07)
# EXACTLY — across the size sweep the only thing that varies is the backbone.
if [ "$HEBBIAN" = "1" ]; then
    EXP_NAME="pareto_${MODEL_SIZE}_hebbian"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,${MODEL_FAMILY},vision,pareto,${MODEL_SIZE},hebbian,${SOCIAL_MODULE}}"
    HEBBIAN_ARGS=(
        --hebbian
        --hebbian-eta-0 0.005
        --hebbian-reward-norm 50
        --hebbian-decay 0.005
        --hebbian-gamma 0.2
        --social-module "$SOCIAL_MODULE"
    )
else
    EXP_NAME="pareto_${MODEL_SIZE}_base"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,${MODEL_FAMILY},vision,pareto,${MODEL_SIZE},no_hebbian}"
    HEBBIAN_ARGS=()
fi

echo "── new_exp_pareto: MODEL_SIZE=$MODEL_SIZE HEBBIAN=$HEBBIAN → $EXP_NAME ──"
echo "   model: $MODEL_LLM"

# ${arr[@]+"${arr[@]}"} — expands to nothing when the base arm leaves the
# array empty (bash 4.2 on the login nodes).
run_exp "$EXP_NAME" "$MODEL_LLM" \
    ${HEBBIAN_ARGS[@]+"${HEBBIAN_ARGS[@]}"} \
    --simultaneous
