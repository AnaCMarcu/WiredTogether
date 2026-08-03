#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_gemma4.sh — run the suite on ONE Gemma 4 model, with vision ON.
#
# What differs from the Qwen suite:
#   MODEL_LLM         one model everywhere (default gemma-4-E4B-it) instead of
#                     the old 2B/9B pair — so exp02/exp08 are dropped below,
#                     they would be byte-identical re-runs of exp01/exp07
#   LLM_VISION_MODE   "vision": Gemma 4 is multimodal, so the Craftium frame
#                     that multi_agent_craftium.py already attaches to every
#                     agent message is actually consumed. Under Qwen it was
#                     silently dropped. Set "text" for a like-for-like
#                     comparison against the Qwen numbers.
#   WT_IMAGE          the Gemma 4 container (transformers new enough to know
#                     the architecture); wiredtogether.sif stays untouched
#   RUN_GROUP         gemma4 → runs/gemma4/<exp>/seed_<N>/, so nothing mixes
#                     with runs/final/ or runs/medium2k/
#
# These runs are NOT poolable with the Qwen aggregation: different base model
# AND a modality the earlier runs never had. Treat runs/gemma4/ as its own
# suite.
#
# Prerequisites (both one-time, both on the login node):
#   sbatch hpc/daic/build_image_gemma4.sbatch     # build the image
#   bash   hpc/daic/download_gemma4.sh            # stage the weights
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1 bash submit_gemma4.sh     # 1 exp × 1 seed, 1 ep × 150 steps
#   DRY_RUN=1 bash submit_gemma4.sh   # print what the full suite would submit
#   bash submit_gemma4.sh             # submit the full suite
#
# Idempotent, like submit_medium2k.sh: an exp/seed whose final_metrics.json
# already exists on PRB is skipped, so re-running fills in only what is missing.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

# ── Gemma 4 configuration (all env-overridable) ─────────────────────────────
: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=gemma4}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=2000}"
: "${WANDB_PROJECT:=gemma4_wired_together}"
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

# The nine distinct conditions. exp02_llm_9b and exp08_llm_9b_social_prompt are
# omitted: with a single model they are exp01 and exp07 again. Restore the size
# ablation by staging a second checkpoint and setting MODEL_9B to it.
EXPS=(
    exp01_llm_2b
    exp03_mappo
    exp04_ippo
    exp05_mappo_hebbian
    exp06_ippo_hebbian
    exp07_llm_2b_social_prompt
    exp09_llm_9b_allied_all
    exp10_llm_9b_allied_pair
    exp11_llm_9b_allied_none
)
SEEDS=(42 123 456)

# Smoke test: one short run to prove the image loads Gemma 4, the processor
# accepts the frames, and a JSON action comes back — before committing days of
# GPU to the suite. Lands in runs/gemma4_smoke/ and skips wandb.
if [ "${SMOKE:-0}" = "1" ]; then
    EXPS=(exp01_llm_2b)
    SEEDS=(42)
    RUN_GROUP=gemma4_smoke
    EPISODES=1
    MAX_STEPS=${SMOKE_STEPS:-150}
    export RUN_GROUP EPISODES MAX_STEPS
    export WANDB=0
    : "${QOS:=short}"
    : "${TIME:=4:00:00}"
fi

# Optional sbatch overrides (empty = keep each exp file's baked-in directives).
#
# GPU: Gemma 4 E4B is ~8B params on disk, so bf16 weights + the vision tower +
# KV cache land around 16-18 GB. A bare `--gres=gpu:1` can schedule onto a
# turing/V100-16GB node and OOM at load. Pin a big card if that happens:
#     GPU=gpu:a40:1 bash submit_gemma4.sh
SBATCH_OVERRIDES=()
[ -n "${QOS:-}" ]  && SBATCH_OVERRIDES+=(--qos="$QOS")
[ -n "${TIME:-}" ] && SBATCH_OVERRIDES+=(--time="$TIME")
[ -n "${GPU:-}" ]  && SBATCH_OVERRIDES+=(--gres="$GPU")
[ -n "${MEM:-}" ]  && SBATCH_OVERRIDES+=(--mem="$MEM")

echo "== submit_gemma4.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  vision    : $LLM_VISION_MODE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  exps      : ${#EXPS[@]} (${EXPS[*]})"
echo "  seeds     : ${SEEDS[*]}"
[ ${#SBATCH_OVERRIDES[@]} -gt 0 ] && echo "  overrides : ${SBATCH_OVERRIDES[*]}"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "======================"

# Pre-flight: both prerequisites, checked before anything is queued.
missing=0
if [ ! -f "$WT_IMAGE" ]; then
    echo "ERROR: image not found: $WT_IMAGE" >&2
    echo "       sbatch hpc/daic/build_image_gemma4.sbatch" >&2
    missing=1
fi
if [ ! -f "$MODEL_LLM/config.json" ]; then
    echo "ERROR: weights not staged: $MODEL_LLM/config.json" >&2
    echo "       bash hpc/daic/download_gemma4.sh" >&2
    missing=1
fi
[ "$missing" = "1" ] && exit 1

n_queued=0
n_skipped=0
for exp in "${EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ -f "$REPO/runs/$RUN_GROUP/$exp/seed_$seed/final_metrics.json" ]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $exp  seed_$seed"
        else
            jobid=$(SEED=$seed sbatch --parsable \
                ${SBATCH_OVERRIDES[@]:+"${SBATCH_OVERRIDES[@]}"} \
                "$exp.sbatch")
            echo "queued  $exp  seed_$seed  →  job $jobid"
        fi
        n_queued=$((n_queued + 1))
    done
done
echo "── done: $n_queued submitted, $n_skipped already complete ──"
echo "Track with:   squeue -u \$USER"
echo "Watch load:   tail -f slurm_logs/${EXPS[0]}_*.out"
echo "Cancel all:   scancel -u \$USER"
