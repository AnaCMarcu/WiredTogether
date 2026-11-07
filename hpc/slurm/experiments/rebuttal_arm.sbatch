#!/bin/bash
#SBATCH --job-name=rebuttal
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/rebuttal_%j.out
#SBATCH --error=slurm_logs/rebuttal_%j.err

# ───────────────────────────────────────────────────────────────────────────
# rebuttal_arm — ONE arm of the ICLR rebuttal ablations. The existing arm
# files hard-code their flags and experiment name, so the ablations need one
# launcher that takes both from the environment.
#
# Submit through submit_rebuttal.sh, which sets REB_ARM, REB_ARGS, SEED,
# RUN_GROUP and LLM_SERVER per arm.
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE="${WT_WORKSPACE:?set WT_WORKSPACE to your cluster workspace (it holds WiredTogether/, images/ and models/)}"
for _req in REB_ARM REB_ARGS SEED RUN_GROUP; do
    if [ -z "${!_req:-}" ]; then
        echo "ERROR: $_req is required; submit through submit_rebuttal.sh" >&2
        exit 1
    fi
done

: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=rebuttal}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/slurm/experiments/_common.sh"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-rebuttal,${REB_ARM}}"

echo "── rebuttal_arm: $REB_ARM seed=$SEED server=${LLM_SERVER:-in-process} ──"
echo "   args: $REB_ARGS"

# shellcheck disable=SC2086  # REB_ARGS is a flag list and must word-split.
run_exp "$REB_ARM" "$MODEL_LLM" $REB_ARGS
