#!/bin/bash
#SBATCH --job-name=new-exp-0-gemma
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/new_exp_0_gemma_%j.out
#SBATCH --error=slurm_logs/new_exp_0_gemma_%j.err
#SBATCH --exclude=cor1,influ[2-3]

# ───────────────────────────────────────────────────────────────────────────
# new_exp_0_gemma — Gemma 4 LLM agents, with and without the Hebbian social
# module. Same shape as exp01_llm_2b (plain LLM, no RL), at 3 episodes × 1000
# steps, in its own run group and wandb project.
#
# ONE file, TWO conditions, selected by HEBBIAN (they write to different run
# dirs, so both can be in flight at once):
#
#   HEBBIAN=0 → new_exp_0_gemma_base      plain LLM            (= exp01 shape)
#   HEBBIAN=1 → new_exp_0_gemma_hebbian   + plastic Hebbian graph feeding the
#                                           social module      (= exp07 shape)
#
# Both arms, all three seeds (6 jobs):
#   cd $REPO/hpc/daic/experiments
#   for s in 42 123 456; do for h in 0 1; do
#       SEED=$s HEBBIAN=$h sbatch new_exp_0_gemma.sbatch
#   done; done
#
# One arm, one seed:
#   SEED=42 HEBBIAN=1 sbatch new_exp_0_gemma.sbatch
#
# The --exclude header lists influ[2-3] directly (not just cor1) so a plain
# `sbatch new_exp_0_gemma.sbatch` is safe. Both influ GPU nodes are 11264 MiB
# RTX 2080 Tis — under _common.sh's 15000 MiB preflight threshold — so jobs
# landing there abort at startup with "GPU PREFLIGHT FAILED". Submitting
# through gpu_filter.sh (which reads bad_gpu_nodes.txt) achieves the same and
# stays current as the preflight appends new offenders; this header is the
# belt for direct submissions:
#     source gpu_filter.sh
#     sbatch ${GPU_FILTER_FLAGS[@]:+"${GPU_FILTER_FLAGS[@]}"} new_exp_0_gemma.sbatch
#
# Scheduled on medium qos / 36 h rather than the long-qos 168 h the other exp
# files ask for: 3 × 1000 steps is a fraction of the final suite's 5 × 2500 and
# medium queues sooner. If a vision-on run turns out slower than that fits,
# give one arm more room without editing this file:
#   sbatch --qos=long --time=168:00:00 new_exp_0_gemma.sbatch
#
# The Hebbian hyper-parameters mirror exp07 exactly, so the only differences
# from that established condition are the base model and the vision input.
# SOCIAL_MODULE=bias switches the coupling from prompt text to the logit-bias
# path (the exp09-11 mechanism) if you want the tighter integration instead.
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project: nothing pools with runs/final/, runs/medium*/
# or the runs/gemma4/ suite. These are set BEFORE sourcing _common.sh because
# it binds WANDB_PROJECT at source time — assigning it afterwards would leave
# the runs logging to final_wired_together.
: "${RUN_GROUP:=new_exp_0_gemma}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=new_exp_0_gemma}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

HEBBIAN="${HEBBIAN:-1}"
SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"

if [ "$HEBBIAN" = "1" ]; then
    EXP_NAME="new_exp_0_gemma_hebbian"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,hebbian,${SOCIAL_MODULE},new_exp_0}"
    HEBBIAN_ARGS=(
        --hebbian
        --hebbian-eta-0 0.005
        --hebbian-reward-norm 50
        --hebbian-decay 0.005
        --hebbian-gamma 0.2
        --social-module "$SOCIAL_MODULE"
    )
else
    EXP_NAME="new_exp_0_gemma_base"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,no_hebbian,new_exp_0}"
    HEBBIAN_ARGS=()
fi

echo "── new_exp_0_gemma: HEBBIAN=$HEBBIAN → $EXP_NAME ──"

# ${arr[@]+"${arr[@]}"} — expands to nothing when the base arm leaves the array
# empty, instead of an empty string argument (bash 4.2 on the login nodes).
run_exp "$EXP_NAME" "$MODEL_LLM" \
    ${HEBBIAN_ARGS[@]+"${HEBBIAN_ARGS[@]}"} \
    --simultaneous
