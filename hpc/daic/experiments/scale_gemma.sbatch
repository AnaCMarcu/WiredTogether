#!/bin/bash
#SBATCH --job-name=scale-gemma
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/scale_gemma_%j.out
#SBATCH --error=slurm_logs/scale_gemma_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# scale_gemma — agent-count scaling suite (compute vs. cooperative milestones).
#
# ONE file, parameterised by NUM_AGENTS (required) and HEBBIAN (default 0):
#
#   NUM_AGENTS=<N>  HEBBIAN=0 → scale_gemma_base_n<N>      plain Gemma LLM
#   NUM_AGENTS=<N>  HEBBIAN=1 → scale_gemma_hebbian_n<N>   + plastic Hebbian
#                                                            (exp07 hyper-params)
#
# The environment is IDENTICAL for every N (same chambers, same tasks, same
# milestone rewards, same team-level max):
#   * --team-scaling is the master switch: truthful N-dependent prompt text
#     + the collision-free generic Ch1 spawn row. Every OTHER suite leaves
#     it off and renders the historical 3-agent prompts byte-identically.
#   * --ch4-mob-count 3 pins the Ch4 zombie count to the 3-agent design —
#     without it the chamber spawns min(N, 6) zombies and the env would
#     change with team size.
#   * Chamber 3 is the ONLY room that scales: one isolation cell + switch
#     per agent (rotational unlock), exactly as designed. Milestone reward
#     VALUES are unchanged from the 3-agent setting.
#   * Boss HP (60), anvils (2), Ch1 resources: fixed by config.lua already.
#
# Budget: EPISODES × MAX_STEPS defaults to 3 × 500 (the scaling suite's
# 500-step episodes; chamber timers fire at 100/200/300/400).
#
# Normally submitted through submit_agent_scaling.sh, which sweeps
# N ∈ {2,3,4,5,6,9} × seeds and scales --mem/--time with N. Manual single job:
#   NUM_AGENTS=6 SEED=42 sbatch --mem=64GB scale_gemma.sbatch
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

if [ -z "${NUM_AGENTS:-}" ]; then
    echo "ERROR: NUM_AGENTS is required (e.g. NUM_AGENTS=6 SEED=42 sbatch scale_gemma.sbatch)" >&2
    exit 1
fi

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project (bound at _common.sh source time).
: "${RUN_GROUP:=agent_scaling}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=500}"
: "${WANDB_PROJECT:=agent_scaling}"
: "${CH4_MOBS:=3}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT NUM_AGENTS

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

HEBBIAN="${HEBBIAN:-0}"
SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"

if [ "$HEBBIAN" = "1" ]; then
    EXP_NAME="scale_gemma_hebbian_n${NUM_AGENTS}"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,hebbian,scaling,n${NUM_AGENTS}}"
    HEBBIAN_ARGS=(
        --hebbian
        --hebbian-eta-0 0.005
        --hebbian-reward-norm 50
        --hebbian-decay 0.005
        --hebbian-gamma 0.2
        --social-module "$SOCIAL_MODULE"
    )
else
    EXP_NAME="scale_gemma_base_n${NUM_AGENTS}"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,no_hebbian,scaling,n${NUM_AGENTS}}"
    HEBBIAN_ARGS=()
fi

echo "── scale_gemma: NUM_AGENTS=$NUM_AGENTS HEBBIAN=$HEBBIAN → $EXP_NAME ──"

# ${arr[@]+"${arr[@]}"} — expands to nothing when the base arm leaves the array
# empty, instead of an empty string argument (bash 4.2 on the login nodes).
run_exp "$EXP_NAME" "$MODEL_LLM" \
    ${HEBBIAN_ARGS[@]+"${HEBBIAN_ARGS[@]}"} \
    --team-scaling \
    --ch4-mob-count "$CH4_MOBS" \
    --simultaneous
