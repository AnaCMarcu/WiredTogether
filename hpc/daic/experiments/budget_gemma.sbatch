#!/bin/bash
#SBATCH --job-name=budget-gemma
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/budget_gemma_%j.out
#SBATCH --error=slurm_logs/budget_gemma_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# budget_gemma — ONE cell of the communication-budget sweep:
#     N ∈ {3,5,7}  ×  budget ∈ {0, 600, 2600, 10400} tokens/agent/episode
#                  ×  arm ∈ {base, hebbian}
# Gemma 4 E4B (vision), 3 ep × 1000 steps, team scaling on, Ch4 mobs pinned
# to 3, simultaneous stepping — the agent-count scaling invariants — plus:
#   --comm-budget-tokens $COMM_BUDGET   per-agent, per-episode token budget;
#                                       messaging becomes OPTIONAL in the
#                                       prompts and is muted in code once the
#                                       budget is spent (env/comm_budget.py)
#   --comm-budget-msg-cap 32            one message costs at most 32 tokens
#   --comm-reward-scale 0               BOTH arms: no per-message reward and
#                                       no m_comm_* payout, so budget cannot
#                                       be spent on milestone farming and the
#                                       three-factor rule cannot bond on it
# hebbian arm = the Hebbian-2.0 block of scale_gemma_3f.sbatch (three_factor
# + signed death LTD, social module "prompt"); base arm = no Hebbian, no
# social module. Both arms see the same budget line and comm rule.
#
# Lands in runs/comm_budget/budget_gemma_<arm>_n<N>_b<B>/seed_<S>/ + wandb
# project comm_budget. Normally submitted through submit_comm_budget.sh.
# Manual:
#   NUM_AGENTS=5 COMM_BUDGET=2600 ARM=hebbian SEED=42 \
#       sbatch --mem=48GB --qos=long --time=72:00:00 budget_gemma.sbatch
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

for _req in NUM_AGENTS COMM_BUDGET ARM; do
    if [ -z "${!_req:-}" ]; then
        echo "ERROR: $_req is required (e.g. NUM_AGENTS=3 COMM_BUDGET=800 ARM=base SEED=42 sbatch budget_gemma.sbatch)" >&2
        exit 1
    fi
done
case "$ARM" in
    base|hebbian) ;;
    *) echo "ERROR: ARM must be 'base' or 'hebbian' (got '$ARM')" >&2; exit 1 ;;
esac
case "$COMM_BUDGET" in
    ''|*[!0-9]*) echo "ERROR: COMM_BUDGET must be a non-negative integer (got '$COMM_BUDGET')" >&2; exit 1 ;;
esac

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

: "${RUN_GROUP:=comm_budget}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=comm_budget}"
: "${CH4_MOBS:=3}"
: "${COMM_MSG_CAP:=32}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT NUM_AGENTS

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

SOCIAL_MODULE="${SOCIAL_MODULE:-prompt}"

if [ "$ARM" = "hebbian" ]; then
    EXP_NAME="budget_gemma_hebbian_n${NUM_AGENTS}_b${COMM_BUDGET}"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,hebbian,three_factor,fdecay,comm_budget,b${COMM_BUDGET},n${NUM_AGENTS}}"
    # Hebbian 2.0: byte-identical to scale_gemma_3f.sbatch / exp_qwen_three_factor_fdecay.
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
else
    EXP_NAME="budget_gemma_base_n${NUM_AGENTS}_b${COMM_BUDGET}"
    export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,no_hebbian,comm_budget,b${COMM_BUDGET},n${NUM_AGENTS}}"
    HEBBIAN_ARGS=()
fi

echo "── budget_gemma: NUM_AGENTS=$NUM_AGENTS COMM_BUDGET=$COMM_BUDGET ARM=$ARM → $EXP_NAME ──"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    ${HEBBIAN_ARGS[@]+"${HEBBIAN_ARGS[@]}"} \
    --team-scaling \
    --ch4-mob-count "$CH4_MOBS" \
    --simultaneous \
    --comm-budget-tokens "$COMM_BUDGET" \
    --comm-budget-msg-cap "$COMM_MSG_CAP" \
    --comm-reward-scale 0
