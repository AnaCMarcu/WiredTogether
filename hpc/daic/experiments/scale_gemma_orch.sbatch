#!/bin/bash
#SBATCH --job-name=scale-gemma-orch
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/scale_gemma_orch_%j.out
#SBATCH --error=slurm_logs/scale_gemma_orch_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# scale_gemma_orch — the centralized orchestrator at an arbitrary team size.
#
# Why this file exists: new_exp_orchestrator.sbatch has no agent-count knob
# and its EXP_NAME (new_exp_0_gemma_orch_<variant>_<mode>) carries no N, so
# running it with NUM_AGENTS=6 would write into the 3-agent run dirs. It also
# omits the scaling family's two environment invariants, which would make an
# N=6 orchestrator run non-comparable to scale_gemma_hebbian_n6:
#     --team-scaling        N-templated prompts (see team_scaling.py)
#     --ch4-mob-count 3     Ch4 difficulty pinned, not scaled with N
#
# This is therefore the orchestrator arm OF THE SCALING SWEEP: same knobs as
# scale_gemma_3f.sbatch, with the orchestrator replacing the Hebbian rule as
# the coupling. Lands in runs/agent_scaling_orch/scale_gemma_orch_<variant>_
# n<N>/seed_<S>/ + wandb agent_scaling_orch, so nothing pools with either
# runs/agent_scaling_3f/ (Hebbian arm) or runs/orchestrator/ (the 3-agent
# published arm).
#
# Defaults follow the PUBLISHED orchestrator row, which is the villager DAG
# variant in advisory mode at cadence 8 (make_final_table.ROWS names
# new_exp_0_gemma_orch_villager_advisory) — so an N=3 run of this file is the
# scaling-invariant counterpart of that row, and N=6 extends it.
#
# EPISODES x MAX_STEPS defaults to 3 x 1000, matching both the published
# orchestrator runs and the 1000-step scaling sweep (RUN_GROUP=
# agent_scaling_3f_1k), which is what the N=6 Hebbian comparison uses.
#
# The #SBATCH headers above are sized for N=6 at 3 x 1000 steps (64GB / 10
# CPUs, ~64 h projected at 0.14 + 0.19*N min/step, 96 h with the 1.5x margin
# the scaling launcher uses). Override per N:
#   N<=3  --mem=32GB --cpus-per-task=8  --qos=medium --time=36:00:00
#   N<=5  --mem=48GB --cpus-per-task=8  --qos=long   --time=72:00:00
#   N>=7  --mem=96GB --cpus-per-task=12 --qos=long   --time=168:00:00
#
# Three seeds at N=6:
#   cd $REPO/hpc/daic/experiments
#   for s in 42 123 456; do
#       NUM_AGENTS=6 SEED=$s sbatch scale_gemma_orch.sbatch
#   done
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

if [ -z "${NUM_AGENTS:-}" ]; then
    echo "ERROR: NUM_AGENTS is required (e.g. NUM_AGENTS=6 SEED=42 sbatch scale_gemma_orch.sbatch)" >&2
    exit 1
fi
case "$NUM_AGENTS" in
    ''|*[!0-9]*|0|1)
        echo "ERROR: NUM_AGENTS must be an integer >= 2 (got '${NUM_AGENTS}')." >&2
        exit 1
        ;;
esac

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project (bound at _common.sh source time).
: "${RUN_GROUP:=agent_scaling_orch}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=agent_scaling_orch}"
: "${CH4_MOBS:=3}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT NUM_AGENTS

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

# Same validation as new_exp_orchestrator.sbatch — refuse to burn GPU-hours
# on a typo'd arm name.
ORCH_VARIANT="${ORCH_VARIANT:-villager}"
case "$ORCH_VARIANT" in
    task|social|plan|villager) ;;
    *)
        echo "ERROR: ORCH_VARIANT must be 'task', 'social', 'plan' or 'villager' (got '${ORCH_VARIANT}')." >&2
        exit 1
        ;;
esac
ORCH_MODE="${ORCH_MODE:-advisory}"
case "$ORCH_MODE" in
    advisory|bias) ;;
    *)
        echo "ERROR: ORCH_MODE must be 'advisory' or 'bias' (got '${ORCH_MODE}')." >&2
        exit 1
        ;;
esac
if [ "$ORCH_VARIANT" = "villager" ] && [ "$ORCH_MODE" = "bias" ]; then
    echo "ERROR: villager issues task assignments, not comm directives — bias mode is invalid; use ORCH_MODE=advisory." >&2
    exit 1
fi
# Cadence 8 = the Hebbian anchor's social-interval default (matched call
# rate). A different value makes the arm non-comparable to the anchors.
ORCH_CADENCE="${ORCH_CADENCE:-8}"
case "$ORCH_CADENCE" in
    ''|*[!0-9]*|0)
        echo "ERROR: ORCH_CADENCE must be a positive integer (got '${ORCH_CADENCE}')." >&2
        exit 1
        ;;
esac

EXP_NAME="scale_gemma_orch_${ORCH_VARIANT}_n${NUM_AGENTS}"
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,orchestrator,${ORCH_VARIANT},${ORCH_MODE},cadence${ORCH_CADENCE},scaling,n${NUM_AGENTS}}"

ORCH_ARGS=(
    --orchestrator
    --orchestrator-variant "$ORCH_VARIANT"
    --orchestrator-mode "$ORCH_MODE"
    --orchestrator-cadence "$ORCH_CADENCE"
)
# Villager only: node timeout. 60 is the published setting for 1000-step
# runs. NOTE (measured): at N=3 a 60-step timeout dissolves nearly every
# staffed team before it completes anything — 74% of teams freed by timeout
# vs 3.4% by success. Raise it deliberately via ORCH_NODE_TIMEOUT if the
# N=6 runs are meant to show teams persisting.
if [ "$ORCH_VARIANT" = "villager" ]; then
    ORCH_ARGS+=(--orchestrator-node-timeout-steps "${ORCH_NODE_TIMEOUT:-60}")
fi

echo "── scale_gemma_orch: NUM_AGENTS=$NUM_AGENTS variant=$ORCH_VARIANT mode=$ORCH_MODE cadence=$ORCH_CADENCE → $EXP_NAME ──"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    "${ORCH_ARGS[@]}" \
    --team-scaling \
    --ch4-mob-count "$CH4_MOBS" \
    --simultaneous
