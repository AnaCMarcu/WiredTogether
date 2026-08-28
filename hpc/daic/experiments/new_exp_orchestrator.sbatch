#!/bin/bash
#SBATCH --job-name=new-exp-orch
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/new_exp_orch_%j.out
#SBATCH --error=slurm_logs/new_exp_orch_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# new_exp_orchestrator — O2 centralized task-ledger orchestrator baseline on
# the new_exp_0_gemma suite (Gemma 4 E4B, vision, 3 ep × 1000 steps).
#
# EXACTLY the new_exp_0_gemma arm shape (same model, image, vision path,
# episodes/steps, --simultaneous), but the Hebbian coupling is REPLACED by
# the orchestrator: one non-embodied coordinator LLM call every
# ORCH_CADENCE steps (+ milestone/chamber/death event triggers) that keeps a
# within-episode task ledger and issues per-agent comm_target/help
# directives. --orchestrator and --hebbian are mutually exclusive in code,
# so this file passes no Hebbian flags at all.
#
# Comparison set (analysis reads both roots):
#   new_exp_0_gemma_base      runs/new_exp_0_gemma/   no coupling
#   new_exp_0_gemma_hebbian   runs/new_exp_0_gemma/   Hebbian + social module
#   new_exp_0_gemma_orch_*    runs/orchestrator/      THIS file
#
# ONE knob names the arm, mirroring the social module's prompt/bias split:
#   ORCH_MODE=advisory (default) → new_exp_0_gemma_orch_advisory
#       directives go into the same {social_directive} prompt slot the
#       social module uses (= the anchors' SOCIAL_MODULE=prompt coupling)
#   ORCH_MODE=bias               → new_exp_0_gemma_orch_bias
#       additionally overrides communication_target at the routing site
#       (= the exp09-11 / SOCIAL_MODULE=bias mechanism)
#
# ORCH_CADENCE defaults to 8 = the Hebbian arm's social-interval default, so
# the two conditions are matched on social-reasoning call frequency.
#
# One run:
#   SEED=42 sbatch new_exp_orchestrator.sbatch
#   SEED=42 ORCH_MODE=bias sbatch new_exp_orchestrator.sbatch
# Smoke / whole sweep: use submit_orchestrator.sh (idempotent + in-queue
# dedup, SMOKE=1 mode).
#
# Per-call audit artifacts land in the run dir under orchestrator/
# (calls.jsonl, compliance.jsonl, maps/*.png); token counts appear in
# log.txt both as the standard [LocalModel usage] lines (already counted by
# scripts/compute_flops.py) and as tagged [Orchestrator usage] lines so the
# orchestrator's share is separable.
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
# Vision matters twice here: agents see frames AND the orchestrator attaches
# its schematic top-down map PNG (falls back to a text block on a
# text-only model, but that changes the condition — don't, silently).
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"

# Own run group + wandb project — nothing pools with runs/new_exp_0_gemma/.
# _common.sh binds WANDB_PROJECT at source time, so these go BEFORE the
# source line.
: "${RUN_GROUP:=orchestrator}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=orchestrator_wired_together}"
export WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source "$WORKSPACE/WiredTogether/hpc/daic/experiments/_common.sh"

# The two knobs. Validated here as well as in code — refuse to burn
# GPU-hours on a typo'd arm name.
ORCH_MODE="${ORCH_MODE:-advisory}"
case "$ORCH_MODE" in
    advisory|bias) ;;
    *)
        echo "ERROR: ORCH_MODE must be 'advisory' or 'bias' (got '${ORCH_MODE}')." >&2
        echo "       e.g. SEED=42 ORCH_MODE=bias sbatch new_exp_orchestrator.sbatch" >&2
        exit 1
        ;;
esac

# ORCH_VARIANT selects the arm family:
#   task   (default) — O2 task-ledger orchestrator (map + event digest)
#   social           — centralized social deliberation, information-matched
#                      to the Hebbian rule (pair digest only)
#   plan             — social + task visibility + per-agent plan notes into
#                      the auto-curriculum (upper baseline)
#   villager         — VillagerAgent-style DAG orchestration: decomposer +
#                      allocator LLMs, HARD task assignments constraining
#                      the auto-curriculum; no comm routing. Produces
#                      dag.jsonl/assignments.jsonl instead of compliance/maps.
ORCH_VARIANT="${ORCH_VARIANT:-task}"
case "$ORCH_VARIANT" in
    task|social|plan|villager) ;;
    *)
        echo "ERROR: ORCH_VARIANT must be 'task', 'social', 'plan' or 'villager' (got '${ORCH_VARIANT}')." >&2
        echo "       e.g. SEED=42 ORCH_VARIANT=villager sbatch new_exp_orchestrator.sbatch" >&2
        exit 1
        ;;
esac
if [ "$ORCH_VARIANT" = "villager" ] && [ "$ORCH_MODE" = "bias" ]; then
    echo "ERROR: villager issues task assignments, not comm directives — bias mode is invalid; use ORCH_MODE=advisory." >&2
    exit 1
fi

# Cadence 8 = the Hebbian anchor's social-interval default (matched call
# rate). Passing a different value makes the arm non-comparable to the
# anchors — do that only as a deliberate cadence ablation.
ORCH_CADENCE="${ORCH_CADENCE:-8}"
case "$ORCH_CADENCE" in
    ''|*[!0-9]*|0)
        echo "ERROR: ORCH_CADENCE must be a positive integer (got '${ORCH_CADENCE}')." >&2
        exit 1
        ;;
esac

# Arm naming: the task variant keeps its original name
# (new_exp_0_gemma_orch_<mode>) so completed runs and the submit script's
# idempotency checks stay valid; social/plan get the variant in the name.
if [ "$ORCH_VARIANT" = "task" ]; then
    EXP_NAME="new_exp_0_gemma_orch_${ORCH_MODE}"
else
    EXP_NAME="new_exp_0_gemma_orch_${ORCH_VARIANT}_${ORCH_MODE}"
fi
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,orchestrator,${ORCH_VARIANT},${ORCH_MODE},cadence${ORCH_CADENCE},new_exp_0}"

ORCH_ARGS=(
    --orchestrator
    --orchestrator-variant "$ORCH_VARIANT"
    --orchestrator-mode "$ORCH_MODE"
    --orchestrator-cadence "$ORCH_CADENCE"
)
# Villager only: node timeout. Default 60 suits the 1000-step runs (chamber
# phases of ~200 steps); a 150-step smoke teleports every 30 steps, so pass
# ORCH_NODE_TIMEOUT=20 there or every Ch1 node outlives its chamber.
if [ "$ORCH_VARIANT" = "villager" ]; then
    ORCH_ARGS+=(--orchestrator-node-timeout-steps "${ORCH_NODE_TIMEOUT:-60}")
fi

echo "── new_exp_orchestrator: ORCH_VARIANT=$ORCH_VARIANT ORCH_MODE=$ORCH_MODE cadence=$ORCH_CADENCE → $EXP_NAME ──"

run_exp "$EXP_NAME" "$MODEL_LLM" \
    "${ORCH_ARGS[@]}" \
    --simultaneous
