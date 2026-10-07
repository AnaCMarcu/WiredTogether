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

# ───────────────────────────────────────────────────────────────────────────
# new_exp_orchestrator — centralised orchestration baseline (VillagerAgent-
# style) on the new_exp_0_gemma suite (Gemma 4 E4B, vision, 3 ep × 1000 steps).
#
# EXACTLY the new_exp_0_gemma arm shape (same model, image, vision path,
# episodes/steps, --simultaneous), but the Hebbian coupling is REPLACED by
# the orchestrator: a decomposer LLM proposes milestone-verified subtasks
# into a dependency DAG and an allocator LLM HARD-assigns ready subtasks to
# free agents, at most once every 8 steps (--orchestrator-decompose-min-
# interval, matched to the social module's T_soc). --orchestrator and
# --hebbian are mutually exclusive in code, so this file passes no Hebbian
# flags at all.
#
# Comparison set (analysis reads both roots):
#   new_exp_0_gemma_base       runs/new_exp_0_gemma/   no coupling
#   new_exp_0_gemma_hebbian3f  runs/new_exp_0_gemma/   social plasticity
#   new_exp_0_gemma_orch_villager_advisory   runs/orchestrator/   THIS file
#
# One run:
#   SEED=42 sbatch new_exp_orchestrator.sbatch
# Whole sweep: submit_orchestrator.sh (idempotent + in-queue dedup, SMOKE=1).
#
# Per-call audit artifacts land in the run dir under orchestrator/
# (calls.jsonl, dag.jsonl, assignments.jsonl, task_compliance.jsonl); token
# counts appear in log.txt both as the standard [LocalModel usage] lines
# (already counted by analysis/compute_flops.py) and as tagged
# [Orchestrator usage] lines so the orchestrator's share is separable.
# ───────────────────────────────────────────────────────────────────────────

WORKSPACE="${WT_WORKSPACE:?set WT_WORKSPACE to your cluster workspace (it holds WiredTogether/, images/ and models/)}"
# Gemma 4 needs the newer container + the vision path; set BEFORE sourcing
# _common.sh, which defaults WT_IMAGE to the original (pre-Gemma) image.
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

source "${WT_REPO:-$WORKSPACE/WiredTogether}/hpc/slurm/experiments/_common.sh"

# ORCH_VARIANT selects the orchestrator:
#   villager (default) — the SOFT orchestrator above (peer messaging left on).
#   hmas2              — the HARD orchestrator: HMAS-2 (Chen et al., ICRA
#                        2024) adapted to WIRE — every step a central planner
#                        assigns subtasks, assigned agents check them ('I
#                        Agree' or an objection), <= 3 revision rounds;
#                        hub-and-spoke comms (no agent-to-agent messages);
#                        curriculum pinned to the assignment. Writes
#                        orchestrator/hmas2.jsonl. A planner call + checks run
#                        every step: give it a longer --time (the submit
#                        script asks for 96 h; the Snellius shim caps 120 h).
ORCH_VARIANT="${ORCH_VARIANT:-villager}"
case "$ORCH_VARIANT" in
    villager|hmas2) ;;
    *)
        echo "ERROR: ORCH_VARIANT must be 'villager' or 'hmas2' (got '${ORCH_VARIANT}')." >&2
        exit 1
        ;;
esac

# The arm name predates the removal of the other orchestrator variants and
# modes; it is kept (…_<variant>_advisory) so villager reruns land beside the
# reported runs.
EXP_NAME="new_exp_0_gemma_orch_${ORCH_VARIANT}_advisory"
export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-llm,gemma4,vision,orchestrator,${ORCH_VARIANT},new_exp_0}"

ORCH_ARGS=(
    --orchestrator
    --orchestrator-variant "$ORCH_VARIANT"
)
# Villager only — node timeout. Default 60 suits the 1000-step runs (chamber
# phases of ~200 steps); a 150-step smoke teleports every 30 steps, so pass
# ORCH_NODE_TIMEOUT=20 there or every Ch1 node outlives its chamber.
if [ "$ORCH_VARIANT" = "villager" ]; then
    ORCH_ARGS+=(--orchestrator-node-timeout-steps "${ORCH_NODE_TIMEOUT:-60}")
fi

echo "── new_exp_orchestrator: variant=$ORCH_VARIANT → $EXP_NAME ──"

run_exp "$EXP_NAME" "$MODEL_LLM"     "${ORCH_ARGS[@]}"     --simultaneous
