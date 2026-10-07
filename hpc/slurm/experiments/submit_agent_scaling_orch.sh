#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_agent_scaling_orch.sh — the agent-count scaling sweep's ORCHESTRATOR
# arms: N ∈ {3,5,7} by default, 3 ep × 1000 steps, Gemma 4 E4B, through
# scale_gemma_orch.sbatch (--team-scaling, --ch4-mob-count 3, --simultaneous).
# Lands in runs/agent_scaling_orch/scale_gemma_orch_<variant>_n<N>/.
#
# Variants (VARIANTS, default "hmas2"):
#   hmas2     the HARD orchestrator: HMAS-2 (Chen et al., ICRA 2024) adapted —
#             every step a central planner assigns subtasks and the assigned
#             agents check them (<= 3 revision rounds); hub-and-spoke comms
#             (no agent-to-agent messages); curriculum pinned to the
#             assignment.
#   villager  the published VillagerAgent port (the soft orchestrator: peer
#             messaging left on).
# Existing villager cells (e.g. the N=6 point) are skipped via
# final_metrics.json, so VARIANTS="hmas2 villager" NS="3 5 6 7" only fills
# gaps.
#
# QoS/time policy: same as submit_agent_scaling_3f.sh — projection from the
# measured fit min/step ≈ 0.14 + 0.19·N, x1.5 margin, ladder-rounded, medium
# up to 36 h — times a per-variant factor: hmas2 runs a planner call plus
# per-agent checks every step, budgeted at 1.4x until the smoke measures it
# (orchestrator/hmas2.jsonl latency_s). Runs are not resumable: a run that
# hits its wall is lost, so re-measure before queueing N >= 7.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/slurm/experiments
#   SMOKE=1   bash submit_agent_scaling_orch.sh   # N=3, seed 42, 1 ep × 150
#   SMOKE=1 VARIANTS="hmas2 villager" bash submit_agent_scaling_orch.sh
#   DRY_RUN=1 bash submit_agent_scaling_orch.sh   # print what would be submitted
#   bash submit_agent_scaling_orch.sh             # hmas2, N 3/5/7 × 3 seeds = 9 jobs
#   NS="3 6" SEEDS="42" VARIANTS="hmas2 villager" bash submit_agent_scaling_orch.sh
#
# Idempotent: a cell whose final_metrics.json already exists on PRB is
# skipped, and one already sitting in the Slurm queue (same job name) is not
# resubmitted. A refused sbatch (e.g. QOSMaxSubmitJobPerUserLimit) is
# reported as FAILED — re-run the same command once the queue drains.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE="${WT_WORKSPACE:?set WT_WORKSPACE to your cluster workspace (it holds WiredTogether/, images/ and models/)}"
REPO="$WORKSPACE/WiredTogether"

_EXPLICIT_RUN_GROUP="${RUN_GROUP:+1}"
_EXPLICIT_EPISODES="${EPISODES:+1}"
_EXPLICIT_MAX_STEPS="${MAX_STEPS:+1}"
_EXPLICIT_NS="${NS:+1}"
_EXPLICIT_SEEDS="${SEEDS:+1}"
_EXPLICIT_TIMEOUT="${ORCH_NODE_TIMEOUT:+1}"

: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=agent_scaling_orch}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=agent_scaling_orch}"
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

NS_LIST=(${NS:-3 5 7})
SEEDS_LIST=(${SEEDS:-42 123 456})
VARIANT_LIST=(${VARIANTS:-hmas2})

for v in "${VARIANT_LIST[@]}"; do
    case "$v" in
        hmas2|villager) ;;
        *) echo "ERROR: VARIANTS may only contain 'hmas2' and 'villager' (got '$v')" >&2; exit 1 ;;
    esac
done

# SMOKE supplies DEFAULTS, never overrides.
if [ "${SMOKE:-0}" = "1" ]; then
    [ -z "$_EXPLICIT_NS" ]        && NS_LIST=(3)
    [ -z "$_EXPLICIT_SEEDS" ]     && SEEDS_LIST=(42)
    [ -z "$_EXPLICIT_RUN_GROUP" ] && RUN_GROUP=agent_scaling_orch_smoke
    [ -z "$_EXPLICIT_EPISODES" ]  && EPISODES=1
    [ -z "$_EXPLICIT_MAX_STEPS" ] && MAX_STEPS=${SMOKE_STEPS:-150}
    # 150-step smokes teleport every ~30 steps; villager's 60-step node
    # timeout would never resolve a DAG node inside one chamber.
    [ -z "$_EXPLICIT_TIMEOUT" ]   && ORCH_NODE_TIMEOUT=20
    export RUN_GROUP EPISODES MAX_STEPS ORCH_NODE_TIMEOUT
    export WANDB=0
fi
[ -n "${ORCH_NODE_TIMEOUT:-}" ] && export ORCH_NODE_TIMEOUT

# Per-N resources (memory/CPUs by client count, as in the 3f sweep) and wall
# time with the per-variant factor (percent of the base fit).
resources_for_n() {
    local n="$1" variant="$2"
    if   [ "$n" -le 3 ]; then R_MEM=32GB;  R_CPUS=8
    elif [ "$n" -le 5 ]; then R_MEM=48GB;  R_CPUS=8
    elif [ "$n" -le 6 ]; then R_MEM=64GB;  R_CPUS=10
    else                      R_MEM=96GB;  R_CPUS=12
    fi

    local time_pct=100
    [ "$variant" = "hmas2" ] && time_pct=140
    # Integer arithmetic in centi-minutes per step (bash has no floats).
    local steps=$(( EPISODES * MAX_STEPS ))
    local centi=$(( (14 + 19 * n) * time_pct / 100 ))
    R_EST_H=$(( centi * steps / 6000 ))          # projected hours
    local req_h=$(( R_EST_H * 3 / 2 ))           # 1.5x margin
    [ "$req_h" -lt 8 ] && req_h=8
    local tier
    for tier in 24 36 48 72 96 120 144 168; do
        [ "$req_h" -le "$tier" ] && break
    done
    if [ "$req_h" -gt 168 ]; then
        echo "WARNING: $variant N=$n projected ${R_EST_H}h; 1.5x margin" \
             "(${req_h}h) exceeds long-qos 168h cap — requesting 168h" >&2
    fi
    R_TIME="${tier}:00:00"
    if [ "$tier" -le 36 ]; then R_QOS=medium; else R_QOS=long; fi

    R_MEM="${MEM:-$R_MEM}"; R_CPUS="${CPUS:-$R_CPUS}"
    R_QOS="${QOS:-$R_QOS}"; R_TIME="${TIME:-$R_TIME}"
}

echo "== submit_agent_scaling_orch.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  variants  : ${VARIANT_LIST[*]}"
echo "  N sweep   : ${NS_LIST[*]}"
echo "  seeds     : ${SEEDS_LIST[*]}"
echo "  timeout   : ${ORCH_NODE_TIMEOUT:-60} steps per villager DAG node"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "=================================="

# Pre-flight: image + weights staged.
missing=0
if [ ! -f "$WT_IMAGE" ]; then
    echo "ERROR: image not found: $WT_IMAGE" >&2
    echo "       sbatch hpc/slurm/build_image_gemma4.sbatch" >&2
    missing=1
fi
if [ ! -f "$MODEL_LLM/config.json" ]; then
    echo "ERROR: weights not staged: $MODEL_LLM/config.json" >&2
    echo "       stage the weights (Snellius: bash hpc/snellius/setup.sh models gemma4)" >&2
    missing=1
fi
[ "$missing" = "1" ] && exit 1

queued_names=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

n_queued=0
n_skipped=0
n_failed=0
for v in "${VARIANT_LIST[@]}"; do
    for n in "${NS_LIST[@]}"; do
        resources_for_n "$n" "$v"
        exp="scale_gemma_orch_${v}_n${n}"
        for seed in "${SEEDS_LIST[@]}"; do
            jobname="${RUN_GROUP}-${exp}_s${seed}"
            if [ -f "$REPO/runs/$RUN_GROUP/$exp/seed_$seed/final_metrics.json" ]; then
                n_skipped=$((n_skipped + 1))
                continue
            fi
            if printf '%s\n' "$queued_names" | grep -qx "$jobname"; then
                echo "in queue     $exp  seed_$seed  — skipping"
                n_skipped=$((n_skipped + 1))
                continue
            fi
            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "would queue  $exp  seed_$seed  (mem=$R_MEM cpus=$R_CPUS qos=$R_QOS time=$R_TIME, est ${R_EST_H}h)"
            else
                jobid=$(NUM_AGENTS=$n SEED=$seed ORCH_VARIANT=$v ORCH_MODE=advisory \
                    sbatch --parsable \
                    --job-name="$jobname" \
                    --mem="$R_MEM" --cpus-per-task="$R_CPUS" \
                    --qos="$R_QOS" --time="$R_TIME" \
                    scale_gemma_orch.sbatch)
                if [ -z "$jobid" ]; then
                    echo "FAILED  $exp  seed_$seed  — not submitted; re-run this command later" >&2
                    n_failed=$((n_failed + 1))
                    continue
                fi
                echo "queued  $exp  seed_$seed  →  job $jobid  (mem=$R_MEM qos=$R_QOS time=$R_TIME)"
            fi
            n_queued=$((n_queued + 1))
        done
    done
done
echo "── done: $n_queued submitted, $n_skipped skipped, $n_failed FAILED ──"
[ "$n_failed" -gt 0 ] && echo "   re-run the same command once the queue drains; finished/queued cells are skipped" >&2
echo "Track with:   squeue -u \$USER -o \"%.10i %.52j %.8T %.12M %.12l %R\""
echo "Then locally: pull runs/$RUN_GROUP and summarise orchestrator/hmas2.jsonl"
echo "              (python analysis/make_hmas2_load.py --runs-root <root>)"
