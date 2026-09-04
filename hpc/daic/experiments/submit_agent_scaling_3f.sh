#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_agent_scaling_3f.sh — the agent-count scaling sweep's HEBBIAN arm on
# the NEW rule (three_factor + signed death LTD): N ∈ {2,3,4,5,6,9}, 3 ep ×
# 500 steps, Gemma 4 E4B. Lands in runs/agent_scaling_3f/.
#
# The BASE arm is rule-independent and is NOT handled here — fill any missing
# base runs with the existing script (it skips finished ones):
#   ARMS="0" SEEDS="42 123 456" bash submit_agent_scaling.sh
# and merge the two roots locally for make_scaling_fig.py.
#
# QoS/time policy: "as small as possible, varied per N" — 1.5× the measured
# per-step projection (submit_agent_scaling.sh used 2×), rounded to a ladder
# rung, medium wherever ≤ 36 h, else long with a LEAN wall (shorter walls
# backfill sooner). At 3 ep × 500 steps this gives:
#     N=2 → medium 24 h   N=3 → medium 36 h   N=4 → medium 36 h
#     N=5 → long   48 h   N=6 → long   48 h   N=9 → long   72 h
# The 1.5× margin is a deliberate schedule-speed trade: the projection comes
# from a base-arm smoke, and a run that hits the wall is lost (no resumable
# state). If an N=4 seed TIMEOUTs, resubmit it with QOS=long TIME=48:00:00.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1   bash submit_agent_scaling_3f.sh   # N=2 and N=9, seed 42, 1 ep × 150
#   DRY_RUN=1 bash submit_agent_scaling_3f.sh   # print what would be submitted
#   bash submit_agent_scaling_3f.sh             # full sweep: 6 N × 3 seeds = 18 jobs
#   NS="2 9" SEEDS="42" bash submit_agent_scaling_3f.sh   # subset override
#   SEEDS="789 1011 1213" bash submit_agent_scaling_3f.sh # seed extension
#
# Idempotent: an N/seed whose final_metrics.json already exists on PRB is
# skipped, and one already sitting in the Slurm queue (same job name) is not
# resubmitted.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

_EXPLICIT_RUN_GROUP="${RUN_GROUP:+1}"
_EXPLICIT_EPISODES="${EPISODES:+1}"
_EXPLICIT_MAX_STEPS="${MAX_STEPS:+1}"
_EXPLICIT_NS="${NS:+1}"
_EXPLICIT_SEEDS="${SEEDS:+1}"

: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=agent_scaling_3f}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=500}"
: "${WANDB_PROJECT:=agent_scaling_3f}"
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

NS_LIST=(${NS:-2 3 4 5 6 9})
# Three seeds by default (results-first); the extension trio pools in later
# via the idempotent skip: SEEDS="789 1011 1213" bash submit_agent_scaling_3f.sh
SEEDS_LIST=(${SEEDS:-42 123 456})

# SMOKE supplies DEFAULTS, never overrides.
if [ "${SMOKE:-0}" = "1" ]; then
    [ -z "$_EXPLICIT_NS" ]        && NS_LIST=(2 9)
    [ -z "$_EXPLICIT_SEEDS" ]     && SEEDS_LIST=(42)
    [ -z "$_EXPLICIT_RUN_GROUP" ] && RUN_GROUP=agent_scaling_3f_smoke
    [ -z "$_EXPLICIT_EPISODES" ]  && EPISODES=1
    [ -z "$_EXPLICIT_MAX_STEPS" ] && MAX_STEPS=${SMOKE_STEPS:-150}
    export RUN_GROUP EPISODES MAX_STEPS
    export WANDB=0
fi

# Per-N resources. Memory/CPUs depend only on client count (tabulated); wall
# time is derived from the measured fit min/step ≈ 0.14 + 0.19·N with a 1.5×
# margin (see header), ladder-rounded; qos = medium up to 36 h, long above.
resources_for_n() {
    local n="$1"
    if   [ "$n" -le 3 ]; then R_MEM=32GB;  R_CPUS=8
    elif [ "$n" -le 5 ]; then R_MEM=48GB;  R_CPUS=8
    elif [ "$n" -le 6 ]; then R_MEM=64GB;  R_CPUS=10
    else                      R_MEM=96GB;  R_CPUS=12
    fi

    # Integer arithmetic in centi-minutes per step (bash has no floats).
    local steps=$(( EPISODES * MAX_STEPS ))
    local centi=$(( 14 + 19 * n ))
    R_EST_H=$(( centi * steps / 6000 ))          # projected hours
    local req_h=$(( R_EST_H * 3 / 2 ))           # 1.5x margin (lean, see header)
    [ "$req_h" -lt 8 ] && req_h=8
    local tier
    for tier in 24 36 48 72 96 120 144 168; do
        [ "$req_h" -le "$tier" ] && break
    done
    if [ "$req_h" -gt 168 ]; then
        echo "WARNING: N=$n projected ${R_EST_H}h; 1.5x margin (${req_h}h) exceeds" \
             "long-qos 168h cap — requesting 168h" >&2
    fi
    R_TIME="${tier}:00:00"
    if [ "$tier" -le 36 ]; then R_QOS=medium; else R_QOS=long; fi

    R_MEM="${MEM:-$R_MEM}"; R_CPUS="${CPUS:-$R_CPUS}"
    R_QOS="${QOS:-$R_QOS}"; R_TIME="${TIME:-$R_TIME}"
}

echo "== submit_agent_scaling_3f.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  N sweep   : ${NS_LIST[*]}"
echo "  seeds     : ${SEEDS_LIST[*]}"
echo "  rule      : three_factor + death LTD (hebbian arm only)"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "============================="

# Pre-flight: image + weights staged.
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

queued_names=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

n_queued=0
n_skipped=0
for n in "${NS_LIST[@]}"; do
    resources_for_n "$n"
    exp="scale_gemma_hebbian_n${n}"
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
            jobid=$(NUM_AGENTS=$n SEED=$seed sbatch --parsable \
                --job-name="$jobname" \
                --mem="$R_MEM" --cpus-per-task="$R_CPUS" \
                --qos="$R_QOS" --time="$R_TIME" \
                scale_gemma_3f.sbatch)
            echo "queued  $exp  seed_$seed  →  job $jobid  (mem=$R_MEM qos=$R_QOS time=$R_TIME)"
        fi
        n_queued=$((n_queued + 1))
    done
done
echo "── done: $n_queued submitted, $n_skipped skipped ──"
echo "Track with:   squeue -u \$USER"
echo "Base arm:     ARMS=\"0\" SEEDS=\"42 123 456\" bash submit_agent_scaling.sh"
echo "Then locally: merge runs_from_daic/agent_scaling_3f (hebbian) with"
echo "              runs_from_daic/agent_scaling (base) into one root for"
echo "              python make_scaling_fig.py --runs-root <merged root>"
