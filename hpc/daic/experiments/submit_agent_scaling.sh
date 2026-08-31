#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_agent_scaling.sh — the agent-count scaling sweep on Gemma 4 E4B
# ("Gemma 4B"): N ∈ {2,3,4,5,6,9} agents, identical environment, 3 ep × 500
# steps. Produces the data for make_scaling_fig.py (x = inference FLOPs,
# y = cooperative milestones; one point per team size).
#
# Design invariants (enforced by scale_gemma.sbatch → --ch4-mob-count 3):
#   * same tasks and same team-level max milestones for every N — Ch4 is
#     pinned at 3 zombies, boss/anvils/Ch1 resources are N-independent;
#   * Chamber 3 is the only room that scales: one cell + switch per agent,
#     with the 3-agent milestone reward values unchanged;
#   * one Luanti client per agent, so job memory/time scale with N below.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1   bash submit_agent_scaling.sh   # N=2 and N=9, seed 42, 1 ep × 150
#   DRY_RUN=1 bash submit_agent_scaling.sh   # print what would be submitted
#   bash submit_agent_scaling.sh             # full sweep: 6 N × 6 seeds × BOTH arms (72 jobs)
#   SEEDS="789 1011 1213" bash submit_agent_scaling.sh   # only the 3-seed extension (36 jobs)
#   ARMS="0" bash submit_agent_scaling.sh    # base (non-Hebbian) arm only
#   NS="2 9" SEEDS="42" bash submit_agent_scaling.sh   # subset override
#
# Idempotent: an N/arm/seed whose final_metrics.json already exists on PRB is
# skipped, and one already sitting in the Slurm queue (same job name) is not
# resubmitted.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

# ── Configuration (all env-overridable) ─────────────────────────────────────
# Record which knobs the CALLER set before any default is applied, so the
# SMOKE block below can supply defaults without clobbering an explicit
# value. (It used to plain-assign RUN_GROUP, which silently discarded
# `RUN_GROUP=foo SMOKE=1 bash submit_agent_scaling.sh` and sent the run
# back to the group whose final_metrics.json then made it "skipped".)
_EXPLICIT_RUN_GROUP="${RUN_GROUP:+1}"
_EXPLICIT_EPISODES="${EPISODES:+1}"
_EXPLICIT_MAX_STEPS="${MAX_STEPS:+1}"
_EXPLICIT_NS="${NS:+1}"
_EXPLICIT_SEEDS="${SEEDS:+1}"

: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=agent_scaling}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=500}"
: "${WANDB_PROJECT:=agent_scaling}"
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

NS_LIST=(${NS:-2 3 4 5 6 9})
# Six seeds: the original three plus the extension trio pre-registered for
# the medium suite (789 1011 1213, submit_seed_extension.sh), so pooled
# episodes per point go 9 -> 18. Idempotent: seeds whose final_metrics.json
# exists are skipped, so the full list is safe to resubmit at any time.
SEEDS_LIST=(${SEEDS:-42 123 456 789 1011 1213})
# Both arms by default: the base-vs-Hebbian scaling-curve comparison is the
# point of the figure. ARMS="0" for base only, ARMS="1" for Hebbian only.
ARMS_LIST=(${ARMS:-0 1})        # 0 = base, 1 = hebbian

# SMOKE supplies DEFAULTS, never overrides: each knob is only forced when
# the caller did not set it, so `RUN_GROUP=... SMOKE=1` re-tests into a
# fresh group instead of colliding with the previous smoke's results.
if [ "${SMOKE:-0}" = "1" ]; then
    # cheapest + most demanding client count
    [ -z "$_EXPLICIT_NS" ]        && NS_LIST=(2 9)
    [ -z "$_EXPLICIT_SEEDS" ]     && SEEDS_LIST=(42)
    [ -z "$_EXPLICIT_RUN_GROUP" ] && RUN_GROUP=agent_scaling_smoke
    [ -z "$_EXPLICIT_EPISODES" ]  && EPISODES=1
    [ -z "$_EXPLICIT_MAX_STEPS" ] && MAX_STEPS=${SMOKE_STEPS:-150}
    export RUN_GROUP EPISODES MAX_STEPS
    export WANDB=0
fi

# Per-N resources: each agent is a Luanti client + its share of LLM batch
# state. 3 agents fit the 32GB/8cpu defaults (gemma suite); 6 clients needed
# 64GB in expB_merged_transplant; 9 gets headroom on both axes. GPU is NOT
# the constraint -- the smokes measured 15.88 GB at N=9, identical to N=3,
# because the model is shared rather than per-agent.
#
# Wall time is the real constraint. It grows with N (more LLM calls per env
# step) AND with the step budget, so it is DERIVED rather than tabulated —
# a fixed table silently under-requests the moment MAX_STEPS changes. At
# 3 ep x 1000 steps the old N<=3 tier (medium/36h) would have given N=3 a
# 35.5h job against a 36h cap: a ~1.01x margin, i.e. a near-certain loss of
# the whole run at the wall.
#
# Measured on the 2026-08-20 smokes (base arm, 121 steps):
#     N=2 -> 0.52 min/step, N=9 -> 1.86 min/step
#     fit:  min/step ~= 0.14 + 0.19*N   (near-linear in N)
# Requested wall time = 2x that projection, rounded up to a SLURM ladder
# rung and capped at long-qos's 168h. Worked examples:
#     3 ep x  500 -> N=2 13h(->36) N=4 23h(->72)  N=6 32h(->72)  N=9 47h(->96)
#     3 ep x 1000 -> N=2 26h(->72) N=4 46h(->96)  N=6 64h(->144)
# Memory/CPU depend only on the client count, so they stay tabulated. GPU is
# NOT the constraint -- the smokes measured 15.88 GB at N=9, identical to
# N=3, because the model is shared rather than per-agent.
# All overridable: MEM=… TIME=… QOS=… CPUS=….
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
    local req_h=$(( R_EST_H * 2 ))               # 2x safety margin
    [ "$req_h" -lt 12 ] && req_h=12
    local tier
    for tier in 36 72 96 120 144 168; do
        [ "$req_h" -le "$tier" ] && break
    done
    if [ "$req_h" -gt 168 ]; then
        echo "WARNING: N=$n projected ${R_EST_H}h; 2x margin (${req_h}h) exceeds" \
             "long-qos 168h cap — requesting 168h, margin only" \
             "$(( 168 * 10 / (R_EST_H > 0 ? R_EST_H : 1) ))/10x" >&2
    fi
    R_TIME="${tier}:00:00"
    if [ "$tier" -le 36 ]; then R_QOS=medium; else R_QOS=long; fi

    R_MEM="${MEM:-$R_MEM}"; R_CPUS="${CPUS:-$R_CPUS}"
    R_QOS="${QOS:-$R_QOS}"; R_TIME="${TIME:-$R_TIME}"
}

echo "== submit_agent_scaling.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  N sweep   : ${NS_LIST[*]}"
echo "  seeds     : ${SEEDS_LIST[*]}"
echo "  arms      : ${ARMS_LIST[*]} (0=base 1=hebbian)"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "============================="

# Pre-flight: image + weights staged (same checks as submit_gemma4.sh).
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
    for arm in "${ARMS_LIST[@]}"; do
        if [ "$arm" = "1" ]; then exp="scale_gemma_hebbian_n${n}"; else exp="scale_gemma_base_n${n}"; fi
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
                jobid=$(NUM_AGENTS=$n HEBBIAN=$arm SEED=$seed sbatch --parsable \
                    --job-name="$jobname" \
                    --mem="$R_MEM" --cpus-per-task="$R_CPUS" \
                    --qos="$R_QOS" --time="$R_TIME" \
                    scale_gemma.sbatch)
                echo "queued  $exp  seed_$seed  →  job $jobid  (mem=$R_MEM qos=$R_QOS)"
            fi
            n_queued=$((n_queued + 1))
        done
    done
done
echo "── done: $n_queued submitted, $n_skipped skipped ──"
echo "Track with:   squeue -u \$USER"
echo "Then locally: python make_scaling_fig.py --runs-root runs/agent_scaling"
