#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_comm_budget.sh — the communication-budget sweep.
#
#   N ∈ {3,5,7}  ×  budget ∈ {0, 800, 3200, 12800} tokens/agent/episode
#                ×  arm ∈ {base, hebbian}  ×  seeds
#
# Gemma 4 E4B, 3 ep × 1000 steps, one budget_gemma.sbatch job per cell.
# Lands in runs/comm_budget/budget_gemma_<arm>_n<N>_b<B>/seed_<S>/.
#
# Ladder (per agent per episode; ~16 model tokens per short message):
#     0      zero      control
#     800    low       ≈ 50 msgs  ≈ 5 % of steps (the observed request rate)
#     3200   medium    ≈ 200 msgs ≈ 20 % of steps
#     12800  high      ≈ 800 msgs ≈ 80 % of steps (binds only for every-step chatter)
# The token values assume 16 tokens/message; re-pin them with
#     python analysis/calibrate_comm_budget.py --model $MODEL_LLM
# before the first submission (message counts are the invariant).
#
# Wall-time policy: same fit as submit_agent_scaling_3f.sh (min/step ≈
# 0.14 + 0.19·N, ×1.5 margin, ladder-rounded; medium ≤ 36 h else long). At
# 3 ep × 1000 steps every N lands on long: N=3 → 72 h, N=5 → 96 h, N=7 → 120 h.
# Budgeted cells should run faster than the fit (shorter inboxes, empty
# comm fields), so the margin is generous.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1   bash submit_comm_budget.sh     # N=3, b=800, both arms, seed 42, 1 ep × 150
#   DRY_RUN=1 bash submit_comm_budget.sh     # print what would be submitted
#   SEEDS="42" bash submit_comm_budget.sh    # PILOT: 3 N × 4 budgets × 2 arms = 24 jobs
#   SEEDS="123 456" bash submit_comm_budget.sh   # extension: +48 jobs
#   NS="3" BUDGETS="0 800" ARMS="hebbian" bash submit_comm_budget.sh  # subset
#
# Idempotent: a cell whose final_metrics.json already exists on PRB is
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
_EXPLICIT_BUDGETS="${BUDGETS:+1}"
_EXPLICIT_ARMS="${ARMS:+1}"
_EXPLICIT_SEEDS="${SEEDS:+1}"

: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=comm_budget}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=comm_budget}"
: "${COMM_MSG_CAP:=32}"
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT COMM_MSG_CAP

NS_LIST=(${NS:-3 5 7})
BUDGET_LIST=(${BUDGETS:-0 800 3200 12800})
ARM_LIST=(${ARMS:-base hebbian})
# Pilot = one seed; extend with SEEDS="123 456" (finished cells are skipped).
SEEDS_LIST=(${SEEDS:-42})

# SMOKE supplies DEFAULTS, never overrides.
if [ "${SMOKE:-0}" = "1" ]; then
    [ -z "$_EXPLICIT_NS" ]        && NS_LIST=(3)
    [ -z "$_EXPLICIT_BUDGETS" ]   && BUDGET_LIST=(800)
    [ -z "$_EXPLICIT_ARMS" ]      && ARM_LIST=(base hebbian)
    [ -z "$_EXPLICIT_SEEDS" ]     && SEEDS_LIST=(42)
    [ -z "$_EXPLICIT_RUN_GROUP" ] && RUN_GROUP=comm_budget_smoke
    [ -z "$_EXPLICIT_EPISODES" ]  && EPISODES=1
    [ -z "$_EXPLICIT_MAX_STEPS" ] && MAX_STEPS=${SMOKE_STEPS:-150}
    export RUN_GROUP EPISODES MAX_STEPS
    export WANDB=0
fi

# Per-N resources — identical policy to submit_agent_scaling_3f.sh.
resources_for_n() {
    local n="$1"
    if   [ "$n" -le 3 ]; then R_MEM=32GB;  R_CPUS=8
    elif [ "$n" -le 5 ]; then R_MEM=48GB;  R_CPUS=8
    elif [ "$n" -le 6 ]; then R_MEM=64GB;  R_CPUS=10
    else                      R_MEM=96GB;  R_CPUS=12
    fi

    local steps=$(( EPISODES * MAX_STEPS ))
    local centi=$(( 14 + 19 * n ))
    R_EST_H=$(( centi * steps / 6000 ))
    local req_h=$(( R_EST_H * 3 / 2 ))
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

echo "== submit_comm_budget.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  N sweep   : ${NS_LIST[*]}"
echo "  budgets   : ${BUDGET_LIST[*]}  (tokens/agent/episode, msg cap $COMM_MSG_CAP)"
echo "  arms      : ${ARM_LIST[*]}"
echo "  seeds     : ${SEEDS_LIST[*]}"
echo "  rewards   : --comm-reward-scale 0 on every cell"
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
    for b in "${BUDGET_LIST[@]}"; do
        for arm in "${ARM_LIST[@]}"; do
            exp="budget_gemma_${arm}_n${n}_b${b}"
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
                    jobid=$(NUM_AGENTS=$n COMM_BUDGET=$b ARM=$arm SEED=$seed sbatch --parsable \
                        --job-name="$jobname" \
                        --mem="$R_MEM" --cpus-per-task="$R_CPUS" \
                        --qos="$R_QOS" --time="$R_TIME" \
                        budget_gemma.sbatch)
                    echo "queued  $exp  seed_$seed  →  job $jobid  (mem=$R_MEM qos=$R_QOS time=$R_TIME)"
                fi
                n_queued=$((n_queued + 1))
            done
        done
    done
done
echo "── done: $n_queued submitted, $n_skipped skipped ──"
echo "Track with:   squeue -u \$USER -o \"%.10i %.52j %.8T %.12M %.12l %R\""
echo "Then locally: bash pull_new.sh  (runs_from_daic/comm_budget) and"
echo "              python analysis/make_budget_fig.py"
