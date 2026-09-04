#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_pareto_social_3f.sh — social-interval Pareto sweep on the NEW
# Hebbian rule (three_factor + signed death LTD), Gemma 4 E4B, vision.
# Replaces submit_pareto_social.sh's reward_modulated sweep.
#
# Six plot points; only x = 0 is reused from the old suite:
#
#   x = 0 (no module)      new_exp_0_gemma_base        existing (rule-free)
#   interval 2             new_exp_0_gemma_si3f2       this script
#   interval 8 (default)   new_exp_0_gemma_si3f8       this script (the old
#                          hebbian anchor was reward_modulated; hebbian3f has
#                          no death LTD — neither anchors the new rule)
#   interval 20            new_exp_0_gemma_si3f20      this script
#   interval 50            new_exp_0_gemma_si3f50      this script
#   interval 100           new_exp_0_gemma_si3f100     this script
#
# 5 intervals × 3 seeds = 15 jobs, qos medium (36 h, baked into the sbatch).
# New arms land in runs/pareto_social_3f/ + wandb pareto_social_3f.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1   bash submit_pareto_social_3f.sh   # si3f2 × seed 42, 1 ep × 150
#   DRY_RUN=1 bash submit_pareto_social_3f.sh   # print what would be submitted
#   bash submit_pareto_social_3f.sh             # full sweep (15 jobs)
#
#   INTERVALS="2 8 20" bash submit_pareto_social_3f.sh   # subset
#   SEEDS="789 1011 1213" bash submit_pareto_social_3f.sh # seed extension
#
# Idempotent (final_metrics.json skip) + in-queue dedup (--job-name carries
# group/arm/seed): rerunning the SAME command is always safe and only fills
# in what is neither finished nor queued.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

# ── Pinned to the new_exp_0_gemma anchors (env-overridable, but overriding
#    model/image/steps makes the new points non-comparable to the existing
#    base runs — don't, unless you re-run those too) ────────────────────────
: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=pareto_social_3f}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=pareto_social_3f}"
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

# Interval 8 IS in the default grid this time — no existing arm carries the
# new rule at the default interval, so it is a sweep point, not a duplicate.
INTERVALS=(${INTERVALS:-2 8 20 50 100})
SEEDS=(${SEEDS:-42 123 456})

# Smoke: the cheapest end (interval 2 = max social-call rate) for one short
# run — proves the flags (incl. --hebbian-death-ltd) reach the module before
# committing 15 × 36 GPU-hours. Lands in runs/pareto_social_3f_smoke/.
if [ "${SMOKE:-0}" = "1" ]; then
    INTERVALS=(2)
    SEEDS=(42)
    RUN_GROUP=pareto_social_3f_smoke
    EPISODES=1
    MAX_STEPS=${SMOKE_STEPS:-150}
    export RUN_GROUP EPISODES MAX_STEPS
    export WANDB=0
    : "${QOS:=short}"
    : "${TIME:=4:00:00}"
fi

SBATCH_OVERRIDES=()
[ -n "${QOS:-}" ]  && SBATCH_OVERRIDES+=(--qos="$QOS")
[ -n "${TIME:-}" ] && SBATCH_OVERRIDES+=(--time="$TIME")
[ -n "${GPU:-}" ]  && SBATCH_OVERRIDES+=(--gres="$GPU")
[ -n "${MEM:-}" ]  && SBATCH_OVERRIDES+=(--mem="$MEM")
# A command-line --exclude REPLACES the sbatch file's baked-in
# "--exclude=cor1", so always include cor1 in the list.
[ -n "${EXCLUDE:-}" ] && SBATCH_OVERRIDES+=(--exclude="$EXCLUDE")

echo "== submit_pareto_social_3f.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  vision    : $LLM_VISION_MODE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  intervals : ${INTERVALS[*]}  (anchor: base = no module; rule = 3f+fdecay)"
echo "  seeds     : ${SEEDS[*]}"
[ ${#SBATCH_OVERRIDES[@]} -gt 0 ] && echo "  overrides : ${SBATCH_OVERRIDES[*]}"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "============================="

# Pre-flight: image + staged weights, checked before anything is queued.
missing=0
if [ ! -f "$WT_IMAGE" ]; then
    echo "ERROR: image not found: $WT_IMAGE" >&2
    missing=1
fi
if [ ! -f "$MODEL_LLM/config.json" ]; then
    echo "ERROR: weights not staged: $MODEL_LLM/config.json" >&2
    missing=1
fi
[ "$missing" = "1" ] && exit 1

# In-queue dedup: the final_metrics skip only sees FINISHED runs, so a rerun
# while jobs are pending would double-submit them. New submissions carry
# --job-name=<group>-<arm>_s<seed>; anything already in squeue under that
# name (or under the sbatch file's baked-in name, from a manual submit —
# seed unknown, so it conservatively blocks the whole sweep) is skipped.
QUEUED_NAMES=$(squeue -u "${USER:-$(whoami)}" -h -o %j 2>/dev/null || true)

n_queued=0
n_skipped=0
n_inqueue=0
n_failed=0
for si in "${INTERVALS[@]}"; do
    exp="new_exp_0_gemma_si3f${si}"
    for seed in "${SEEDS[@]}"; do
        if [ -f "$REPO/runs/$RUN_GROUP/$exp/seed_$seed/final_metrics.json" ]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        jobname="${RUN_GROUP}-${exp}_s${seed}"
        if printf '%s\n' "$QUEUED_NAMES" | grep -Fqx "$jobname"; then
            echo "in queue     $exp  seed_$seed"
            n_inqueue=$((n_inqueue + 1))
            continue
        fi
        if printf '%s\n' "$QUEUED_NAMES" | grep -Fqx "new-exp-0-gemma-si3f"; then
            echo "in queue*    $exp  seed_$seed  (manually submitted job, arm/seed unknown — sweep blocked until it drains)"
            n_inqueue=$((n_inqueue + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $exp  seed_$seed"
            n_queued=$((n_queued + 1))
        else
            jobid=$(SEED=$seed SOCIAL_INTERVAL=$si sbatch --parsable --job-name="$jobname" \
                ${SBATCH_OVERRIDES[@]:+"${SBATCH_OVERRIDES[@]}"} \
                new_exp_0_gemma_si3f.sbatch)
            if [ -n "$jobid" ]; then
                echo "queued  $exp  seed_$seed  →  job $jobid"
                n_queued=$((n_queued + 1))
            else
                # sbatch printed its error to stderr already. Abort instead of
                # spamming more failures. If it is the per-user QOS submission
                # cap, nothing is wrong — rerun this SAME command as the queue
                # drains; dedup makes that safe.
                echo "FAILED  $exp  seed_$seed  — sbatch rejected the job" >&2
                n_failed=$((n_failed + 1))
                break 2
            fi
        fi
    done
done
echo "── done: $n_queued submitted, $n_skipped already complete, $n_inqueue in queue, $n_failed failed ──"
[ "$n_failed" -gt 0 ] && exit 1
echo "Track with:   squeue -u \$USER"
echo "Watch load:   tail -f slurm_logs/new_exp_0_gemma_si3f_*.out"
