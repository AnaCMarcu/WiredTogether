#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_pareto.sh — launch the Gemma 4 size sweep (new_exp_pareto.sbatch)
# with per-size SLURM resources.
#
# Suite: {e2b, 12b, 26b, 31b} × {base, hebbian} × 6 seeds = 48 jobs.
# The E4B (4.5B) point REUSES the finished new_exp_0_gemma runs — add
# SIZES="... e4b" only if you deliberately want it re-run under this group.
#
# Results:  runs/pareto_gemma4/pareto_<size>_<arm>/seed_<N>/
# W&B:      project pareto_gemma4
#
# Usage (from the repo root on DAIC):
#   bash hpc/daic/experiments/submit_pareto.sh                 # full suite
#   DRY_RUN=1 bash hpc/daic/experiments/submit_pareto.sh       # print, no submit
#   SMOKE=1  bash hpc/daic/experiments/submit_pareto.sh        # 1 ep × 50 steps,
#                                                              # seed 42, per size+arm
#   SIZES="12b" bash hpc/daic/experiments/submit_pareto.sh     # one size only
#   SIZES="31b" GPU=gpu:1 GPU_CONSTRAINT=a100 bash ...         # if DAIC has 80GB A100s
#   SEEDS="42 123 456" bash ...                                # subset of seeds
#
# SMOKE FIRST for 12b/26b/31b: it validates (a) the checkpoint loads and
# shards, (b) the vision path engages (log.txt: "Loaded VISION model"),
# (c) VRAM headroom, and (d) steps/hour — extrapolate to 3×1000 and check it
# fits the TIME below BEFORE committing 12 jobs per size. Measure with:
#   grep -c "INFO ep=" runs/pareto_gemma4_smoke/<exp>/seed_42/log.txt   # steps done
#
# Idempotent + queue-deduped (same as submit_cofiring.sh): an exp/seed with
# final_metrics.json is skipped, and one whose job name is still in squeue is
# skipped — rerunning this same command is always safe (e.g. after the QOS
# submission cap).
#
# Global overrides QOS/TIME/GPU/MEM/EXCLUDE replace the per-size defaults for
# EVERY submitted size — use them with a single-size SIZES=... invocation.
# GPU_CONSTRAINT adds --constraint (find feature names: sinfo -N -h -o "%N %G %f").
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${RUN_GROUP:=pareto_gemma4}"
: "${WANDB_PROJECT:=pareto_gemma4}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
export WT_IMAGE RUN_GROUP WANDB_PROJECT EPISODES MAX_STEPS

SIZES=(${SIZES:-e2b 12b 26b 31b})
SEEDS=(${SEEDS:-42 123 456 789 1011 1213})
ARMS=(${ARMS:-base hebbian})

if [ "${SMOKE:-0}" = "1" ]; then
    SEEDS=(42)
    RUN_GROUP=pareto_gemma4_smoke
    EPISODES=1
    MAX_STEPS=${SMOKE_STEPS:-50}
    export RUN_GROUP EPISODES MAX_STEPS
    export WANDB=0
fi

# Per-size SLURM resources (rationale in new_exp_pareto.sbatch's header;
# E4B wall-clock measured 19-38 h for 3×1000, the rest scaled from it).
size_resources() {
    case "$1" in
        e2b) S_GPU=gpu:1 S_MEM=32G S_QOS=medium S_TIME=36:00:00 ;;
        e4b) S_GPU=gpu:1 S_MEM=32G S_QOS=long   S_TIME=48:00:00 ;;
        12b) S_GPU=gpu:1 S_MEM=48G S_QOS=long   S_TIME=96:00:00 ;;
        26b) S_GPU=gpu:2 S_MEM=64G S_QOS=long   S_TIME=96:00:00 ;;
        31b) S_GPU=gpu:2 S_MEM=64G S_QOS=long   S_TIME=168:00:00 ;;
        *)   echo "ERROR: unknown size '$1'" >&2; return 1 ;;
    esac
    # Global env overrides win over the per-size defaults.
    S_GPU="${GPU:-$S_GPU}"; S_MEM="${MEM:-$S_MEM}"
    S_QOS="${QOS:-$S_QOS}"; S_TIME="${TIME:-$S_TIME}"
}

model_path() {
    case "$1" in
        e2b) echo "$WORKSPACE/models/gemma-4-E2B-it" ;;
        e4b) echo "$WORKSPACE/models/gemma-4-E4B-it" ;;
        12b) echo "$WORKSPACE/models/gemma-4-12B-it" ;;
        26b) echo "$WORKSPACE/models/gemma-4-26B-A4B-it" ;;
        31b) echo "$WORKSPACE/models/gemma-4-31B-it" ;;
    esac
}

echo "== submit_pareto.sh =="
echo "  image     : $WT_IMAGE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES × $MAX_STEPS steps"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  sizes     : ${SIZES[*]}"
echo "  arms      : ${ARMS[*]}"
echo "  seeds     : ${SEEDS[*]}"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "======================"

# Pre-flight: image + every selected size's staged weights, before anything
# is queued. Also warn when a checkpoint lacks the multimodal processor —
# that run would silently load TEXT-ONLY and void the perception axis.
missing=0
if [ ! -f "$WT_IMAGE" ]; then
    echo "ERROR: image not found: $WT_IMAGE" >&2
    missing=1
fi
for size in "${SIZES[@]}"; do
    mp="$(model_path "$size")"
    if [ ! -f "$mp/config.json" ]; then
        echo "ERROR: weights not staged for $size: $mp/config.json" >&2
        echo "       MODEL=google/${mp##*/} sbatch hpc/daic/download_gemma4.sbatch" >&2
        missing=1
    elif [ ! -f "$mp/processor_config.json" ] && [ ! -f "$mp/preprocessor_config.json" ]; then
        echo "ERROR: $size has no processor config — would load TEXT-ONLY (no perception)." >&2
        missing=1
    fi
done
[ "$missing" = "1" ] && exit 1

QUEUED_NAMES=$(squeue -u "${USER:-$(whoami)}" -h -o %j 2>/dev/null || true)

n_queued=0
n_skipped=0
n_inqueue=0
n_failed=0
for size in "${SIZES[@]}"; do
    size_resources "$size" || exit 1
    for arm in "${ARMS[@]}"; do
        case "$arm" in
            base)    h=0 ;;
            hebbian) h=1 ;;
            *) echo "ERROR: unknown arm '$arm'" >&2; exit 1 ;;
        esac
        exp="pareto_${size}_${arm}"
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
            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "would queue  $exp  seed_$seed  (${S_GPU} ${S_MEM} ${S_QOS} ${S_TIME})"
                n_queued=$((n_queued + 1))
            else
                jobid=$(MODEL_SIZE=$size HEBBIAN=$h SEED=$seed \
                    sbatch --parsable --job-name="$jobname" \
                        --gres="$S_GPU" --mem="$S_MEM" \
                        --qos="$S_QOS" --time="$S_TIME" \
                        ${EXCLUDE:+--exclude="$EXCLUDE"} \
                        ${GPU_CONSTRAINT:+--constraint="$GPU_CONSTRAINT"} \
                        new_exp_pareto.sbatch)
                if [ -n "$jobid" ]; then
                    echo "queued  $exp  seed_$seed  →  job $jobid  (${S_GPU} ${S_QOS} ${S_TIME})"
                    n_queued=$((n_queued + 1))
                else
                    # sbatch printed its error already. Abort instead of
                    # spamming more failures; the QOS submission cap is the
                    # benign cause — rerun this SAME command as the queue
                    # drains (dedup makes it safe).
                    echo "FAILED  $exp  seed_$seed  — sbatch rejected the job" >&2
                    n_failed=$((n_failed + 1))
                    break 3
                fi
            fi
        done
    done
done
echo "── done: $n_queued submitted, $n_skipped already complete, $n_inqueue in queue, $n_failed failed ──"
[ "$n_failed" -gt 0 ] && exit 1
echo "Track with:   squeue -u \$USER"
