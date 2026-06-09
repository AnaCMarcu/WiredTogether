#!/bin/bash
# Submit WiredTogether experiments to SLURM in parallel on DelftBlue.
#
# Mirrors hpc/daic/experiments/submit_all.sh — same flags, same filtering
# semantics, same multi-seed array support. Different cluster, same UX.
#
# The actual seed VALUES come from _common.sh's hardcoded array
# (SEEDS=(42 123 456) at the top of that file). This script only controls
# HOW MANY of those seeds to run, via N_SEEDS.
#
# Usage (from DelftBlue login node, or anywhere with sbatch access):
#
#   bash hpc/delft_blue/experiments/submit_all.sh
#       — submit every exp*.sh with seed 42 (single job per experiment).
#
#   N_SEEDS=2 bash hpc/delft_blue/experiments/submit_all.sh
#       — submit every exp*.sh as a SLURM job array with 2 tasks.
#         Tasks 0,1 land on seeds 42,123 (from _common.sh's array).
#
#   N_SEEDS=3 bash hpc/delft_blue/experiments/submit_all.sh
#       — full 3-seed sweep: tasks 0,1,2 → seeds 42, 123, 456.
#
#   ONLY="exp1_,exp4_" bash hpc/delft_blue/experiments/submit_all.sh
#       — submit only the named experiments. Comma-separated substring
#         match against the .sh filename. Use a trailing "_" to avoid
#         "exp1" also matching "exp10".
#
#   SKIP="exp9_,exp11_" bash hpc/delft_blue/experiments/submit_all.sh
#       — submit everything except these.
#
#   DRY_RUN=1 bash hpc/delft_blue/experiments/submit_all.sh
#       — print the sbatch command for each experiment without submitting.
#         Sanity check before burning N × 36h of GPU.
#
#   WANDB=0 bash hpc/delft_blue/experiments/submit_all.sh
#       — disable W&B logging for this batch (otherwise inherited from env).
#
# Per-experiment overrides (e.g. custom WANDB_EXTRA_TAGS) belong in each
# .sh file. Env vars set on the submit_all.sh command line are passed
# through to sbatch via --export=ALL,VAR=value.
#
# Results land in:
#     /scratch/$USER/WiredTogether/runs/legacy/<exp_name>/seed_<seed>/
# SLURM .out logs land at:
#     /scratch/$USER/WiredTogether/slurm_logs/<exp_name>-<jobid>.out
# (or <exp_name>-<jobid>_<arrayidx>.out for array jobs).
#
# Re-submit one experiment with a chosen seed:
#     SEED=123 sbatch hpc/delft_blue/experiments/exp4_mappo_hebbian.sh

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same experiment set as DAIC. Keep in lock-step so cross-cluster
# comparisons hit identical configs.
EXPERIMENTS=(
    # "exp1_llm.sh"
    "exp3_mappo.sh"
    "exp4_mappo_hebbian.sh"
    "exp5_ippo.sh"
    "exp6_ippo_hebbian.sh"
    # "exp7_llm_9b.sh"
    # "exp9_llm_2b_social_prompt.sh"
    # "exp11_llm_9b_social_prompt.sh"
)

# Resolved seed values for human-readable reporting only. Must match the
# hardcoded SEEDS=(42 123 456) array in _common.sh.
SEED_VALUES_REFERENCE=(42 123 456)

N_SEEDS="${N_SEEDS:-1}"
if ! [[ "$N_SEEDS" =~ ^[1-3]$ ]]; then
    echo "ERROR: N_SEEDS must be 1, 2, or 3 (got '$N_SEEDS'). To use other" >&2
    echo "       seeds, edit the SEEDS=(...) array in _common.sh and bump" >&2
    echo "       SEED_VALUES_REFERENCE here so reporting matches." >&2
    exit 1
fi

# Comma-separated substring match against the .sh filename.
matches_csv() {
    local needle="$1" csv="$2"
    local IFS=','
    for token in $csv; do
        token=$(echo "$token" | xargs)
        [ -z "$token" ] && continue
        if [[ "$needle" == *"$token"* ]]; then
            return 0
        fi
    done
    return 1
}

FILTERED=()
for exp in "${EXPERIMENTS[@]}"; do
    if [ -n "${ONLY:-}" ] && ! matches_csv "$exp" "$ONLY"; then
        continue
    fi
    if [ -n "${SKIP:-}" ] &&   matches_csv "$exp" "$SKIP"; then
        continue
    fi
    FILTERED+=("$exp")
done

SEEDS_USED=("${SEED_VALUES_REFERENCE[@]:0:$N_SEEDS}")

echo "== submit_all.sh =="
echo "  experiments : ${#FILTERED[@]} of ${#EXPERIMENTS[@]}"
echo "  N_SEEDS     : $N_SEEDS  (seeds=${SEEDS_USED[*]} from _common.sh)"
if [ -n "${ONLY:-}" ]; then echo "  only        : $ONLY"; fi
if [ -n "${SKIP:-}" ]; then echo "  skip        : $SKIP"; fi
if [ "${DRY_RUN:-0}" = "1" ]; then echo "  mode        : DRY_RUN (no submission)"; fi
echo "==================="

# --array spec when more than one seed.
SBATCH_ARRAY=()
if [ "$N_SEEDS" -gt 1 ]; then
    SBATCH_ARRAY=(--array="0-$((N_SEEDS-1))")
fi

# Vars to forward into the job's environment. Each is single-token-safe
# (no spaces, no commas) so it passes cleanly through sbatch --export.
EXPORT_VARS=()
[ -n "${WANDB:-}" ]              && EXPORT_VARS+=("WANDB=$WANDB")
[ -n "${WANDB_PROJECT:-}" ]      && EXPORT_VARS+=("WANDB_PROJECT=$WANDB_PROJECT")
[ -n "${WANDB_MODE:-}" ]         && EXPORT_VARS+=("WANDB_MODE=$WANDB_MODE")
# WANDB_EXTRA_TAGS may contain commas; pass it via env inheritance only.
# (Each .sh file re-exports its own default if unset.)
if [ ${#EXPORT_VARS[@]} -gt 0 ]; then
    EXPORT_LIST=$(IFS=,; echo "${EXPORT_VARS[*]}")
    EXPORT_FLAG="--export=ALL,$EXPORT_LIST"
else
    EXPORT_FLAG="--export=ALL"
fi

submitted=0
for exp in "${FILTERED[@]}"; do
    script="$EXP_DIR/$exp"
    if [ ! -f "$script" ]; then
        echo "MISSING: $script — skipping" >&2
        continue
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  [dry] sbatch ${SBATCH_ARRAY[*]:-} $EXPORT_FLAG $exp"
        continue
    fi
    jobid=$(sbatch --parsable \
        ${SBATCH_ARRAY[@]:+"${SBATCH_ARRAY[@]}"} \
        "$EXPORT_FLAG" \
        "$script")
    if [ "$N_SEEDS" -gt 1 ]; then
        echo "  $exp  →  job $jobid  (array 0-$((N_SEEDS-1)), seeds=${SEEDS_USED[*]})"
    else
        echo "  $exp  →  job $jobid  (seed=${SEEDS_USED[0]})"
    fi
    submitted=$((submitted+1))
done

echo "================================="
echo "  submitted : $submitted job(s)"
echo "Track with:    squeue -u \$USER"
echo "Tail any log:  tail -f /scratch/\$USER/WiredTogether/slurm_logs/<exp_name>-<jobid>.out"
echo "Cancel all:    scancel -u \$USER"
