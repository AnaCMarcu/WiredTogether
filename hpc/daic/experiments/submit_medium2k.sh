#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_medium2k.sh — medium2k sweep for the NON-frozen conditions.
#
# The frozen-graph LLM conditions (exp09/10/11, --hebbian-freeze) are already
# running under RUN_GROUP=medium2k. This launcher fills in the remaining eight
# conditions (exp01–exp08) with the SAME medium2k configuration so every run
# pools into runs/medium2k/:
#
#   RUN_GROUP=medium2k          → results in runs/medium2k/<exp>/seed_<N>/
#   EPISODES=3, MAX_STEPS=2000  → 3 episodes × ≤2000 steps
#   SEEDS=(42 123 456)          → the three original medium-suite seeds
#
# Do NOT change RUN_GROUP / EPISODES / MAX_STEPS below, or the new runs cannot
# be pooled with the frozen-graph medium2k data. (The sbatch files themselves
# default to the FINAL suite — 5 ep × 2500 steps, runs/final/ — which is why we
# override here rather than in each file.)
#
# Idempotent: skips any exp/seed whose runs/medium2k/.../final_metrics.json
# already exists on PRB, so re-running only submits what is still missing.
#
# Every setting is env-overridable, e.g. to schedule faster on medium qos:
#   QOS=medium TIME=36:00:00 bash submit_medium2k.sh
# or to re-check the WANDB project against the frozen-graph runs:
#   WANDB_PROJECT=final_wired_together bash submit_medium2k.sh
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   DRY_RUN=1 bash submit_medium2k.sh   # print what would be submitted
#   bash submit_medium2k.sh             # submit everything missing
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

REPO=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether

# ── medium2k configuration (env-overridable, defaults must match the frozen
#    graph exp09/10/11 runs already in flight) ────────────────────────────────
: "${RUN_GROUP:=medium2k}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=2000}"
: "${WANDB_PROJECT:=medium2k_wired_together}"
export RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

# Optional sbatch overrides (empty = keep each exp file's baked-in
# --qos=long / --time=168:00:00 directive).
SBATCH_OVERRIDES=()
[ -n "${QOS:-}" ]  && SBATCH_OVERRIDES+=(--qos="$QOS")
[ -n "${TIME:-}" ] && SBATCH_OVERRIDES+=(--time="$TIME")

# The eight NON-frozen conditions. exp09/10/11 (frozen graph) are deliberately
# excluded — they are already running.
EXPS=(
    exp01_llm_2b
    exp02_llm_9b
    exp03_mappo
    exp04_ippo
    exp05_mappo_hebbian
    exp06_ippo_hebbian
    exp07_llm_2b_social_prompt
    exp08_llm_9b_social_prompt
)
SEEDS=(42 123 456)

echo "== submit_medium2k.sh =="
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : $WANDB_PROJECT"
echo "  exps      : ${#EXPS[@]} (${EXPS[*]})"
echo "  seeds     : ${SEEDS[*]}"
[ ${#SBATCH_OVERRIDES[@]} -gt 0 ] && echo "  overrides : ${SBATCH_OVERRIDES[*]}"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "========================"

n_queued=0
n_skipped=0
for exp in "${EXPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ -f "$REPO/runs/$RUN_GROUP/$exp/seed_$seed/final_metrics.json" ]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $exp  seed_$seed"
        else
            jobid=$(SEED=$seed sbatch --parsable \
                ${SBATCH_OVERRIDES[@]:+"${SBATCH_OVERRIDES[@]}"} \
                "$exp.sbatch")
            echo "queued  $exp  seed_$seed  →  job $jobid"
        fi
        n_queued=$((n_queued + 1))
    done
done
echo "── done: $n_queued submitted, $n_skipped already complete ──"
echo "Track with:   squeue -u \$USER"
echo "Cancel all:   scancel -u \$USER"
