#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_orchestrator.sh — O2 centralized orchestrator baseline on the
# new_exp_0_gemma suite (Gemma 4 E4B, vision, 3 ep × 1000 steps).
#
# Arms (per mode) land in runs/orchestrator/new_exp_0_gemma_orch_<mode>/
# seed_<S>/ + wandb project orchestrator_wired_together; the comparison
# anchors (new_exp_0_gemma_base / _hebbian) stay in runs/new_exp_0_gemma/
# and are never touched here — analyses read both roots.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1   bash submit_orchestrator.sh   # advisory × seed 42, 1 ep × 150
#                                           # steps, wandb off, 4 h walltime
#   DRY_RUN=1 bash submit_orchestrator.sh   # print what would be submitted
#   bash submit_orchestrator.sh             # advisory × 3 seeds = 3 jobs
#
#   MODES="advisory bias" bash submit_orchestrator.sh   # both couplings
#   SEEDS="42 123 456 789 1011 1213" bash submit_orchestrator.sh
#
# QOS: this account tops out at medium, so everything (smoke included)
# submits under the sbatch file's qos=medium; the smoke just asks for a 4 h
# walltime. QOS=short works too if the account has it: QOS=short SMOKE=1 ...
#
# Idempotent (final_metrics.json skip) + in-queue dedup (--job-name carries
# group/arm/seed), same as submit_cofiring.sh / submit_pareto_social.sh:
# rerunning the SAME command is always safe and only fills in what is
# neither finished nor queued.
#
# Smoke pass/fail — after the job drains, check the smoke run dir
# (runs/orchestrator_smoke/new_exp_0_gemma_orch_advisory/seed_42/):
#   run.log        has "[FEATURES] Orchestrator:     ENABLED [advisory]"
#                  and "[Orchestrator usage] prompt_tokens=..." lines
#   orchestrator/  calls.jsonl (~19 calls at cadence 8 over 150 steps, more
#                  with event triggers; "failed": false on most),
#                  compliance.jsonl, maps/*.png
#   final_metrics.json exists (episode completed end-to-end)
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

# ── Pinned to the new_exp_0_gemma anchors (env-overridable, but overriding
#    model/image/steps makes the arm non-comparable to the existing
#    base/hebbian runs — don't, unless you re-run those too) ────────────────
: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=orchestrator}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=orchestrator_wired_together}"
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

MODES=(${MODES:-advisory})
SEEDS=(${SEEDS:-42 123 456})
# Arm families: task (O2 task-ledger), social (Hebbian-matched centralized
# deliberation), plan (social + curriculum plan notes; upper baseline),
# villager (VillagerAgent-style DAG task orchestration; hard assignments,
# advisory mode only). Smoke example: SMOKE=1 VARIANTS=villager
VARIANTS=(${VARIANTS:-task})

# Smoke: one advisory seed, one short episode — proves the orchestrator call
# fires on Gemma 4 (map image attaches, JSON validates, directives reach the
# prompts, calls.jsonl/compliance.jsonl fill up) before committing
# 3 × 36 GPU-hours. Lands in runs/orchestrator_smoke/, wandb off.
if [ "${SMOKE:-0}" = "1" ]; then
    MODES=(advisory)
    SEEDS=(42)
    # VARIANTS is honoured in smoke mode: SMOKE=1 VARIANTS="social plan" ...
    RUN_GROUP=orchestrator_smoke
    EPISODES=1
    MAX_STEPS=${SMOKE_STEPS:-150}
    export RUN_GROUP EPISODES MAX_STEPS
    export WANDB=0
    # Account QOS ceiling is medium — keep the smoke there and just cap the
    # walltime; override with QOS=short if the account has it.
    : "${QOS:=medium}"
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

echo "== submit_orchestrator.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  vision    : $LLM_VISION_MODE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  variants  : ${VARIANTS[*]}"
echo "  modes     : ${MODES[*]}  (anchors: base=no coupling, hebbian=W(t))"
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
for variant in "${VARIANTS[@]}"; do
for mode in "${MODES[@]}"; do
    # Naming mirrors the sbatch file: the task variant keeps its original
    # arm name so finished runs keep matching the idempotency check.
    if [ "$variant" = "task" ]; then
        exp="new_exp_0_gemma_orch_${mode}"
    else
        exp="new_exp_0_gemma_orch_${variant}_${mode}"
    fi
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
        if printf '%s\n' "$QUEUED_NAMES" | grep -Fqx "new-exp-orch"; then
            echo "in queue*    $exp  seed_$seed  (manually submitted job, arm/seed unknown — sweep blocked until it drains)"
            n_inqueue=$((n_inqueue + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $exp  seed_$seed"
            n_queued=$((n_queued + 1))
        else
            jobid=$(SEED=$seed ORCH_MODE=$mode ORCH_VARIANT=$variant \
                sbatch --parsable --job-name="$jobname" \
                ${SBATCH_OVERRIDES[@]:+"${SBATCH_OVERRIDES[@]}"} \
                new_exp_orchestrator.sbatch)
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
                break 3
            fi
        fi
    done
done
done
echo "── done: $n_queued submitted, $n_skipped already complete, $n_inqueue in queue, $n_failed failed ──"
[ "$n_failed" -gt 0 ] && exit 1
echo "Track with:   squeue -u \$USER"
echo "Watch load:   tail -f slurm_logs/new_exp_orch_*.out"
