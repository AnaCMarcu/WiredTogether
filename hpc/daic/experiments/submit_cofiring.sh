#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_cofiring.sh — Experiment 2: co-firing without enforced communication.
#
# 6 arms × 3 seeds, 3 ep × 1000 steps (the medium protocol), RUN_GROUP=cofiring
# so nothing pools with final/medium2k/gemma4/legacy.
#
#   exp20_cofire_prc          menu comm            credit comm
#   exp21_cofire_pro          menu obs   (MUTE)    credit obs
#   exp22_cofire_pri          menu imit  (MUTE)    credit imit
#   exp23_cofire_prcoi        menu all             credit all  (Goal-1 primary:
#                             per-channel growth attribution within the run)
#   exp27_cofire_anchor       legacy (enforced comm) — reference + non-regression
#   exp28_cofire_null         menu none  (MUTE)    credit none     (pr floor)
#
# The credit-family arms (full menu, single credited channel) were REMOVED
# 2026-08-05 at the user's request — prcoi keeps ALL co-firing signals, and
# channel ranking comes from its within-run attribution (cofiring_events.jsonl
# + cofire_attribution), cross-checked against the affordance arms exp20-22.
#
# MODEL PIN (deliberate): the LLM backbone decision is DEFERRED, and
# _common.sh silently defaults MODEL_LLM to gemma-4-E4B-it — which is NOT the
# model behind the Experiment-1 numbers. This launcher therefore REFUSES to
# run unless MODEL_LLM (and WT_IMAGE) are exported explicitly:
#
#   # Gemma-2-9B (Experiment-1 comparable, validated image):
#   MODEL_LLM=$WORKSPACE/models/gemma-2-9b-it \
#   WT_IMAGE=$WORKSPACE/images/wiredtogether.sif \
#     bash submit_cofiring.sh
#
#   # Gemma-4-E4B (newest; Exp-2 self-contained via the exp27 anchor):
#   MODEL_LLM=$WORKSPACE/models/gemma-4-E4B-it \
#   WT_IMAGE=$WORKSPACE/images/wiredtogether_gemma4.sif \
#     bash submit_cofiring.sh
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   MODEL_LLM=... WT_IMAGE=... SMOKE=1 bash submit_cofiring.sh
#       # smoke = prcoi AND pri × seed 42, 1 ep × 150 steps, no wandb.
#       # GATES before the full sweep (check runs/cofiring_smoke/):
#       #   - act mix non-degenerate in prcoi social_acts.jsonl (no act >85%)
#       #   - c_comm/c_obs/c_imit all fire in prcoi cofiring_events.jsonl
#       #   - pri: imitation gate passes sometimes (gate="passed" rows exist,
#       #     imitation_abort_rate < 1.0) — else pri is dead on arrival
#       #   - tail -1 hebbian_snapshots.jsonl: off-diag W in 0.2-0.8
#   MODEL_LLM=... WT_IMAGE=... DRY_RUN=1 bash submit_cofiring.sh
#   MODEL_LLM=... WT_IMAGE=... bash submit_cofiring.sh
#
# Idempotent: an exp/seed whose final_metrics.json already exists is skipped.
# Pre-registration: seeds 42/123/456 are binding — dropping seeds post hoc is
# seed-shopping. Goal 2 is reported as effect sizes with CIs (n=9 episodes/arm
# is descriptive, no significance claim).
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

# ── Hard pre-flight: no silent model default ────────────────────────────────
if [ -z "${MODEL_LLM:-}" ] || [ -z "${WT_IMAGE:-}" ]; then
    echo "ERROR: MODEL_LLM and WT_IMAGE must be set explicitly — the backbone" >&2
    echo "       decision is deferred and no default may make it. See the"      >&2
    echo "       header of this script for the two candidate invocations."      >&2
    exit 1
fi
export MODEL_LLM WT_IMAGE

: "${RUN_GROUP:=cofiring}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=cofiring_wired_together}"
export RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

EXPS=(
    exp20_cofire_prc
    exp21_cofire_pro
    exp22_cofire_pri
    exp23_cofire_prcoi
    exp27_cofire_anchor
    exp28_cofire_null
)
SEEDS=(42 123 456)

# Smoke: THREE short jobs —
#   prcoi  : act mix + all-channel coverage (the model must actually choose)
#   pri    : the fragile replay-gate machinery (in prcoi it can dodge imitate)
#   anchor : legacy mode on the NEW code (live non-regression: no choice
#            sidecars, empty social_act_metrics, messages still flow)
# After they finish, validate ALL gates in one shot:
#   python scripts/check_cofiring_smoke.py --runs-root runs/cofiring_smoke
if [ "${SMOKE:-0}" = "1" ]; then
    EXPS=(exp23_cofire_prcoi exp22_cofire_pri exp27_cofire_anchor)
    SEEDS=(42)
    RUN_GROUP=cofiring_smoke
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
# Node exclusion. NOTE: a command-line --exclude REPLACES the sbatch files'
# baked-in "--exclude=cor1", so always include cor1 in the list, e.g.
#   EXCLUDE=cor1,influ2,influ3 bash submit_cofiring.sh
[ -n "${EXCLUDE:-}" ] && SBATCH_OVERRIDES+=(--exclude="$EXCLUDE")

echo "== submit_cofiring.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  exps      : ${#EXPS[@]} (${EXPS[*]})"
echo "  seeds     : ${SEEDS[*]}"
[ ${#SBATCH_OVERRIDES[@]} -gt 0 ] && echo "  overrides : ${SBATCH_OVERRIDES[*]}"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE (gates in header comment)"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "========================"

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
echo "Watch load:   tail -f slurm_logs/${EXPS[0]}_*.out"
echo "Cancel all:   scancel -u \$USER"
