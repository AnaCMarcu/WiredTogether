#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_cofiring_3f.sh — Experiment 2 (RQ2 co-firing) re-run on the NEW
# Hebbian rule (three_factor + signed death LTD, "Hebbian 2.0"), with the
# delivery-symmetric wiring of the cofiring_bidi suite.
#
# Relation to submit_cofiring.sh: that launcher's BIDI=1 branch OVERWRITES
# EXTRA_ARGS and narrows the sweep to the four choice arms, so it cannot also
# carry the new rule's flags. This is its 3f sibling — same arms, same
# protocol, one rule change — mirroring submit_pareto_social_3f.sh.
#
# ALL SEVEN arms (the full sweep, not BIDI's four):
#   exp20_cofire_prc      choice comm            credit comm
#   exp21_cofire_pro      choice obs             credit obs
#   exp22_cofire_pri      choice imit            credit imit
#   exp29_cofire_prco     choice comm,obs        credit comm,obs
#   exp23_cofire_prcoi    choice comm,obs,imit   credit all (primary)
#   exp27_cofire_anchor   LEGACY enforced comm   reference / non-regression
#   exp28_cofire_null     choice none            credit none (pr floor)
#
# What every arm gets on top of its own flags (appended via EXTRA_ARGS, which
# each exp*.sbatch already expands after its own arguments — argparse takes
# the LAST occurrence, so the eta-0/decay below deliberately override the
# 0.005/0.005 baked into the arm files):
#
#   delivery symmetry (as cofiring_bidi):
#     --social-act-rewards      obs/imit paid exactly like messages
#     --social-bidirectional    one obs/imit event credits BOTH W directions
#     --comm-distance-free      a message co-fires at any distance
#   Hebbian 2.0 (byte-identical to new_exp_0_gemma_si3f / exp34-35):
#     --hebbian-mode three_factor  --hebbian-eta-0 0.001  --hebbian-decay 0.001
#     --hebbian-eligibility-rho 0.9  --hebbian-coact-floor 0.25
#     --hebbian-death-ltd 0.05  --hebbian-death-cap 10
#     --hebbian-reward-norm 50  --hebbian-gamma 0.2  (already the arm values)
#
# ANCHOR EXCEPTION: exp27 runs in legacy act mode, and cli.py hard-refuses
# "--social-act-rewards requires --social-act-mode choice". That one arm
# therefore gets the bidirectional/distance-free + 3f flags WITHOUT
# --social-act-rewards. Nothing to pay there: legacy has no obs/imit acts.
#
# Lands in runs/cofiring_bidi_3f/ + wandb cofiring_bidi_3f_wired_together.
# NEVER pool with cofiring_bidi (different update rule) or cofiring_actrew
# (different wiring rule) — cofiring_bidi is the rule-only comparison.
#
# Model: defaults to the gemma-4-E4B-it pin that cofiring_bidi actually ran
# (verified from its log.txt), because that suite is this one's comparison
# target. Override MODEL_LLM/WT_IMAGE only if you re-run cofiring_bidi too.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1   bash submit_cofiring_3f.sh   # 3 arms x seed 42, 1 ep x 250
#   DRY_RUN=1 bash submit_cofiring_3f.sh   # print without submitting
#   bash submit_cofiring_3f.sh             # 7 arms x 3 seeds = 21 jobs
#
#   EXPS="exp23_cofire_prcoi" bash submit_cofiring_3f.sh    # subset
#   SEEDS="789 1011 1213"     bash submit_cofiring_3f.sh    # seed extension
#
# Idempotent (final_metrics.json skip) + in-queue dedup (--job-name carries
# group/arm/seed): rerunning the SAME command is always safe.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${RUN_GROUP:=cofiring_bidi_3f}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=cofiring_bidi_3f_wired_together}"
export MODEL_LLM WT_IMAGE RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT
# LLM_VISION_MODE is deliberately left unset: _common.sh defaults it to
# "auto", which is what cofiring_bidi ran (every seed logs vision=True on
# gemma-4-E4B). Setting it explicitly would diverge from that suite.

if [ -n "${EXPS:-}" ]; then
    EXPS=($EXPS)
else
    EXPS=(
        exp20_cofire_prc
        exp21_cofire_pro
        exp22_cofire_pri
        exp29_cofire_prco
        exp23_cofire_prcoi
        exp27_cofire_anchor
        exp28_cofire_null
    )
fi
SEEDS=(${SEEDS:-42 123 456})

BIDI_ARGS="--social-act-rewards --social-bidirectional --comm-distance-free"
BIDI_ARGS_LEGACY="--social-bidirectional --comm-distance-free"
HEB2_ARGS="--hebbian-mode three_factor --hebbian-eta-0 0.001 --hebbian-reward-norm 50 --hebbian-decay 0.001 --hebbian-eligibility-rho 0.9 --hebbian-coact-floor 0.25 --hebbian-death-ltd 0.05 --hebbian-death-cap 10 --hebbian-gamma 0.2"

# exp27 is the only legacy-act-mode arm — see ANCHOR EXCEPTION above.
extra_args_for() {
    case "$1" in
        exp27_cofire_anchor) echo "$BIDI_ARGS_LEGACY $HEB2_ARGS" ;;
        *)                   echo "$BIDI_ARGS $HEB2_ARGS" ;;
    esac
}

# Smoke: prcoi (act mix + all three channels fire), pri (imitation actually
# gets adopted) and anchor (legacy purity on the new rule). Validate with
#   python hpc/diagnostics/check_cofiring_smoke.py --runs-root runs/cofiring_bidi_3f_smoke
if [ "${SMOKE:-0}" = "1" ]; then
    EXPS=(exp23_cofire_prcoi exp22_cofire_pri exp27_cofire_anchor)
    SEEDS=(42)
    RUN_GROUP=cofiring_bidi_3f_smoke
    EPISODES=1
    MAX_STEPS=${SMOKE_STEPS:-250}
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
# A command-line --exclude REPLACES the sbatch files' baked-in
# "--exclude=cor1", so always include cor1 in the list.
[ -n "${EXCLUDE:-}" ] && SBATCH_OVERRIDES+=(--exclude="$EXCLUDE")

echo "== submit_cofiring_3f.sh =="
echo "  model     : $MODEL_LLM"
echo "  image     : $WT_IMAGE"
echo "  run_group : $RUN_GROUP"
echo "  episodes  : $EPISODES"
echo "  max_steps : $MAX_STEPS"
echo "  wandb     : ${WANDB:-1} (project=$WANDB_PROJECT)"
echo "  rule      : Hebbian 2.0 (three_factor + death LTD) + bidi/act-rewards"
echo "  exps      : ${#EXPS[@]} (${EXPS[*]})"
echo "  seeds     : ${SEEDS[*]}"
[ ${#SBATCH_OVERRIDES[@]} -gt 0 ] && echo "  overrides : ${SBATCH_OVERRIDES[*]}"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "==========================="

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

QUEUED_NAMES=$(squeue -u "${USER:-$(whoami)}" -h -o %j 2>/dev/null || true)

n_queued=0
n_skipped=0
n_inqueue=0
n_failed=0
for exp in "${EXPS[@]}"; do
    if [ ! -f "$exp.sbatch" ]; then
        echo "ERROR: no such arm: $exp.sbatch" >&2
        exit 1
    fi
    arm_args="$(extra_args_for "$exp")"
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
        if printf '%s\n' "$QUEUED_NAMES" | grep -Fqx "${exp//_/-}"; then
            echo "in queue*    $exp  seed_$seed  (legacy-named job, seed unknown — arm blocked until it drains)"
            n_inqueue=$((n_inqueue + 1))
            continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "would queue  $exp  seed_$seed"
            n_queued=$((n_queued + 1))
        else
            jobid=$(SEED=$seed EXTRA_ARGS="$arm_args" \
                sbatch --parsable --job-name="$jobname" \
                ${SBATCH_OVERRIDES[@]:+"${SBATCH_OVERRIDES[@]}"} \
                "$exp.sbatch")
            if [ -n "$jobid" ]; then
                echo "queued  $exp  seed_$seed  →  job $jobid"
                n_queued=$((n_queued + 1))
            else
                # sbatch already printed the reason. The usual one is the
                # per-user QOS submission cap — nothing is wrong, just rerun
                # this same command as the queue drains; dedup makes it safe.
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
echo "Rule check:   grep -o '\"hebbian_mode\": \"[a-z_]*\"' runs/$RUN_GROUP/*/seed_*/config.json"
