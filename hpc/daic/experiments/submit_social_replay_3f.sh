#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_social_replay_3f.sh — weight-gated experience sharing (Eq. 7) on the
# NEW Hebbian rule ("Hebbian 2.0": three_factor + signed death LTD).
#
# The 3f sibling of submit_social_replay.sh: same two lanes, same protocol
# (3 ep x 1000 steps), same rho=0.3 replay mixture — the update rule is the
# only thing that changes. Arms exp36 (MAPPO) / exp37 (IPPO) exist as their
# own sbatch files because exp30/exp31 do NOT expand ${EXTRA_ARGS}, so the
# rule cannot be injected into them from a launcher.
#
#   qwen    MODEL=Qwen3.5-2B, wiredtogether.sif, vision auto (= ON; the
#           model is a VL model and the medium suite ran vision=True)
#           → runs/social_replay_3f_qwen/
#   gemma4  MODEL=gemma-4-E4B-it, wiredtogether_gemma4.sif, vision ON
#           → runs/social_replay_3f_gemma4/
#
# Comparisons: each lane's rule-only baseline is the SAME lane of
# runs/social_replay_* (exp30/exp31). Never pool the two rules.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1   bash submit_social_replay_3f.sh   # 1 arm x 1 seed per lane, 150 steps
#   DRY_RUN=1 bash submit_social_replay_3f.sh   # print without submitting
#   bash submit_social_replay_3f.sh             # both lanes, both arms, 3 seeds
#   LANES=gemma4 bash submit_social_replay_3f.sh          # one lane
#   EXPS="exp36_mappo_hebbian_replay_3f" bash submit_social_replay_3f.sh
#   SEEDS="789 1011 1213" bash submit_social_replay_3f.sh # seed extension
#
# GPU/memory (inherited from the exp30/31 experience, do not lower):
#   Gemma 4 E4B needs a big card (GPU=gpu:a40:1) AND --mem=64GB AND
#   RL_UPDATE_STAGGER=1 — unstaggered Gemma RL runs hang the Minetest bridge
#   after their ~40-min update rounds while RSS climbs to OOM
#   (RLConfig.update_stagger). The Qwen lane stays unstaggered on purpose:
#   its completed exp30/exp31 seeds ran that way and this suite must match.
#
# Idempotent (final_metrics.json skip) + in-queue dedup (--job-name carries
# group/arm/seed): rerunning the SAME command is always safe.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

if [ -n "${EXPS:-}" ]; then
    EXPS=($EXPS)
else
    EXPS=(exp36_mappo_hebbian_replay_3f exp37_ippo_hebbian_replay_3f)
fi
SEEDS=(${SEEDS:-42 123 456})
LANES="${LANES:-qwen gemma4}"

: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=social_replay_3f}"

if [ "${SMOKE:-0}" = "1" ]; then
    EXPS=(exp36_mappo_hebbian_replay_3f)
    SEEDS=(42)
    EPISODES=1
    MAX_STEPS=${SMOKE_STEPS:-150}
    export WANDB=0
    : "${QOS:=short}"
    : "${TIME:=4:00:00}"
fi

QUEUED_NAMES=$(squeue -u "${USER:-$(whoami)}" -h -o %j 2>/dev/null || true)

n_queued=0
n_skipped=0
n_inqueue=0
n_failed=0
for lane in $LANES; do
    case "$lane" in
        qwen)
            LANE_MODEL="$WORKSPACE/models/Qwen3.5-2B"
            LANE_IMAGE="$WORKSPACE/images/wiredtogether.sif"
            LANE_GROUP="social_replay_3f_qwen"
            LANE_GPU="${GPU:-}"
            LANE_MEM="${MEM:-48GB}"      # Qwen RL seeds have OOM'd at 32GB
            LANE_STAGGER=0
            LANE_VISION="auto"           # VL model → sniff turns vision ON
            ;;
        gemma4)
            LANE_MODEL="$WORKSPACE/models/gemma-4-E4B-it"
            LANE_IMAGE="$WORKSPACE/images/wiredtogether_gemma4.sif"
            LANE_VISION="vision"
            LANE_GROUP="social_replay_3f_gemma4"
            LANE_GPU="${GPU:-gpu:a40:1}"
            LANE_MEM="${MEM:-64GB}"      # hang leaks RAM; 32GB died at 1.5 d
            LANE_STAGGER=1               # env-idle hang fix
            ;;
        *)
            echo "ERROR: unknown lane '$lane' (qwen|gemma4)" >&2; exit 1
            ;;
    esac
    [ "${SMOKE:-0}" = "1" ] && LANE_GROUP="${LANE_GROUP}_smoke"

    if [ ! -f "$LANE_IMAGE" ]; then
        echo "ERROR: image not found: $LANE_IMAGE — skipping lane $lane" >&2
        continue
    fi
    if [ ! -f "$LANE_MODEL/config.json" ]; then
        echo "ERROR: weights not staged: $LANE_MODEL — skipping lane $lane" >&2
        continue
    fi

    SBATCH_OVERRIDES=()
    [ -n "${QOS:-}" ]     && SBATCH_OVERRIDES+=(--qos="$QOS")
    [ -n "${TIME:-}" ]    && SBATCH_OVERRIDES+=(--time="$TIME")
    [ -n "$LANE_GPU" ]    && SBATCH_OVERRIDES+=(--gres="$LANE_GPU")
    [ -n "$LANE_MEM" ]    && SBATCH_OVERRIDES+=(--mem="$LANE_MEM")
    # A command-line --exclude REPLACES the sbatch files' baked-in
    # "--exclude=cor1", so always include cor1 in the list.
    [ -n "${EXCLUDE:-}" ] && SBATCH_OVERRIDES+=(--exclude="$EXCLUDE")

    echo "== lane $lane: model=$LANE_MODEL group=$LANE_GROUP vision=$LANE_VISION stagger=$LANE_STAGGER =="
    [ "${DRY_RUN:-0}" = "1" ] && echo "   (DRY_RUN — nothing will be submitted)"
    for exp in "${EXPS[@]}"; do
        if [ ! -f "$exp.sbatch" ]; then
            echo "ERROR: no such arm: $exp.sbatch" >&2
            exit 1
        fi
        for seed in "${SEEDS[@]}"; do
            if [ -f "$REPO/runs/$LANE_GROUP/$exp/seed_$seed/final_metrics.json" ]; then
                n_skipped=$((n_skipped + 1))
                continue
            fi
            jobname="${LANE_GROUP}-${exp}_s${seed}"
            if printf '%s\n' "$QUEUED_NAMES" | grep -Fqx "$jobname"; then
                echo "in queue     $exp  seed_$seed  ($lane)"
                n_inqueue=$((n_inqueue + 1))
                continue
            fi
            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "would queue  $exp  seed_$seed  ($lane)"
                n_queued=$((n_queued + 1))
            else
                jobid=$(SEED=$seed RUN_GROUP=$LANE_GROUP \
                    EPISODES=$EPISODES MAX_STEPS=$MAX_STEPS \
                    WANDB_PROJECT=$WANDB_PROJECT \
                    MODEL_LLM=$LANE_MODEL MODEL_2B=$LANE_MODEL \
                    WT_IMAGE=$LANE_IMAGE LLM_VISION_MODE=$LANE_VISION \
                    RL_UPDATE_STAGGER=$LANE_STAGGER \
                    sbatch --parsable --job-name="$jobname" \
                    ${SBATCH_OVERRIDES[@]:+"${SBATCH_OVERRIDES[@]}"} \
                    "$exp.sbatch")
                if [ -n "$jobid" ]; then
                    echo "queued  $exp  seed_$seed  ($lane)  →  job $jobid"
                    n_queued=$((n_queued + 1))
                else
                    # sbatch already printed the reason; the usual one is the
                    # per-user QOS submission cap — rerun this same command as
                    # the queue drains, dedup makes that safe.
                    echo "FAILED  $exp  seed_$seed  ($lane) — sbatch rejected the job" >&2
                    n_failed=$((n_failed + 1))
                    break 3
                fi
            fi
        done
    done
done
echo "── done: $n_queued submitted, $n_skipped already complete, $n_inqueue in queue, $n_failed failed ──"
[ "$n_failed" -gt 0 ] && exit 1
echo "Success signature in log.txt:  'social replay — N neighbour transitions'"
echo "Rule check:  grep -o '\"hebbian_mode\": \"[a-z_]*\"' runs/social_replay_3f_*/*/seed_*/config.json"
echo "Track with:  squeue -u \$USER"
