#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_social_replay.sh — weight-gated experience sharing (Eq. 7) arms.
#
# Submits exp30 (MAPPO+Heb+replay) and exp31 (IPPO+Heb+replay) on BOTH
# reasoning cores, as two separated lanes:
#
#   qwen    MODEL=Qwen3.5-2B, wiredtogether.sif, vision auto (= ON, as in
#           the medium suite: Qwen3.5-2B is a VL model)
#           → runs/social_replay_qwen/, poolable against the medium-suite
#             exp05/exp06 baselines (same model, same 3×1000 budget)
#   gemma4  MODEL=gemma-4-E4B-it, wiredtogether_gemma4.sif, vision ON
#           → runs/social_replay_gemma4/, poolable against the runs/gemma4/
#             exp05/exp06 baselines
#
# The replay arms differ from exp05/exp06 by exactly one flag
# (--hebbian-rho 0.3), so each lane's baseline is its suite's exp05/exp06.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#   SMOKE=1  bash submit_social_replay.sh   # 1 exp × 1 seed per lane, 150 steps
#   DRY_RUN=1 bash submit_social_replay.sh  # print without submitting
#   bash submit_social_replay.sh            # both lanes, both arms, 3 seeds
#   LANES=gemma4 bash submit_social_replay.sh   # one lane only (qwen|gemma4)
#   SEEDS="789 1011 1213" bash submit_social_replay.sh   # seed extension
#
# Idempotent: an exp/seed whose final_metrics.json already exists is skipped.
# GPU: Gemma 4 E4B needs a big card — GPU=gpu:a40:1 is the default there.
# Gemma lane also sets RL_UPDATE_STAGGER=1 and --mem=64GB: unstaggered Gemma
# RL runs hang the Minetest bridge after their ~40-min update rounds (see
# RLConfig.update_stagger). The Qwen lane stays unstaggered on purpose — its
# completed seeds ran that way and the extension seeds must match them.
# ────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
REPO="$WORKSPACE/WiredTogether"

EXPS=(exp30_mappo_hebbian_replay exp31_ippo_hebbian_replay)
SEEDS=(${SEEDS:-42 123 456})
LANES="${LANES:-qwen gemma4}"

: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=social_replay_wired_together}"

if [ "${SMOKE:-0}" = "1" ]; then
    EXPS=(exp30_mappo_hebbian_replay)
    SEEDS=(42)
    EPISODES=1
    MAX_STEPS=${SMOKE_STEPS:-150}
    export WANDB=0
    : "${QOS:=short}"
    : "${TIME:=4:00:00}"
fi

n_queued=0
n_skipped=0
for lane in $LANES; do
    case "$lane" in
        qwen)
            LANE_MODEL="$WORKSPACE/models/Qwen3.5-2B"
            LANE_IMAGE="$WORKSPACE/images/wiredtogether.sif"
            LANE_GROUP="social_replay_qwen"
            LANE_GPU="${GPU:-}"
            LANE_MEM="${MEM:-48GB}"      # Qwen RL seeds have OOM'd at 32GB before
            LANE_STAGGER=0
            # "auto" = the medium suite's setting: Qwen3.5-2B is a VL model and
            # the sniff turns vision ON (every exp03-06 medium run logs
            # "vision=True"). The first 6 replay seeds (2026-08-23/24) were
            # launched with "text" by mistake and are kept aside as
            # runs/social_replay_qwen_textonly — NOT comparable to the
            # baselines (no frames -> perception grounding collapses).
            LANE_VISION="auto"
            ;;
        gemma4)
            LANE_MODEL="$WORKSPACE/models/gemma-4-E4B-it"
            LANE_IMAGE="$WORKSPACE/images/wiredtogether_gemma4.sif"
            LANE_VISION="vision"
            LANE_GROUP="social_replay_gemma4"
            LANE_GPU="${GPU:-gpu:a40:1}"
            LANE_MEM="${MEM:-64GB}"      # hang leaks RAM; 32GB died at 1.5 d
            LANE_STAGGER=1               # env-idle hang fix (RLConfig.update_stagger)
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

    echo "== lane $lane: model=$LANE_MODEL group=$LANE_GROUP =="
    for exp in "${EXPS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if [ -f "$REPO/runs/$LANE_GROUP/$exp/seed_$seed/final_metrics.json" ]; then
                n_skipped=$((n_skipped + 1))
                continue
            fi
            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "would queue  $exp  seed_$seed  ($lane)"
            else
                jobid=$(SEED=$seed RUN_GROUP=$LANE_GROUP \
                    EPISODES=$EPISODES MAX_STEPS=$MAX_STEPS \
                    WANDB_PROJECT=$WANDB_PROJECT \
                    MODEL_LLM=$LANE_MODEL MODEL_2B=$LANE_MODEL \
                    WT_IMAGE=$LANE_IMAGE LLM_VISION_MODE=$LANE_VISION \
                    RL_UPDATE_STAGGER=$LANE_STAGGER \
                    sbatch --parsable \
                    ${SBATCH_OVERRIDES[@]:+"${SBATCH_OVERRIDES[@]}"} \
                    "$exp.sbatch")
                echo "queued  $exp  seed_$seed  ($lane)  →  job $jobid"
            fi
            n_queued=$((n_queued + 1))
        done
    done
done
echo "── done: $n_queued submitted, $n_skipped already complete ──"
echo "Success signature in log.txt:  'social replay — N neighbour transitions'"
echo "Track with:   squeue -u \$USER"
