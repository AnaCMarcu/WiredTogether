#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_transplant_2x2.sh — the memory x bond cells of the transplant study.
#
# Phase B of submit_transplant.sh runs cell A only: memories and bonds are
# both transplanted, and the Transplant/Shuffled contrast varies whether the
# memories are TRUE. That leaves the bond factor untested, so a reader can
# still say "you have shown agents remember who they met; why do I need the
# Hebbian graph?". These three arms close that:
#
#   cell   memory      bond        arm                      submitted by
#   A      retained    retained    expB_merged_transplant   submit_transplant.sh
#          fabricated  retained    expB_merged_shuffled     submit_transplant.sh
#   B      retained    reset       expB_memory_only         THIS SCRIPT
#   C      reset       retained    expB_bond_only           THIS SCRIPT
#   D      reset       reset       expB_neither             THIS SCRIPT
#
# "Bond reset" means the flat matrix written by `merge_pair_runs.py uniform-w`,
# whose mean off-diagonal equals the merged matrix's exactly. Bond MASS is
# held constant across all five arms; only its distribution changes. The
# plain Hebbian default (init_weight 0.1) is deliberately not used — it is
# both structureless and weaker, which would confound the two.
#
# "Memory reset" means fresh agents: no --agent-state-init at all.
#
# Everything else is pinned to cell A: 6 agents, --start-chamber 3, 3 episodes
# x 1000 steps, the reward_modulated rule at the exp08 settings,
# --social-module prompt, and seeds 42/123/456 so the cells pair seed-wise.
#
# THE RULE. No --hebbian-mode is passed, so these arms take the flag's default,
# reward_modulated (Variant B) -- exactly what cell A recorded in its
# config.json, and the rule behind 213 of the 225 Hebbian runs in the suite.
# Do NOT retarget these at --hebbian-mode three_factor: the cells only mean
# something against cell A, so changing the rule for B/C/D alone destroys the
# comparison. Moving the whole family to three_factor means re-running Phase A
# too, since Phase A is what produces the transplanted W.
# Beware: "legacy" is a SEPARATE mode in that flag's choices (legacy /
# coactivity / reward_modulated / three_factor) and is NOT this rule.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#
#   # 1. build the flat matrix from the existing transplant merge (once)
#   bash submit_transplant_2x2.sh uniform
#
#   # 2. smoke the three arms: 2 episodes x 150 steps, no wandb
#   SMOKE=1 bash submit_transplant_2x2.sh phaseB
#
#   # 3. the real thing: 3 arms x 3 seeds = 9 jobs, ~50-90h each
#   bash submit_transplant_2x2.sh phaseB
#
#   DRY_RUN=1 ... to print the sbatch calls without submitting.
#   ARMS="bond_only neither" ... to submit a subset.
#   PHASEB_SEEDS="123" ... to backfill a single seed.
# ────────────────────────────────────────────────────────────────────────────
set -u
# Resolve once, BEFORE the cd: a later $(dirname "$0") would be relative to
# the new cwd and silently miss gpu_filter.sh when invoked by path.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p slurm_logs

REPO=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether
PHASE="${1:-phaseB}"

export RUN_GROUP="${RUN_GROUP:-pair_bonding}"
export WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

MERGED_DIR="${MERGED_DIR:-$REPO/runs/$RUN_GROUP/merged/transplant}"
UNIFORM_DIR="${UNIFORM_DIR:-$REPO/runs/$RUN_GROUP/merged/uniform}"

case "$PHASE" in
    uniform)
        if [ ! -f "$MERGED_DIR/merged_W.json" ]; then
            echo "FATAL: no merged_W.json under $MERGED_DIR — run" >&2
            echo "       submit_transplant.sh phaseA and merge first." >&2
            exit 1
        fi
        echo "── Building the magnitude-matched flat W ──"
        PYTHONPATH="$REPO/src" python3 "$REPO/src/mindforge/tools/merge_pair_runs.py" \
            uniform-w --from-w "$MERGED_DIR/merged_W.json" --out-dir "$UNIFORM_DIR"
        echo
        echo "Now: SMOKE=1 bash submit_transplant_2x2.sh phaseB"
        exit 0
        ;;
    phaseB)
        ;;
    *)
        echo "usage: bash submit_transplant_2x2.sh {uniform|phaseB}" >&2
        exit 1
        ;;
esac

source "$HERE/gpu_filter.sh"
echo "GPU filter: --exclude=$GPU_EXCLUDE${GPU_CONSTRAINT:+ --constraint=$GPU_CONSTRAINT}"

# SMOKE=1 shrinks everything to a pipeline check. Enough to prove that the
# fresh-agent arms start without a manifest and that the flat W reaches the
# graph — without a real training run.
if [ "${SMOKE:-0}" = "1" ]; then
    export EPISODES="${EPISODES:-2}"
    export MAX_STEPS="${MAX_STEPS:-150}"
    export WANDB="${WANDB:-0}"
    PHASEB_SEEDS="${PHASEB_SEEDS:-42}"
    # Smoke output goes to its OWN run group. log.txt and llm_logs/*.log are
    # opened in append mode, so a 150-step smoke followed by a 1000-step run in
    # the same directory silently doubles the token counts and leaves stale
    # config.json values behind — that mixup has happened twice on this suite.
    # MERGED_DIR/UNIFORM_DIR were resolved above and still point at the real
    # merge outputs.
    export RUN_GROUP="${RUN_GROUP}_smoke"
    echo "SMOKE mode: EPISODES=$EPISODES MAX_STEPS=$MAX_STEPS WANDB=$WANDB seeds=$PHASEB_SEEDS"
    echo "SMOKE output group: runs/$RUN_GROUP (kept apart from the real runs)"
else
    export EPISODES="${EPISODES:-3}"
    export MAX_STEPS="${MAX_STEPS:-1000}"
    PHASEB_SEEDS="${PHASEB_SEEDS:-42 123 456}"
fi

ARMS="${ARMS:-memory_only bond_only neither}"

# Per-arm input requirements. memory_only needs both; bond_only needs only
# the learned matrix; neither needs only the flat one.
_needs_merged() { case "$1" in memory_only|bond_only) return 0 ;; *) return 1 ;; esac; }
_needs_uniform() { case "$1" in memory_only|neither) return 0 ;; *) return 1 ;; esac; }

for arm in $ARMS; do
    if [ ! -f "expB_${arm}.sbatch" ]; then
        echo "FATAL: unknown arm '$arm' (no expB_${arm}.sbatch)" >&2
        exit 1
    fi
    if _needs_merged "$arm" && [ ! -f "$MERGED_DIR/merged_W.json" ]; then
        echo "FATAL: arm '$arm' needs $MERGED_DIR/merged_W.json" >&2
        exit 1
    fi
    if _needs_merged "$arm" && [ "$arm" = "memory_only" ] \
            && [ ! -f "$MERGED_DIR/merged_manifest.json" ]; then
        echo "FATAL: arm 'memory_only' needs $MERGED_DIR/merged_manifest.json" >&2
        exit 1
    fi
    if _needs_uniform "$arm" && [ ! -f "$UNIFORM_DIR/uniform_W.json" ]; then
        echo "FATAL: arm '$arm' needs $UNIFORM_DIR/uniform_W.json —" >&2
        echo "       run 'bash submit_transplant_2x2.sh uniform' first." >&2
        exit 1
    fi
done

_submit() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "would queue: sbatch ${GPU_FILTER_FLAGS[*]} $*"
    else
        # shellcheck disable=SC2068
        sbatch ${GPU_FILTER_FLAGS[@]:+"${GPU_FILTER_FLAGS[@]}"} "$@"
    fi
}

echo "── memory x bond cells: arms [$ARMS], seeds [$PHASEB_SEEDS] ──"
echo "   MERGED_DIR=$MERGED_DIR"
echo "   UNIFORM_DIR=$UNIFORM_DIR"
for _seed in $PHASEB_SEEDS; do
    for arm in $ARMS; do
        # Subshell: a var-assignment prefix on a shell FUNCTION can persist in
        # bash, which would leak one arm's paths into the next.
        ( export SEED="$_seed" MERGED_DIR="$MERGED_DIR" UNIFORM_DIR="$UNIFORM_DIR"
          _submit "expB_${arm}.sbatch" )
    done
done

echo
echo "Verify once they start:"
echo "  grep -c '\[TRANSPLANT\] agent_' runs/$RUN_GROUP/expB_memory_only/seed_*/run.log   # expect 6"
echo "  grep -L '\[TRANSPLANT\]' runs/$RUN_GROUP/expB_bond_only/seed_*/run.log            # expect all (fresh agents)"
echo "  grep '\[HEBBIAN\]' runs/$RUN_GROUP/expB_neither/seed_*/run.log | head"
