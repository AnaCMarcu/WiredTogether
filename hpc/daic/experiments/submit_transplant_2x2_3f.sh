#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# submit_transplant_2x2_3f.sh — the memory x bond cells on the THREE-FACTOR
# rule. Three-factor twin of submit_transplant_2x2.sh.
#
#   cell   memory      bond        arm (all _3f)
#   A      retained    retained    expB_merged_transplant_3f
#          fabricated  retained    expB_merged_shuffled_3f   (A's truth control)
#   B      retained    reset       expB_memory_only_3f
#   C      reset       retained    expB_bond_only_3f
#   D      reset       reset       expB_neither_3f
#
# WHY ALL FIVE, not just B/C/D. The design is a set of contrasts — A-B, A-C,
# C-D, B-D. A cell is only interpretable against the others, so a three_factor
# B compared against a reward_modulated A would measure the RULE, not the
# factor under test. Cell A therefore moves too. 5 arms x 3 seeds = 15 jobs.
#
# WHY PHASE A IS NOT RE-RUN. The transplanted W and memories still come from
# the reward_modulated pair runs under runs/pair_bonding/merged/. Phase A is
# the relationship-FORMATION stage; this suite asks how the new rule treats a
# relationship that already exists. Deliberate, and worth stating in the paper:
# the bonds handed to Phase B were grown under the old rule.
#
# The rule (matches submit_cofiring_3f.sh and exp36):
#   --hebbian-mode three_factor --hebbian-eta-0 0.001 --hebbian-decay 0.001
#   --hebbian-eligibility-rho 0.9 --hebbian-coact-floor 0.25
#   --hebbian-death-ltd 0.05 --hebbian-death-cap 10
#
# Output group is pair_bonding_3f, so nothing mixes with the reward_modulated
# cells in pair_bonding/. Inputs are read from pair_bonding/merged/ either way.
#
# Usage (from the DAIC login node):
#   cd $REPO/hpc/daic/experiments
#
#   # the flat matrix, if submit_transplant_2x2.sh uniform has not built it
#   bash submit_transplant_2x2_3f.sh uniform
#
#   # smoke: one seed, all five arms, 2 eps x 150 steps, no wandb
#   SMOKE=1 bash submit_transplant_2x2_3f.sh phaseB
#
#   # the real suite: 15 jobs
#   bash submit_transplant_2x2_3f.sh phaseB
#
#   DRY_RUN=1 ...                to print the sbatch calls without submitting
#   ARMS="bond_only neither" ... to submit a subset
#   PHASEB_SEEDS="123" ...       to backfill a single seed
# ────────────────────────────────────────────────────────────────────────────
set -u
# Resolve once, BEFORE the cd: a later $(dirname "$0") would be relative to
# the new cwd and silently miss gpu_filter.sh when invoked by path.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p slurm_logs

REPO=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether
PHASE="${1:-phaseB}"

export RUN_GROUP="${RUN_GROUP:-pair_bonding_3f}"
export WANDB_PROJECT="${WANDB_PROJECT:-transplant_wired_together}"

# Phase A inputs live in the ORIGINAL group and are shared with the
# reward_modulated cells. Keep this separate from RUN_GROUP, which only names
# where output goes.
PHASEA_GROUP="${PHASEA_GROUP:-pair_bonding}"
MERGED_ROOT="${MERGED_ROOT:-$REPO/runs/$PHASEA_GROUP/merged}"

case "$PHASE" in
    uniform)
        if [ ! -f "$MERGED_ROOT/transplant/merged_W.json" ]; then
            echo "FATAL: no merged_W.json under $MERGED_ROOT/transplant" >&2
            exit 1
        fi
        echo "── Building the magnitude-matched flat W ──"
        PYTHONPATH="$REPO/src" python3 "$REPO/src/mindforge/tools/merge_pair_runs.py" \
            uniform-w --from-w "$MERGED_ROOT/transplant/merged_W.json" \
            --out-dir "$MERGED_ROOT/uniform"
        echo
        echo "Now: SMOKE=1 bash submit_transplant_2x2_3f.sh phaseB"
        exit 0
        ;;
    phaseB)
        ;;
    *)
        echo "usage: bash submit_transplant_2x2_3f.sh {uniform|phaseB}" >&2
        exit 1
        ;;
esac

source "$HERE/gpu_filter.sh"
echo "GPU filter: --exclude=$GPU_EXCLUDE${GPU_CONSTRAINT:+ --constraint=$GPU_CONSTRAINT}"

if [ "${SMOKE:-0}" = "1" ]; then
    export EPISODES="${EPISODES:-2}"
    export MAX_STEPS="${MAX_STEPS:-150}"
    export WANDB="${WANDB:-0}"
    PHASEB_SEEDS="${PHASEB_SEEDS:-42}"
    # Smoke output goes to its OWN group: log.txt and llm_logs/*.log open in
    # append mode, so a 150-step smoke followed by a 1000-step run in the same
    # directory silently doubles the token counts.
    export RUN_GROUP="${RUN_GROUP}_smoke"
    echo "SMOKE mode: EPISODES=$EPISODES MAX_STEPS=$MAX_STEPS WANDB=$WANDB seeds=$PHASEB_SEEDS"
    echo "SMOKE output group: runs/$RUN_GROUP"
else
    export EPISODES="${EPISODES:-3}"
    export MAX_STEPS="${MAX_STEPS:-1000}"
    PHASEB_SEEDS="${PHASEB_SEEDS:-42 123 456}"
fi

ARMS="${ARMS:-merged_transplant merged_shuffled memory_only bond_only neither}"

# Which merge subdirectory each arm reads. merged_shuffled is the one arm that
# takes the shuffled seating rather than the transplant one.
_merged_dir_for() {
    case "$1" in
        merged_shuffled) echo "$MERGED_ROOT/shuffled" ;;
        *)               echo "$MERGED_ROOT/transplant" ;;
    esac
}
_needs_manifest() {
    case "$1" in merged_transplant|merged_shuffled|memory_only) return 0 ;;
                 *) return 1 ;; esac
}
_needs_merged_w() {
    case "$1" in merged_transplant|merged_shuffled|bond_only) return 0 ;;
                 *) return 1 ;; esac
}
_needs_uniform() {
    case "$1" in memory_only|neither) return 0 ;; *) return 1 ;; esac
}

for arm in $ARMS; do
    if [ ! -f "expB_${arm}_3f.sbatch" ]; then
        echo "FATAL: unknown arm '$arm' (no expB_${arm}_3f.sbatch)" >&2
        exit 1
    fi
    d="$(_merged_dir_for "$arm")"
    if _needs_merged_w "$arm" && [ ! -f "$d/merged_W.json" ]; then
        echo "FATAL: arm '$arm' needs $d/merged_W.json" >&2
        exit 1
    fi
    if _needs_manifest "$arm" && [ ! -f "$d/merged_manifest.json" ]; then
        echo "FATAL: arm '$arm' needs $d/merged_manifest.json" >&2
        exit 1
    fi
    if _needs_uniform "$arm" && [ ! -f "$MERGED_ROOT/uniform/uniform_W.json" ]; then
        echo "FATAL: arm '$arm' needs $MERGED_ROOT/uniform/uniform_W.json —" >&2
        echo "       run 'bash submit_transplant_2x2_3f.sh uniform' first." >&2
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

echo "── three-factor memory x bond: arms [$ARMS], seeds [$PHASEB_SEEDS] ──"
echo "   inputs from $MERGED_ROOT (reward_modulated Phase A, deliberately)"
echo "   output to   runs/$RUN_GROUP"
for _seed in $PHASEB_SEEDS; do
    for arm in $ARMS; do
        # Subshell: a var-assignment prefix on a shell FUNCTION can persist in
        # bash, which would leak one arm's paths into the next.
        ( export SEED="$_seed" \
                 MERGED_DIR="$(_merged_dir_for "$arm")" \
                 UNIFORM_DIR="$MERGED_ROOT/uniform"
          _submit "expB_${arm}_3f.sbatch" )
    done
done

echo
echo "Verify once they start:"
echo "  grep -o '\"hebbian_mode\": \"[a-z_]*\"' runs/$RUN_GROUP/*/seed_*/config.json | sort -u"
echo "  # expect three_factor for every run"
