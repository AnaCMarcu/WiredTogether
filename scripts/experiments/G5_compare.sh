#!/bin/bash
#SBATCH --job-name=G5-compare
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/G5-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/G5-%j.err
# Submit: sbatch G5_compare.sh  (CPU-only, ~5 min runtime)
#
# Reads grpo_metrics.jsonl from G1-G4 runs and produces the headline
# ablation figure + summary JSON. Re-runnable: it just reads existing
# files, doesn't touch the GPU.
#
# Usage:
#   sbatch G5_compare.sh                       # default SEED=42 dirs
#   SEED=123 sbatch G5_compare.sh              # different seed
#   SEEDS_TO_AGGREGATE="42 123 456" sbatch G5_compare.sh   # multi-seed (TODO)
#
# Outputs to /scratch/$USER/WiredTogether/reports/grpo_ablation/

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

SEED="${SEED:-42}"
RUNS_ROOT="/scratch/${USER}/WiredTogether/runs/grpo"
REPORT_DIR="/scratch/${USER}/WiredTogether/reports/grpo_ablation/seed_${SEED}"
mkdir -p "$REPORT_DIR"

# Collect metrics paths — only include experiments that actually produced a JSONL.
METRIC_PATHS=()
LABELS=()
for tag in G2 G2b G3a G3b G4; do
    metrics="$RUNS_ROOT/${tag}/seed_${SEED}/grpo_metrics.jsonl"
    if [ -f "$metrics" ]; then
        METRIC_PATHS+=("$metrics")
        LABELS+=("$tag")
    else
        echo "WARN: missing $metrics — skipping"
    fi
done

if [ "${#METRIC_PATHS[@]}" -eq 0 ]; then
    echo "ERROR: no GRPO metrics files found under $RUNS_ROOT for seed=$SEED" >&2
    echo "       Did G1..G4 finish?" >&2
    exit 1
fi

echo "== G5 compare =="
echo "Seed:    $SEED"
echo "Inputs:  ${METRIC_PATHS[*]}"
echo "Labels:  ${LABELS[*]}"
echo "Output:  $REPORT_DIR"
echo "================"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

python scripts/compare_modes.py \
    --grpo-metrics "${METRIC_PATHS[@]}" \
    --labels "${LABELS[@]}" \
    --output-dir "$REPORT_DIR" \
    --window 20 \
    --final-window 50

echo "G5 done — figures in $REPORT_DIR"
