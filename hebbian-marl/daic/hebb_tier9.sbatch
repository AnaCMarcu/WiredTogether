#!/bin/bash
#SBATCH --job-name=hebb-tier9
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --output=hebb-tier9-%A_%a.out
#SBATCH --error=hebb-tier9-%A_%a.err

# ── DAIC port of hebbian-marl/scripts/slurm/hebb_tier9.sh ──
# Tier 9 sweep: 3 variants × 5 seeds (indices 0-14).
# Original wall: ~50-60 min per task on DelftBlue compute-p1.
#
# Submit one index for smoke test:
#   sbatch --array=0 hebbian-marl/daic/hebb_tier9.sbatch
#
# Submit full sweep:
#   sbatch --array=0-14 hebbian-marl/daic/hebb_tier9.sbatch

set -euo pipefail

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
IMG="$WORKSPACE/images/hebbian-marl.sif"
HEBB_REPO="$WORKSPACE/WiredTogether/hebbian-marl"
RESULTS_DIR="$WORKSPACE/WiredTogether/runs/hebbian-marl"

mkdir -p "$RESULTS_DIR/logs"

echo "== hebb-tier9 (DAIC) =="
echo "Host:       $(hostname)"
echo "Image:      $IMG"
echo "Repo:       $HEBB_REPO"
echo "ResultsDir: $RESULTS_DIR"
echo "JobID:      ${SLURM_JOB_ID:-N/A}"
echo "ArrayID:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "============================"

apptainer exec \
    --bind /tmp:/tmp \
    --bind /tudelft.net:/tudelft.net \
    --env HEBBIAN_RESULTS_DIR="$RESULTS_DIR" \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --pwd "$HEBB_REPO" \
    "$IMG" \
    python scripts/run_experiments.py \
        --tier 9 \
        --single-index "${SLURM_ARRAY_TASK_ID}" \
        --parallel 1

echo "hebb-tier9 task ${SLURM_ARRAY_TASK_ID} done"
