#!/bin/bash
#SBATCH --job-name=hebb-tier11
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --output=hebb-tier11-%A_%a.out
#SBATCH --error=hebb-tier11-%A_%a.err

# ── DAIC port of hebbian-marl/scripts/slurm/hebb_tier11.sh ──
# Tier 11: same 3 variants × 5 seeds as tier 9 but with force_coop=True
# (Foraging-Comm-10x10-3p-3f-coop-v3). 15 indices.
#
# Submit one index for smoke test:
#   sbatch --array=0 hebbian-marl/daic/hebb_tier11.sbatch
#
# Submit full sweep:
#   sbatch --array=0-14 hebbian-marl/daic/hebb_tier11.sbatch

set -euo pipefail

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
IMG="$WORKSPACE/images/hebbian-marl.sif"
HEBB_REPO="$WORKSPACE/WiredTogether/hebbian-marl"
RESULTS_DIR="$WORKSPACE/WiredTogether/runs/hebbian-marl"

mkdir -p "$RESULTS_DIR/logs"

echo "== hebb-tier11 (DAIC) =="
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
        --tier 11 \
        --single-index "${SLURM_ARRAY_TASK_ID}" \
        --parallel 1

echo "hebb-tier11 task ${SLURM_ARRAY_TASK_ID} done"
