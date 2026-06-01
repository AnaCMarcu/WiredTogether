#!/bin/bash
#SBATCH --job-name=wt-smoke
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/smoke_%j.out
#SBATCH --error=slurm_logs/smoke_%j.err

set -eu

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
IMG="$WORKSPACE/images/wiredtogether.sif"
SCRIPT="$WORKSPACE/WiredTogether/daic/smoke_test.py"

echo "[smoke] node:   $(hostname)"
echo "[smoke] image:  $IMG"
echo "[smoke] script: $SCRIPT"
echo

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    "$IMG" \
    python "$SCRIPT"
