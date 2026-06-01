#!/bin/bash
#SBATCH --job-name=hebb-image-build
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --output=slurm_logs/hebb_build_%j.out
#SBATCH --error=slurm_logs/hebb_build_%j.err

set -eu

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
DEF_FILE="$WORKSPACE/WiredTogether/hebbian-marl/daic/hebbian-marl.def"
IMG_DIR="$WORKSPACE/images"
IMG_FILE="$IMG_DIR/hebbian-marl.sif"
BUILD_TMP="/tmp/$USER/apptainer-build-hebb"

mkdir -p "$IMG_DIR" "$BUILD_TMP"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$BUILD_TMP/cache"

echo "[build] node:     $(hostname)"
echo "[build] def file: $DEF_FILE"
echo "[build] output:   $IMG_FILE"
echo "[build] start:    $(date)"
echo

apptainer build --fakeroot "$IMG_FILE" "$DEF_FILE"

echo
echo "[build] end:        $(date)"
echo "[build] image size:"
du -sh "$IMG_FILE"
