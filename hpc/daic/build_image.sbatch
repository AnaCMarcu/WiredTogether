#!/bin/bash
#SBATCH --job-name=wt-image-build
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=slurm_logs/build_%j.out
#SBATCH --error=slurm_logs/build_%j.err
#SBATCH --exclude=cor1

set -eu

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
DEF_FILE="$WORKSPACE/WiredTogether/hpc/daic/wiredtogether.def"
IMG_DIR="$WORKSPACE/images"
IMG_FILE="$IMG_DIR/wiredtogether.sif"
BUILD_TMP="/tmp/$USER/apptainer-build"

mkdir -p "$IMG_DIR" "$BUILD_TMP" "$WORKSPACE/models"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$BUILD_TMP/cache"

echo "[build] node:      $(hostname)"
echo "[build] def file:  $DEF_FILE"
echo "[build] output:    $IMG_FILE"
echo "[build] tmp:       $BUILD_TMP"
echo "[build] start:     $(date)"
echo

apptainer build --fakeroot --force "$IMG_FILE" "$DEF_FILE"

echo
echo "[build] end:       $(date)"
echo "[build] image size:"
du -sh "$IMG_FILE"
