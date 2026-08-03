#!/bin/bash
#SBATCH --job-name=wt-image-build-gemma4
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=slurm_logs/build_gemma4_%j.out
#SBATCH --error=slurm_logs/build_gemma4_%j.err
#SBATCH --exclude=cor1

# Builds images/wiredtogether_gemma4.sif from wiredtogether_gemma4.def.
# The original wiredtogether.sif is NOT touched — the Qwen runs stay
# reproducible, and _common.sh picks between them via WT_IMAGE.

set -eu

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
DEF_FILE="$WORKSPACE/WiredTogether/hpc/daic/wiredtogether_gemma4.def"
IMG_DIR="$WORKSPACE/images"
IMG_FILE="$IMG_DIR/wiredtogether_gemma4.sif"
BUILD_TMP="/tmp/$USER/apptainer-build-gemma4-${SLURM_JOB_ID:-nojob}"

mkdir -p "$IMG_DIR" "$BUILD_TMP" "$WORKSPACE/models"
export APPTAINER_TMPDIR="$BUILD_TMP"
export APPTAINER_CACHEDIR="$BUILD_TMP/cache"
export TMPDIR="$BUILD_TMP"

# DAIC requires every job to clean up what it writes to the node's /tmp — an
# apptainer build leaves a multi-GB cache + FUSE session files behind. Fire on
# any exit, including a SLURM cancel/timeout (SIGTERM).
trap "rm -rf '$BUILD_TMP' 2>/dev/null || true" EXIT INT TERM

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

echo
echo "[build] transformers / gemma4 check inside the finished image:"
apptainer exec "$IMG_FILE" python -c \
    "import transformers; print('transformers', transformers.__version__)"
