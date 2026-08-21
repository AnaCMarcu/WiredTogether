#!/bin/bash
#SBATCH --job-name=wt-image-build-torch26
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
# Relative to the SUBMISSION dir, and resolved BEFORE the script body runs, so
# the directory must already exist. hpc/daic/experiments/slurm_logs/ is created
# by the submit_* launchers; the repo root has no slurm_logs/.
# => SUBMIT THIS FROM THE REPO ROOT.
#SBATCH --output=hpc/daic/experiments/slurm_logs/build_torch26_%j.out
#SBATCH --error=hpc/daic/experiments/slurm_logs/build_torch26_%j.err
#SBATCH --exclude=cor1

# Builds images/wiredtogether_gemma4_torch26.sif from
# wiredtogether_gemma4_torch26.def — the torch>=2.6 image that 26b/31b need
# (see that def's header for the failure it fixes).
#
# NEITHER existing image is touched: wiredtogether.sif keeps the Qwen runs
# reproducible and wiredtogether_gemma4.sif keeps the e2b/e4b/12b runs
# reproducible. _common.sh picks between all three via WT_IMAGE.
#
# --time is 3h rather than the 2h of build_image_gemma4.sbatch: the cu126 torch
# wheel is a larger download than cu121 and the build has one extra guard.

set -eu

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
DEF_FILE="$WORKSPACE/WiredTogether/hpc/daic/wiredtogether_gemma4_torch26.def"
IMG_DIR="$WORKSPACE/images"
IMG_FILE="$IMG_DIR/wiredtogether_gemma4_torch26.sif"
BUILD_TMP="/tmp/$USER/apptainer-build-torch26-${SLURM_JOB_ID:-nojob}"

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
echo "[build] stack check inside the finished image:"
# The def already fails the build on torch < 2.6; this re-states it in the job
# log so the version is visible without opening the image.
apptainer exec "$IMG_FILE" python -c \
    "import torch, transformers; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| transformers', transformers.__version__)"

echo
echo "[build] NEXT: verify it fixes the big checkpoints BEFORE launching runs —"
echo "  WT_IMAGE=$IMG_FILE MODEL=gemma-4-31B-it CONFIGS=sdpa_auto \\"
echo "      sbatch hpc/daic/probe_nan.sbatch"
echo "  want: 'logits finite : True' plus a non-empty greedy/sample line"
