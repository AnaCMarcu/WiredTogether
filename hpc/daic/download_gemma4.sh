#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Stage Gemma 4 weights into the PRB workspace.
#
# RUN THIS ON THE LOGIN NODE — compute nodes run with HF_HUB_OFFLINE=1
# (see _common.sh), so nothing downloads from inside a job.
#
#   bash hpc/daic/download_gemma4.sh                 # default: gemma-4-E4B-it
#   MODEL=google/gemma-4-12B-it bash hpc/daic/download_gemma4.sh
#
# Gemma 4 is Apache-2.0, but the HF repo may still ask you to accept terms
# once in the browser. If the download 401/403s, log in first:
#     huggingface-cli login          (or: export HF_TOKEN=hf_...)
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
IMG="${WT_IMAGE:-$WORKSPACE/images/wiredtogether_gemma4.sif}"

MODEL="${MODEL:-google/gemma-4-E4B-it}"
DEST="$WORKSPACE/models/${MODEL##*/}"

echo "== download_gemma4.sh =="
echo "  model : $MODEL"
echo "  dest  : $DEST"
echo "  image : $IMG"
echo "========================"

if [ ! -f "$IMG" ]; then
    echo "ERROR: $IMG not found — build it first:" >&2
    echo "       sbatch hpc/daic/build_image_gemma4.sbatch" >&2
    exit 1
fi

mkdir -p "$DEST"

# --local-dir gives a plain directory of weights (no blob/symlink cache
# layout), which is what LLM_MODEL_PATH expects and what survives an
# HF_HUB_OFFLINE=1 job. Resumable: re-run after an interrupted transfer.
# The GGUF/QAT sibling files are quantised variants we don't use.
apptainer exec \
    --bind /tudelft.net:/tudelft.net \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    "$IMG" \
    huggingface-cli download "$MODEL" \
        --local-dir "$DEST" \
        --exclude "*.gguf" "*.pth" "original/*"

echo
echo "== staged files =="
du -sh "$DEST"
ls -1 "$DEST" | head -20

# Fail loudly here rather than at step 0 of a 168 h run.
for f in config.json; do
    if [ ! -f "$DEST/$f" ]; then
        echo "ERROR: $DEST/$f missing — download incomplete." >&2
        exit 1
    fi
done

if [ -f "$DEST/preprocessor_config.json" ]; then
    echo "OK: preprocessor_config.json present — the vision path will engage."
else
    echo "WARN: no preprocessor_config.json — this checkpoint will load TEXT-ONLY." >&2
fi

echo
echo "Point runs at it with:"
echo "    MODEL_LLM=$DEST WT_IMAGE=$IMG sbatch hpc/daic/experiments/exp01_llm_2b.sbatch"
echo "(or just use hpc/daic/experiments/submit_gemma4.sh, which sets both)"
