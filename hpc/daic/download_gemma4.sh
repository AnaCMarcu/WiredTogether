#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Stage Gemma 4 weights into the PRB workspace.
#
# PREFER hpc/daic/download_gemma4.sbatch FOR A FULL MODEL. Running this on the
# login node gets SIGKILLed part-way ("Killed" after a few hundred MB): hf_xet
# reconstructs chunks in parallel and at ~120 MB/s its memory use exceeds what
# a login shell is allowed. The sbatch does the same download on a compute
# node, which has internet and a real memory allocation. This script stays
# useful for small artefacts and for resuming by hand.
#
#   sbatch hpc/daic/download_gemma4.sbatch           # recommended
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
#
# `hf` vs `huggingface-cli`: recent huggingface_hub renamed the CLI, and the
# old name is now a hard error rather than an alias — which is why the first
# run of this script downloaded nothing. Pick whichever the image has.
#
# No --exclude: `hf download` takes optional positional FILENAMES after the
# repo id, and the exclude patterns landed there instead ("Ignoring --exclude
# since filenames have been explicitly set" → 0 files fetched). Nothing needs
# excluding anyway — the quantised GGUF/QAT weights live in separate repos
# (google/gemma-4-*-it-qat-*), not in this one.
apptainer exec \
    --bind /tudelft.net:/tudelft.net \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    "$IMG" \
    sh -c '
        if command -v hf >/dev/null 2>&1; then
            set -- hf download "$@"
        else
            set -- huggingface-cli download "$@"
        fi
        echo "[download] $*"
        exec "$@"
    ' sh "$MODEL" --local-dir "$DEST"

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

# Either filename means the multimodal processor is present; Gemma 4 ships
# processor_config.json, the Qwen-VL generation used preprocessor_config.json.
if [ -f "$DEST/preprocessor_config.json" ] || [ -f "$DEST/processor_config.json" ]; then
    echo "OK: processor config present — the vision path can engage."
else
    echo "WARN: no processor config — this checkpoint would load TEXT-ONLY." >&2
fi

echo
echo "Point runs at it with:"
echo "    MODEL_LLM=$DEST WT_IMAGE=$IMG sbatch hpc/daic/experiments/exp01_llm_2b.sbatch"
echo "(or just use hpc/daic/experiments/submit_gemma4.sh, which sets both)"
