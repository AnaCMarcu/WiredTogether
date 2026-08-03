#!/bin/bash
#SBATCH --job-name=gemma4-download
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=slurm_logs/gemma4_download_%j.out
#SBATCH --error=slurm_logs/gemma4_download_%j.err
#SBATCH --exclude=cor1

# Stage Gemma 4 weights from a COMPUTE node.
#
# Why not the login node: hf_xet reconstructs chunks in parallel and, at the
# ~120 MB/s the transfer actually runs at, its memory use trips the login
# node's cgroup limit — the process is SIGKILLed part-way through ("Killed"
# after a few hundred MB). Compute nodes have outbound internet (the image
# build pip-installs and git-clones from one) and a real memory allocation.
#
#   sbatch hpc/daic/download_gemma4.sbatch
#   MODEL=google/gemma-4-12B-it sbatch hpc/daic/download_gemma4.sbatch
#
# Resumable: hf keeps .incomplete parts under <dest>/.cache/huggingface/, so
# re-submitting after a kill or timeout continues rather than restarting.
# Set HF_HUB_DISABLE_XET=1 to fall back to plain HTTP if xet misbehaves.

set -euo pipefail

WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu
IMG="${WT_IMAGE:-$WORKSPACE/images/wiredtogether_gemma4.sif}"

MODEL="${MODEL:-google/gemma-4-E4B-it}"
DEST="$WORKSPACE/models/${MODEL##*/}"

echo "== gemma4 download =="
echo "  node  : $(hostname)"
echo "  model : $MODEL"
echo "  dest  : $DEST"
echo "  image : $IMG"
echo "  start : $(date)"
echo "====================="

mkdir -p "$DEST"

apptainer exec \
    --bind /tudelft.net:/tudelft.net \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    ${HF_HUB_DISABLE_XET:+--env HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"} \
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
ls -1 "$DEST"

# Fail loudly here rather than at step 0 of a training run.
if [ ! -f "$DEST/config.json" ]; then
    echo "ERROR: $DEST/config.json missing — download incomplete." >&2
    exit 1
fi
if ! ls "$DEST"/*.safetensors >/dev/null 2>&1; then
    echo "ERROR: no .safetensors in $DEST — download incomplete." >&2
    exit 1
fi
# Either filename means the multimodal processor is present; Gemma 4 ships
# processor_config.json, the Qwen-VL generation used preprocessor_config.json.
if [ -f "$DEST/preprocessor_config.json" ] || [ -f "$DEST/processor_config.json" ]; then
    echo "OK: processor config present — the vision path can engage."
else
    echo "WARN: no processor config — this checkpoint would load TEXT-ONLY." >&2
fi

echo "  end   : $(date)"
