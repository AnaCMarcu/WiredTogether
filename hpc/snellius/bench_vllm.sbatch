#!/bin/bash
#SBATCH --job-name=wt-bench-vllm
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/bench_vllm_%j.out
#SBATCH --error=slurm_logs/bench_vllm_%j.err

# In-process HF generate() vs a vLLM server on the same A100, same calls
# (see bench_vllm.py). Before submitting, on the login node:
#     bash hpc/snellius/setup.sh vllm          # the server image (no compute)
#     bash hpc/snellius/setup.sh vllm-check    # Gemma supported? (no compute)
# then, from the repository root with env.sh sourced:
#     sbatch hpc/snellius/bench_vllm.sbatch
# The comparison table is at the end of slurm_logs/bench_vllm_<jobid>.out;
# raw numbers and the server log go to runs/bench/vllm_<jobid>/.
#
# Ordered to fail cheaply: every file is checked in the first seconds, and the
# vLLM half (the part most likely to fail) runs before the HF baseline, so a
# broken server costs minutes, not the whole allocation.

set -uo pipefail
: "${WT_WORKSPACE:?source hpc/snellius/env.sh before submitting}"

REPO="$WT_WORKSPACE/WiredTogether"
WT_IMG="${WT_IMAGE:-$WT_WORKSPACE/images/wiredtogether_gemma4.sif}"
MODEL="${MODEL_LLM:-$WT_WORKSPACE/models/gemma-4-E4B-it}"
# setup.sh vllm builds a sandbox directory; a vllm.sif works too.
if [ -n "${VLLM_IMAGE:-}" ]; then VLLM_IMG="$VLLM_IMAGE"
elif [ -d "$WT_WORKSPACE/images/vllm" ]; then VLLM_IMG="$WT_WORKSPACE/images/vllm"
else VLLM_IMG="$WT_WORKSPACE/images/vllm.sif"; fi
OUT="$REPO/runs/bench/vllm_${SLURM_JOB_ID:-nojob}"
SCR="${TMPDIR:-/tmp/$USER}/bench_vllm_${SLURM_JOB_ID:-nojob}"
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))

# ── 0. preflight: seconds, before any GPU work ────────────────────────────
fail=0
for f in "$WT_IMG" "$MODEL/config.json" "$REPO/hpc/snellius/bench_vllm.py"; do
    [ -e "$f" ] || { echo "!! missing: $f"; fail=1; }
done
[ -e "$VLLM_IMG" ] || { echo "!! missing vLLM image: $VLLM_IMG (run setup.sh vllm)"; fail=1; }
command -v curl >/dev/null || { echo "!! curl not found on $(hostname)"; fail=1; }
[ "$fail" = 0 ] || { echo "!! preflight failed — nothing was run"; exit 1; }

mkdir -p "$OUT" "$SCR"
BINDS=(--bind "$WT_WORKSPACE:$WT_WORKSPACE" --bind "$SCR:$SCR")
for b in ${WT_BIND:+${WT_BIND//,/ }}; do BINDS+=(--bind "$b:$b"); done

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
    rm -rf "$SCR"
}
trap cleanup EXIT

echo "== bench_vllm  node=$(hostname)  model=$MODEL  port=$PORT"
echo "   vllm image: $VLLM_IMG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── 1. vLLM server ─────────────────────────────────────────────────────────
echo "── [1/3] starting vLLM (first start compiles; allow ~5-10 min)  $(date +%T)"
apptainer exec --nv "${BINDS[@]}" \
    --env HF_HUB_OFFLINE=1 --env HOME="$SCR" \
    --env XDG_CACHE_HOME="$SCR/cache" --env VLLM_CACHE_ROOT="$SCR/vllm_cache" \
    "$VLLM_IMG" vllm serve "$MODEL" \
        --served-model-name wt \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
    > "$OUT/vllm_server.log" 2>&1 &
SERVER_PID=$!

healthy=0
for _ in $(seq 1 90); do                       # up to 15 min
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then healthy=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 10
done
if [ "$healthy" != 1 ]; then
    echo "!! vLLM did not come up — stopping here (HF baseline skipped). Server log tail:"
    tail -40 "$OUT/vllm_server.log"
    exit 1
fi
echo "   vLLM healthy  $(date +%T)"

echo "── [2/3] vLLM client"
if ! apptainer exec "${BINDS[@]}" "$VLLM_IMG" \
        python3 -u "$REPO/hpc/snellius/bench_vllm.py" vllm \
        --url "http://127.0.0.1:$PORT" --json "$OUT/vllm.json"; then
    echo "!! vLLM client failed — server log tail:"
    tail -40 "$OUT/vllm_server.log"
    exit 1
fi
kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
sleep 5                                        # let the GPU memory drain

# ── 3. today's path: in-process transformers, batch 1 ─────────────────────
echo "── [3/3] HF in-process baseline  $(date +%T)"
apptainer exec --nv "${BINDS[@]}" \
    --env PYTHONPATH="$REPO/src" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env LLM_VISION_MODE="${LLM_VISION_MODE:-auto}" \
    "$WT_IMG" python -u "$REPO/hpc/snellius/bench_vllm.py" hf \
        --model "$MODEL" --json "$OUT/hf.json" \
    || { echo "!! HF baseline failed (vLLM numbers are saved in $OUT/vllm.json)"; exit 1; }

apptainer exec "${BINDS[@]}" "$WT_IMG" \
    python "$REPO/hpc/snellius/bench_vllm.py" compare "$OUT/hf.json" "$OUT/vllm.json" \
    | tee "$OUT/summary.txt"
echo "done  $(date +%T)"
