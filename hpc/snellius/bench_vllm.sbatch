#!/bin/bash
#SBATCH --job-name=wt-bench-vllm
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/bench_vllm_%j.out
#SBATCH --error=slurm_logs/bench_vllm_%j.err

# In-process HF generate() vs a vLLM server on the same A100, same calls
# (see bench_vllm.py). Needs images/vllm.sif (setup.sh vllm). Submit from the
# repository root after sourcing env.sh:
#     sbatch hpc/snellius/bench_vllm.sbatch
# The comparison table is at the end of slurm_logs/bench_vllm_<jobid>.out;
# raw numbers and the server log go to runs/bench/vllm_<jobid>/.

set -uo pipefail
: "${WT_WORKSPACE:?source hpc/snellius/env.sh before submitting}"

REPO="$WT_WORKSPACE/WiredTogether"
WT_IMG="${WT_IMAGE:-$WT_WORKSPACE/images/wiredtogether_gemma4.sif}"
VLLM_IMG="${VLLM_IMAGE:-$WT_WORKSPACE/images/vllm.sif}"
MODEL="${MODEL_LLM:-$WT_WORKSPACE/models/gemma-4-E4B-it}"
OUT="$REPO/runs/bench/vllm_${SLURM_JOB_ID:-nojob}"
SCR="${TMPDIR:-/tmp/$USER}/bench_vllm_${SLURM_JOB_ID:-nojob}"
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
mkdir -p "$OUT" "$SCR"

BINDS=(--bind "$WT_WORKSPACE:$WT_WORKSPACE" --bind "$SCR:$SCR")
for b in ${WT_BIND//,/ }; do BINDS+=(--bind "$b:$b"); done

echo "== bench_vllm  node=$(hostname)  model=$MODEL  port=$PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── 1. today's path: in-process transformers, batch 1 ─────────────────────
echo "── [1/3] HF in-process baseline"
apptainer exec --nv "${BINDS[@]}" \
    --env PYTHONPATH="$REPO/src" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env LLM_VISION_MODE="${LLM_VISION_MODE:-auto}" \
    "$WT_IMG" python -u "$REPO/hpc/snellius/bench_vllm.py" hf \
        --model "$MODEL" --json "$OUT/hf.json" \
    || { echo "!! HF baseline failed"; exit 1; }

# ── 2. vLLM server on the same GPU (the HF process has exited) ─────────────
echo "── [2/3] starting vLLM (first start compiles; allow ~5-10 min)"
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
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -rf "$SCR"' EXIT

for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "!! vLLM server exited during startup — last lines of its log:"
        tail -40 "$OUT/vllm_server.log"
        exit 1
    fi
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null \
    || { echo "!! vLLM not healthy after 20 min"; tail -40 "$OUT/vllm_server.log"; exit 1; }
grep -m1 -iE "vllm (api server )?version|Starting vLLM" "$OUT/vllm_server.log" || true

echo "── [3/3] vLLM client"
apptainer exec "${BINDS[@]}" "$VLLM_IMG" \
    python3 -u "$REPO/hpc/snellius/bench_vllm.py" vllm \
        --url "http://127.0.0.1:$PORT" --json "$OUT/vllm.json" \
    || { echo "!! vLLM client failed"; tail -40 "$OUT/vllm_server.log"; exit 1; }

apptainer exec "${BINDS[@]}" "$VLLM_IMG" \
    python3 "$REPO/hpc/snellius/bench_vllm.py" compare "$OUT/hf.json" "$OUT/vllm.json" \
    | tee "$OUT/summary.txt"
