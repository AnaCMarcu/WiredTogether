#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/quick_test-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/quick_test-%j.out

# ── quick_test.sh ────────────────────────────────────────────────────────────
# 1-hour smoke test to verify the warmup_noop ConnectionError
# (server-dies-1s-after-listening) is no longer happening.
#
# Minimal config: 1 episode × 50 steps, LLM-only, no RL/Hebbian.
# Expected wall-clock: ~5 min warmup + ~15-20 min stepping = under 30 min.
#
# Submit:
#   sbatch scripts/experiments/quick_test.sh
#   SEED=99 sbatch --export=ALL scripts/experiments/quick_test.sh
#
# Pass criteria — in the slurm .out you should see:
#   [FEATURES] Ch1 timeout: 50 env steps (Python primary)...   ← Lua-safety-factor fix live
#   * MT server is ready!
#   * Pre-listened on 3 channel sockets
#   * Starting client 0/1/2 ...
#   * All clients loaded (...s).
#   Episode 1/1 ... step=1/50 ...                              ← stepping started
#
# Fail symptom — same ConnectionError as before:
#   ConnectionError: Failed to receive from MT. Connection closed by peer
# Then check the corresponding minetest-srv-*/debug.txt for the
# "Server: Shutting down" line that's been killing the runs.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail
# Do NOT use -e: we want the pkill below to be best-effort, not fatal.

# Kill any stale minetest/luanti processes left over from previous failed runs
# on this same compute node. They were the leading hypothesis for the port-52800
# bind race. Safe: only affects this user's own processes.
echo "── pre-flight: clearing stale minetest/luanti processes (best-effort) ──"
pkill -9 -u "$USER" -f "minetest" 2>/dev/null || true
pkill -9 -u "$USER" -f "luanti"   2>/dev/null || true
sleep 5   # let kernel release ports / file handles before we start a fresh server
echo "── pre-flight done ──"

source "/scratch/${USER}/WiredTogether/scripts/experiments/_common.sh"

EXP_NAME="quick_test"
SEED="${SEED:-42}"
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

# Use the 2B model — fastest LLM inference so we get the most env steps per hour.
export LLM_MODEL_PATH="$MODEL_2B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 1 \
    --max-steps 50 \
    --warmup-time 300 \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    --simultaneous \
    2>&1 | tee "$RUN_DIR/run.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "── ${EXP_NAME} python exit code: $EXIT_CODE  (0 = success) ──"
exit "$EXIT_CODE"
