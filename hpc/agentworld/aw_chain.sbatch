#!/bin/bash
#SBATCH --job-name=aw-chain
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/aw_chain_%j.out
#SBATCH --error=slurm_logs/aw_chain_%j.err
# ────────────────────────────────────────────────────────────────────────────
# One chained AgentWorld run (chains/run_chain.py). Submit through
# submit_aw_chain.sh from the repository root (slurm_logs/ is relative), which
# sets --time per chain; the header's 2 h fits metallurgy_4.
#
# Environment: CHAIN (default chains/metallurgy_4.yaml), ARM, SEED / array index,
# RUN_GROUP, STEPS, MAX_ROUNDS, RANDOM_AGENT=1, CHAIN_ARGS (more run_chain.py
# flags, split on spaces, e.g. CHAIN_ARGS="--hebbian-decay 0.005"), plus
# chain_common.sh's settings.
# Passed if the log ends with "python exit: 0"; the run's chain_summary.json
# then holds every step's verifier result and the four checks.
# ────────────────────────────────────────────────────────────────────────────
source "${WT_REPO:-${WT_WORKSPACE:-$WORKSPACE}/WiredTogether}/hpc/agentworld/chain_common.sh"
EXTRA=()
[ -n "${STEPS:-}" ] && EXTRA+=(--steps "$STEPS")
[ -n "${MAX_ROUNDS:-}" ] && EXTRA+=(--max-rounds "$MAX_ROUNDS")
[ -n "${CHAIN_ARGS:-}" ] && read -r -a _chain_args <<<"$CHAIN_ARGS" && EXTRA+=("${_chain_args[@]}")
if [ -n "${RANDOM_AGENT:-}" ]; then
    EXTRA+=(--random-agent)
    export LLM_SERVER=none AGENT_CONFIG="${AGENT_CONFIG:-chains/configs/random_plumbing.yaml}"
fi
run_chain_aw "${CHAIN:-chains/metallurgy_4.yaml}" ${EXTRA[@]+"${EXTRA[@]}"}
