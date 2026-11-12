#!/bin/bash
#SBATCH --job-name=aw-mf-chain
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/aw_mf_chain_%j.out
#SBATCH --error=slurm_logs/aw_mf_chain_%j.err
# ────────────────────────────────────────────────────────────────────────────
# One chained AgentWorld run with MindForge agents (chains/run_mf_chain.py).
# Submit through submit_aw_mf_chain.sh from the repository root (slurm_logs/ is
# relative), which always sets --time.
#
# Environment: CHAIN (default chains/pendant_staff_4.yaml), ARM (base | hebbian),
# SEED / array index, RUN_GROUP, MF_ARGS (more run_mf_chain.py flags, split on
# spaces, e.g. MF_ARGS="--hebbian-decay 0.005 --hold-complete"), PLUMBING=1 (no
# model, 4 rounds per step), plus mf_chain_common.sh's settings.
# Passed if the log ends with "python exit: 0"; the run's chain_summary.json
# then holds every step's verifier result.
# ────────────────────────────────────────────────────────────────────────────
source "${WT_REPO:-${WT_WORKSPACE:-$WORKSPACE}/WiredTogether}/hpc/agentworld/mf_chain_common.sh"
# Rounds padded to >= 20 s, as every earlier MindForge AgentWorld run (aw_smoke.sbatch's
# MIN_ROUND_S): the game runs in real time (trees regrow, mobs respawn).
EXTRA=(--min-round-seconds "${MIN_ROUND_S:-20}")
[ -n "${MF_ARGS:-}" ] && read -r -a _mf_args <<<"$MF_ARGS" && EXTRA+=("${_mf_args[@]}")
if [ -n "${PLUMBING:-}" ]; then
    EXTRA+=(--plumbing --rounds 4 --min-round-seconds 0)
    export LLM_SERVER=none
fi
run_mf_chain_aw "${CHAIN:-chains/pendant_staff_4.yaml}" ${EXTRA[@]+"${EXTRA[@]}"}
