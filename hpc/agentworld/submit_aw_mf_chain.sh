#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Submit chained AgentWorld runs with MindForge agents. Run from the repository
# root on the login node after `source hpc/snellius/env.sh`:
#
#   bash hpc/agentworld/submit_aw_mf_chain.sh plumbing  # no model, every step, 4 rounds each
#   bash hpc/agentworld/submit_aw_mf_chain.sh check     # hebbian arm, seed 42, whole chain
#   bash hpc/agentworld/submit_aw_mf_chain.sh pilot     # every arm (base, hebbian) per array index
#
#   ARRAY=0-2 REPEAT=2 MF_ARGS="--hebbian-decay 0.005 --hold-complete" \
#       RUN_GROUP=mf_chains_ps4 bash hpc/agentworld/submit_aw_mf_chain.sh pilot
#
# REPEAT submits the same array that many times (independent runs of each seed;
# every run gets its own folder, named by job id). --time is always set (Snellius
# checks the budget against it): 03:00:00 for a chain, or TIME=HH:MM:SS.
# Pace: 55 rounds padded to >= 20 s each is ~20 min per unsolved step, so a
# 4-step chain is <= ~1.5 h plus vLLM start-up and one server restart per step.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
MODE="${1:-pilot}"
CHAIN="${CHAIN:-chains/pendant_staff_4.yaml}"
ARMS="${ARMS:-base hebbian}"
ARRAY="${ARRAY:-0}"                       # 0 = seed 42; 0-2 = 42, 123, 456
REPEAT="${REPEAT:-1}"
MF_ARGS="${MF_ARGS:-}"
JOB=hpc/agentworld/aw_mf_chain.sbatch
[ -f "$JOB" ] || { echo "run from the repository root" >&2; exit 1; }
mkdir -p slurm_logs
submit() {   # <time> <array> <exports>
    echo "sbatch --time=$1 --array=$2 --export=ALL,$3 $JOB   [MF_ARGS=$MF_ARGS]"
    MF_ARGS="$MF_ARGS" sbatch --time="$1" --array="$2" --export="ALL,$3" "$JOB"
}
case "$MODE" in
    plumbing) submit "${TIME:-00:45:00}" 0 "CHAIN=$CHAIN,ARM=hebbian,PLUMBING=1,RUN_GROUP=${RUN_GROUP:-mf_chains_plumbing}" ;;
    check)    submit "${TIME:-03:00:00}" 0 "CHAIN=$CHAIN,ARM=hebbian,RUN_GROUP=${RUN_GROUP:-mf_chains_check}" ;;
    pilot)    for _r in $(seq 1 "$REPEAT"); do
                  for arm in $ARMS; do
                      submit "${TIME:-03:00:00}" "$ARRAY" "CHAIN=$CHAIN,ARM=$arm,RUN_GROUP=${RUN_GROUP:-mf_chains}"
                  done
              done ;;
    *) echo "usage: $0 plumbing|check|pilot" >&2; exit 2 ;;
esac
