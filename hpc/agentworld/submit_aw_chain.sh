#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# Submit chained AgentWorld runs. Run from the repository root on the login node
# after `source hpc/snellius/env.sh` (puts the sbatch shim on PATH):
#
#   bash hpc/agentworld/submit_aw_chain.sh plumbing  # random agent, all steps, 4 rounds, no model
#   bash hpc/agentworld/submit_aw_chain.sh check     # step 1 with the model (hebbian arm)
#   bash hpc/agentworld/submit_aw_chain.sh pilot     # every arm, seed 42, full chain
#
#   CHAIN=chains/metallurgy_5.yaml bash hpc/agentworld/submit_aw_chain.sh pilot
#   ARRAY=0-2 ARMS="memory hebbian" bash hpc/agentworld/submit_aw_chain.sh pilot
#
# --time is always set (Snellius checks the budget against the requested
# limit at job start): defaults below, or TIME=HH:MM:SS. Estimates from
# AgentWorld's serial runner: a harvest or attack ~36 s, other actions ~0.1-1 s,
# plus the model call; 3 agents ~15-20 min and 5 agents ~60-75 min per failed
# 55-round task, plus vLLM start-up.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail
MODE="${1:-pilot}"
CHAIN="${CHAIN:-chains/metallurgy_4.yaml}"
ARMS="${ARMS:-reference memory hebbian}"
ARRAY="${ARRAY:-0}"                       # 0 = seed 42; 0-2 = 42, 123, 456
case "$CHAIN" in
    *metallurgy_5*) FULL=05:00:00; ONE=02:00:00 ;;
    *)              FULL=02:00:00; ONE=01:00:00 ;;
esac
JOB=hpc/agentworld/aw_chain.sbatch
[ -f "$JOB" ] || { echo "run from the repository root" >&2; exit 1; }
mkdir -p slurm_logs
submit() {   # <time> <exports>
    echo "sbatch --time=$1 --array=$ARRAY --export=ALL,$2 $JOB"
    sbatch --time="$1" --array="$ARRAY" --export="ALL,$2" "$JOB"
}
case "$MODE" in
    plumbing) submit "${TIME:-$ONE}" "CHAIN=$CHAIN,ARM=hebbian,RANDOM_AGENT=1,MAX_ROUNDS=4,RUN_GROUP=${RUN_GROUP:-chains_plumbing}" ;;
    check)    submit "${TIME:-$ONE}" "CHAIN=$CHAIN,ARM=hebbian,STEPS=1,RUN_GROUP=${RUN_GROUP:-chains_check}" ;;
    pilot)    for arm in $ARMS; do
                  submit "${TIME:-$FULL}" "CHAIN=$CHAIN,ARM=$arm,RUN_GROUP=${RUN_GROUP:-chains_pilot}"
              done ;;
    *) echo "usage: $0 plumbing|check|pilot" >&2; exit 2 ;;
esac
