#!/bin/bash
#SBATCH --job-name=aw-scale
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/aw_scale_%A_%a.out
#SBATCH --error=slurm_logs/aw_scale_%A_%a.err
# One scaling-sweep cell. Submit through submit_aw_scaling.sh (from the repo
# root), which sets ARM, REPLICAS and TASK; the seed comes from the array index.
# Optional: ROUNDS, EPISODES, TEAM_SPACING, VIDEO, MAX_IN_FLIGHT.
source "${WT_REPO:-$WORKSPACE/WiredTogether}/hpc/agentworld/aw_common.sh"
: "${ARM:?}" "${REPLICAS:?}" "${TASK:?}"
EXTRA=()
[ -n "${ROUNDS:-}" ] && EXTRA+=(--rounds "$ROUNDS")
run_aw "${ARM}_${TASK}_x${REPLICAS}" \
    --tasks "$TASK" --replicas "$REPLICAS" --team-spacing "${TEAM_SPACING:-20}" \
    --arm "$ARM" --episodes "${EPISODES:-3}" --video "${VIDEO:-team}" \
    --max-in-flight "${MAX_IN_FLIGHT:-256}" ${EXTRA[@]+"${EXTRA[@]}"}
