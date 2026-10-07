#!/bin/bash
# E2 scaling sweep (UNTESTED): N = 3 × replicas of a 3-agent task.
#     WT_WORKSPACE=... TASK=task_02_arrow_production bash hpc/agentworld/submit_aw_scaling.sh
# DAIC login is password-only, so run this yourself on the login node.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TASK="${TASK:-task_02_arrow_production}"
for REPLICAS in ${REPLICAS_LIST:-1 4 7 10 17 34}; do      # N = 3 12 21 30 51 102
    for ARM in ${ARMS:-base hebbian shuffled prompt_only}; do
        sbatch --export=ALL,ARM="$ARM",REPLICAS="$REPLICAS",TASK="$TASK" \
               --job-name="aw_${ARM}_x${REPLICAS}" "$HERE/aw_scale.sbatch"
    done
done
