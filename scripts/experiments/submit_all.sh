#!/bin/bash
# Submit all 6 experiments to SLURM in parallel.
#
# Usage (from anywhere on DelftBlue):
#     bash /scratch/$USER/WiredTogether/scripts/experiments/submit_all.sh
#
# Each experiment runs on its own GPU node, single seed (42). Results land in:
#     /scratch/$USER/WiredTogether/runs/legacy/<exp_name>/seed_42/
# bundled together: log.txt, final_metrics.json, episodes/, gifs/, run.log.
#
# The SLURM .out file for each experiment also lives at:
#     /scratch/$USER/WiredTogether/slurm_logs/<exp_name>-<jobid>.out
# (stderr is merged into .out — there is no .err file).
#
# To re-submit just one experiment:
#     sbatch scripts/experiments/exp4_mappo_hebbian.sh

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENTS=(
    "exp1_llm.sh"
    "exp2_llm_hebbian.sh"
    "exp3_mappo.sh"
    "exp4_mappo_hebbian.sh"
    "exp5_ippo.sh"
    "exp6_ippo_hebbian.sh"
)

echo "== Submitting 6 experiments =="
for exp in "${EXPERIMENTS[@]}"; do
    script="$EXP_DIR/$exp"
    if [ ! -f "$script" ]; then
        echo "MISSING: $script — skipping" >&2
        continue
    fi
    jobid=$(sbatch --parsable "$script")
    echo "  $exp  →  job $jobid"
done
echo "=============================="
echo "Track with:    squeue -u \$USER"
echo "Tail any log:  tail -f /scratch/\$USER/WiredTogether/slurm_logs/<exp_name>-<jobid>.out"
