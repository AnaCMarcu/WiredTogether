#!/bin/bash
# Submit all WiredTogether experiments to SLURM in parallel on DAIC.
#
# Usage (from DAIC login or compute node):
#     bash daic/experiments/submit_all.sh
#
# Each experiment runs on its own GPU node, single seed (42 by default).
# Results land in:
#     $WORKSPACE/WiredTogether/runs/legacy/<exp_name>/seed_42/
# bundled with: log.txt, final_metrics.json, episodes/, gifs/, run.log, work_artifacts/.
#
# Slurm .out / .err for each experiment land in the CWD as
# <exp_name>_<jobid>.out / .err.
#
# Re-submit just one experiment:
#     sbatch daic/experiments/exp4_mappo_hebbian.sbatch

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENTS=(
    "exp1_llm.sbatch"
    "exp2_llm_hebbian.sbatch"
    "exp3_mappo.sbatch"
    "exp4_mappo_hebbian.sbatch"
    "exp5_ippo.sbatch"
    "exp6_ippo_hebbian.sbatch"
    "exp7_llm_9b.sbatch"
    "exp8_llm_9b_hebbian.sbatch"
    "exp9_llm_2b_social_prompt.sbatch"
    "exp10_llm_2b_social_bias.sbatch"
    "exp11_llm_9b_social_prompt.sbatch"
    "exp12_llm_9b_social_bias.sbatch"
    "exp14_llm_hebbian_roles.sbatch"
    "exp15_llm_roles.sbatch"
)

echo "== Submitting ${#EXPERIMENTS[@]} experiments =="
for exp in "${EXPERIMENTS[@]}"; do
    script="$EXP_DIR/$exp"
    if [ ! -f "$script" ]; then
        echo "MISSING: $script — skipping" >&2
        continue
    fi
    jobid=$(sbatch --parsable "$script")
    echo "  $exp  →  job $jobid"
done
echo "================================="
echo "Track with:    squeue -u \$USER"
echo "Tail any log:  tail -f <exp_name>_<jobid>.out"
