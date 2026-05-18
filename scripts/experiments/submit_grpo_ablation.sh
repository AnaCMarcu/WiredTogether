#!/bin/bash
# Convenience launcher: submits the full GRPO ablation grid as one batch.
#
# Submits G2, G3a, G3b, G4 as parallel SLURM array jobs (one array per
# experiment, indexed by seed), then submits G5_compare with a dependency
# on all of them finishing.
#
# Usage:
#   bash scripts/experiments/submit_grpo_ablation.sh           # default seeds 0..2
#   SEEDS="0 1 2 3 4" bash scripts/experiments/submit_grpo_ablation.sh
#
# This is NOT a SLURM job itself — run it from a login node.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPTS_DIR=/scratch/${USER:-acmarcu}/WiredTogether/scripts/experiments
SEEDS="${SEEDS:-0 1 2}"
N_SEEDS=$(echo "$SEEDS" | wc -w)
ARRAY_RANGE="0-$((N_SEEDS - 1))"

echo "== GRPO ablation grid submission =="
echo "Scripts dir: $SCRIPTS_DIR"
echo "Seeds:       $SEEDS  (array $ARRAY_RANGE)"
echo "===================================="

submit_one() {
    local script_name="$1"
    local job_id
    job_id=$(sbatch --parsable --array="$ARRAY_RANGE" \
                    "${SCRIPTS_DIR}/${script_name}")
    echo "$job_id"
}

# Fire-and-forget the four training experiments in parallel.
G2_ID=$(submit_one "G2_grpo_multi_agent.sh")
echo "  G2 (multi-agent, 3B):           $G2_ID"
G3A_ID=$(submit_one "G3a_grpo_hebbian_diffusion.sh")
echo "  G3a (diffusion only, 4a):       $G3A_ID"
G3B_ID=$(submit_one "G3b_grpo_hebbian_composition.sh")
echo "  G3b (composition only, 4b):     $G3B_ID"
G4_ID=$(submit_one "G4_grpo_hebbian_full.sh")
echo "  G4 (full Hebbian, 4a+4b):       $G4_ID"

# Optional 3A team-reward ablation (smaller priority — submit if you want it).
# G2B_ID=$(submit_one "G2b_grpo_multi_agent_team_reward.sh")
# echo "  G2b (multi-agent, 3A team):     $G2B_ID"

# G5 comparison runs after the slowest finishes. ``afterany`` so we still
# get a partial figure if one variant crashed.
DEP="afterany:${G2_ID}:${G3A_ID}:${G3B_ID}:${G4_ID}"
G5_ID=$(sbatch --parsable --dependency="$DEP" \
               "${SCRIPTS_DIR}/G5_compare.sh")
echo "  G5 (compare, runs after all):   $G5_ID"

echo ""
echo "Monitor:    squeue -u \$USER"
echo "Output:     /scratch/\$USER/WiredTogether/runs/grpo/"
echo "Reports:    /scratch/\$USER/WiredTogether/reports/grpo_ablation/"
