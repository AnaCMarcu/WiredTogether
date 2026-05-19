#!/bin/bash
# Convenience launcher for the full 11-method Phase B+ thesis grid.
#
# Submits 7 legacy ablations (M1, L1, L2, M2, M3, M4, M5) + 5 GRPO
# ablations (G2, G2b, G3a, G3b, G4) as parallel SLURM array jobs.
# Followed by a dependent post-hoc job that translates legacy runs and
# builds the cross-stack results via build_results.py.
#
# Usage:
#   bash scripts/experiments/submit_full_grid.sh                # default seeds 0..4
#   SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/experiments/submit_full_grid.sh
#
# NOT a SLURM job itself — runs on a login node.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPTS_DIR=/scratch/${USER:-acmarcu}/WiredTogether/scripts/experiments
PROJECT_DIR=/scratch/${USER:-acmarcu}/WiredTogether
SEEDS="${SEEDS:-0 1 2 3 4}"
N_SEEDS=$(echo "$SEEDS" | wc -w)
ARRAY_RANGE="0-$((N_SEEDS - 1))"

echo "== Phase B+ full-grid submission =="
echo "Scripts dir: $SCRIPTS_DIR"
echo "Seeds:       $SEEDS  (array $ARRAY_RANGE)"
echo "===================================="

submit_one() {
    local script_name="$1"
    sbatch --parsable --array="$ARRAY_RANGE" \
           "${SCRIPTS_DIR}/${script_name}"
}

# ── Legacy variants ──
M1_ID=$(submit_one "M1_plain_llm.sh");           echo "  M1 (plain LLM):         $M1_ID"
L1_ID=$(submit_one "L1_llm_hebbian_prompt.sh");  echo "  L1 (LLM + Heb prompt):  $L1_ID"
L2_ID=$(submit_one "L2_llm_hebbian_propagation.sh"); echo "  L2 (L1 + reward prop):  $L2_ID"
M2_ID=$(submit_one "E2_mappo.sh");               echo "  M2 (MAPPO):             $M2_ID"
M3_ID=$(submit_one "E5_hebbian.sh");             echo "  M3 (MAPPO + Hebbian):   $M3_ID"
M4_ID=$(submit_one "M4_ippo.sh");                echo "  M4 (IPPO):              $M4_ID"
M5_ID=$(submit_one "M5_ippo_hebbian.sh");        echo "  M5 (IPPO + Hebbian):    $M5_ID"

# ── GRPO variants ──
G2_ID=$(submit_one "G2_grpo_multi_agent.sh");       echo "  G2 (GRPO):              $G2_ID"
G2B_ID=$(submit_one "G2b_grpo_multi_agent_team_reward.sh"); echo "  G2b (GRPO team):        $G2B_ID"
G3A_ID=$(submit_one "G3a_grpo_hebbian_diffusion.sh"); echo "  G3a (GRPO + diff):     $G3A_ID"
G3B_ID=$(submit_one "G3b_grpo_hebbian_composition.sh"); echo "  G3b (GRPO + comp):   $G3B_ID"
G4_ID=$(submit_one "G4_grpo_hebbian_full.sh");      echo "  G4 (GRPO + full Heb):   $G4_ID"

# ── Post-hoc cross-stack comparison job ──
# Depends `afterany` on all training jobs so a partial figure renders
# even if a few seeds crash.
DEP="afterany:${M1_ID}:${L1_ID}:${L2_ID}:${M2_ID}:${M3_ID}:${M4_ID}:${M5_ID}:${G2_ID}:${G2B_ID}:${G3A_ID}:${G3B_ID}:${G4_ID}"

# Inline submission — no separate script file, just a one-shot job.
CROSS_ID=$(sbatch --parsable --dependency="$DEP" \
  --job-name=full-grid-results \
  --partition=compute --time=00:30:00 --ntasks=1 --cpus-per-task=2 \
  --mem-per-cpu=4G --account=education-eemcs-msc-dsait \
  --output=/scratch/%u/WiredTogether/slurm_logs/full-grid-results-%j.out \
  --error=/scratch/%u/WiredTogether/slurm_logs/full-grid-results-%j.err \
  --wrap="cd $PROJECT_DIR && export PYTHONPATH=\$PROJECT_DIR/src:\${PYTHONPATH:-} && \
          python scripts/build_results.py \
              --grpo /scratch/\$USER/WiredTogether/runs/grpo \
              --legacy /scratch/\$USER/WiredTogether/runs \
              --out /scratch/\$USER/WiredTogether/results \
              --ablations M1,L1,L2,M2,M3,M4,M5,G2,G2b,G3a,G3b,G4 \
              --baseline M1 \
              --bootstrap 10000 --window 50 --rolling-window 20")
echo "  full-grid-results (after all): $CROSS_ID"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Outputs: /scratch/\$USER/WiredTogether/runs/  +  results/"
echo "Reports: /scratch/\$USER/WiredTogether/results/tables/ + results/cross_ablation/plots/"
