#!/bin/bash
#SBATCH --job-name=exp15-llm-roles
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp15_llm_roles-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp15_llm_roles-%j.out
# Submit:  sbatch scripts/experiments/exp15_llm_roles.sh
# Different seed: sbatch --array=1 ...   (uses SEEDS[1]=123 from _common.sh)
#                 SEED=99 sbatch --export=ALL ...

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# LLM + ROLE-DIFFERENTIATED agents, NO Hebbian, NO RL training. This is the
# heterogeneous counterpart to exp1_llm and completes the 2×2 ablation:
#
#                    homogeneous     heterogeneous
#   ┌──────────────────────────────────────────────┐
#   │ no Hebbian │   exp1_llm    │  exp15_llm_roles│
#   │ + Hebbian  │ exp2_llm_hebb │  exp14_llm_hebb_roles │
#   └──────────────────────────────────────────────┘
#
# The 2×2 cleanly attributes any cooperation_score / milestone-progress
# delta to (a) the role asymmetry on its own, (b) the Hebbian mechanism
# on its own, or (c) the interaction of the two.
#
# All non-role settings are IDENTICAL to exp1_llm. The only difference is
# --team-mode heterogeneous + --roles. Keep the comparison clean.
#
# Tag-driven run dir: runs/legacy/exp15_llm_roles/seed_42/
#
# Role assignment is positional:
#   agent_0 → hunter     (combat focus)
#   agent_1 → harvester  (dig / collect / lead Ch2 anvil pair-coop)
#   agent_2 → scouter    (Ch3 puzzle coordination, scanning)
# Keep the order stable across seeds so role-to-spawn mapping is consistent.

EXP_NAME="exp15_llm_roles"
# Honor seed inherited from _common.sh (array index, SEED env var, or default 42).
SEED="${SEED:-42}"
RUN_DIR="/scratch/${USER}/WiredTogether/runs/legacy/${EXP_NAME}/seed_${SEED}"
mkdir -p "$RUN_DIR"

export LLM_MODEL_PATH="$MODEL_2B"

python -u multi_agent_craftium.py \
    --num-agents 3 \
    --episodes 3 \
    --max-steps 1500 \
    --warmup-time 300 \
    --team-mode heterogeneous \
    --roles hunter,harvester,scouter \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"
