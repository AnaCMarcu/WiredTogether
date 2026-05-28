#!/bin/bash
#SBATCH --job-name=exp14-llm-hebbian-roles
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/exp14_llm_hebbian_roles-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/exp14_llm_hebbian_roles-%j.out
# Submit:  sbatch scripts/experiments/exp14_llm_hebbian_roles.sh
# Different seed: sbatch --array=1 ...   (uses SEEDS[1]=123 from _common.sh)
#                 SEED=99 sbatch --export=ALL ...

source "/scratch/acmarcu/WiredTogether/scripts/experiments/_common.sh"

# LLM + Hebbian + ROLE-DIFFERENTIATED agents. The hypothesis from the LBF
# experiments: in symmetric/homogeneous teams Hebbian weights have nothing
# distinct to encode (all agent pairs are interchangeable), so the bond
# matrix collapses to a uniform value and the routing decisions become
# random. Asymmetric roles (hunter / harvester / scouter) give the bonds
# something to track — "which pairing works for which chamber" becomes a
# learnable signal.
#
# All Hebbian hyperparameters are IDENTICAL to exp2_llm_hebbian. The only
# difference is --team-mode heterogeneous + --roles. This keeps the
# comparison exp2 vs exp14 a clean "what does role asymmetry add?".
#
# Tag-driven run dir: runs/legacy/exp14_llm_hebbian_roles/seed_42/
#
# Role assignment is positional:
#   agent_0 → hunter     (combat focus)
#   agent_1 → harvester  (dig / collect / lead Ch2 anvil pair-coop)
#   agent_2 → scouter    (Ch3 puzzle coordination, scanning)
# Keep the order stable across seeds so role-to-spawn mapping is consistent.

EXP_NAME="exp14_llm_hebbian_roles"
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
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2 \
    --seed "$SEED" \
    --experiment-id "$EXP_NAME" \
    --tag "$EXP_NAME" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "$EXP_NAME done (seed=$SEED)"
