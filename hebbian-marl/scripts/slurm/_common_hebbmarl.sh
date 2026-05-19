#!/bin/bash
# ============================================================
# Common environment setup for hebbian-marl SLURM submissions.
# Pattern-matched to WiredTogether's scripts/experiments/_common.sh
# but stripped of LLM / Craftium / GPU-specific lines that don't
# apply to the CPU-only hebbian-marl pipeline.
#
# Source from per-tier sbatch files:
#   source "$(dirname "$0")/_common_hebbmarl.sh"
# ============================================================

# Repo layout: this file lives at hebbian-marl/scripts/slurm/
HEBB_REPO=/scratch/acmarcu/WiredTogether/hebbian-marl
ENV_PREFIX=/scratch/acmarcu/.conda/envs/WiredTogether

# ── Modules & Conda ──
module purge
module load 2025
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# ── Redirect hebbian-marl results to /scratch ──
# The launcher (scripts/run_experiments.py) reads HEBBIAN_RESULTS_DIR for
# RESULTS_DIR; main.py reads it for the sacred FileStorageObserver and
# tb_logs path. Single env var routes sacred + bonds + launcher logs +
# tb_logs all to the same tree.
export HEBBIAN_RESULTS_DIR=/scratch/${USER}/WiredTogether/runs/hebbian-marl
mkdir -p "$HEBBIAN_RESULTS_DIR/logs"

# Single-threaded torch — main.py sets th.set_num_threads(1). Match it
# at the OpenMP/MKL layer so we don't oversubscribe the allocated CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /scratch/${USER}/WiredTogether/slurm_logs

cd "$HEBB_REPO"

echo "== hebbian-marl SLURM job =="
echo "Host:         $(hostname)"
echo "Date:         $(date)"
echo "Python:       $(which python)"
echo "Repo:         $HEBB_REPO"
echo "ResultsDir:   $HEBBIAN_RESULTS_DIR"
echo "JobID:        ${SLURM_JOB_ID:-N/A}"
echo "ArrayID:      ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "ArrayCount:   ${SLURM_ARRAY_TASK_COUNT:-N/A}"
echo "============================"
