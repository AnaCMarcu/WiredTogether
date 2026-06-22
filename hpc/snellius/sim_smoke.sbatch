#!/bin/bash
#SBATCH --job-name=sim_smoke
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei14435
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/sim_smoke_%j.out
#SBATCH --error=slurm_logs/sim_smoke_%j.err

# Snellius pre-launch smoke — mirrors the final-suite plastic-Hebbian config
# (exp05: --simultaneous + --rl centralized critic + reward_modulated Hebbian
# with the Oja homeostatic decay) on a SHORT budget, to validate the Snellius
# port end-to-end (apptainer image + A100 + $TMPDIR scratch + headless render)
# before committing SBU to the full suite. RUN_GROUP=smoke keeps it out of
# runs/final/; WANDB=0 so we don't need the offline-sync dance for a throwaway.
#
# After it finishes:
#   grep -E "python exit|Experiment complete|Traceback" slurm_logs/sim_smoke_*.out
#   tail -1 runs/smoke/suite_smoke/seed_*/hebbian_snapshots.jsonl
# Want: exit 0 (port + --rl death path OK) AND a DIFFERENTIATED W (off-diagonal
# ~0.2-0.8, varied) — NOT all ~0.95 (raise --hebbian-decay) / not all ~0.

source /gpfs/work2/0/prjs1879/WiredTogether/hpc/snellius/experiments/_common.sh

EPISODES=2 MAX_STEPS=150 WANDB=0 RUN_GROUP=smoke \
    run_exp "suite_smoke" "$MODEL_2B" \
    --simultaneous \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2
