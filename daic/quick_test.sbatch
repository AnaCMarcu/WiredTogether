#!/bin/bash
#SBATCH --job-name=fast_rl
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/fast%j.out
#SBATCH --error=slurm_logs/fast%j.err

# 200-step MAPPO+Hebbian smoke test (action-level MAPPO with centralized
# critic + Hebbian social plasticity, NO social-module prompt block).
# Exercises the full RL update path on a tiny episode budget so we can
# verify plumbing (gradient steps fire, critic loss not NaN, wandb gets
# rl/* metrics) in ~30 min instead of waiting for a real 36h job.
#
# MODEL = 9B. RLLayer now uses a process-wide SHARED base model + per-agent
# LoRA adapter (Option 1 in the OOM follow-up), so 3 agents × 9B costs
# 1 × 18 GB + 3 × ~1 MB adapter weights + small per-agent value heads —
# fits comfortably on V100 (32 GB). If you ever OOM again, drop to 2B.
#
# Uses the same apptainer + wandb + env setup as the real experiments
# via _common.sh (pre-flight pkill, /tmp cwd, EGL software fallback,
# model paths, wandb wiring, archive rsync).

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

run_exp "fast_rl2" "$MODEL_9B" \
    --episodes 2 \
    --max-steps 100 \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_9B" \
    --rl-update-interval 16 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2
