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
# MODEL = 2B. Each agent's RLLayer loads its own AutoModelForCausalLM
# instance, so the GPU holds 3× the base model. 9B × 3 ≈ 54 GB OOMs on
# the V100 (32 GB); 2B × 3 ≈ 12 GB fits comfortably. exp4_mappo_hebbian
# uses 2B for the same reason; use exp7/exp8 (9B inference-only, no RL
# adapter per agent) when you specifically need 9B behaviour.
#
# Uses the same apptainer + wandb + env setup as the real experiments
# via _common.sh (pre-flight pkill, /tmp cwd, EGL software fallback,
# model paths, wandb wiring, archive rsync).

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/daic/experiments/_common.sh

run_exp "fast_rl" "$MODEL_2B" \
    --episodes 2 \
    --max-steps 100 \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 16 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-ltp 0.01 \
    --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 \
    --hebbian-beta 1.0 \
    --hebbian-rho 0.3 \
    --hebbian-gamma 0.2
