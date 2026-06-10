#!/bin/bash
#SBATCH --job-name=sim_smoke
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=cor1
#SBATCH --output=slurm_logs/sim_smoke_%j.out
#SBATCH --error=slurm_logs/sim_smoke_%j.err

# Pre-launch smoke that MIRRORS the final-suite plastic-Hebbian config
# (exp05_mappo_hebbian: --simultaneous + --rl centralized critic + the
# reward_modulated Hebbian graph with the Oja homeostatic decay) on a SHORT
# budget. Validates the two things that never ran before we commit GPU-weeks
# to the suite:
#   1. the --rl terminal-death-feed path (Ch5 deaths -> store_reward done=True),
#   2. the homeostatic Hebbian dynamics (does W differentiate or saturate?).
# ONLY the budget differs from exp05 (2x150 vs 5x2500). RUN_GROUP=smoke keeps
# results out of runs/final/. 2x150 still drives the team through all five
# chambers (timers fire at 20/40/60/80% of max-steps -> Ch5 at step ~120) and
# accumulates bonds across 2 episodes (bonds persist, they are not reset).
#
# After it finishes, inspect the final bond matrix:
#     tail -1 runs/smoke/suite_smoke/seed_*/hebbian_snapshots.jsonl
# Want a DIFFERENTIATED W (off-diagonal ~0.2-0.8, varied) — NOT all ~0.95
# (still saturating -> raise --hebbian-decay) and NOT all ~0 (over-pruned ->
# lower --hebbian-decay or raise --hebbian-eta-0).

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# wandb off + RUN_GROUP=smoke for a throwaway diagnostic (override WANDB=1 if wanted).
EPISODES=2 MAX_STEPS=150 WANDB="${WANDB:-1}" RUN_GROUP=smoke \
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
