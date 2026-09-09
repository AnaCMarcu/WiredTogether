#!/bin/bash
#SBATCH --job-name=exp36-mappo-replay-3f
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp36_mappo_hebbian_replay_3f_%j.out
#SBATCH --error=slurm_logs/exp36_mappo_hebbian_replay_3f_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# exp36_mappo_hebbian_replay_3f — exp30 (MAPPO + Hebbian + weight-gated
# experience sharing, Eq. 7 at rho=0.3) on the NEW Hebbian rule
# ("Hebbian 2.0": three_factor + signed death LTD).
#
# Differs from exp30_mappo_hebbian_replay.sbatch in the update rule ONLY —
# --rl/--rl-critic-mode/--rl-update-interval/--rl-lr and --hebbian-rho 0.3
# are byte-identical, so the pair isolates the rule:
#
#   exp30  reward_modulated  eta0 0.005, lambda 0.005
#   exp36  three_factor      eta0 0.001, lambda 0.001, rho_e 0.9,
#                            coact floor 0.25, death LTD 0.05 (cap 10)
#
# The rule parameters are byte-identical to new_exp_0_gemma_si3f.sbatch and
# exp34/35 (exp_qwen_three_factor_fdecay), so this arm's bonds are directly
# comparable with the rest of the Hebbian 2.0 suite.
#
# NOTE on the interaction that makes this arm interesting: --hebbian-rho
# gates WHOSE transitions enter the PPO pool by bond weight W, and the
# three-factor rule changes how W is grown (eligibility trace converted by
# reward, decayed by death blame). The replay mixture therefore tracks a
# different bond graph than it did under exp30 — that is the whole point.
#
# Submit via submit_social_replay_3f.sh, which sets RUN_GROUP / model /
# image / stagger per lane (Qwen 2B vs Gemma 4 E4B). One run by hand:
#   SEED=42 RUN_GROUP=social_replay_3f_qwen MODEL_LLM=... sbatch exp36_...sbatch
# ───────────────────────────────────────────────────────────────────────────

# Set before sourcing: _common.sh binds WANDB_PROJECT and the image at
# source time (see new_exp_0_gemma_si3f.sbatch for the same ordering).
RUN_GROUP="${RUN_GROUP:-social_replay_3f}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-social_replay_3f}"
export RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-mappo,hebbian,centralized_critic,social_replay,three_factor,fdecay}"

# MAPPO + Hebbian 2.0 + social replay — exp30's flags with the 3f rule.
run_exp "exp36_mappo_hebbian_replay_3f" "$MODEL_LLM" \
    --rl \
    --rl-critic-mode centralized \
    --rl-model-path "$MODEL_LLM" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-mode three_factor \
    --hebbian-eta-0 0.001 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.001 \
    --hebbian-eligibility-rho 0.9 \
    --hebbian-coact-floor 0.25 \
    --hebbian-death-ltd 0.05 \
    --hebbian-death-cap 10 \
    --hebbian-gamma 0.2 \
    --hebbian-rho 0.3 --simultaneous
