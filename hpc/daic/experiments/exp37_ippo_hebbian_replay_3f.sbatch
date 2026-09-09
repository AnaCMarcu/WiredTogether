#!/bin/bash
#SBATCH --job-name=exp37-ippo-replay-3f
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp37_ippo_hebbian_replay_3f_%j.out
#SBATCH --error=slurm_logs/exp37_ippo_hebbian_replay_3f_%j.err
#SBATCH --exclude=cor1

# ───────────────────────────────────────────────────────────────────────────
# exp37_ippo_hebbian_replay_3f — exp31 (IPPO + Hebbian + weight-gated
# experience sharing, Eq. 7 at rho=0.3) on the NEW Hebbian rule
# ("Hebbian 2.0": three_factor + signed death LTD).
#
# Differs from exp31_ippo_hebbian_replay.sbatch in the update rule ONLY.
# In independent-critic mode the neighbour GAE-on-copy uses the neighbour's
# stored value baseline (rl_layer.ppo_update._gae_on_copy) — unchanged here;
# only the bond graph that gates the mixture is computed differently.
#
# Rule parameters byte-identical to exp36 / new_exp_0_gemma_si3f / exp34-35.
#
# Submit via submit_social_replay_3f.sh (sets RUN_GROUP / model / image /
# stagger per lane). See exp36 for the full rationale.
# ───────────────────────────────────────────────────────────────────────────

RUN_GROUP="${RUN_GROUP:-social_replay_3f}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-social_replay_3f}"
export RUN_GROUP EPISODES MAX_STEPS WANDB_PROJECT

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-ippo,hebbian,independent_critic,social_replay,three_factor,fdecay}"

# IPPO + Hebbian 2.0 + social replay — exp31's flags with the 3f rule.
run_exp "exp37_ippo_hebbian_replay_3f" "$MODEL_LLM" \
    --rl \
    --rl-critic-mode independent \
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
