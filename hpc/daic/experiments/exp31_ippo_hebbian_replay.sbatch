#!/bin/bash
#SBATCH --job-name=exp31-ippo-hebbian-replay
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp31_ippo_hebbian_replay_%j.out
#SBATCH --error=slurm_logs/exp31_ippo_hebbian_replay_%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

# Social-replay suite: exp06 (IPPO+Hebbian) plus weight-gated experience
# sharing (Eq. 7) at rho=0.3. In independent-critic mode the neighbour
# GAE-on-copy uses the neighbour's stored value baseline (see
# rl_layer.ppo_update._gae_on_copy). Submit via submit_social_replay.sh.
RUN_GROUP="${RUN_GROUP:-social_replay}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-social_replay_wired_together}"

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-ippo,hebbian,independent_critic,social_replay}"

# IPPO + Hebbian + social replay — exp06's flags + --hebbian-rho 0.3.
run_exp "exp31_ippo_hebbian_replay" "$MODEL_2B" \
    --rl \
    --rl-critic-mode independent \
    --rl-model-path "$MODEL_2B" \
    --rl-update-interval 64 \
    --rl-lr 3e-4 \
    --hebbian \
    --hebbian-eta-0 0.005 \
    --hebbian-reward-norm 50 \
    --hebbian-decay 0.005 \
    --hebbian-gamma 0.2 \
    --hebbian-rho 0.3 --simultaneous
