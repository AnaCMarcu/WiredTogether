#!/bin/bash
#SBATCH --job-name=aw-smoke
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/aw_smoke_%j.out
#SBATCH --error=slurm_logs/aw_smoke_%j.err
# ────────────────────────────────────────────────────────────────────────────
# First AgentWorld job: one 3-agent task, the Hebbian arm, one short episode,
# Gemma-4-E4B on vLLM, plus the replay video. Submit from the repository root
# (slurm_logs/ is relative), after hpc/agentworld/setup_aw.sh all:
#
#     cd $WT_WORKSPACE/WiredTogether
#     sbatch hpc/agentworld/aw_smoke.sbatch
#
# Overrides: TASK, ARM, ROUNDS, REPLICAS, EPISODES, VIDEO, MIN_ROUND_S.
# ROUNDS defaults to AgentWorld's 55. MIN_ROUND_S pads every round so the
# real-time game (trees regrow, mobs respawn) advances about as much per round
# as in AgentWorld's serial runner; 20 s is a stand-in until that runner's
# pace is measured.
# Passed if the log ends with "python exit: 0" and
# runs_aw/smoke/<name>/seed_42/episode_1/ holds summary.json, trajectories/,
# replay/state.jsonl and videos/team_t00.mp4.
# ────────────────────────────────────────────────────────────────────────────
source "${WT_REPO:-$WORKSPACE/WiredTogether}/hpc/agentworld/aw_common.sh"
RUN_GROUP="${RUN_GROUP:-smoke}"
TASK="${TASK:-task_01_magic_staff}"
ARM="${ARM:-hebbian}"
ROUNDS="${ROUNDS:-55}"
MIN_ROUND_S="${MIN_ROUND_S:-20}"
REPLICAS="${REPLICAS:-1}"
run_aw "${ARM}_${TASK}_x${REPLICAS}" \
    --tasks "$TASK" --replicas "$REPLICAS" --arm "$ARM" \
    --episodes "${EPISODES:-1}" --rounds "$ROUNDS" --video "${VIDEO:-team}" \
    --min-round-seconds "$MIN_ROUND_S" --max-in-flight 64
