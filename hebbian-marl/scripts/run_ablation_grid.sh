#!/usr/bin/env bash
# Run the full hebbian-marl ablation grid.
#
# Usage:
#   bash scripts/run_ablation_grid.sh
#
# Each variant × seed is launched as a separate process. PARALLEL_CAP
# limits concurrency. Output goes to epymarl/results/sacred/{name}/{key}/{n}
# and bonds.jsonl snapshots go to results/bonds/{name}/seed_{seed}.jsonl.
#
# Env vars:
#   PARALLEL_CAP=4                            (max concurrent processes)
#   T_MAX=5000000                             (env steps per run)
#   ENV_KEY="Foraging-Comm-10x10-3p-3f-v3"    (LBF variant)
#   SEEDS="0 1 2 3 4"                         (space-separated seed list)
#   VARIANTS="ippo_baseline seac hebb_s ..."  (algorithm yamls to run)

set -e

# Resolve the script's directory; the EPyMARL entry point is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EPYMARL_ROOT="${REPO_ROOT}/epymarl"

# Defaults — override via env vars on the command line.
PARALLEL_CAP="${PARALLEL_CAP:-4}"
T_MAX="${T_MAX:-5000000}"
ENV_KEY="${ENV_KEY:-Foraging-Comm-10x10-3p-3f-v3}"
SEEDS_DEFAULT="0 1 2 3 4"
SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
VARIANTS_DEFAULT="ippo_baseline seac hebb_r hebb_s hebb_rs hebb_s_nocomm hebb_s_commonly"
VARIANTS="${VARIANTS:-$VARIANTS_DEFAULT}"

# Pick python from the local venv if present, else fall back to PATH.
if [ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]; then
    PY="${REPO_ROOT}/.venv/Scripts/python.exe"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PY="${REPO_ROOT}/.venv/bin/python"
else
    PY="python"
fi

echo "[ablation] variants: ${VARIANTS}"
echo "[ablation] seeds:    ${SEEDS}"
echo "[ablation] t_max:    ${T_MAX}"
echo "[ablation] env:      ${ENV_KEY}"
echo "[ablation] parallel: ${PARALLEL_CAP}"

cd "${EPYMARL_ROOT}"

run_one() {
    local variant="$1"
    local seed="$2"
    local label="${variant}_seed${seed}"
    echo "[ablation] launching ${label}"
    "${PY}" src/main.py \
        --config="${variant}" --env-config=lbf_comm \
        with env_args.key="${ENV_KEY}" \
             env_args.time_limit=50 \
             seed="${seed}" \
             t_max="${T_MAX}" \
             use_cuda=False \
             name="${label}" \
        > "results/${label}.log" 2>&1
}

for v in $VARIANTS; do
    for s in $SEEDS; do
        run_one "$v" "$s" &
        # cap concurrency
        while [ "$(jobs -r | wc -l)" -ge "${PARALLEL_CAP}" ]; do
            sleep 1
        done
    done
done

wait
echo "[ablation] all runs finished"
