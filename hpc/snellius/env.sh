# shellcheck shell=bash
# ────────────────────────────────────────────────────────────────────────────
# Snellius environment for WiredTogether. Source it on a login node before
# submitting anything:
#
#     source $WT_WORKSPACE/WiredTogether/hpc/snellius/env.sh
#
# It reads your settings from ~/.wiredtogether_snellius (written by
# hpc/snellius/setup.sh init), exports what hpc/slurm/ expects, and puts the
# sbatch shim (hpc/snellius/bin/sbatch) first on PATH, so every
# hpc/slurm/experiments/*.sbatch and submit_*.sh runs on Snellius unchanged.
# ────────────────────────────────────────────────────────────────────────────

WT_SN_CONFIG="${WT_SN_CONFIG:-$HOME/.wiredtogether_snellius}"
if [ -f "$WT_SN_CONFIG" ]; then
    # shellcheck disable=SC1090
    . "$WT_SN_CONFIG"
fi

if [ -z "${WT_WORKSPACE:-}" ]; then
    echo "[wt-snellius] WT_WORKSPACE is not set. Run hpc/snellius/setup.sh init first." >&2
    return 1 2>/dev/null || exit 1
fi

# Bind mounts must use the real path: /projects/<id> is a symlink into /gpfs,
# and apptainer does not resolve a symlinked bind source reliably.
WT_WORKSPACE="$(readlink -f "$WT_WORKSPACE")"
export WT_WORKSPACE
# The .sbatch files source _common.sh through $WORKSPACE, not $WT_WORKSPACE.
export WORKSPACE="$WT_WORKSPACE"

# Shared images/ or models/ may be symlinks into a project directory outside
# the workspace; WT_BIND makes those targets visible inside the container.
if [ -n "${WT_SHARED:-}" ]; then
    WT_SHARED="$(readlink -f "$WT_SHARED")"
    export WT_SHARED
    case ",${WT_BIND:-}," in
        *",$WT_SHARED,"*) ;;
        *) export WT_BIND="${WT_BIND:+$WT_BIND,}$WT_SHARED" ;;
    esac
fi

# SLURM settings the shim applies to every submission.
export WT_SN_PARTITION="${WT_SN_PARTITION:-gpu_a100}"
export WT_SN_ACCOUNT="${WT_SN_ACCOUNT:-}"
export WT_SN_MAX_TIME="${WT_SN_MAX_TIME:-120:00:00}"
export WT_SN_CPUS="${WT_SN_CPUS:-18}"

# Default image and model: Gemma 4 E4B, as in the paper's main suite.
export WT_IMAGE="${WT_IMAGE:-$WT_WORKSPACE/images/wiredtogether_gemma4.sif}"
export MODEL_LLM="${MODEL_LLM:-$WT_WORKSPACE/models/gemma-4-E4B-it}"

# The image built on the other cluster bakes that cluster's storage path into
# its %environment as the Hugging Face cache (HF_HOME etc.), which does not
# exist here, so any cache write fails with "Read-only file system". The
# APPTAINERENV_ prefix overrides %environment inside every container.
export APPTAINERENV_HF_HOME="$WT_WORKSPACE/models/.hf_home"
export APPTAINERENV_HF_HUB_CACHE="$WT_WORKSPACE/models/.hf_home/hub"
export APPTAINERENV_TRANSFORMERS_CACHE="$WT_WORKSPACE/models/.hf_home/transformers"
mkdir -p "$WT_WORKSPACE/models/.hf_home" 2>/dev/null || true

# Compute nodes have no internet: log W&B offline and skip the end-of-job
# upload. Sync later from a login node with hpc/snellius/sync_wandb.sh.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_AUTOSYNC="${WANDB_AUTOSYNC:-0}"

case ":$PATH:" in
    *":$WT_WORKSPACE/WiredTogether/hpc/snellius/bin:"*) ;;
    *) export PATH="$WT_WORKSPACE/WiredTogether/hpc/snellius/bin:$PATH" ;;
esac

echo "[wt-snellius] workspace=$WT_WORKSPACE partition=$WT_SN_PARTITION account=${WT_SN_ACCOUNT:-<default>} max_time=$WT_SN_MAX_TIME"
