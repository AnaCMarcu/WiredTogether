#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# One-time Snellius setup for WiredTogether. Run on a LOGIN node (compute
# nodes have no internet), from the clone at $WT_WORKSPACE/WiredTogether.
#
#   bash hpc/snellius/setup.sh init  [--account ACC] [--shared DIR]
#   bash hpc/snellius/setup.sh image [gemma4|qwen] [--from FILE.sif]
#   bash hpc/snellius/setup.sh models [gemma4|qwen|all]
#   bash hpc/snellius/setup.sh vllm   [TAG]          (optional: vLLM benchmark image)
#   bash hpc/snellius/setup.sh check
#
# See hpc/snellius/README.md for the full walkthrough.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd -P)"
CONFIG="${WT_SN_CONFIG:-$HOME/.wiredtogether_snellius}"

die()  { echo "[setup] ERROR: $*" >&2; exit 1; }
info() { echo "[setup] $*"; }

load_env() {
    # shellcheck disable=SC1091
    . "$HERE/env.sh" >/dev/null || die "run '$0 init' first"
}

image_path() {
    case "$1" in
        gemma4) echo "$WT_WORKSPACE/images/wiredtogether_gemma4.sif" ;;
        qwen)   echo "$WT_WORKSPACE/images/wiredtogether.sif" ;;
        *)      die "unknown image '$1' (gemma4|qwen)" ;;
    esac
}

# ── init: write the per-user config and the workspace layout ──────────────
cmd_init() {
    local account="" shared=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --account) account="$2"; shift 2 ;;
            --shared)  shared="$2";  shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done

    # The launchers expect the repository at $WT_WORKSPACE/WiredTogether.
    [ "$(basename "$REPO")" = "WiredTogether" ] \
        || die "clone the repository into a directory named WiredTogether (found $REPO)"
    local ws
    ws="$(dirname "$REPO")"

    if [ -z "$account" ] && command -v sacctmgr >/dev/null 2>&1; then
        account="$(sacctmgr -nP show assoc user="$USER" format=account 2>/dev/null | head -1 || true)"
    fi

    cat > "$CONFIG" <<EOF
# WiredTogether on Snellius — read by hpc/snellius/env.sh.
WT_WORKSPACE="$ws"
WT_SN_ACCOUNT="$account"
WT_SHARED="$shared"
EOF
    info "wrote $CONFIG"

    mkdir -p "$ws/slurm_logs" "$REPO/slurm_logs" "$REPO/hpc/slurm/experiments/slurm_logs"
    local d
    for d in images models; do
        if [ -n "$shared" ]; then
            mkdir -p "$shared/$d"
            if [ -e "$ws/$d" ] && [ ! -L "$ws/$d" ]; then
                info "$ws/$d already exists as a real directory; leaving it (not linking to $shared/$d)"
            else
                ln -sfn "$(readlink -f "$shared/$d")" "$ws/$d"
                info "$ws/$d -> $shared/$d"
            fi
        else
            mkdir -p "$ws/$d"
        fi
    done

    load_env
    info "account: ${account:-<SLURM default>}   (check budgets with: accinfo)"
    info "next: bash hpc/snellius/setup.sh image gemma4   then   models gemma4"
}

# ── image: copy a known .sif (preferred) or build one from the recipe ─────
cmd_image() {
    load_env
    local which="gemma4" from=""
    while [ $# -gt 0 ]; do
        case "$1" in
            gemma4|qwen) which="$1"; shift ;;
            --from) from="$2"; shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done
    local out def
    out="$(image_path "$which")"
    mkdir -p "$(dirname "$out")"

    if [ -n "$from" ]; then
        [ -f "$from" ] || die "no such file: $from"
        info "copying $from -> $out"
        cp "$from" "$out"
    else
        if [ "$which" = gemma4 ]; then def="$REPO/hpc/slurm/wiredtogether_gemma4.def"
        else                           def="$REPO/hpc/slurm/wiredtogether.def"; fi
        # Login-node /tmp is small; keep the multi-GB build cache in the workspace.
        local tmp="$WT_WORKSPACE/.apptainer_build_$$"
        mkdir -p "$tmp"
        trap "rm -rf '$tmp'" EXIT
        export APPTAINER_TMPDIR="$tmp" APPTAINER_CACHEDIR="$tmp/cache"
        info "building $out from $def (20-60 min)"
        # %files paths in the recipe are relative to the repo root.
        (cd "$REPO" && apptainer build --fakeroot --force "$out" "$def") \
            || die "build failed. If --fakeroot is not allowed here, build the image elsewhere and rerun with --from FILE.sif"
    fi
    info "sha256 (record it next to your results):"
    sha256sum "$out" | tee "$out.sha256"
}

# ── vllm: the official vLLM server image (for bench_vllm.sbatch) ──────────
cmd_vllm() {
    load_env
    local tag="${1:-latest}"
    local out="$WT_WORKSPACE/images/vllm.sif"
    local tmp="$WT_WORKSPACE/.apptainer_pull_$$"
    mkdir -p "$tmp" "$(dirname "$out")"
    trap "rm -rf '$tmp'" EXIT
    export APPTAINER_TMPDIR="$tmp" APPTAINER_CACHEDIR="$tmp/cache"
    info "pulling docker://vllm/vllm-openai:$tag -> $out (several GB, 10-30 min)"
    apptainer pull --force "$out" "docker://vllm/vllm-openai:$tag"
    info "vLLM version in the image:"
    apptainer exec "$out" python3 -c "import vllm; print(vllm.__version__)" | tee "$out.version"
    sha256sum "$out" | tee "$out.sha256"
}

# ── models: download weights from the Hugging Face Hub ────────────────────
cmd_models() {
    load_env
    local which="${1:-gemma4}" repos=()
    case "$which" in
        gemma4) repos=(google/gemma-4-E4B-it) ;;
        qwen)   repos=(Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B) ;;
        all)    repos=(google/gemma-4-E4B-it Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B) ;;
        *) die "unknown model set '$which' (gemma4|qwen|all)" ;;
    esac
    # The sentence encoder for the message codebook is always needed.
    repos+=(sentence-transformers/all-MiniLM-L6-v2)

    local img="$WT_IMAGE"
    [ -f "$img" ] || img="$(image_path qwen)"
    [ -f "$img" ] || die "no image yet — run '$0 image' first (the download uses its huggingface_hub)"

    # The image sets HF_HOME to the workspace, so a token saved by a
    # host-side `hf auth login` would not be found inside it — pass it on.
    if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
        HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    fi
    if [ -z "${HF_TOKEN:-}" ]; then
        info "WARNING: no HF_TOKEN. Gemma is gated: accept its licence on huggingface.co,"
        info "         then 'export HF_TOKEN=hf_...' and rerun."
    fi

    local r
    for r in "${repos[@]}"; do
        local dest="$WT_WORKSPACE/models/${r##*/}"
        info "$r -> $dest"
        apptainer exec --bind "$WT_WORKSPACE:$WT_WORKSPACE" \
            ${WT_BIND:+--bind "$WT_BIND"} \
            --env HF_TOKEN="${HF_TOKEN:-}" \
            "$img" python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
" "$r" "$dest"
    done
}

# ── check: everything a job needs, verified from the login node ───────────
cmd_check() {
    load_env
    local ok=1
    chk() { if eval "$2"; then echo "  ok    $1"; else echo "  MISS  $1"; ok=0; fi; }
    echo "[check] workspace $WT_WORKSPACE"
    chk "repository at \$WT_WORKSPACE/WiredTogether"  "[ -f '$WT_WORKSPACE/WiredTogether/hpc/slurm/experiments/_common.sh' ]"
    chk "apptainer on PATH"                           "command -v apptainer >/dev/null"
    chk "sbatch shim first on PATH"                   "[ \"\$(command -v sbatch)\" = '$WT_WORKSPACE/WiredTogether/hpc/snellius/bin/sbatch' ]"
    chk "image $WT_IMAGE"                             "[ -f '$WT_IMAGE' ]"
    chk "model $MODEL_LLM"                            "[ -f '$MODEL_LLM/config.json' ]"
    chk "sentence encoder all-MiniLM-L6-v2"           "[ -d '$WT_WORKSPACE/models/all-MiniLM-L6-v2' ]"
    chk "slurm_logs/ in the repository"               "[ -d '$WT_WORKSPACE/WiredTogether/slurm_logs' ]"
    if [ -f "$WT_IMAGE" ]; then
        chk "image imports craftium + transformers"   "apptainer exec '$WT_IMAGE' python -c 'import craftium, transformers' 2>/dev/null"
    fi
    if [ -f "$HOME/.netrc" ] && grep -q api.wandb.ai "$HOME/.netrc"; then
        echo "  ok    W&B credentials in ~/.netrc (only needed for sync_wandb.sh)"
    else
        echo "  note  no W&B login in ~/.netrc — fine with WANDB=0; needed to sync offline runs"
    fi
    echo "[check] dry-run of a submission:"
    (cd "$WT_WORKSPACE/WiredTogether" && WT_SN_DRY_RUN=1 sbatch hpc/slurm/experiments/exp01_llm_2b.sbatch 2>&1 | tail -1)
    [ "$ok" = 1 ] && echo "[check] all good" || { echo "[check] fix the MISS lines above"; return 1; }
}

case "${1:-}" in
    init)   shift; cmd_init "$@" ;;
    image)  shift; cmd_image "$@" ;;
    models) shift; cmd_models "$@" ;;
    vllm)   shift; cmd_vllm "$@" ;;
    check)  shift; cmd_check "$@" ;;
    *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
