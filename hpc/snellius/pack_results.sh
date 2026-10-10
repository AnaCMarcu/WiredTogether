#!/usr/bin/env bash
# pack_results.sh - build ONE tarball of the finished runs on Snellius, for
# hpc/snellius/fetch_results.sh to pull to a laptop with a single scp.
#
# Run on a login node (no budget used), from anywhere:
#     bash ~/wt/WiredTogether/hpc/snellius/pack_results.sh            # every group
#     bash ~/wt/WiredTogether/hpc/snellius/pack_results.sh comm_budget_vllm smoke_vllm
#
# Packs every runs/<group>/<arm>/seed_<N> that has final_metrics.json, plus
# runs/bench/. Runs still in progress, and set-aside dirs such as
# seed_123.failed_zmq, are listed but left out. Videos (gifs/, and the
# run_artifacts/ tree) and per-call prompt logs (llm_logs/) stay on Snellius;
# WITH_LLM_LOGS=1 includes llm_logs/. log.txt and vllm_server.log ARE included:
# per-step timing and the FLOPs accounting read them. LIGHT=1 leaves log.txt
# out too (two thirds of a run's size; no figure script reads it), for a laptop
# short on disk; the full log stays on Snellius for a later pull.
#
# A group may be a symlink into another workspace, e.g. to ship the
# hard-orchestrator checkout's runs in the same tarball:
#     ln -sfn ~/wt_orch/WiredTogether/runs/orchestrator runs/orchestrator_vllm
set -euo pipefail
: "${WT_WORKSPACE:?source hpc/snellius/env.sh first}"
RUNS="$WT_WORKSPACE/WiredTogether/runs"
OUT="$HOME/wt_snellius_pull.tgz"
LIST="$HOME/wt_snellius_pull.list"

cd "$RUNS"
if [ "$#" -gt 0 ]; then groups=("$@"); else
    groups=(); for g in */; do g="${g%/}"; [ "$g" = bench ] || groups+=("$g"); done
fi

: > "$LIST"
n_done=0; n_open=0
for g in ${groups[@]+"${groups[@]}"}; do
    [ -d "$g" ] || { echo "no group runs/$g - skipped" >&2; continue; }
    for d in "$g"/*/seed_*; do
        [ -d "$d" ] || continue
        case "${d##*/}" in seed_*.*) echo "  set aside, not packed: $d"; continue ;; esac
        if [ -f "$d/final_metrics.json" ]; then
            echo "$d" >> "$LIST"; n_done=$((n_done + 1))
        else
            echo "  unfinished, not packed: $d"; n_open=$((n_open + 1))
        fi
    done
done
[ "$#" -eq 0 ] && [ -d bench ] && echo bench >> "$LIST"

[ -s "$LIST" ] || { echo "nothing to pack"; exit 1; }
excl=(--exclude=gifs --exclude=checkpoints --exclude=work_artifacts --exclude=wandb)
[ "${WITH_LLM_LOGS:-0}" = 1 ] || excl+=(--exclude=llm_logs)
[ "${LIGHT:-0}" = 1 ] && excl+=(--exclude=log.txt)
tar czf "$OUT" "${excl[@]}" -T "$LIST"

echo
echo "packed $n_done finished runs ($n_open unfinished left out) into $OUT:"
ls -lh "$OUT"
cut -d/ -f1,2 "$LIST" | sort | uniq -c
echo
echo "Next, from Git Bash on your laptop:"
echo "    bash /c/Users/marcu/OneDrive/Documente/GitHub/WiredTogether-snellius/hpc/snellius/fetch_results.sh"
echo "Then back here:  rm -f $OUT $LIST"
