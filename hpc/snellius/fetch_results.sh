#!/usr/bin/env bash
# fetch_results.sh - pull the tarball built by pack_results.sh and file each
# run into the laptop's RQ-grouped dataset (runs_from_daic/ by default).
#
# Run from Git Bash on the laptop; ONE scp = one Snellius password:
#     bash hpc/snellius/fetch_results.sh
#
# Filing (group names are kept, so analysis/paths.group() finds them):
#   comm_budget_vllm, and any other non-smoke group -> <dest>/compute/<group>
#   smoke, smoke_vllm, pilot_* and *_smoke groups   -> <dest>/smoke/snellius_<group>
#   bench/                                          -> <dest>/smoke/snellius_bench
# A run already on the laptop is replaced, so the script can be re-run.
#
# Overrides: SNELLIUS=user@host  DEST=<dataset root>  KEEP_TGZ=1
set -euo pipefail
SNELLIUS="${SNELLIUS:-amarcu@snellius.surf.nl}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${DEST:-$HERE/../WiredTogether/runs_from_daic}"
DEST="$(cd "$DEST" && pwd)"
TGZ="$DEST/wt_snellius_pull.tgz"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

if [ ! -f "$TGZ" ]; then
    echo "Fetching wt_snellius_pull.tgz (one password prompt)..."
    scp "$SNELLIUS:~/wt_snellius_pull.tgz" "$TGZ"
else
    echo "Reusing already-downloaded $TGZ"
fi
ls -lh "$TGZ"
echo "Free space before extracting: $(df -h "$DEST" | awk 'NR==2 {print $4}')"

tar xzf "$TGZ" -C "$STAGE"

n=0
for g in "$STAGE"/*/; do
    g="$(basename "$g")"
    case "$g" in
        bench) mkdir -p "$DEST/smoke"; rm -rf "$DEST/smoke/snellius_bench"
               mv "$STAGE/bench" "$DEST/smoke/snellius_bench"
               echo "  bench -> smoke/snellius_bench"; continue ;;
        smoke|smoke_*|pilot*|*_smoke) target="$DEST/smoke/snellius_$g" ;;
        *) target="$DEST/compute/$g" ;;
    esac
    for d in "$STAGE/$g"/*/seed_*; do
        [ -d "$d" ] || continue
        arm="$(basename "$(dirname "$d")")"; seed="$(basename "$d")"
        mkdir -p "$target/$arm"; rm -rf "$target/$arm/$seed"; mv "$d" "$target/$arm/$seed"
        n=$((n + 1))
    done
    echo "  $g -> ${target#$DEST/}"
done
echo "filed $n runs under $DEST"

[ "${KEEP_TGZ:-0}" = 1 ] || { rm -f "$TGZ"; echo "removed the local tarball"; }
echo "On Snellius, clean up with:  rm -f ~/wt_snellius_pull.tgz ~/wt_snellius_pull.list"
