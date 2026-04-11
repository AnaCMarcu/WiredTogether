#!/bin/bash
#SBATCH --job-name=wt_chain
#SBATCH --partition=compute
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/%u/WiredTogether/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/WiredTogether/slurm_logs/%x-%j.err

# ── chain_jobs.sh ─────────────────────────────────────────────────────────────
#
# Submits a first job and N continuation jobs chained via --dependency=afterany.
# Each job runs for the SLURM time limit (8h), saves a checkpoint at the end,
# and the next job picks it up automatically.
#
# Usage (either works):
#   sbatch scripts/chain_jobs.sh          ← uses default NUM_CONT=2
#   bash   scripts/chain_jobs.sh 3        ← 1 first + 3 continuations
#
# Pass NUM_CONT via --export when using sbatch:
#   sbatch --export=ALL,NUM_CONT=4 scripts/chain_jobs.sh
# ─────────────────────────────────────────────────────────────────────────────
#
# Example — 1 first job + 3 continuations (4 × 8h = up to 32 h of compute):
#   bash scripts/chain_jobs.sh 3
#
# The first job is always run_first.sh.
# All continuation jobs use run_continue.sh.
# ─────────────────────────────────────────────────────────────────────────────

NUM_CONT=${1:-2}   # number of continuation jobs (default: 2)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting first job..."
FIRST_JOB=$(sbatch --parsable "${SCRIPT_DIR}/run_first.sh")
echo "  First job ID: $FIRST_JOB"

PREV_JOB=$FIRST_JOB
for i in $(seq 1 "$NUM_CONT"); do
    echo "Submitting continuation $i (after job $PREV_JOB)..."
    CONT_JOB=$(sbatch --parsable \
        --dependency=afterany:"$PREV_JOB" \
        "${SCRIPT_DIR}/run_continue.sh")
    echo "  Continuation $i job ID: $CONT_JOB"
    PREV_JOB=$CONT_JOB
done

echo ""
echo "Chain submitted:"
echo "  First job:         $FIRST_JOB"
echo "  Last continuation: $PREV_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /scratch/\$USER/WiredTogether/slurm_logs/wt_first-${FIRST_JOB}.out"
