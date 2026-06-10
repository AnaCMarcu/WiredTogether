#!/bin/bash
# ────────────────────────────────────────────────────────────────────────────
# tmp_cleanup_test.sbatch — prove a DAIC job leaves NO orphans in the node's
# /tmp after it finishes.
#
# Why this exists: HPC support flagged (2026-06) that our jobs left orphaned
# files in the compute nodes' /tmp — including Apptainer FUSE/squashfuse mounts
# their automated cleanup can't remove. run_exp() in experiments/_common.sh now
# scopes ALL node-local scratch under /tmp/$USER (WORK_DIR + APPTAINER_TMPDIR /
# TMPDIR) and removes it on exit (explicit post-salvage cleanup + an EXIT/INT/
# TERM trap that also covers SLURM cancel/timeout). This job verifies that:
# it runs a tiny smoke through the REAL launch path, then inspects /tmp on the
# SAME node and prints a PASS/FAIL verdict into this job's .out on PRB — so the
# result survives even though the node's /tmp is unreachable once the job ends.
#
# Submit:  mkdir -p slurm_logs && sbatch hpc/daic/tmp_cleanup_test.sbatch
# Read:    grep TMP-CHECK slurm_logs/tmpchk<jobid>.out
# Success: two PASS lines + an empty/absent /tmp/$USER.
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=tmpchk
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/tmpchk%j.out
#SBATCH --error=slurm_logs/tmpchk%j.err
#SBATCH --exclude=cor1

source /tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/hpc/daic/experiments/_common.sh

echo "[TMP-CHECK] node=$(hostname) user=$USER job=${SLURM_JOB_ID:-?}"
echo "[TMP-CHECK] BEFORE: $(ls -A /tmp/$USER 2>/dev/null | tr '\n' ' ')"

# Tiny smoke through the real launch path (1 episode, 3 steps, wandb off). The
# ~5-min VoxeLibre media-load floor means this still takes ~8-12 min — the 1h
# wall above is deliberate headroom so it finishes NORMALLY (we want to verify
# the normal-completion cleanup, which is the exact thing support complained
# about). The cancel/timeout path uses the same rm -rf via the trap.
EPISODES=1 MAX_STEPS=3 WANDB=0 run_exp tmpchk "$MODEL_2B"

echo "[TMP-CHECK] ===== AFTER run_exp returned ====="
echo "[TMP-CHECK] contents of /tmp/$USER:"
ls -la /tmp/$USER/ 2>/dev/null || echo "[TMP-CHECK]   /tmp/$USER gone (fully clean)"
if ls -d /tmp/$USER/*tmpchk* >/dev/null 2>&1; then
    echo "[TMP-CHECK] FAIL: scratch dirs left behind"
else
    echo "[TMP-CHECK] PASS: WORK_DIR + TMP_ROOT cleaned"
fi
if findmnt -t fuse.squashfuse 2>/dev/null | grep -q "$USER"; then
    echo "[TMP-CHECK] FAIL: leftover squashfuse mount"
else
    echo "[TMP-CHECK] PASS: no FUSE mounts"
fi
