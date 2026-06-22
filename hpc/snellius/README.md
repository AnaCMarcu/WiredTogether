# Snellius (SURF) setup

Third cluster, alongside `hpc/daic/` (Apptainer) and `hpc/delft_blue/` (conda).
Snellius reuses the **same Apptainer image** as DAIC, so the experiment code is
identical; only the wrapper (`experiments/_common.sh`) differs.

## Confirmed environment (2026-06)
| | value |
|---|---|
| GPU partition | `gpu_a100` (4× A100 / node), max walltime **5 days (120h)** |
| Apptainer | `/usr/bin/apptainer` (no module needed) |
| Node scratch | `$TMPDIR = /scratch-local/<jobid>` — **auto-wiped** by SLURM (no /tmp cleanup needed) |
| Project space | `/projects/prjs1879` → `/gpfs/work2/0/prjs1879` (2.5 PB, use the **real** path in binds) |
| Account | `tdsei14435` (budget `EINF-18147`, shared with oshirekar) |

## One-time setup (on an `int` node)
```bash
WS=/projects/prjs1879
mkdir -p $WS/wiredtogether/{images,models}
DAIC=acmarcu@login.daic.tudelft.nl
DAIC_WS=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu

rsync -avP $DAIC:$DAIC_WS/images/wiredtogether.sif $WS/wiredtogether/images/
rsync -avP $DAIC:$DAIC_WS/models/                  $WS/wiredtogether/models/
cd $WS && git clone https://github.com/AnaCMarcu/WiredTogether.git
cd WiredTogether && git checkout wired_final && mkdir -p slurm_logs
```

## Validate the port (do this BEFORE the full suite)
```bash
cd /projects/prjs1879/WiredTogether
sbatch hpc/snellius/sim_smoke.sbatch
# when it clears (~15 min):
grep -E "python exit|Experiment complete|Traceback" slurm_logs/sim_smoke_*.out
tail -1 runs/smoke/suite_smoke/seed_*/hebbian_snapshots.jsonl   # W should be ~0.2-0.8, varied
```

## Key differences from DAIC
- **wandb runs OFFLINE** — Snellius compute nodes have **no internet**. Real runs
  write `wandb/offline-run-*` under `run_artifacts/.../work_artifacts/`. Sync them
  from an `int` node afterward:
  ```bash
  IMG=/projects/prjs1879/wiredtogether/images/wiredtogether.sif
  apptainer exec $IMG python -m wandb sync \
    /projects/prjs1879/WiredTogether/run_artifacts/final/*/seed_*/work_artifacts/wandb/offline-run-*
  ```
- **Scratch is `$TMPDIR`** (auto-cleaned) — the DAIC `/tmp`-orphan problem does not
  exist here.
- **SBU budget is finite and shared.** An A100-GPU-hour costs SBU; size runs
  deliberately (fewer seeds / a subset) rather than blindly mirroring DAIC's
  `submit_all` at 168h × many jobs. Check remaining budget with `accinfo`.

## TODO
- Per-experiment sbatch files (`experiments/exp01..exp11`) + `submit_all.sh` are
  **not yet ported** — added after `sim_smoke` validates the port, by adapting the
  DAIC ones (swap the SLURM header to `gpu_a100`/`--gpus=1`/120h/`--account`, source
  this `_common.sh`).
