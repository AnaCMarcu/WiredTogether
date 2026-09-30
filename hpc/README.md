# Cluster launchers

Every experiment in the paper ran on a SLURM cluster inside an Apptainer image. `slurm/` holds the
image recipes and one launcher per experimental condition.

`slurm/experiments/_common.sh` does everything shared: picks the image, exports `PYTHONPATH`,
`CRAFTIUM_ENV_DIR` and the model paths, masks `/dev/dri` so rendering stays on the CPU, allocates a
per-job Luanti server port from `SLURM_JOB_ID`, cleans the node's `/tmp` on exit, and calls
`multi_agent_craftium.py`. A per-experiment `.sbatch` only sets the condition's flags.

Set `WT_WORKSPACE` to a directory that holds this repository as `WiredTogether/`, the images in
`images/` and the model weights in `models/`. Then:

```bash
export WT_WORKSPACE=/path/to/workspace
sbatch hpc/slurm/build_image_gemma4.sbatch                  # once; build_image.sbatch for the Qwen image
sbatch hpc/slurm/experiments/exp03_mappo.sbatch             # one condition, one seed (SEED=...)
bash   hpc/slurm/experiments/submit_orchestrator.sh         # a whole family across seeds
```

[../docs/experiments.md](../docs/experiments.md) maps each condition in the paper to its launcher.

Two things to know before submitting:

- Long conditions need a 72 h wall time (`QOS=long TIME=72:00:00`); the deliberation-interval runs
  took 19–35 h.
- Move a failed run's directory aside before re-running it. `log.txt` and `llm_logs/*.log` are
  append-mode, so a rerun in place doubles the token counts the FLOPs accounting reports.

`experiments/bad_gpu_nodes.txt` and `experiments/gpu_filter.sh` exclude nodes whose GPU is too
small for the action-selection model; the list ships empty.

On Snellius (SURF), follow [snellius/README.md](snellius/README.md): an sbatch shim adapts these
same launchers, so nothing in `slurm/` needs editing.
