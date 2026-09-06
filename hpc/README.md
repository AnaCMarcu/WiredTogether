# Cluster launchers

Every experiment in the paper was run on **DAIC** (`daic/`); the other two directories are
launchers kept for the clusters this project also ran on at earlier stages.

| Directory | Runtime | Status |
|---|---|---|
| `daic/` | Apptainer image built from `wiredtogether*.def` | The suite behind the paper |
| `delft_blue/` | conda environment, 8-hour chunked chains (`chain_jobs.sh`) | Earlier generation of the same arms |
| `snellius/` | The DAIC image, rsynced; W&B offline | Smoke test only |

## DAIC

`experiments/_common.sh` does everything shared: picks the Apptainer image, exports `PYTHONPATH`,
`CRAFTIUM_ENV_DIR` and the model paths, masks `/dev/dri` so rendering stays on the CPU, allocates a
per-job Luanti server port from `SLURM_JOB_ID`, cleans the node's `/tmp` on exit, and calls
`multi_agent_craftium.py`. A per-experiment `.sbatch` only sets the arm's flags.

```bash
sbatch hpc/daic/experiments/exp05_mappo_hebbian.sbatch   # one arm, seeds from the array
bash  hpc/daic/experiments/submit_all.sh                 # a whole family
```

Build the image once with `build_image.sbatch` (or `build_image_gemma4*.sbatch` for the Gemma
runs); `download_gemma4.sbatch` fetches the weights.

Two things to know before submitting:

- Use `QOS=long TIME=72:00:00` for the long arms. Successful `pareto_social` runs took 19–35 h, and
  seed 123 needs roughly 20% more LLM calls than any other seed.
- Move a failed run's directory aside before re-running it. `log.txt` and `llm_logs/*.log` are
  append-mode, so a rerun in place doubles the token counts the FLOPs accounting reports.

`experiments/bad_gpu_nodes.txt` + `experiments/gpu_filter.sh` exclude nodes whose GPU is too small for the action-selection
model; `probe_nan.sbatch` is the diagnostic for the NaN-logits failure seen on the largest
checkpoints under two-GPU sharding.
