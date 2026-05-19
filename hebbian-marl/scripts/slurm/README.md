# hebbian-marl on the HPC cluster

Cheat sheet for submitting tier-0/tier-5/tier-1 runs to SLURM. Pattern-matched
to WiredTogether's existing `scripts/experiments/_common.sh` + `E5_hebbian.sh`
convention.

## Layout

```
hebbian-marl/scripts/slurm/
├── _common_hebbmarl.sh   # module load + conda activate + HEBBIAN_RESULTS_DIR
├── hebb_tier0.sh         # 5 diagnostic runs, 500k steps each
├── hebb_tier5.sh         # 5 preview runs, 1M steps each
└── hebb_tier1.sh         # 15 headline runs, 5M steps each
```

All three sbatch files:
- partition: `compute` (CPU-only — hebbian-marl uses `use_cuda=False`)
- account: `education-eemcs-msc-dsait`
- 2 CPUs, 8 GB RAM, no GPU
- SLURM logs at `/scratch/$USER/WiredTogether/slurm_logs/hebb-tier{N}-%A_%a.{out,err}`
- Training outputs at `/scratch/$USER/WiredTogether/runs/hebbian-marl/{sacred,bonds,logs,tb_logs}/`

## One-time setup

On the HPC login node:

```bash
# 1. Pull the latest WiredTogether (which now contains hebbian-marl/)
ssh <cluster>
cd /scratch/$USER
git clone <wiredtogether-remote> WiredTogether  # or `git pull` if you already have it

# 2. Install hebbian-marl deps into the existing WiredTogether conda env.
#    The deps are small (lbforaging, gymnasium, sacred, tensorboard).
module load 2025 miniconda3
conda activate /scratch/$USER/.conda/envs/WiredTogether
cd /scratch/$USER/WiredTogether/hebbian-marl
pip install -e ".[dev]"

# 3. Confirm the import works
python -c "from hebbian_module import HebbianConfig, HebbianSocialGraph; print('OK')"
python -c "import lbforaging; print('lbforaging', lbforaging.__version__)"

# 4. Confirm the tests pass on the cluster
pytest tests/ -q
```

## Submission workflow

Always do `--list` first to confirm array bounds:

```bash
cd /scratch/$USER/WiredTogether/hebbian-marl

# Tier 0 — diagnostics (5 runs, ~15 min each)
python scripts/run_experiments.py --tier 0 --list
sbatch --array=0-4 scripts/slurm/hebb_tier0.sh

# Tier 5 — preview (5 runs, ~25 min each)
python scripts/run_experiments.py --tier 5 --list
sbatch --array=0-4 scripts/slurm/hebb_tier5.sh

# Tier 1 — headline grid (15 runs, ~2 h each)
python scripts/run_experiments.py --tier 1 --list
sbatch --array=0-14 scripts/slurm/hebb_tier1.sh
```

Subset submission (e.g., just the hebb_s seeds of tier 1):

```bash
sbatch --array=0-4 scripts/slurm/hebb_tier1.sh
```

Retry only failed/missing tasks (e.g., tasks 7 and 12):

```bash
sbatch --array=7,12 scripts/slurm/hebb_tier1.sh
```

## Monitoring

```bash
# Queue status
squeue -u $USER

# Detailed completion + exit codes
sacct -u $USER -X --format=JobID,JobName,State,ExitCode,Elapsed --starttime today

# Tail one in-flight job's stdout
tail -f /scratch/$USER/WiredTogether/slurm_logs/hebb-tier1-<jobid>_<arrayidx>.out

# Tail the per-run launcher log (more readable than slurm_logs)
tail -f /scratch/$USER/WiredTogether/runs/hebbian-marl/logs/hebb_s_seed0.log

# Launcher status table — what's done, what's pending
python scripts/experiment_status.py --tier 1
```

## Retrieving results

Once all tasks finish, sync results back for local plotting:

```bash
# From your local machine:
rsync -av --progress \
    <user>@<cluster>:/scratch/<user>/WiredTogether/runs/hebbian-marl/ \
    ./WiredTogether/hebbian-marl/results/

# Then locally:
cd WiredTogether/hebbian-marl
python scripts/plot_tier0.py            # for tier 0
python scripts/plot_results.py          # for tier 1 — emits Wilcoxon p-values
```

## Why CPU-only is fine here

hebbian-marl trains feed-forward PPO on Level-Based Foraging — small env,
small networks, no replay buffer to GPU-batch. Throughput is roughly
~800 env-steps/sec on a single CPU thread. Putting it on `gpu-a100` would
waste a GPU node and bottleneck on the queue. The Hebbian module is
numpy-only (no torch dependency).

## Why we don't share `_common.sh` with the WiredTogether experiments

WiredTogether's `scripts/experiments/_common.sh` sets up LLM model paths,
Craftium SDL flags, and `cd` into `src/mindforge/`. None of those apply
here. The local `_common_hebbmarl.sh` keeps the conda+module load pattern
and adds `HEBBIAN_RESULTS_DIR` — nothing more.

## Architecture: how output redirection works

`HEBBIAN_RESULTS_DIR` is read by two places:
1. The launcher ([scripts/run_experiments.py](../run_experiments.py)) reads it
   to set `RESULTS_DIR`, where `runs.jsonl` and `logs/<label>.log` land.
2. [epymarl/src/main.py](../../epymarl/src/main.py) reads it to set
   `results_path`, which is the FileStorageObserver root. It also injects
   `config_dict["local_results_path"] = results_path`, which the
   [HebbianRunner](../../epymarl/src/runners/hebbian_runner.py) reads for
   the `bonds/<label>/seed_<n>.jsonl` path, and which
   [epymarl/src/run.py](../../epymarl/src/run.py) reads for `tb_logs/`.

So setting one env var routes sacred + bonds + launcher logs + tb_logs
into one tree.
