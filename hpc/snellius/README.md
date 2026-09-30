# Running WiredTogether on Snellius

Snellius runs the same Apptainer image and the same launchers as the rest of `hpc/slurm/`. Nothing
in `hpc/slurm/experiments/` is copied or edited for it. Three pieces adapt the launchers:

| File | What it does |
|---|---|
| `env.sh` | Sets the workspace, image, model and W&B defaults, and puts the sbatch shim on `PATH`. Source it in every login shell. |
| `bin/sbatch` | Rewrites each submission for Snellius. It sets `--partition=gpu_a100 --gpus=1 --cpus-per-task=18`, your `--account`, and caps `--time` at 120 h. It drops `--qos`, `--gres` and `--exclude`, and puts job scratch on `$TMPDIR`. Everything else passes through. |
| `setup.sh` | One-time setup, in four steps: `init`, `image`, `models`, `check`. |
| `sync_wandb.sh` | Uploads offline W&B runs from a login node. |
| `bench_vllm.sbatch` | Optional benchmark: in-process HF `generate()` vs a vLLM server on the same A100. |

Every `sbatch hpc/slurm/experiments/<arm>.sbatch` and every `submit_*.sh` family script works as
documented in [../../docs/experiments.md](../../docs/experiments.md).

## Before you start

- **Snellius access** through SURF (a login and a budget), with your SSH public key uploaded in
  the SURF user portal. Then `ssh <login>@snellius.surf.nl`.
- **A Hugging Face token** with the Gemma 4 licence accepted on
  [huggingface.co/google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it). This is only
  needed to download the weights.
- *(optional)* **A W&B account** if you want the runs on wandb.ai as well as on disk.

`accinfo` shows the budget and `accuse` shows your own usage. One A100 is billed as a quarter
node, roughly 128 SBU per GPU-hour; check SURF's current rates. A full 2,500-step × 5-episode arm
takes about one to four days on one A100. Size a sweep before submitting it.

## Setup (once, on a login node)

Compute nodes have **no internet**, so everything that downloads runs on the login node.

```bash
# 1. Workspace: a directory in your project space. It will hold WiredTogether/, images/ and models/.
export WT=/projects/<project-id>/$USER/wt
mkdir -p $WT && cd $WT
git clone <repository-url> WiredTogether      # the directory MUST be named WiredTogether
cd WiredTogether && git checkout snellius

# 2. Write ~/.wiredtogether_snellius and create the layout.
#    --account defaults to your first SLURM account (see: accinfo).
#    --shared DIR puts images/ and models/ in DIR (as symlinks), so collaborators on the
#    same project share one copy (~30 GB) instead of downloading their own.
bash hpc/snellius/setup.sh init --account <account> [--shared /projects/<project-id>/shared]

# 3. The container image.
#    Preferred: copy the exact .sif the paper used. That is the only way to get identical
#    dependency versions, because the recipe floor-pins and would pull newer packages today.
bash hpc/snellius/setup.sh image gemma4 --from /path/to/wiredtogether_gemma4.sif
#    Otherwise, build it from the recipe (20–60 min, needs apptainer --fakeroot):
bash hpc/snellius/setup.sh image gemma4
#    Both print a sha256 and save it next to the image. Keep it with your results.

# 4. Model weights (Gemma 4 E4B + the sentence encoder; `qwen` or `all` for the Qwen arms).
export HF_TOKEN=hf_...
bash hpc/snellius/setup.sh models gemma4

# 5. Verify. Every line should say ok, and the dry-run should print a gpu_a100 submission.
bash hpc/snellius/setup.sh check
```

Add this to `~/.bashrc` so every login shell is ready:

```bash
source /projects/<project-id>/$USER/wt/WiredTogether/hpc/snellius/env.sh
```

To use W&B, put your key in `~/.netrc` (`chmod 600 ~/.netrc`):

```
machine api.wandb.ai
  login user
  password <your-wandb-key>
```

## First job: a smoke test (~30 GPU-minutes)

Run it before any real run. A Gemma step takes about 45 s with 3 agents, so 20 steps plus about
10 min of warm-up and model loading finish in about 30 min. The 1.5 h limit is only a ceiling;
you pay for the time the job actually runs. Submit from the repository root, because the `slurm_logs/` paths are
relative:

```bash
cd $WT_WORKSPACE/WiredTogether
EPISODES=1 MAX_STEPS=20 RUN_GROUP=smoke WANDB=0 \
    sbatch --time=01:30:00 --job-name=wt-smoke hpc/slurm/experiments/exp01_llm_2b.sbatch

squeue -u $USER                                   # PD = queued, R = running
grep -E "python exit|Traceback|PREFLIGHT" slurm_logs/exp01_llm_2b_*.out
```

It passed if you see `python exit: 0` and there is a `runs/smoke/exp01_llm_2b/seed_123/` directory
with `log.txt` and `episodes/`. Then run `accuse` to see what the job cost.

## Real runs

```bash
# One arm, one seed:
SEED=42 sbatch hpc/slurm/experiments/new_exp_0_gemma.sbatch

# One arm, the three default seeds (42, 123, 456) as an array:
sbatch --array=0-2 hpc/slurm/experiments/new_exp_0_gemma.sbatch

# A whole family, e.g. the communication-budget grid (see the header of each submit script):
bash hpc/slurm/experiments/submit_comm_budget.sh
```

The family scripts skip runs whose `final_metrics.json` already exists, so re-running one after a
failure only resubmits what is missing. Before you submit a family, count its jobs × hours × ~128
SBU.

Results go to `runs/<group>/<arm>/seed_<N>/` (small; this is what `analysis/` reads). Heavy
artifacts go to `run_artifacts/...`. Copy results to your machine with rsync:

```bash
rsync -avP <login>@snellius.surf.nl:/projects/<project-id>/<user>/wt/WiredTogether/runs/<group>/ runs/<group>/
```

To send the W&B logs, run `bash hpc/snellius/sync_wandb.sh [<group>]` on a login node after the jobs
finish.

## Optional: benchmark vLLM against today's inference path

This checks whether serving the model with vLLM would speed up the LLM calls. It costs one job of
about 1 GPU-hour. It uses the same two call shapes as real runs: action selection (about 4,500
prompt tokens plus a frame, 150 out) and a belief update (about 470 in, 55 out). Each shape is timed
for rounds of 1, 3 and 9 agents.

```bash
bash hpc/snellius/setup.sh vllm                 # once, login node, no budget: unpacks the image
bash hpc/snellius/setup.sh vllm-check           # login node, no budget: must print OK
sbatch hpc/snellius/bench_vllm.sbatch
tail -15 slurm_logs/bench_vllm_*.out            # the comparison table
```

`vllm-check` catches the likely failure, a vLLM that does not know Gemma 4, before any GPU time is
spent. In the job, all files are checked first, and the vLLM half runs before the HF baseline. If
the server still fails to start, the job stops within minutes and prints the server log.

## Running LLM-only arms on vLLM (experimental)

`LLM_SERVER=vllm` starts a vLLM server inside the job, on the same GPU, and the agents send it the
same requests they would make to the in-process model (`src/mindforge/agent_modules/remote_model_client.py`):
same prompts, sampling and post-processing, and the same log lines. Only the random draws differ.
It refuses `--rl` arms, which train LoRA on the in-process weights. Agents' calls run concurrently,
so vLLM can batch them, only in a checkout whose CLI has `--llm-batch`. Otherwise they stay
sequential and gain only the faster decode and the prefix cache.

```bash
# 1. no GPU, no budget: the client against a fake server, inside the image on the login node
apptainer exec --bind $WT_WORKSPACE $WT_IMAGE env PYTHONPATH=$WT_WORKSPACE/WiredTogether/src     python hpc/snellius/check_remote_client.py $MODEL_LLM          # must end "N/N checks passed"

# 2. a 20-step in-game run on vLLM (compare with the smoke test's run.log / log.txt)
LLM_SERVER=vllm EPISODES=1 MAX_STEPS=20 RUN_GROUP=smoke_vllm WANDB=0     sbatch --time=01:30:00 --job-name=wt-smoke-vllm hpc/slurm/experiments/exp01_llm_2b.sbatch
```

The server log is written to `runs/<group>/<arm>/seed_<N>/vllm_server.log`. Tunables:
`VLLM_GPU_UTIL` (default 0.80), `VLLM_MAX_MODEL_LEN` (default 16384) and `VLLM_IMAGE`.

## Things that differ from other clusters

- **The walltime limit is 120 h.** The shim caps longer requests. A capped arm that does not finish
  leaves no `final_metrics.json`. Lower `EPISODES` or `MAX_STEPS`, or split the run.
- **W&B runs offline** (`WANDB_MODE=offline`, set by `env.sh`) and the end-of-job upload is skipped
  (`WANDB_AUTOSYNC=0`). Use `sync_wandb.sh` from a login node. To skip W&B entirely, use `WANDB=0`.
- **Scratch is the per-job `$TMPDIR`** (node-local, wiped by SLURM at job end). Nothing is left in
  `/tmp`.
- **Qwen arms** need the Qwen image and weights:
  `WT_IMAGE=$WT_WORKSPACE/images/wiredtogether.sif MODEL_LLM=$WT_WORKSPACE/models/Qwen3.5-2B sbatch ...`
  (after `setup.sh image qwen` and `setup.sh models qwen`).
- **Gemma RL arms** (MAPPO/IPPO with `--rl`) need `RL_UPDATE_STAGGER=1` and 64 GB of memory. Without the stagger,
  runs can hang the Luanti bridge after an update round. `submit_social_replay.sh` sets both; for
  any other Gemma `--rl` arm use `RL_UPDATE_STAGGER=1 sbatch --mem=64GB <file>.sbatch`.
- **Move a failed run's directory aside before re-running it.** `log.txt` and `llm_logs/*.log` are
  append-mode.
- **Check a submission without spending anything:** `WT_SN_DRY_RUN=1 sbatch <file>.sbatch` prints the
  rewritten script and the final `sbatch` command.
- **Overrides:** `WT_SN_PARTITION=gpu_h100` (about 1.5× the SBU per GPU-hour), `WT_SN_MAX_TIME`,
  `WT_SN_CPUS`, and `WT_SN_ACCOUNT` all go through `env.sh`.
