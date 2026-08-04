# 11 — Operations: Local Runs, HPC, Artifacts, Testing

**Source files:** `README.md`, `hpc/daic/` (`experiments/*.sbatch`, `experiments/_common.sh`, `wiredtogether.def`, `build_image.sbatch`, `sim_smoke.sbatch`), `hpc/delft_blue/` (`experiments/*.sh`, `chain_jobs.sh`), `hpc/snellius/` (`README.md`, `experiments/_common.sh`, `sim_smoke.sbatch`), `src/mindforge/run_layout.py`, `src/mindforge/multi_agent_craftium.py` (CLI/checkpoint args), `src/marl_craftium/openworld_multi_agents.py`, `make_results.py`, `pyproject.toml`
**Paper sections:** Experimental setup / conditions table; Appendix on compute
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

## 1. Local quickstart

| Step | Command | Notes |
|---|---|---|
| Engine fork | `git clone https://github.com/AnaCMarcu/craftium_wired_together.git craftium && pip install -e ./craftium` | Builds the Luanti binary; not vendored (README.md:99-107) |
| Python env | `poetry install` or `conda env create -f environment.yml` | Both lockfiles exist at repo root (README.md:111-114) |
| Smoke run | `cd src/mindforge && PYTHONPATH=../ python multi_agent_craftium.py --num-agents 3 --episodes 3 --max-steps 100 --warmup-time 60 --team-mode homogeneous-agent` | README.md:118-123 |
| Full run | add `--rl --rl-critic-mode centralized --rl-model-path <Qwen> --hebbian --hebbian-gamma 0.2` | README.md:127-133; flag glossary README.md:135-146 |

`PYTHONPATH` must include `src/` (HPC scripts export `PYTHONPATH=$REPO/src`; the README smoke uses `../` from inside `src/mindforge/`). **Windows cannot run the game** — craftium's multi-agent server (`MTServerOnly`/`MTClientOnly`) is Linux-only, so local work on this machine is code + unit tests only; anything touching the env runs on a cluster (tests needing the binary are marked `needs_game`, see §5).

## 2. HPC

### Clusters

| Cluster | Dir | Runtime | Scheduler config | Differs |
|---|---|---|---|---|
| DAIC | `hpc/daic/` | Apptainer `wiredtogether.sif` | `--partition=general --qos=long --time=168:00:00 --gres=gpu:1 --mem=32GB --exclude=cor1` | Container-first; /tmp cleanup mandated by HPC support; wandb online via `~/.netrc` |
| DelftBlue | `hpc/delft_blue/` | `module load 2025 + miniconda3` + conda env at `/scratch/acmarcu/.conda/envs/WiredTogether` (`experiments/_common.sh:15-24`) | account `education-eemcs-msc-dsait` | No container; older `.sh` experiment suite (exp0–exp15 incl. roles exp14/15 and social-bias exp10/12); `chain_jobs.sh` for 8h chunked chains |
| Snellius (SURF) | `hpc/snellius/` | **Same** Apptainer image as DAIC, rsynced over | `gpu_a100`, max 120h, account `tdsei14435` (`README.md:7-14`) | wandb **offline** (no internet on compute nodes; sync from int node); `$TMPDIR=/scratch-local/<jobid>` auto-wiped, so no /tmp problem; only `sim_smoke` ported, exp01–exp11 TODO (`hpc/snellius/README.md:53-57`) |

### DAIC final experiment matrix (`hpc/daic/experiments/exp01..exp11.sbatch`)

All: 3 agents, `RUN_GROUP=final`, `EPISODES=5`, `MAX_STEPS=2500`, `--simultaneous`, wandb project `final_wired_together`, seeds 42/123/456 via SLURM array. Plastic-Hebbian flags map to paper symbols: `--hebbian-eta-0 0.005` (eta_0), `--hebbian-reward-norm 50` (R), `--hebbian-decay 0.005` (homeostatic lambda for these runs), `--hebbian-gamma 0.2` (gamma_d). RL flags: `--rl-update-interval 64 --rl-lr 3e-4`.

| Exp | Condition | Model | Key flags beyond common |
|---|---|---|---|
| exp01_llm_2b | LLM-only baseline | 2B | — |
| exp02_llm_9b | LLM-only, model-size companion | 9B | — |
| exp03_mappo | MAPPO (shared centralized critic) | 2B | `--rl --rl-critic-mode centralized` |
| exp04_ippo | IPPO (per-agent critics) | 2B | `--rl --rl-critic-mode independent` |
| exp05_mappo_hebbian | **Headline**: MAPPO + plastic Hebbian | 2B | exp03 + hebbian flags above |
| exp06_ippo_hebbian | IPPO + plastic Hebbian | 2B | exp04 + hebbian flags |
| exp07_llm_2b_social_prompt | LLM + Hebbian + prompt coupling (no RL) | 2B | hebbian flags + `--social-module prompt` |
| exp08_llm_9b_social_prompt | 9B companion to exp07 | 9B | same |
| exp09_llm_9b_allied_all | Frozen topology: all allied, W=0.8 off-diag | 9B | `--hebbian-freeze --hebbian-preset uniform --hebbian-bond-strong 0.8 --hebbian-gamma 0.0 --social-module bias` |
| exp10_llm_9b_allied_pair | Frozen: agents 0–1 bonded 0.8, agent 2 excluded (`WEAK=0.0` env override) | 9B | `--hebbian-preset pair --hebbian-bond-weak $WEAK`, rest as exp09 |
| exp11_llm_9b_allied_none | Null control: W ≡ 0, same code path as exp09/10 | 9B | `--hebbian-preset uniform --hebbian-bond-strong 0.0`, rest as exp09 |

Submission: `bash hpc/daic/experiments/submit_all.sh` (filters `ONLY=`/`SKIP=`, `N_SEEDS=1..3` → job array over `SEEDS=(42 123 456)`, `DRY_RUN=1` preview) — `submit_all.sh:52-64,103-132`. Single re-run: `SEED=123 sbatch hpc/daic/experiments/exp05_mappo_hebbian.sbatch`. Pre-suite validation: `hpc/daic/sim_smoke.sbatch` mirrors exp05 at 2×150 steps under `RUN_GROUP=smoke` and prints how to read the final W snapshot (want differentiated ~0.2–0.8 off-diagonals, not saturated ~0.95 or pruned ~0).

### `run_exp` contract (`hpc/daic/experiments/_common.sh:51-311`)

`run_exp <EXP_NAME> <LLM_MODEL_PATH> [extra python args...]` — every sbatch is just env overrides + one `run_exp` call appending condition flags.

| Aspect | Mechanism | Anchor |
|---|---|---|
| Output trees | results → `runs/$RUN_GROUP/<exp>/seed_<N>/`; heavy artifacts (debug.txt, intermediate gifs, offline wandb) → parallel `run_artifacts/.../work_artifacts/` via post-run rsync salvage | `_common.sh:59-65,262-283` |
| Scratch + /tmp cleanup | `WORK_DIR=/tmp/$USER/<exp>_<jobid>` + scoped `APPTAINER_TMPDIR`/`TMPDIR`; `trap "rm -rf ..." EXIT INT TERM` removes both on any exit (HPC-support requirement; on timeout the trap deletes WORK_DIR **before** salvage — those artifacts are lost by design) | `_common.sh:66-87` |
| Node sharing | node-wide `pkill minetest/luanti` pre-flight intentionally removed (killed sibling jobs); safe because each job gets a **unique `mt_server_port` = 49152 + SLURM_JOB_ID % 16000** (random fallback off-SLURM) | `_common.sh:89-100`; `openworld_multi_agents.py:112-117` |
| Headless rendering | empty dir bind-mounted **over** `/dev/dri` (DAIC GPUs are compute-only; Mesa otherwise refuses software fallback) + `LIBGL_ALWAYS_SOFTWARE=1`/`GALLIUM_DRIVER=llvmpipe`/`EGL_PLATFORM=surfaceless` + xvfb-run (or manual Xvfb when xauth missing) wrapper | `_common.sh:134-247` |
| Process reaping | `apptainer exec --pid` so python is pid 1 — kernel reaps leftover MT/Xvfb procs, preventing the FUSE-orphan hang | `_common.sh:166-175` |
| Model/env exports | `LLM_MODEL_PATH` (→ mismatch context below), `ST_MODEL_NAME`/`SENTENCE_TRANSFORMERS_HOME` (local MiniLM), `HF_HUB_OFFLINE=1`, `CRAFTIUM_ENV_DIR=$REPO/src/marl_craftium/craftium-envs/five-chambers`, `WIREDTOGETHER_RUNS_ROOT=$REPO/runs`, `WIREDTOGETHER_RUN_GROUP=$RUN_GROUP` (the shell's suite name, so python writes to the same tree the shell reports), `PYTHONPATH=$REPO/src` | `_common.sh:181-208` |
| Fixed python args | `--num-agents 3 --warmup-time 300 --seed $SEED --experiment-id/--tag $EXP_NAME`, `${EPISODES:-3}`/`${MAX_STEPS:-1000}` env-overridable | `_common.sh:248-258` |
| wandb | on by default (`WANDB=1`), auth via `~/.netrc`; tags auto = `exp_<name>,seed_<N>` + `WANDB_EXTRA_TAGS`; run id = `<group>_<exp>_seed_<N>` (bare `<exp>_seed_<N>` for `legacy`, preserving pre-grouping ids) so a new `RUN_GROUP` starts a **new** wandb run instead of `resume="allow"`-ing the previous suite's, and the group is also set as the wandb Group field; offline-run dirs auto-`wandb sync`ed post-job | `_common.sh:28-48,104-116,295-302` |

> PAPER MISMATCH — base model is config-driven via `LLM_MODEL_PATH` (Qwen3.5-2B/9B on HPC), see PAPER_INCONSISTENCIES.md #8.

### Apptainer image

`sbatch hpc/daic/build_image.sbatch` → `apptainer build --fakeroot` from `hpc/daic/wiredtogether.def` into `$WORKSPACE/images/wiredtogether.sif`. The def file (python:3.12-slim base): xvfb+xauth, CUDA-12.1 torch, the craftium **cp312 wheel** (bundles the luanti binary) with two wheel files overwritten from the fork's `daic-changes` branch (`minetest.py`, `multiagent_env.py` — the wheel's `_create_mt_dirs` lacks the `isfile()` branch), then the pyproject dependency floors with `gymnasium==0.29.1` pinned by craftium (`wiredtogether.def:55-103`). Snellius reuses the same `.sif` (rsync, `hpc/snellius/README.md:16-27`).

## 3. Chunked-job resume

| Piece | Behavior | Anchor |
|---|---|---|
| Cadence | `--checkpoint-interval 500` (default) saves every N in-episode steps to `runs/<id>/checkpoints/step_NNNNNN/` (`run_state.json` + hebbian/curricula/rl subdirs) | `multi_agent_craftium.py:299-313,632-703,2419` |
| Resume | `--resume <checkpoint_dir>` restores cognitive/RL/Hebbian state and continues from saved episode (episode restarts at step 0; `resume_step` currently unused); `--resume-skip-warmup` skips media-load detection when the VoxeLibre cache is warm; skill DBs are preserved on resume, wiped on fresh runs | `multi_agent_craftium.py:305-310,417-471,1143-1166` |
| wandb identity | run id defaults to the sanitised run_id with `resume='allow'`, so chunked SLURM jobs land in **one** W&B run; `--wandb-id` overrides explicitly | `multi_agent_craftium.py:120-123,875-882` |
| Chaining | DelftBlue `hpc/delft_blue/chain_jobs.sh`: 1 first + N continuation jobs via `--dependency=afterany`, each 8h chunk checkpoints and the next resumes | `chain_jobs.sh:34-52` |

## 4. Artifacts

- **`runs/<run_id>/` layout** — single source of truth is the `run_layout.py` module docstring (`src/mindforge/run_layout.py:1-22`): `config.json`, `log.txt`, `episodes/ep_NNNN/{step_log,event_log,messages}.jsonl + summary.json`, `checkpoints/`, `plots/`, `hebbian_snapshots.jsonl` (one episode-end W per line), `final_metrics.json`, `final_summary.txt`. Consumed run-side as described in 07-orchestrator.md and 09-metrics-and-evaluation.md. HPC runs land under `runs/<RUN_GROUP>/<exp>/seed_<N>/` (the legacy tagged layout is `RunPaths.create_tagged`, `run_layout.py:113-155`).
- **`runs_from_daic/legacy/`** — pulled results of the 9-run legacy sweep: `ANALYSIS_REPORT.md` plus per-experiment dirs (`exp1_llm_2B`, `exp3_mappo`, `exp4_mappo_hebbian`, `exp6_ippo_hebbian`, `exp7_llm_9b`, `exp9_llm_2b_social_prompt`, `exp11_llm_9b_social_prompt`, `exp16_llm_9b_allied_all`, `exp17_llm_9b_allied_pair`, `opt_test_7`, `suite_smoke`). Naming follows the **old** DelftBlue-era numbering, not the final exp01–exp11 matrix above.
- **Paper assets** — `python make_results.py --runs-root runs/final --out paper_assets` reads each condition's `final_metrics.json` and emits `table_rows.tex` (tab:main_results, tab:steps_to_milestone, tab:topology_ablation, tab:graph_stats), `summary.csv`, and `milestone_progression.pdf` / `milestone_timeline.pdf` / `bond_evolution.pdf` (`make_results.py:1-35`); the condition→dir registry is the `CONDITIONS` table at the top of the script. Aggregation conventions (coop milestones = distinct Ch2–Ch5 team milestones; task return excludes `hebbian_diffuse`) are documented in its docstring — see 09-metrics-and-evaluation.md.

## 5. Testing

Suite under `tests/` (13 modules + `conftest.py`); scope and provenance in `tests/README.md`. `pyproject.toml:67-79` sets `testpaths = ["tests"]` and `addopts = "-m 'not needs_game and not needs_llm'"`, so a plain `pytest` runs only what works without the Luanti binary or model weights (i.e. everything runnable on this Windows machine); markers: `slow`, `needs_game`, `needs_llm`. The `testpaths`-without-tests repo-hygiene gap is PAPER_INCONSISTENCIES.md #14 — now FIXED by this suite, which also surfaced the two post-commit source fixes named in the header. Hyperparameter defaults are pinned by `tests/test_paper_defaults.py`; component-level pins are cross-referenced from each sibling doc (02-hebbian-graph.md, 03-rl-layer.md, 06-rewards.md).
