# AgentWorld runs

MindForge (+ Hebbian) agents in [AgentWorld](https://github.com/openagents-org/agentworld)
(a Kaetram 2D MMORPG benchmark), driven by our own harness in `src/agentworld/`.
The WIRE/Craftium stack is untouched (`tests/test_aw_freeze.py` guards it).

**Status (2026-10-03): built and tested offline only.** Every piece runs
against an in-memory fake server (`tests/fake_kaetram.py`) and a scripted
LLM. Nothing here has run against the real game server or on a cluster yet.
P0 below is the next step.

## Layout

| Path | What |
|---|---|
| `src/agentworld/` | harness: tasks, executor (their `KaetramGameTools`, unmodified), scheduler, comms (DMs + board + gated reading), progress reward, verifier wrapper, MindForge agent, replay videos, CLI |
| `src/hebbian/multichannel.py` | three-factor rule over channels spat / comm / xfer / combat (+ read); parity-tested against `graph.py` |
| `src/mindforge/multi_agent_agentworld.py` | entry point |
| `analysis/agentworld/` | `calibrate_rule.py`, `make_bond_recovery.py`, `make_aw_progress_curves.py` |
| `hpc/agentworld/` | this folder: fetch script, server image, job scripts |

## Local run (Windows or Linux, needs Node 20 + Yarn 4)

```bash
git clone https://github.com/openagents-org/agentworld ../agentworld
git -C ../agentworld checkout $(cat hpc/agentworld/AGENTWORLD_COMMIT)
(cd ../agentworld && corepack enable && HUSKY=0 yarn install && cp .env.defaults .env)
export AGENTWORLD_ROOT=$(realpath ../agentworld)
# API anchor (any OpenAI-compatible endpoint), small world:
export LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_MODEL=anthropic/claude-haiku-4.5
PYTHONPATH=src python -m mindforge.multi_agent_agentworld \
    --tasks task_02_arrow_production --arm hebbian --episodes 1 \
    --max-in-flight 8 --out runs_aw/local/smoke
```

The harness starts the server itself (`yarn workspace @kaetram/server start`) on
ports from `agentworld.server.job_ports`, restarts it between episodes, writes
`episode_k/{trajectories,messages.jsonl,events.jsonl,replay/state.jsonl,
hebbian_W.npy,summary.json,videos/*.mp4}`.

Videos for any episode afterwards:

```bash
PYTHONPATH=src python -m agentworld.replay runs_aw/local/smoke/episode_1 --team t00
PYTHONPATH=src python -m agentworld.replay runs_aw/local/smoke/episode_1 --agent 2 --reader-view
PYTHONPATH=src python -m agentworld.replay runs_aw/.../episode_1 --world
```

## Snellius: first test job

Needs the existing Snellius setup (`hpc/snellius/README.md`): the workspace,
`images/wiredtogether_gemma4.sif`, `images/vllm`, `models/gemma-4-E4B-it` and
`models/all-MiniLM-L6-v2`.

```bash
# login node
source $WT_WORKSPACE/WiredTogether/hpc/snellius/env.sh
cd $WT_WORKSPACE/WiredTogether
git fetch origin && git checkout agentworld && git pull

# once, login node, no budget (~5–10 min): AgentWorld at the pinned commit,
# a node:20-bookworm sandbox, yarn install, and a 3-minute server boot check.
bash hpc/agentworld/setup_aw.sh all          # must end with "OK: game server answered"

# the smoke job: 3 agents, task_01_magic_staff, Hebbian arm, 15 rounds, video (~1 GPU-hour max)
sbatch hpc/agentworld/aw_smoke.sbatch
squeue -u $USER
tail -f slurm_logs/aw_smoke_*.out
```

**If the main checkout is busy on another branch** (jobs running from it), run
AgentWorld from a second checkout instead of switching branches under those jobs:

```bash
cd $WT_WORKSPACE/WiredTogether && git fetch origin
git worktree add $WT_WORKSPACE/WiredTogether-agentworld agentworld
cd $WT_WORKSPACE/WiredTogether-agentworld && git pull && mkdir -p slurm_logs
source hpc/snellius/env.sh
export WT_REPO=$PWD                         # the AgentWorld scripts run from this checkout
type -a sbatch                              # exactly ONE .../hpc/snellius/bin/sbatch, then the real one
sbatch hpc/agentworld/aw_smoke.sbatch       # the shim exports WT_REPO into the job
```

env.sh puts the MAIN checkout's sbatch shim on PATH. Only if `type -a sbatch`
shows no shim at all (the main checkout is on a branch without
hpc/snellius/bin) add this worktree's: `export PATH="$PWD/hpc/snellius/bin:$PATH"`.
Never have two shims on PATH: each takes the other for the real sbatch and they
recurse until mktemp fails with "File name too long" (nothing is submitted).

Passed if the log ends with `python exit: 0` and
`runs_aw/smoke/hebbian_task_01_magic_staff_x1/seed_42/episode_1/` holds
`summary.json`, `trajectories/`, `replay/state.jsonl`, `hebbian_W.npy` and
`videos/team_t00.mp4`. `server.log` and `vllm_server.log` sit next to `run.log`.
Overrides: `TASK=... ARM=base ROUNDS=30 REPLICAS=4 sbatch hpc/agentworld/aw_smoke.sbatch`.

## Sweeps

```bash
TASK=task_01_magic_staff bash hpc/agentworld/submit_aw_scaling.sh   # from the repo root
```

`aw_common.sh` starts vLLM (prefix caching on), supervises the game server on
the host (restarts it when the harness touches the restart file), and runs the
harness in the Gemma image with `LLM_BACKEND=vllm`. The game-server runtime is
`images/node20` + the checkout's `node_modules` (`setup_aw.sh`), or a
self-contained `images/agentworld.sif` built from `agentworld.def` where
`apptainer build --fakeroot` works.

## P0 checklist (before any real experiment)

1. Server starts from the image; `PORT` / `API_PORT` / `SKIP_DATABASE` env vars
   override `.env`; startup and restart times.
2. Run their own runner on task 02 with the API model; keep the trajectory as
   `tests/fixtures/agentworld/golden_traj.json` and check our trajectories
   score identically (`verify()`), including the radius-1 post-action observation.
3. Spawn points: about 1 in 6 task spawns sit on empty map cells (task 02's
   (388, 3) among them). Where do teleports there actually land?
4. `team_spacing` offsets land on walkable tiles for the tasks used in E2.
5. 100 scripted bots × observe + act: Node thread saturation, observe latency
   vs radius.
6. Attack / collect synchronous or not; death semantics (items kept? respawn?).
7. Live capture: can a web-client login follow agents; do mobs attack it;
   does headless WebGL render the tilesets (else record locally).
8. vLLM throughput for Gemma-4-E4B at concurrency 1/32/128 (`hpc/snellius/bench_vllm.py`).
9. N=10 pilot: bonds differentiate with the calibrated rates
   (`analysis/agentworld/calibrate_rule.py` defaults).
