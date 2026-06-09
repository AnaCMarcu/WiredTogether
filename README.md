# WiredTogether

Hebbian Social Learning integrated over reinforcement learning with MindForge based agents in a multi-agent cooperative Craftium environment.

Agents act in **Five Chambers**, a hand-designed 5-room cooperative dungeon
where progress requires both individual skill (Ch1) and inter-agent coordination
(Ch2 anvil breaking, Ch3 switch puzzle, Ch4 zombie combat, Ch5 boss fight).

The agents combine:

- a **MindForge** LLM-driven cognitive stack (perception, beliefs, auto-curriculum, critic, skill memory),
- a **modular RL layer** with PPO + LoRA fine-tuning over discrete + macro actions,
- **MAPPO** with a shared centralised critic on a joint state vector (compact features + sentence-transformer embeddings of the agents' last actions/messages),
- a **Hebbian social-plasticity graph** that learns inter-agent bonds from spatial co-activity, communication, and shared advantage, and uses them for reward diffusion + social replay.

---

## Project layout

```
src/
├── rl_layer/                  Modular RL layer (PPO actor + optional centralised critic)
│   ├── config.py
│   ├── heads.py               ActionHead, ValueHead, RunningMeanStd
│   ├── trajectory_buffer.py   Per-agent rollout buffer + GAE
│   ├── ippo.py                Per-agent PPO step (policy + optional value loss)
│   ├── ppo_update.py          Update orchestration (GAE + social replay + PPO loop)
│   ├── token_opt.py           Agent-decided token-level fine-tuning
│   ├── persistence.py         Save / load checkpoint helpers
│   ├── centralized_critic.py  Shared MAPPO critic (joint state → V)
│   └── rl_layer.py            RLLayer orchestrator (~350 lines, delegates to the above)
│
├── hebbian/                   Social-plasticity graph (independent of rl_layer)
│   ├── config.py
│   └── graph.py
│
├── mindforge/                 LLM-driven agent stack + main loop
│   ├── multi_agent_craftium.py    Entry point
│   ├── custom_agent.py
│   ├── custom_environment_craftium.py
│   ├── env/                       Env-side utilities
│   │   ├── communication_rewards.py
│   │   ├── cooperation_metric.py
│   │   └── episode_logger.py
│   ├── agent_modules/             LLM-side: actions, beliefs, curriculum, critic, …
│   └── prompts/                   Text templates
│
└── marl_craftium/             Patched MARL Craftium env wrapper + five-chambers env
    └── craftium-envs/five-chambers/    Bedrock-walled dungeon (Lua mods + VoxeLibre game)
```

---

## Quick start

### 0. Get the Craftium engine fork

The five-chambers env runs on a **patched Craftium/Luanti fork** that is *not* vendored in this
repository (it is large and built per-platform). Clone it into `craftium/` at the repo root —
the env wrapper (`src/marl_craftium/_bootstrap.py`) expects it there and puts it on `sys.path`
so `import craftium` resolves:

```bash
git clone https://github.com/AnaCMarcu/craftium_wired_together.git craftium
# build / install per that repo's README (it compiles the Luanti engine), e.g.:
#   pip install -e ./craftium
```

The five-chambers world itself (Lua mods + the VoxeLibre game) **is** shipped here under
`src/marl_craftium/craftium-envs/five-chambers/`; only the engine fork is external.

### 1. Python environment + smoke test

```bash
# Conda / Poetry env from pyproject.toml
poetry install
# OR: conda env create -f environment.yml && conda activate wiredtogether

# Run a tiny smoke test (3 agents, 3 episodes, 100 steps, no RL)
cd src/mindforge
PYTHONPATH=../ python multi_agent_craftium.py \
    --num-agents 3 --episodes 3 --max-steps 100 \
    --warmup-time 60 --team-mode homogeneous-agent
```

### HPC (DelftBlue SLURM)


| Script | What it runs |
|---|---|
| `small.sh` | 3 agents, 3 episodes, 100 steps — quick sanity check |
| `test.sh` | 6 agents, 3 episodes, 200 steps |
| `rl.sh` | 3 agents + RL (action-mode MAPPO) |
| `rl_hebbian.sh` | RL + Hebbian social plasticity |
| `rl_token_opt.sh` | RL + agent-decided token-level fine-tuning |
| `rl_hebbian_survival.sh` | RL + Hebbian + phased survival difficulty |
| `run_first.sh` / `run_continue.sh` | Long runs split across SLURM jobs (8h each) |
| `mindforge_slurm.sh` | Generic launcher |

All scripts:

- `cd "$PROJECT_DIR"` first
- `export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"` so `rl_layer`, `hebbian`, and `mindforge.env` resolve as top-level packages
- `cd src/mindforge` then `python multi_agent_craftium.py …`

Submit with e.g. `sbatch scripts/small.sh`.

---

## CLI flags worth knowing

```text
--num-agents 3                      # team size (homogeneous role only)
--episodes 3 --max-steps 100        # episode count and per-episode horizon

--no-communication                  # disable inter-agent chat (default ON, targeted 1-to-1)
--belief-interval 5                 # refresh perception/partner/interaction beliefs every N steps
--critic-interval 20                # run task-success critic every N steps

# RL layer
--rl                                # enable MAPPO action-level training
--rl-model-path /path/to/Qwen3.5-9B
--rl-mode action|token              # action: discrete head; token: token-level RLHF
--rl-critic-mode centralized|independent   # default centralized (MAPPO)
--rl-update-interval 256
--rl-lr 1e-4
--rl-auto-token-opt                 # let agents self-trigger token-level fine-tuning

# Hebbian
--hebbian                           # enable social-plasticity graph
--hebbian-radius 5.0                # interaction radius (world units)
--hebbian-rho 0.3                   # social-replay blend factor
--hebbian-gamma 0.2                 # reward-diffusion strength

# Phased difficulty (optional)
--survival-mode --survival-episode 3

# Checkpointing
--checkpoint-dir checkpoints/<id>
--checkpoint-interval 500
--resume <path>
--resume-skip-warmup
```

---

## Five Chambers — environment overview

| Chamber | Mechanic | Milestones |
|---|---|---|
| **Ch1** | Solo learning (12×12 grass room, north door always open) | M1 move, M2 dig 3 blocks, M3 pickup 3 items, M4 dig 5 wood, M5/M6 kill 1/2 animals, M7 dig 3 stone |
| **Ch2** | Cooperative anvil breaking (≥2 simultaneous diggers) | M8–M13 anvil breaks, M14 sword wielded, M15 chestplate equipped — drops `mcl_tools:sword_diamond` and `mcl_armor:chestplate_diamond` |
| **Ch3** | Switch puzzle (3 sealed cells; switch in cell A opens cell B's door, B→C, C→A) | M16 enter cell, M17 switch pressed, M18 door opened, M19 all in communal room |
| **Ch4** | Combat — 3 zombies | M20 enter, M21/M22 first/all kills, M23 all-alive bonus |
| **Ch5** | Boss fight — single 60 HP zombie | M24 enter, M25 first damage, M26 boss half-HP, M27 defeated, M28 all-alive bonus |
| **Comm** | Targeted 1-to-1 chat rewarded in every chamber | `m_comm_ch1..ch5` (Ch1 has the highest milestone bonus to bootstrap chat) |

The Lua side (`src/marl_craftium/craftium-envs/five-chambers/mods/five_chambers/`) builds
the world with VoxelManip, tracks milestones, writes JSONL events that Python polls.

---

## Checkpoint / resume

Every episode end and every `--checkpoint-interval` steps within an episode, the
run saves its full state. A new SLURM job can restore that state and continue
without re-running completed episodes.

What is saved:

| File | Contents |
|---|---|
| `run_state.json` | Episode/step counter, metric history, CLI args |
| `hebbian_graph.json` | Hebbian weight matrix, config |
| `rl_agent_{i}/` | LoRA adapter, action/value head weights, optimizer, RMS, counters |
| `agent_{i}_curriculum.json` | Current task, completed/failed task lists |
| `frames_{i}.npy` | Raw observation frames *(optional, `--checkpoint-frames`)* |

The Craftium server is **not** checkpointed — it does a fresh `reset()` on resume.

Layout:

```
checkpoints/<run_id>/
├── latest_checkpoint.txt      ← path to most recent ep*_end dir
├── ep1_end/
│   ├── run_state.json
│   ├── hebbian_graph.json
│   ├── rl_agent_0/  rl_agent_1/  rl_agent_2/
│   ├── agent_0_curriculum.json  …
│   └── (optional) frames_*.npy
├── ep1_step500/  ep1_step1000/  …
└── ep2_end/  …
```

`run_continue.sh` reads `latest_checkpoint.txt` to know where to resume.

**Graceful shutdown.** The run hooks `SIGTERM`/`SIGINT`. When SLURM sends
`SIGTERM` at the end of the time limit, the current step completes and a
`*_shutdown` checkpoint is written before the process exits. The next job
picks it up via `run_continue.sh`. Trigger manually with
`scancel --signal=SIGTERM <job_id>`.

**Chained jobs.** `bash scripts/chain_jobs.sh 3` submits 1 fresh job + 3
continuations (`--dependency=afterany`), giving up to ~32 hours of compute.

---

## RL design at a glance

```
                       ┌── Hebbian graph ──┐
                       │  W[N,N] bonds     │
                       │  reward diffusion │
                       └────────┬──────────┘
                                │
  per agent:                    │ shaped per-agent rewards
  ┌─────────────────────────┐   │
  │ LLM forward → pooled h  │───┼──→ ActionHead → π(a|s)  (per-agent)
  │ + LoRA adapter          │   │     ValueHead → V(s)     (IPPO baseline)
  └─────────────────────────┘   │
                                │
  shared:                       │
  ┌─────────────────────────┐   │
  │ joint state             │   │
  │ ├ compact features      │───┴──→ CentralizedCritic → V_global   (MAPPO)
  │ └ sentence-encoded      │
  │   last action + comm    │
  └─────────────────────────┘
```

**Centralised vs independent critic.** Default is MAPPO (`--rl-critic-mode
centralized`): one shared `V(joint_state)` across all agents. The legacy IPPO
mode (`--rl-critic-mode independent`) restores the per-agent `value_head` —
useful as an ablation/regression guard.

**Hebbian effects on PPO.** The graph diffuses per-agent rewards before they
land in each agent's buffer, and weights social replay during the PPO update so
strongly-bonded teammates' transitions are mixed into each mini-batch.

---

## Reproducing the experiments

The paper's experiment campaigns are SLURM scripts under `scripts/experiments/`. Each sources
`_common.sh` (conda activation, model paths, seed array) and launches `multi_agent_craftium.py`
with a fixed configuration. Submit one campaign across all seeds with `submit_all.sh`, or an
individual experiment with `sbatch scripts/experiments/<name>.sh`.

| Group | Scripts | Configuration |
|---|---|---|
| LLM baselines | `exp1_llm.sh`, `exp7_llm_9b.sh` | LLM agents only (2B / 9B), no RL, no Hebbian |
| LLM + Hebbian | `exp2_llm_hebbian.sh`, `exp8_llm_9b_hebbian.sh` | LLM agents + social-plasticity graph |
| RL | `exp3_mappo.sh`, `exp5_ippo.sh` | MAPPO (centralised critic) / IPPO (per-agent critic) |
| RL + Hebbian | `exp4_mappo_hebbian.sh`, `exp6_ippo_hebbian.sh` | RL + social plasticity |
| Social prompting | `exp9_…_social_prompt.sh`, `exp10/12_…_social_bias.sh`, `exp11_…` | Prompt-/bias-level social signals (2B / 9B) |
| Heterogeneous roles | `exp14_llm_hebbian_roles.sh`, `exp15_llm_roles.sh` | Role-specialised teams (harvester / hunter / scouter) |

`quick_test.sh` is a fast end-to-end check. Post-hoc analysis lives in `scripts/analysis/`
(`analyze_runs.py`, `aggregate_seeds.py`, and the `tier*_analysis.py` plotters); their figures
are regenerated locally and are not committed.

## Tests

```bash
pytest tests/        # pure-python unit tests (Hebbian graph rules, import contracts) — no LLM/engine needed
```

Design notes and derivations live in `docs/` (architecture, RL approaches, Hebbian integration,
five-chambers spec).

## Citation

```bibtex
@software{marcu_wiredtogether,
  author  = {Marcu, Ana Cristiana},
  title   = {WiredTogether: Hebbian Social Learning over Multi-Agent Reinforcement Learning},
  year    = {2026},
  url     = {https://github.com/AnaCMarcu/WiredTogether},
  license = {MIT}
}
```

Released under the [MIT License](LICENSE).

---

## Acknowledgments

- **[Craftium](https://github.com/mikelma/craftium)** — Minetest/Luanti-based RL platform; the
  five-chambers env builds on a [patched fork](https://github.com/AnaCMarcu/craftium_wired_together).
- **VoxeLibre** (formerly MineClone2) — the VoxeLibre game ships as a vendored
  asset for the five-chambers env.
- **MindForge** — the LLM-agent reasoning stack this work extends.
