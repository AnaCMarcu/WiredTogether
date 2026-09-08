# Wired Together

**Reward-Modulated Hebbian Social Plasticity for Emergent Social Intelligence in Multi-Agent Systems**

A multi-agent system modelled as an adaptive network: agents are *neurons*, the social bonds
between them are *synapses*, and reward is the *modulatory signal*. A reward-modulated Hebbian rule
learns a directed bond matrix `W(t) ∈ [0,1]^{N×N}` online from co-firing and outcome salience, and
couples it back to behaviour through reward diffusion, weight-gated experience sharing (RL agents)
and a bond-conditioned social module (LLM agents).

Everything runs in **WIRE** (Wired Inter-agent Reasoning Evaluation), a five-chamber cooperative
environment built on Craftium/Luanti and shipped in this repo.

| Path | Contents |
|---|---|
| `src/hebbian/` | The bond matrix `W`: co-firing, the update rule and its variants, reward diffusion, replay indices |
| `src/rl_layer/` | LoRA-PPO over a frozen LLM actor; MAPPO shared critic or IPPO value heads |
| `src/mindforge/` | The per-agent cognitive stack, the episode loop, the environment adapter |
| `src/marl_craftium/` | PettingZoo wrapper over Craftium plus the WIRE world as Lua mods |
| `src/orchestrator/` | Centralised orchestration baselines |
| `analysis/` | Every table and figure in the paper |
| `hpc/` | SLURM launchers, one per experiment arm |
| `docs/` | Code reference, one document per layer |

## Install

The engine is a patched Craftium/Luanti fork that compiles per platform, so it is not vendored
here. The game itself (VoxeLibre) *is* vendored, under `src/marl_craftium/craftium-envs/`.

```bash
git clone https://github.com/AnaCMarcu/craftium_wired_together.git craftium
pip install -e ./craftium          # builds the Luanti binary; see that repo's README
poetry install                     # or: conda env create -f environment.yml
```

Anything that touches the game is Linux-only; the code and the test suite run anywhere.

Point the wrapper at the WIRE world before running. Without it Craftium falls back to its stock
world, which has no chambers and no milestones — the usual cause of a run that reports zero
cooperative progress.

```bash
export CRAFTIUM_ENV_DIR="$PWD/src/marl_craftium/craftium-envs/wire"
export PYTHONPATH="$PWD/src"
export LLM_MODEL_PATH=/path/to/Qwen3.5-2B   # or LLM_BASE_URL for an OpenAI-compatible endpoint
```

## Run

```bash
python src/mindforge/multi_agent_craftium.py \
    --num-agents 3 --episodes 3 --max-steps 100 --simultaneous
```

`--hebbian` turns on the bond matrix and is the prerequisite for every coupling above it
(`--hebbian-mode`, `--hebbian-gamma`, `--hebbian-rho`, `--social-module`); `--rl` adds the PPO
layer over a frozen actor; `--orchestrator` selects the centralised baselines instead. Runs land in
`runs/<run_id>/`.

[docs/configuration.md](docs/configuration.md) lists every flag with its default and the paper
symbol it carries. [docs/experiments.md](docs/experiments.md) maps each condition in the paper to
its launcher under `hpc/`.

## Reproduce

Analysis scripts read run directories under `runs_from_daic/` and write into `paper_assets/`. The
run artifacts are 46 GB and live outside this repo; [docs/dataset.md](docs/dataset.md) covers how
they are grouped and packaged.

```bash
python analysis/make_final_table.py        # Table 2: cross-model comparison
python analysis/make_pareto_social_fig.py  # Figure 3a: deliberation interval vs compute
```

[analysis/README.md](analysis/README.md) maps every script to the table or figure it produces.

## Tests

```bash
python -m pytest tests -q
```

600 tests, no game binary or model weights required. They pin the hyperparameter defaults against
the paper's tables, the Hebbian update arithmetic, the reward ledger, and the Lua and Python
milestone tables against each other.

MIT licensed. Built on [Craftium](https://github.com/mikelma/craftium),
[VoxeLibre](https://git.minetest.land/VoxeLibre/VoxeLibre) and the MindForge agent stack.
