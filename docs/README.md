# WiredTogether — Code Documentation Index

**Source files:** `docs/` (this index); per-doc source coverage listed below
**Paper sections:** all — each doc names its own sections/tables/equations
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

## 1. What this repo is

Thesis codebase for *"Wired Together: Reward-Modulated Hebbian Social Plasticity for Emergent Social Intelligence in Multi-Agent Systems"* (Ana Cristiana Marcu, TU Delft). Three layers on top of a Craftium/Luanti five-chamber world: an LLM cognitive stack per agent (`src/mindforge`), an optional LoRA-PPO RL layer (`src/rl_layer`), and the thesis contribution — a numpy-only Hebbian social graph W (`src/hebbian`) coupled to the system via reward diffusion (gamma_d) and a social prompt module. The LaTeX paper lives in `paper/` (`main.tex` + `sections/`). These docs are both (a) the author's code reference and (b) the verification record backing the conference submission.

## 2. Doc map

| Doc | Component | Source files covered |
|---|---|---|
| [01-architecture.md](01-architecture.md) | System overview, package map, layer couplings | `src/mindforge/multi_agent_craftium.py`, package `__init__`s, `README.md` |
| [02-hebbian-graph.md](02-hebbian-graph.md) | Bond matrix W: g_i, c_ij, three update modes (eta_0/eta_plus/eta_minus/R), lambda decay, gamma_d diffusion, rho replay indices | `src/hebbian/{config,graph,__init__}.py` + call sites |
| [03-rl-layer.md](03-rl-layer.md) | LoRA-MAPPO/IPPO over a frozen LLM actor; GAE, value clip xi, social replay rho | `src/rl_layer/*` (10 modules) |
| [04-environment-interface.md](04-environment-interface.md) | Craftium/Luanti action + stepping layer, HPC patches, PettingZoo wrapper | `src/mindforge/custom_environment_craftium.py`, `src/marl_craftium/*` |
| [05-five-chambers-world.md](05-five-chambers-world.md) | Lua world mod: chambers, milestones, anvil, switches, deaths | `src/marl_craftium/craftium-envs/five-chambers/mods/five_chambers/*.lua` |
| [06-rewards.md](06-rewards.md) | Full reward ledger: milestones, comm rewards, penalties, JSONL drains, bondable vs total streams | `multi_agent_craftium.py`, `env/communication_rewards.py`, `milestones.lua`, `deaths.lua`, `src/hebbian/graph.py` |
| [07-orchestrator.md](07-orchestrator.md) | Main asyncio training loop, CLI, checkpoint/resume, W&B | `multi_agent_craftium.py`, `run_layout.py`, `wandb_logger.py` |
| [08-cognitive-agent.md](08-cognitive-agent.md) | Per-agent LLM stack: beliefs, critic, curriculum, skills, social module, prompts | `custom_agent.py`, `agent_modules/*`, `prompts/` |
| [09-metrics-and-evaluation.md](09-metrics-and-evaluation.md) | Online recorders, post-hoc coop/comm eval, results tables/plots | `craftium_metric.py`, `coop_eval.py`, `comm_eval.py`, `cooperation_metric.py`, `episode_logger.py`, `make_results.py` |
| [10-configuration.md](10-configuration.md) | All knobs: dataclass defaults vs CLI vs launcher overrides; Tables 6/7 mapping | `hebbian/config.py`, `rl_layer/config.py`, `parse_args`, `hpc/daic/experiments/_common.sh` |
| [11-operations.md](11-operations.md) | Local quickstart, HPC (DAIC/DelftBlue/Snellius), artifacts, testing | `README.md`, `hpc/*`, `run_layout.py`, `make_results.py`, `pyproject.toml` |
| [PAPER_INCONSISTENCIES.md](PAPER_INCONSISTENCIES.md) | Central paper-vs-code consistency record (numbered #1...) | `paper/main.tex` + `paper/sections/` vs all of the above |

## 3. Reading paths

**Understand the code** (top-down, then cross-cutting):
[01-architecture.md](01-architecture.md) → [02-hebbian-graph.md](02-hebbian-graph.md) → [05-five-chambers-world.md](05-five-chambers-world.md) → [06-rewards.md](06-rewards.md) → [04-environment-interface.md](04-environment-interface.md) → [03-rl-layer.md](03-rl-layer.md) → [07-orchestrator.md](07-orchestrator.md) → [08-cognitive-agent.md](08-cognitive-agent.md) → [09-metrics-and-evaluation.md](09-metrics-and-evaluation.md)

**Verify the paper** (mismatch ledger first, then the components mismatches touch):
[PAPER_INCONSISTENCIES.md](PAPER_INCONSISTENCIES.md) → [06-rewards.md](06-rewards.md) → [02-hebbian-graph.md](02-hebbian-graph.md) → [05-five-chambers-world.md](05-five-chambers-world.md)

## 4. Conventions

- **Commit pinning.** Every doc header carries the same line: verified at `52bb302` (branch `wired_final`) plus the six post-commit source fixes surfaced by the test suite (m11→m9 anvil prefix, single-transition GAE guard, `reset()` `_W_history` clear, damage-share id normalisation, `m_door1_open` added to both Python milestone tables) — full list in [PAPER_INCONSISTENCIES.md](PAPER_INCONSISTENCIES.md) #14. Anchors are valid at that state.
- **`file.py:120-135` notation.** Line ranges point at the named symbol in the named file; each was checked by reading that region of source, not inferred.
- **Paper symbols.** g_i (engagement), c_ij (co-activity), W (bond matrix), eta_0/eta_plus/eta_minus (growth/decay rates), lambda (homeostatic decay), gamma_d (diffusion), R (salience normaliser), xi (critic value clip), rho (social replay). Full symbol → config-field glossary: [02-hebbian-graph.md](02-hebbian-graph.md) §8 and the tables in [10-configuration.md](10-configuration.md).
- **Mismatch callouts.** Docs flag paper-vs-code divergences with one-liners ("> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #N"); the full analysis lives only in [PAPER_INCONSISTENCIES.md](PAPER_INCONSISTENCIES.md).

## 5. Tests

A regression suite under `tests/` pins the facts these docs record (hyperparameter defaults, Hebbian update math, reward values, Lua spec constants); scope and provenance in `tests/README.md`. Run with:

```
python -m pytest tests -q
```

`pyproject.toml` default `addopts` excludes `needs_game`/`needs_llm` markers, so the plain invocation works without the Luanti binary or model weights (see [11-operations.md](11-operations.md) §5).
