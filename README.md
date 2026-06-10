# WiredTogether

**Reward-Modulated Hebbian Social Plasticity for Emergent Social Intelligence in Multi-Agent Systems**

A master's thesis project (TU Delft, EEMCS) that treats a multi-agent system as an adaptive
network: agents are *neurons*, the social bonds between them are *synapses*, and team reward
is the *modulatory signal*. A reward-modulated Hebbian rule learns those bonds online, producing "social assemblies", pairs and groups that preferentially collaborate, and uses them as
a structural substrate for reward propagation and targeted communication.

---

## Motivation

Most multi-agent RL and LLM-agent systems treat other agents as transient sources of information:
coordination is achieved through per-step messages, dialogue, or attention that dissolves when the
interaction ends. Human social learning is different, it relies on *persistent relationships* that
determine whom to learn from and what to retain.

This project argues the missing ingredient is **structural plasticity at the population level**.
Just as the brain's capacity to learn comes from synaptic plasticity rather than from any single
neuron, a multi-agent system can be viewed as a graph whose inter-agent bonds are the substrate of
social intelligence. We apply a reward-modulated Hebbian rule, "agents that co-activate and jointly
contribute to reward, bond", to a time-evolving weighted graph `W(t)` over the agents.

### Research questions

- **RQ1** — Does Hebbian plasticity over inter-agent bonds improve cooperative task performance
  versus LLM-only baselines with no social graph?
- **RQ2** — Does reward diffusion across the Hebbian graph propagate learning signals across the
  team, accelerating task completion versus agents that receive only their own reward?

---

## How it works

The system layers three independent pieces on top of a cooperative Craftium environment:

1. **MindForge cognitive stack**: an LLM-driven agent (perception, partner/interaction beliefs,
   auto-curriculum, critic, skill & episodic memory) that maps a first-person frame plus structured
   text context to one discrete action, a thought, and an optional targeted message.
2. **Modular RL layer**: action-level PPO with a frozen base model and per-agent LoRA adapters,
   trained as **MAPPO** with a shared centralised critic over an explicit joint-state encoding
   (compact features + sentence-transformer embeddings of each agent's last action/message). A per-agent-critic **IPPO** mode is kept as an ablation.
3. **Hebbian social-plasticity graph**: a numpy-only, gradient-free module that maintains the
   bond matrix `W(t) ∈ [0,1]^{N×N}`. A two-term update grows bonds from reward-scaled co-activity
   and decays them only when a pair both drifts apart *and* accumulates failure. The graph couples
   into learning via **reward diffusion** (each agent's reward is blended with a bond-weighted
   average of co-active teammates' rewards before it enters PPO) and an inference-time
   **social module** that turns the bond row into a help-request directive in the action prompt.

### The WIRE environment

Experiments run in **WIRE** (Wired Inter-agent Reasoning Evaluation), a five-chamber cooperative
dungeon built on Luanti/Craftium that enforces co-location and shared progression so the Hebbian
co-activity signal reliably fires:

| Chamber | Mechanic |
|---|---|
| **Ch1** | Solo skill acquisition: move, dig, pick up items, kill passive animals |
| **Ch2** | Cooperative anvil breaking: an anvil yields only when ≥2 agents dig it simultaneously |
| **Ch3** | Switch puzzle under partial observability: each agent's switch frees the *next* agent's cell, so escape requires targeted communication |
| **Ch4** | Team combat: clear three zombies, bonus for finishing intact (non-lethal rehearsal) |
| **Ch5** | Cooperative boss fight: a 60-HP boss under genuine (lethal) threat |

Bonds formed in one chamber carry into the next, so cooperation compounds over the episode.

---

## Project layout

```
src/
├── hebbian/                   Social-plasticity graph (numpy-only, no torch)
│   ├── config.py              All bond-rule hyperparameters (fixed constants)
│   └── graph.py               Reward-modulated Hebbian update + reward diffusion
│
├── rl_layer/                  Modular RL layer (PPO actor + optional centralised critic)
│   ├── config.py              RLConfig
│   ├── rl_layer.py            RLLayer orchestrator (delegates to the modules below)
│   ├── centralized_critic.py  Shared MAPPO critic (joint state → V)
│   ├── ippo.py / ppo_update.py  Action- and token-level PPO steps + update loop
│   ├── trajectory_buffer.py   Per-agent rollout buffer
│   ├── heads.py, token_opt.py, persistence.py
│
├── mindforge/                 LLM-driven agent stack + main loop
│   ├── multi_agent_craftium.py    Entry point
│   ├── custom_agent.py
│   ├── custom_environment_craftium.py
│   ├── agent_modules/             Action selection, beliefs, curriculum, critic, metrics, …
│   ├── env/                       Communication rewards, cooperation metric, episode logging
│   └── prompts/                   Prompt templates
│
└── marl_craftium/             Patched Craftium env wrapper + the five-chamber world (Lua mods)
```
---

## Quick start

### 0. Get the Craftium engine fork

The five-chamber world runs on a patched Craftium/Luanti fork that is not vendored here (it
compiles the engine per platform). Clone it into `craftium/` at the repo root:

```bash
git clone https://github.com/AnaCMarcu/craftium_wired_together.git craftium
pip install -e ./craftium   # builds the Luanti engine; see that repo's README
```

### 1. Python environment

```bash
poetry install
# or:  conda env create -f environment.yml && conda activate wiredtogether
```

### 2. Smoke test

```bash
cd src/mindforge
PYTHONPATH=../ python multi_agent_craftium.py \
    --num-agents 3 --episodes 3 --max-steps 100 \
    --warmup-time 60 --team-mode homogeneous-agent
```

### 3. A full run with RL + Hebbian

```bash
cd src/mindforge
PYTHONPATH=../ python multi_agent_craftium.py \
    --num-agents 3 --episodes 5 --max-steps 2500 --simultaneous \
    --rl --rl-critic-mode centralized --rl-model-path /path/to/Qwen3.5-2B \
    --hebbian --hebbian-gamma 0.2 --hebbian-rho 0.3
```

### CLI flags worth knowing

```text
--rl / --rl-critic-mode centralized|independent   enable MAPPO (centralised) or IPPO
--hebbian                                          enable the social-plasticity graph
--hebbian-mode reward_modulated|coactivity|legacy  bond update rule (Variant B is default)
--hebbian-gamma 0.2                                reward-diffusion strength
--hebbian-radius 5.0                               interaction radius (world units)
--social-module none|prompt|bias                   inference-time bond → directive coupling
--hebbian-freeze --hebbian-preset uniform|star|ring|pair   impose a fixed social topology
--resume <path> / --checkpoint-interval 500        chained-job checkpoint / resume
```

---

## Running on HPC

SLURM launchers live under `hpc/` (DelftBlue and DAIC variants). Each per-experiment script sources
a shared `_common.sh` (which wires up the Apptainer image, model paths, headless rendering, seeds,
and W&B) and calls `multi_agent_craftium.py` with a fixed configuration. The experiment suite spans
the comparison matrix used in the thesis:

| Group | Configuration |
|---|---|
| LLM baselines | LLM agents only (2B / 9B), no RL, no Hebbian |
| RL | MAPPO (centralised critic) vs IPPO (per-agent critic) |
| RL + Hebbian | RL plus the social-plasticity graph — the headline claim |
| Social topology | Frozen hardcoded graphs (all-allied / pair / none) for the social-bias ablation |

Submit a whole campaign across seeds with `submit_all.sh`, or a single experiment with
`sbatch hpc/.../<exp_name>.sbatch`.

---

## Outputs

Each run writes everything under `runs/<run_id>/`: per-episode `step_log` / `event_log` /
`messages` JSONL, end-of-episode summaries, `hebbian_snapshots.jsonl` (one `W` matrix per episode),
plots (returns, milestones, bond evolution and asymmetry), checkpoints, and a consolidated
`final_metrics.json`. Optional Weights & Biases logging mirrors the headline curves.

---

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

Released under the MIT License.

## Acknowledgments

- **Craftium** — the Luanti-based RL platform the WIRE environment is built on.
- **VoxeLibre** (formerly MineClone2) — the game shipped as a vendored asset for the env.
- **MindForge** — the LLM-agent reasoning stack this work extends.