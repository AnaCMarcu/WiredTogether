# Code reference

One document per layer. Start with [architecture](architecture.md) for the shape of the system,
then read the layer you need.

| Document | Covers |
|---|---|
| [architecture.md](architecture.md) | The three layers, what one environment step does, how the packages depend on each other |
| [hebbian-graph.md](hebbian-graph.md) | Co-firing, the update rule and its variants, and the three couplings back to behaviour |
| [environment.md](environment.md) | WIRE: the five chambers, the milestone ladder, the Lua ↔ Python interface, extending it |
| [agents.md](agents.md) | The per-agent cognitive stack, prompts, the social module, the orchestrator baselines |
| [rl-layer.md](rl-layer.md) | LoRA-PPO over a frozen actor, MAPPO/IPPO critics, weight-gated experience sharing |
| [configuration.md](configuration.md) | Every flag and config field, its default, and the paper symbol it carries |
| [experiments.md](experiments.md) | Running the paper's conditions, the cluster launchers, and the analysis pipeline |
| [dataset.md](dataset.md) | The run artifacts: how they are grouped, what the release layers hold, how to rebuild them |

Conventions used throughout:

- Symbols follow the paper: `W` (bonds), `c_ij` (co-firing), `g_i` (engagement),
  `η₀/η₊/η₋` (growth and decay rates), `λ` (homeostatic decay), `γ_d` (reward diffusion),
  `ρ` (social replay), `R` (salience normaliser), `ξ` (critic value clip).
- Code is referenced by file and symbol, never by line number.
- Facts that the test suite pins are marked *(pinned)*; changing them fails `pytest tests`.
