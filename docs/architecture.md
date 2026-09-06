# Architecture

## Three layers on one environment

| Layer | Job | Package |
|---|---|---|
| Cognitive stack | First-person frame + structured context → beliefs, task, one discrete action, one targeted message | `src/mindforge` |
| RL layer | Action-level PPO on LoRA adapters over a frozen base model; MAPPO shared critic or IPPO value heads | `src/rl_layer` |
| Social graph | The bond matrix `W(t)`, updated online from co-firing and reward | `src/hebbian` |

The contribution is the third layer plus the couplings that let it change behaviour. Everything
else is the substrate the couplings act on, and each layer runs without the ones above it: the
cognitive stack alone is the zero-shot LLM baseline, adding the RL layer gives MAPPO/IPPO, and
`--hebbian` adds the graph.

`hebbian` and `rl_layer` never import from `mindforge`, so the graph and the RL code can be lifted
out and tested on their own — which is what most of `tests/` does.

## Packages

| Path | Responsibility | Entry point |
|---|---|---|
| `src/hebbian` | `W`, its update variants, reward diffusion, replay indices. numpy only. | `HebbianSocialGraph`, `HebbianConfig` |
| `src/rl_layer` | PPO actor over LoRA, centralised critic, rollout buffer, token-level mode | `RLLayer`, `CentralizedCritic`, `RLConfig` |
| `src/mindforge` | Agent stack (`agent_modules/`), env-side accounting (`env/`), prompts, episode loop | `multi_agent_craftium.py` |
| `src/marl_craftium` | PettingZoo wrapper over the patched Craftium env + the WIRE world | `OpenWorldMultiAgentEnv` |
| `src/orchestrator` | Centralised orchestration baselines (task / social / plan / villager) | `orchestrator.core`, `orchestrator.villager` |

Inside `src/mindforge`:

| Module | Job |
|---|---|
| `multi_agent_craftium.py` | The episode loop — the only place where the layers meet |
| `cli.py` | Every flag, plus `validate_args` for combinations that would silently no-op |
| `agent_factory.py` | Prompt loading, role configs, building the agents |
| `checkpointing.py` | Save/restore for chained cluster jobs (`--resume`) |
| `custom_environment_craftium.py` | The environment adapter: actions in, frames/state/rewards out |
| `env/chamber_state.py`, `env/lua_events.py` | The two halves of the Lua interface: state files and event drains |
| `env/communication_rewards.py`, `env/cooperation_metric.py`, `env/episode_logger.py` | Message validity and payment, pair-level cooperation stats, per-episode JSONL |
| `run_layout.py` | The single source of truth for output paths |

## One environment step

The loop in `multi_agent_craftium.run()` is organised in numbered phases; the order matters and is
the reason several of them exist separately.

1. **Chamber timers.** At each 20% slice of `--max-steps` the team is rescue-teleported forward if
   it has not solved the current chamber. Teleports never grant the chamber's milestones.
2. **Phase 0 — joint state.** MAPPO only: the pre-step joint state is encoded and `V(s_t)` computed
   *before* anyone acts, so the value is `V(s_t)` and not `V(s_{t+1})`.
3. **Social deliberation.** Every `--social-interval` steps each agent's bond row and its recent
   deltas go through the social module; between deliberations the last directive is cached.
4. **Action selection.** All living agents choose on the same `s_t`. Under `--rl` the reasoning is
   generated first and the action is then constrained-decoded from `p(a | prompt, thoughts)`.
   The coroutines run sequentially, not gathered — interleaving corrupts the shared in-process
   model state.
5. **`step_all`.** One environment advance for the whole joint action.
6. **Message routing.** Messages are targeted; invalid targets are rerouted to the strongest bond,
   and under `--social-module bias` the module's `ask_target` overrides routing. The resulting
   `(sender, receiver)` pairs feed communication co-firing.
7. **Phases 1b–1d — reward accounting.** Communication payments and futility penalties, then the
   milestone and death drains from the Lua event logs.
8. **Phase 2 — Hebbian update and diffusion.** Positions, co-firing events, chambers and the
   bond-eligible reward stream go into `HebbianSocialGraph.update`, then rewards are diffused
   along the current graph.
9. **Phases 3–4 — learning and logging.** Diffused rewards enter the metric recorder and the PPO
   buffers, the critic stores its team step, milestone events are recorded, and a graph snapshot is
   taken every `log_graph_every` steps.

Two ordering constraints are load-bearing: the graph is updated *after* the step's rewards are
known but *before* they are stored, so buffers only ever see diffused rewards; and the reward
stream that grows bonds is not the same one that is diffused — deaths are excluded from bond
growth by construction (see [hebbian-graph.md](hebbian-graph.md)).
