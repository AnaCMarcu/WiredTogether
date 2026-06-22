# 07 — Orchestrator: the Main Training Loop

**Source files:** `src/mindforge/multi_agent_craftium.py`, `src/mindforge/run_layout.py`, `src/mindforge/wandb_logger.py`
**Paper sections:** §4 (system architecture), §5/App. B (experimental protocol), Eq. 6–8 (consumed here via `hebbian_graph.update` / `diffuse_rewards` / social replay), Tables 6–7 (hyperparameters surfaced as CLI flags)
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

`multi_agent_craftium.py` is the single entry point that wires environment (04-environment-interface.md), cognitive agents (08-cognitive-agent.md), RL layer (03-rl-layer.md), Hebbian graph W (02-hebbian-graph.md), rewards (06-rewards.md) and metrics (09-metrics-and-evaluation.md) into one asyncio loop. Launched as `python multi_agent_craftium.py [flags]`; `__main__` guard at `multi_agent_craftium.py:2708-2724` (refuses `--social-module` without `--hebbian`).

## 1. Entry points

### CLI (`parse_args`, multi_agent_craftium.py:44-324)

Grouped summary only — the full flag table lives in 10-configuration.md.

| Group | Flags (defaults) | Anchor |
|---|---|---|
| Env / episodes | `--num-agents 3`, `--episodes 1`, `--max-steps 1500`, `--obs-width 320 / --obs-height 180`, `--seed`, `--voxel-obs`, `--ch1-timeout-steps 400` (Lua backstop only) | :46-106, :128-132, :314-323 |
| Stepping / comm | `--simultaneous` (default ON; `--no-simultaneous` = legacy round-robin), `--no-communication`, `--sleep-time 0` | :63-74 |
| LLM cadence | `--belief-interval 5`, `--critic-interval 20`, `--social-interval 8` | :75-80, :248-253 |
| RL | `--rl`, `--rl-model-path`, `--rl-lora-rank 8`, `--rl-update-interval 256`, `--rl-lr 1e-4`, `--rl-mode action\|token`, `--rl-critic-mode centralized\|independent`, `--rl-prompt-max-tokens 512`, `--rl-auto-token-opt` | :133-161 |
| Hebbian | `--hebbian`, `--hebbian-mode reward_modulated\|coactivity\|legacy`; gated knobs eta_plus 0.05, eta_0 0.01, eta_minus 0.025, coop-eps 0.05, coop-window 50, neg-theta 5.0, R=`--hebbian-reward-norm 300`; legacy ltp/ltd/decay; lambda=`--hebbian-decay 0.005` (CLI) vs config 0.0003; gamma_d=`--hebbian-gamma 0.2`, rho=`--hebbian-rho 0.3`, alpha=`--hebbian-alpha 0.5`, radius 5.0, init W_0 0.1, `--hebbian-no-comm-bond`; frozen-preset flags (`--hebbian-freeze`, `--hebbian-preset none\|uniform\|star\|ring\|pair`, bond-strong/weak, hub) | :162-227 |
| Social module | `--social-module none\|prompt\|bias` | :238-247 |
| Logging | `--wandb*` (project, entity, tags, id, upload-artifacts), `--log-interval 10`, `--no-gif`, `--gif-dir auto`, `--gif-interval 300`, `--interpretability` (auto-on with `--hebbian`) | :81-94, :107-127, :233-237, :264-265 |
| Checkpointing | `--checkpoint-dir` (default `runs/<id>/checkpoints/`), `--checkpoint-interval 500`, `--resume <dir>`, `--resume-skip-warmup`, `--checkpoint-frames` | :299-313 |
| Identity / team | `--experiment-id`, `--tag` (tagged layout `runs/legacy/<tag>/seed_<N>/`), `--team-mode homogeneous-agent\|heterogeneous`, `--homogeneous-role`, `--roles` | :254-294 |

NOTE: CLI `--hebbian-rho` defaults to **0.3** and overwrites `HebbianConfig.social_replay_rho` (config-file default 0.0, "disabled until IS correction") at :1040 — the final RL+Hebbian launchers (exp05/exp06) pass no `--hebbian-rho`, so those runs executed with ρ=0.3. > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #9

### Prompt + role assembly

- `load_prompts()` (:327-354) reads `prompts/` (environment, system template, critic, curriculum, skill, four belief-system prompts) plus one role prompt per `ROLE_NAMES = ["agent","hunter","harvester","scouter"]` (:37).
- `build_role_configs()` (:357-412) maps roles to `agent_i`; heterogeneous mode validates `len(roles) == num_agents`.

### `build_agents` wiring (:415-495)

Each `CustomAgent` receives: `ActionSelection`, `AutoCurriculum` (role prompt), `Critic`, `SkillManager` (skill DB wiped unless `is_resume`), `EpisodicMemoryManager`, `BeliefSystem`, the **shared** `CraftiumMetric`, an optional per-agent `RLLayer` and an optional per-agent `SocialModule`. Shared singletons created in `run()`:

| Object | Sharing | Anchor |
|---|---|---|
| `CentralizedCritic` | one V(joint state) for all agents' RLLayers; built only when `--rl --rl-mode action --rl-critic-mode centralized`; turns off per-agent value loss | :1006-1019, :432-436 |
| `HebbianSocialGraph` (W) | one graph; all CLI knobs flow into `HebbianConfig` at :1031-1059; frozen-preset W printed at startup | :1030-1072 |
| `CraftiumMetric` | one instance shared by all agents and the loop | :939-947 |

## 2. Episode loop (`run()`, :832-2705), phase by phase

Per episode (`for episode in range(resume_episode, num_episodes)`, :1166):

| Phase | What happens | Anchor |
|---|---|---|
| Reset | `environment.reset()` + reset of milestone/anvil/death JSONL drain offsets; `agent.on_reset()` for episodes after the first (clears task/beliefs, keeps skills + W) | :1171-1178 |
| Media warmup | poll `warmup_noop()`; per-client frame std-dev > 25 counts as "loaded"; needs 3 consecutive passes after `--warmup-time`; hard cap 900 s, then "Starting anyway"; `signal_warmup_complete()` arms the Lua Ch1 timer | :1180-1232 |
| Per-episode state | per-agent inboxes, `CommunicationTracker`, `CooperationMetric`, `EpisodeLogger`, bond-string cache, streaming per-agent MP4 writers, rolling frame deque (maxlen = gif_interval) | :1234-1286 |
| Chamber timeouts | each chamber gets 20% of `--max-steps`; unconditional force-teleport Ch n→n+1 at 20n% (n=1..4) via `environment.force_chN_teleport()`; logs RESCUE/NUDGE/REGROUP by how many agents had already advanced | :1325-1375 (config print :949-964) |
| Step-loop guards | `--log-interval` stdout/wandb summary; break on `all_done()` or `episode_over()` (Ch5 team wipe flag) | :1377-1412 |
| Frame capture | per-agent frame → streaming MP4 + rolling deque; intermediate GIF+MP4 checkpoint every `--gif-interval` steps | :1414-1430, :1288-1323 |
| Bond cache | bond strings + structured `bond_weights`/`bond_deltas` dicts rebuilt only when `hebbian_graph._step_count` changed (version-cached) | :1443-1478 |
| Phase 0 | encode PRE-step joint state s_t (positions, chambers, HP, inventories, milestones, last actions/comms) and evaluate V(s_t) **before** any agent acts | :1494-1533 |
| Action selection | simultaneous (default): all living agents' `on_messages` awaited **sequentially** (asyncio.gather corrupted shared in-process model state → NaN logits) on the same s_t, then one `environment.step_all(actions)`; env step reward drained once. Each call receives: frame, formatted inbox, reward summary, bond string + weights/deltas, position/status text, current+visited chambers, completed-milestone set, `format_milestone_progress`, chamber state (+ optional voxel summary) | :1535-1628; turn-based fallback `agent_do_action` :525-616, :1666-1695 |
| V(s_t) attach | `set_pending_value_global(v_global_t, joint_state_t)` stamps the just-opened transition so `old_value_global` is V(s_t), not V(s_{t+1}) | :1697-1709 |
| Comm routing | tolerant target parse (`agent_N`/`agentN`); failures (self/all/out-of-range/unparseable) rerouted to argmax-W teammate (Hebbian fallback) or random; `--social-module bias` lets the SocialModule's `ask_target` override routing; message wire-tagged `[in <chamber>]`; inbox capped at num_agents−1 | :1711-1832, bias :1788-1815 |
| Phase 1b | positions snapshot; pitch-cap futile penalty −1.0/event; `comm_tracker.process_step` (+0.5 valid messages, comm milestones; bad-target speakers excluded); per-message records flushed to messages.jsonl with split base/milestone reward; `coop_metric.observe_step` + `ep_logger.log_step` | :1836-1967 |
| Phase 1c | drain `poll_milestone_events()` JSONL into `step_rewards_raw` (authoritative — server-side `craftium.reward()` never reaches Python; see 06-rewards.md); contributor names accept both `agent0` and `agent_0`; anvil co-op diagnostic events forwarded reward-free | :1969-2009 |
| Phase 1d | drain `poll_death_events()` (−10 would-die, −50 Ch5 death; would-die rate-limited to one per agent per episode in the poller) into `step_rewards_raw` > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #6 | :2011-2036 |
| Phase 2 | per-agent one-step advantage delta_t = r_t − V(s_t) (None without RL); chamber-gated `bond_rewards` = milestone + comm + futile only — death drain excluded from growth by construction; `hebbian_graph.update(...)` (Eq. 6, g_i/c_ij inside the graph) then `diffused = hebbian_graph.diffuse_rewards(raw)` (Eq. 8, gamma_d) | :2038-2098 |
| Decomposition | 5 streams: task (env + pitch + milestone drain + death drain), comm_base, comm_milestone, proximity (always 0; bonus removed), hebbian_diffuse = diffused − raw; streams sum to the recorded reward; non-zero rows printed per step | :2119-2171 |
| Phase 3 | live agents: `metric.record_reward(diffused)` + decomposed; `rl_layer.store_reward` closes the pending transition. **Dead-agent booking path:** a terminated agent whose death penalty drained THIS step still gets the −50 booked once (metric + `store_reward(done=True)`) so the terminal reward is neither dropped nor double-counted | :2173-2230 (dead path :2176-2206) |
| RL update | when `rl_layer.should_update()` (buffer ≥ `--rl-update-interval`): `update(neighbour_buffers, hebbian_graph)` — neighbour buffers enable social replay rho (Eq. 7); then optional token-opt, with bond-gated (W > 0.3) social token-opt propagation to teammates | :2232-2289 |
| Phase 3a | centralized critic: `store_step(joint_state_t, mean-alive diffused reward, v_global_t, team_done)`; `update()` on its own cadence; logged as agent_id −1 → `rl/critic/*` in W&B | :2291-2312 |
| Phase 3b | milestone events → `metric.record_milestone_event`, `coop_metric.observe_milestone` / `observe_kill` (m21/m22/m27 → kill targets), event_log.jsonl, `[MILESTONE]` stdout line; Python env step used as the timestamp, never the Lua tick | :2314-2374 |
| Phase 4 | W snapshot every `log_graph_every` steps: `record_graph_snapshot` + compact weight table to stdout | :2376-2405 |
| Housekeeping | `metric.store_timestep`; prev-action/comm snapshot for next s_t encode; periodic checkpoint every `--checkpoint-interval` steps; shutdown checkpoint + clean exit if SIGTERM/SIGINT was flagged | :2407-2465 |
| Episode end | `coop_metric.episode_summary` (handed the final W); per-episode returns snapshotted **before** `metric.end_episode` resets them; W&B episode payload (incl. mean/max off-diagonal bond); `ep_logger.finalize(summary.json)`; W appended to `hebbian_snapshots.jsonl`; `ep<N>_end` checkpoint; MP4 writers closed + tail-window GIF | :2467-2610 |
| Run end | `metric.save_run_metrics()` (final_metrics.json, summary, plots), RL `save()` per agent, `hebbian_graph_final.json`, W&B final summary/artifacts, then a 120 s SIGALRM watchdog around `wandb.finish()` + `environment.close()` (`os._exit(0)` if cleanup hangs) | :2612-2705 |

## 3. Checkpoint / resume

`save_checkpoint` (:619-739) — wrapped in try/except so serialization failure never kills the run:

| File | Contents |
|---|---|
| `run_state.json` | episode/step/global_step counters, full CLI args, serialized `CraftiumMetric` (returns, per-track rewards, milestones, comm counts, graph snapshots) |
| `hebbian_graph.json` | `hebbian_graph.to_dict()` — W matrix + config |
| `rl_agent_{i}/` | LoRA adapter weights + optimizer via `rl_layer.save()` |
| `agent_{i}_curriculum.json` | current task/context, completed/failed task lists |
| `frames_{i}.npy` | raw frames, only with `--checkpoint-frames` |

Cadence: every `--checkpoint-interval` steps (`ep{E}_step{S}/`), on SIGTERM/SIGINT (`..._shutdown/`), and at episode end (`ep{E}_end/`, saved with `episode+1` so resume starts the next episode).

- **SIGTERM handler** (:1127-1140): sets a flag; the loop checkpoints at the end of the current step, closes MP4 writers, finishes W&B, closes the env, returns (:2435-2465).
- **Resume** (:1142-1164 via `load_checkpoint` :742-814): restores metric (into the live `RunPaths` tree), W in place, per-agent RL state and curriculum; loop starts at `resume_episode` (episodes always restart from step 0, :1143-1144). LoRA save dir is re-anchored to the **original** run's `rl_live/` (:978-987, `_resume_run_paths` :498-519). Skill DBs survive because `build_agents(is_resume=True)` skips the wipe.
- **W&B resume**: run id = `_sanitize_id(explicit_id or run_id)` (`[A-Za-z0-9_-]`, ≤64 chars) with `resume="allow"`, so chunked SLURM jobs relaunching the same experiment+seed continue the same W&B run (`wandb_logger.py:11-14, 26-29, 56-65`). All wandb calls are try/except-wrapped — an outage never kills training.

## 4. Run output layout

Verbatim from the `run_layout.py` module docstring (`src/mindforge/run_layout.py:1-22`), the single source of truth — every path is computed through `RunPaths`:

```
runs/<run_id>/
├── config.json              CLI args + git commit + start ts
├── log.txt                  Python logging FileHandler
├── episodes/ep_NNNN/
│   ├── step_log.jsonl       per-step per-agent record
│   ├── event_log.jsonl      milestones, switches, doors, kills, damage
│   ├── messages.jsonl       per-message metadata
│   └── summary.json         end-of-episode summary
├── checkpoints/step_NNNNNN/ run_state.json + hebbian + curricula + rl
├── plots/                   PNGs rendered at run end
├── hebbian_snapshots.jsonl  one episode-end W matrix per line
├── final_metrics.json       consolidated run-level summary
└── final_summary.txt        human-readable digest
```

Additions the orchestrator layers on top: `gifs/` (when `--gif-dir auto`, :884-889), `llm_logs/` (:919-920), `rl_live/` (LoRA adapters, :988), `interpretability.jsonl` (:1244-1250). Note the loop's actual checkpoint subdirs are `ep{E}_step{S}` / `ep{E}_end`, not the docstring's `step_NNNNNN` pattern. Two layouts: default `runs/<exp>_<ts>_<uuid6>/` vs `--tag` → `runs/legacy/<tag>/seed_<N>/` (`RunPaths.create_tagged`, run_layout.py:113-155); a relative root is re-anchored under `$WIREDTOGETHER_RUNS_ROOT` for HPC (run_layout.py:31-49; see 11-operations.md).

## 5. Failure handling

| Failure | Handling | Anchor |
|---|---|---|
| Media never loads | warmup loop times out at 900 s and starts anyway (logged std-devs show which client is stuck) | :1186, :1224-1226 |
| Per-agent LLM/selection error | simultaneous: caught inside `_sim_select`, agent falls back to `{"action": "NoOp"}` — others unaffected; turn-based: `agent_do_action` retries ≤5 times with the error fed back into the prompt, then NoOp | :1600-1604, :592-614 |
| Team wipe (Ch5 permadeath) | `environment.episode_over()` flag breaks the step loop; episode finalization still runs | :1410-1412 |
| SLURM kill | SIGTERM → shutdown checkpoint + finalized (playable) MP4s | :1133-1140, :2435-2465 |
| Hang in final cleanup | 120 s SIGALRM watchdog force-exits after results are persisted | :2688-2705 |
| Checkpoint write error | logged warning, run continues | :738-739 |
| wandb outage | every call no-ops with a warning (`wandb_logger.py`) | wandb_logger.py:3-9 |
