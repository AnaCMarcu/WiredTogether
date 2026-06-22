# 01 — System Architecture

**Source files:** `src/mindforge/multi_agent_craftium.py`, `src/{hebbian,rl_layer,mindforge,marl_craftium}/__init__.py`, `README.md`
**Paper sections:** §3 (system overview), §B (RL layer), Eq. 6–8 (Hebbian update / replay / diffusion), App. C–D
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

## 1. System at a glance

Three independent layers stacked on the WIRE five-chamber world (Craftium/Luanti):

| Layer | What it does | Lives in |
|---|---|---|
| LLM cognitive stack | Frame + text context -> beliefs, curriculum, critic, skills -> one action + thought + optional targeted message | `src/mindforge` |
| RL layer | Action-level PPO (LoRA on frozen base); MAPPO shared critic V(s_t) over joint state, or per-agent IPPO; value clip xi | `src/rl_layer` |
| Hebbian social graph | numpy-only bond matrix W in [0,1]^{NxN}; growth (eta_0 + eta_plus*\|r_bond\|/R)*c_ij*(1-W), failure-gated eta_minus decay, homeostatic lambda*W | `src/hebbian` |

Thesis contribution = the Hebbian graph plus its two couplings: (1) **reward diffusion** — each agent's per-step reward is blended with bond-weighted co-active teammates' rewards (gamma_d) *before* entering metrics and PPO; (2) the **social module** — the agent's bond row (weights + deltas) is rendered into a help-request directive in the action prompt (and, in `bias` mode, overrides message routing). A third, weaker coupling — social replay (rho) mixing neighbour transitions into PPO batches — exists but is paper-App-C.1 material (see #9 below and 03-rl-layer.md).

## 2. Package map

Entry point is run from `src/mindforge` with `PYTHONPATH` pointing at `src/` (README quick start), so both flat imports (`from custom_agent import ...`) and package imports (`from rl_layer import ...`, `from mindforge.env...`) coexist (multi_agent_craftium.py:19-35). `hebbian` and `rl_layer` never import from `mindforge` (src/mindforge/__init__.py:8-10).

| Path | Responsibility | Key entry point |
|---|---|---|
| `src/hebbian` | Bond matrix W, three update modes, diffusion, replay indices; torch-free | `HebbianSocialGraph` (graph.py), `HebbianConfig` (config.py) |
| `src/rl_layer` | PPO actor (LoRA), MAPPO centralised critic, IPPO value head, rollout buffer, token-opt; re-exports hebbian for back-compat (rl_layer/__init__.py:13-15) | `RLLayer` (rl_layer.py), `CentralizedCritic` |
| `src/mindforge` | LLM agent stack (`agent_modules/`), env-side utilities (`env/`), prompts, main loop | `multi_agent_craftium.py::run()` (line 832) |
| `src/marl_craftium` | Patched Craftium wrapper (headless/HPC fixes in `_patched_env.py`) + five-chambers world Lua mods (`craftium-envs/five-chambers/`); `_bootstrap` puts in-tree `craftium/` on sys.path (marl_craftium/__init__.py:10-16) | `OpenWorldMultiAgentEnv` |
| `craftium/` | Vendored Luanti engine fork — NOT in git; cloned + built per README §0 | — |
| `hpc/` | SLURM launchers: `daic/`, `delft_blue/`, `snellius/`; per-experiment scripts source `_common.sh` | `submit_all.sh`, `experiments/*.sbatch` |
| `paper/` | ICML-style LaTeX sources (`main.tex`, `sections/`) | — |
| `runs_from_daic/` | Synced run artifacts (incl. `legacy/ANALYSIS_REPORT.md`) | — |
| `make_results.py` | Aggregates `runs/final/<cond>/seed_<N>/final_metrics.json` -> paper tables/figures (untracked at 52bb302) | `make_results.py:1-25` |
| `tests/` | pytest suite pinning paper facts (see PAPER_INCONSISTENCIES.md) | `tests/conftest.py` |
| `docs/` | This documentation set | `README.md` |

## 3. One environment step

All anchors `multi_agent_craftium.py` unless noted. Step loop body: lines 1340-2465.

1. **Chamber-timeout teleports** — at 20%/40%/60%/80% of `--max-steps`, `force_chN_teleport` fires unconditionally (RESCUE/NUDGE/REGROUP), advancing stragglers (1325-1375; env side custom_environment_craftium.py:1211).
2. **Early exit** — break on `environment.all_done()` or `episode_over()` (Ch5 team wipe) (1406-1412).
3. **Phase 0: joint-state encode** — MAPPO only: pre-step positions/chambers/HP/inventory + milestone sets + previous actions/comms -> `centralized_critic.encode_joint` -> `v_global_t = V(s_t)` (1494-1533). Computed *before* anyone acts so V is V(s_t), not V(s_{t+1}).
4. **Simultaneous action selection** — all living agents choose on shared s_t via `agent.on_messages` (custom_agent.py:232): full cognitive pipeline (beliefs -> curriculum -> action JSON with `action`/`thoughts`/`communication`); in RL mode thoughts are generated first, then the action is constrained-decoded from p(a \| prompt, thoughts) (agent_modules/action_selection.py:79-150). Coroutines run **sequentially, not gathered** — interleaving corrupts the shared in-process model state (1606-1615). Hebbian bond strings/weights/deltas are injected into each prompt (1445-1478, 1579-1597).
5. **`environment.step_all(actions)`** — one env advance for everyone (1622; custom_environment_craftium.py:411); per-agent env-step rewards drained once into `step_rewards_raw` (1624-1628). (Turn-based fallback: `agent_do_action` + per-agent `step`, 525-616, 1666-1695.)
6. **Comm routing** — each emitted message is targeted; bad targets (self/"all"/unparseable) are rerouted to the strongest Hebbian bond (or random without `--hebbian`), tagged with `routing_source` (1770-1786); `--social-module bias` lets the SocialModule's `ask_target` override routing (1795-1815). Messages land in per-agent inboxes; `(sender, recv)` pairs feed Hebbian comm co-activity (1817-1832).
7. **Phase 1b: comm rewards** — `CommunicationTracker.process_step` pays base +0.5/valid message and per-chamber comm milestones; pitch-cap futile penalty -1 per redirected look (1885-1916).
8. **Phase 1c/1d: Lua JSONL drains** — the five_chambers mod's `craftium.reward()` never reaches Python in multi-agent mode, so milestone rewards are drained from `milestone_events.jsonl` (1979-2009; poll at custom_environment_craftium.py:1426) and death/would-die penalties (-10 would-die, -50 Ch5 death) from `death_events.jsonl` (2024-2036; poll at :1550). See 05-five-chambers-world.md, 06-rewards.md.
   > PAPER MISMATCH — comm-milestone values (#1) and would-die scope (#6); see PAPER_INCONSISTENCIES.md.
9. **Phase 2: Hebbian update + diffusion** — per-agent one-step advantages r_t − V(s_t) (2046-2053); chamber-gated `bond_rewards` = milestones + comm + futile, deaths excluded from growth by construction (2078-2088); `hebbian_graph.update(positions, ..., comm_events, chambers, bond_rewards, total_rewards)` (2089-2097), then `diffused_rewards = hebbian_graph.diffuse_rewards(step_rewards_raw)` (2098) implementing r'_i = (1−gamma_d)·r_i + gamma_d·Σ w̄_ij·c_ij·r_j (hebbian/graph.py:847-894).
   > PAPER MISMATCH — unconditional homeostatic lambda·W term in the update; see PAPER_INCONSISTENCIES.md #4.
10. **Reward decomposition + logging** — 5 streams (task, comm_base, comm_milestone, proximity=0, hebbian_diffuse) summing exactly to the diffused value (2119-2142); non-zero streams printed per step (2155-2171).
11. **Phase 3: record diffused reward** — per live agent: `metric.record_reward` + `record_reward_decomposed` (2208-2210), `rl_layer.store_reward(reward, ...)` closing the pending transition (2217-2224); dead agents still book their terminal −50 once, with `done=True` (2187-2206). On `should_update()`: MAPPO update with `neighbour_buffers` + `hebbian_graph` so social replay (rho) can mix bond-weighted neighbour transitions (2235-2247; rl_layer/ppo_update.py:53,141-169). Token-opt may self-trigger and propagate to teammates with bond > 0.3 (2250-2289).
12. **Phase 3a: centralised critic** — stores (s_t, mean alive diffused reward, V(s_t), done) and updates on its own schedule (2291-2312).
13. **Phase 3b/4: metrics** — milestone events -> CraftiumMetric + CooperationMetric + episode_logger JSONL (2314-2374); W snapshot every `log_graph_every` steps printed + recorded (2376-2405); periodic checkpoint every `--checkpoint-interval` (2418-2433).

```mermaid
sequenceDiagram
    participant L as run() loop
    participant A as agents (LLM/RL)
    participant E as CraftiumEnvInterface
    participant X as Luanti srv + N clients
    participant H as HebbianSocialGraph
    participant R as RLLayer / Critic
    L->>R: encode_joint(s_t) -> V(s_t)        %% Phase 0
    L->>A: on_messages(frame, comms, bonds W) %% sequential, all on s_t
    A-->>L: {action, thoughts, communication}
    L->>E: step_all(actions)
    E->>X: TCP actions / frames
    X-->>E: obs + Lua JSONL (milestones, deaths)
    L->>E: poll_milestone_events / poll_death_events
    L->>H: update(positions, comm_events, bond_rewards) ; diffuse_rewards
    H-->>L: diffused r'_i
    L->>R: store_reward(r'_i) ; maybe PPO update (+ social replay rho)
    L->>L: route messages -> inboxes ; log JSONL ; snapshot W
```

## 4. Episode lifecycle

| Stage | What happens | Anchor |
|---|---|---|
| Reset | `environment.reset()` + milestone/anvil/death JSONL offsets reset; from ep2 on, `agent.on_reset()` clears task/chamber/beliefs but keeps skills + Hebbian bonds | 1171-1178; custom_agent.py:593 |
| Warmup | Poll `warmup_noop()` frames until per-client pixel std-dev > 25 for 3 consecutive checks (min `--warmup-time`, max 900 s), then `signal_warmup_complete()` so the Lua Ch1 timer starts | 1181-1232; custom_environment_craftium.py:1159,1176 |
| Step loop | §3 above; each chamber gets a 20%-of-`--max-steps` window (Lua-tick backstop sized by `--ch1-timeout-steps`) | 949-964, 1325-1375 |
| End conditions | `all_done()` \| `episode_over()` (all dead in Ch5, flag file) \| step budget \| SIGTERM/SIGINT -> shutdown checkpoint + clean exit | 1406-1412, 2435-2465 |
| End of episode | Cooperation summary (+ final W), `metric.end_episode`, W&B episode payload, `episode_summary` finalize, append W to `hebbian_snapshots.jsonl`, end-of-ep checkpoint, MP4/GIF flush | 2467-2610 |
| End of run | `final_metrics.json` + summary, RL LoRA save, `hebbian_graph_final.json`, W&B final, 120 s SIGALRM watchdog around `environment.close()` | 2612-2705 |

Checkpoint/resume (`save_checkpoint`/`load_checkpoint`, 619-814) serialises metric state, W, per-agent LoRA + optimiser, and curriculum so chained SLURM jobs continue (`--resume`); resume restarts the in-progress episode from step 0 (1142-1164).

## 5. Process / IPC topology

- **Processes:** one Python process; one Luanti **server** + N Luanti **clients** (one per agent), launched by craftium and patched for headless HPC in `marl_craftium/_patched_env.py:80-91` (all-clients offscreen SDL) with startup diagnostics (:262-291). Engine <-> Python: craftium's TCP protocol (frames in, key/mouse actions out).
- **Lua -> Python:** file IPC. The server-side `five_chambers` mod appends to `milestone_events.jsonl` / `death_events.jsonl` (plus anvil-coop diagnostics); Python polls with persistent offsets each step (custom_environment_craftium.py:1426,1493,1550). Python -> Lua control (forced teleports, warmup-complete) also goes via flag files. This is the *only* reliable reward channel multi-agent — see 04-environment-interface.md.
- **Concurrency:** action selection is written as asyncio coroutines for simultaneous-move semantics, but executed sequentially on the shared in-process model (1606-1615); `step_all` then advances the env once. The legacy turn-based mode (`--no-simultaneous`) steps round-robin.
- **Outputs:** everything under `runs/<run_id>/` (`mindforge/run_layout.py`): step/event/message JSONL, `hebbian_snapshots.jsonl`, `interpretability.jsonl` (auto-on with `--hebbian`, 1244-1250), checkpoints, gifs, `final_metrics.json`; optional W&B mirror.

## 6. Run-mode matrix

All flags in `parse_args()` (44-324). `--social-module` without `--hebbian` exits at startup (2713-2723).

| Dimension | Flag(s) | Options (default first) |
|---|---|---|
| LLM stack | always on; model via env vars `LLM_MODEL_PATH` (local) / `LLM_MODEL` (remote) | — |
| RL | `--rl` (off), `--rl-critic-mode` | `centralized` (MAPPO, shared V over joint state) \| `independent` (IPPO per-agent value head) |
| RL granularity | `--rl-mode` | `action` (PPO head) \| `token` (token-opt only, LLM picks actions) |
| Hebbian | `--hebbian` (off), `--hebbian-mode` | `reward_modulated` (Variant B: (eta_0 + eta_plus·\|r_bond\|/R)·c·(1−W)) \| `coactivity` (Variant A: flat eta_plus·c·(1−W)) \| `legacy` (advantage modulator) |
| Reward diffusion | `--hebbian-gamma` | 0.2; `0.0` = graph measures but never feeds RL/metrics |
| Social module | `--social-module` | `none` (raw bond text only) \| `prompt` (directive in action prompt) \| `bias` (directive also overrides comm routing); cadence `--social-interval 8` |
| Frozen topology | `--hebbian-freeze` + `--hebbian-preset` | `none` \| `uniform` \| `star` \| `ring` \| `pair` (with `--hebbian-bond-strong/weak`, `--hebbian-hub`); pair with `--hebbian-gamma 0` for the LLM-only social-bias ablation |
| Social replay | `--hebbian-rho` | CLI default **0.3** — note this overrides the dataclass default 0.0 (hebbian/config.py:63) for runs launched via this entry point |
| Misc | `--no-communication`, `--simultaneous` (default ON), `--hebbian-no-comm-bond` (delta_comm=0), `--seed`, `--tag` | — |

> PAPER MISMATCH — base model is config-driven, Qwen only via HPC launcher env; see PAPER_INCONSISTENCIES.md #8.
> PAPER MISMATCH — social-replay "rho=0 disabled" claim holds for the dataclass, not the CLI default; see PAPER_INCONSISTENCIES.md #9.

Canonical thesis configurations (HPC scripts under `hpc/*/experiments/`): LLM baseline (no flags); LLM+Hebbian (`--hebbian --hebbian-gamma 0.2`); MAPPO (`--rl`); MAPPO+Hebbian (`--rl --hebbian`) — the headline claim; IPPO+Hebbian (`--rl-critic-mode independent`); frozen-topology ablations (`--hebbian-freeze --hebbian-preset ... --hebbian-gamma 0 --social-module bias`). Details: 10-configuration.md, 11-operations.md.
