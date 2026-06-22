# 04 — Environment Interface (Craftium/Luanti Action & Stepping Layer)
**Source files:** `src/mindforge/custom_environment_craftium.py`, `src/marl_craftium/_actions.py`, `src/marl_craftium/_patched_env.py`, `src/marl_craftium/_bootstrap.py`, `src/marl_craftium/openworld_multi_agents.py`
**Paper sections:** Methods §3 (environment), App. A (action space), App. D (run protocol / chamber timeouts)
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

## 1. Stack

```
Luanti (Minetest) server + N headless clients          (one OS process per agent, VoxeLibre game)
  ^ craftium MarlCraftiumEnv                            (in-tree submodule craftium/, TCP mt_channels)
  ^ _PatchedMarlCraftiumEnv                             src/marl_craftium/_patched_env.py:38
  ^ OpenWorldMultiAgentEnv (PettingZoo ParallelEnv)     src/marl_craftium/openworld_multi_agents.py:18
  ^ CraftiumEnvironmentInterface                        src/mindforge/custom_environment_craftium.py:109
  ^ orchestrator / agents (see 07-orchestrator.md, 08-cognitive-agent.md)
```

- `_bootstrap.py:16-25` — inserts `WiredTogether/craftium/` into `sys.path` so the in-tree craftium submodule is importable even when `pip install -e` was skipped/broken on a compute node. Imported for side effect by both `_patched_env.py:31` and `openworld_multi_agents.py:8`.
- `_PatchedMarlCraftiumEnv` wraps (not forks) upstream with six HPC fixes (`_patched_env.py:1-22`):

| Fix | Anchor | Why |
|---|---|---|
| Binary rename `./bin/minetest` -> `./bin/luanti` in launch cmds | `_patched_env.py:58-78` | New Luanti builds only ship `luanti`; upstream multi-agent classes silently fail on HPC |
| `SDL_VIDEODRIVER=offscreen` on ALL clients | `_patched_env.py:80-91` | Upstream only forces client 0 headless; clients 1+ crash on display-less nodes |
| Merge `os.environ` into subprocess `proc_env` | `_patched_env.py:93-107` | Upstream `env=` replaces the parent env, dropping e.g. `CH1_TIMEOUT_TICKS` read by Lua |
| Persistent media cache symlink (`$SCRATCH/.craftium_media_cache`) | `_patched_env.py:109-142` | Otherwise every reset re-downloads ~700 MB VoxeLibre media (5-60 min) |
| Pre-`listen()` all channel sockets before client launch | `_patched_env.py:265-279` | Upstream listens too late -> client `connect()` gets Connection refused |
| Server-ready polling: read stderr for `"listening on"`, 1200 s ceiling | `_patched_env.py:235-263` | Upstream `time.sleep(5)` far too short for VoxeLibre on HPC |

  Plus diagnostics: server death at launch/init raises with the stderr dump (`_patched_env.py:223-263`); `step_agent` stashes per-agent telemetry `_positions/_velocities/_pitches/_yaws/_dtimes/_voxobs` from the binary channel (`_patched_env.py:160-198`); `warmup_noop()` NoOps every channel without advancing `timesteps` (`_patched_env.py:146-158`).
- `OpenWorldMultiAgentEnv` adapts to PettingZoo ParallelEnv: dict-keyed agents `agent_0..agent_{N-1}`, `Discrete(23)` action space (`openworld_multi_agents.py:89`), Box(H,W,3) uint8 obs. `step()` converts `{agent: int}` -> list of craftium dicts via `_discrete_to_dict` (`openworld_multi_agents.py:182-200`), adds a 0.1/block XZ exploration bonus (`:216-227`), and drops terminated agents from `self.agents` (`:199`). Server port is derived from `SLURM_JOB_ID` to avoid same-node collisions (`:112-117`); `mt_listen_timeout=300_000` ms (`:128`).
- `CraftiumEnvironmentInterface` is what agents talk to: named string actions, guard rails, prompt-state readers, JSONL event drains, and the `step`/`step_all` entry points. The simultaneous-move `step_all()` lives HERE (`custom_environment_craftium.py:411`), not in the PettingZoo wrapper.

## 2. Raw action space (`_actions.py`)

`_DISCRETE_ACTIONS` (`_actions.py:8-16`) holds 22 names; the PettingZoo space is `Discrete(23)` with `0 = NOP` and `i = _DISCRETE_ACTIONS[i-1]`:

| Index | Raw name | Index | Raw name |
|---|---|---|---|
| 0 | NOP (`{}`) | 12-13 | `slot_4`, `slot_5` |
| 1-4 | `forward`, `backward`, `left`, `right` | 14 | `mouse x+` (turn right) |
| 5 | `jump` | 15 | `mouse x-` (turn left) |
| 6 | `sneak` | 16 | `mouse y-` (look DOWN) |
| 7 | `dig` | 17 | `mouse y+` (look UP) |
| 8 | `place` | 18 | `inventory` — RESERVED NOP, returns `{}` |
| 9-11 | `slot_1`..`slot_3` | 19-22 | `drop`, `slot_6`, `slot_7`, `slot_8` |

`_discrete_to_dict` (`_actions.py:22-50`): mouse actions return `{"mouse": [x, y]}` with magnitude `_MOUSE_MOV = 1.0` (~20-30° per step, doubled from upstream 0.5; `_actions.py:18-19`); key actions return `{name: 1, "mouse": [0, 0]}`. Sign convention: Minetest's mouse Y axis is inverted — `y-` is look-down, `y+` look-up (`_actions.py:11`), and `step_agent` re-negates Y and scales by half the obs resolution before sending (`_patched_env.py:172-176`).

## 3. Named action layer (`custom_environment_craftium.py`)

`ACTION_MAP` (`custom_environment_craftium.py:17-40`) — 22 named actions -> raw index. Index 18 (`inventory`) is deliberately unmapped: no named action reaches it (the "inventory hole"), so the LLM can never open the inventory GUI.

| Named | Idx | Named | Idx | Named | Idx |
|---|---|---|---|---|---|
| NoOp | 0 | Dig | 7 | TurnRight | 14 |
| MoveForward | 1 | Place | 8 | TurnLeft | 15 |
| MoveBackward | 2 | Slot1-Slot5 | 9-13 | LookDown / LookUp | 16 / 17 |
| MoveLeft / MoveRight | 3 / 4 | Drop | 19 | Slot6-Slot8 | 20-22 |
| Jump / Sneak | 5 / 6 | | | |

**Synonym recovery** — `_ACTION_ALIASES` (`:44-72`) maps ~60 normalised LLM-invented names to canonical actions instead of clamping them to NoOp. Representative entries: `attack`/`mine`/`punch`/`break` -> Dig; `forward`/`advance`/`step` -> MoveForward; bare `turn` -> TurnRight (direction unspecified defaults right); `lookup`/`tiltup`/`raisecamera` -> LookUp; `placeblock`/`build` -> Place; `throw`/`discard` -> Drop; `hop`/`leap` -> Jump; `wait`/`stay`/`stop`/`nothing` -> NoOp (also covers the misspelling `strafright` -> MoveRight).

**`canonicalize_action`** (`:84-107`) fallback chain, exactly as implemented: (1) exact `ACTION_MAP` key -> (2) format-normalised match (lowercase, strip spaces/`_`/`-`; `_normalize_action_key` `:75-77`) -> (3) alias map -> (4) first whitespace token, retried through normalised-then-alias (rescues `"MoveForward to the door"`) -> (5) `None`. Recoveries are counted per agent (`action_recovered_summary` `:278-285`); unrecoverable names are clamped to NoOp and a warning string is injected into the agent's next `{chamber_state}` prompt (`:524-549`, surfaced at `:771-774`).

## 4. Guard rails

| Guard | Anchor (step / step_all path) | Behaviour | Reward consequence |
|---|---|---|---|
| Pitch clamp | `:605-624` / `:389-408`; limits `:121-122` | Per-agent integer pitch counter; max 2 LookDown / 1 LookUp from level. Violation -> action rewritten to NoOp, `_futile_actions[agent] += 1`. Counter reset to 0 on respawn (`:602-603`) | Trainer drains `consume_futile()` (`:947-956`) and charges **-1.0 per futile event** (`multi_agent_craftium.py:1885-1893`); see 06-rewards.md |
| Idle guard | `_idle_guard` `:220-267`; constants `:119,123` | `_MAX_CONSECUTIVE_IDLE = 1`: the first NoOp in a row is rewritten to MoveForward. Cause recorded as `invalid_action` (clamped name) vs `explicit_noop`, tallied per agent (`idle_force_summary` `:269-276`) | None directly; prevents wasted steps. Invalid-name clamps additionally trigger the prompt warning above |
| Sustained Dig | `_SUSTAINED_TICKS` `:126-129`; loops `:659-670` / `:466-489` | Dig is held for 10 env steps × frameskip 3 = 30 physics ticks — breaks stone in one named action. Rewards summed over all held ticks | Summed tick rewards reported as one step reward |
| Jump expansion | `:653-657` / `:463-464` | Jump becomes a 2-tick macro: jump tick then MoveForward tick (so jumps clear obstacles instead of going straight up) | — |
| Auto-equip on Dig | `_find_best_tool` `:1344-1387`; `:637-651` / `:446-458` | Before a Dig, inventory file is scanned for the best tool (diamond 5 > iron 4 > stone 3 > wood 2 > gold 1) and one extra env tick presses its `SlotN` if not already wielded | Costs one underlying tick |
| Stuck detector | constants `:130-139`; `:563-600` / `:349-384` | Movement actions with <1.0 block XZ displacement for 8 consecutive steps -> warning logged, counters reset. NOTE: `_ESCAPE_SEQUENCE` (`:133-139`) and the `_escape_queue` drain (`:555-562`) exist, but nothing ever ENQUEUES the sequence — the escape macro is currently dead code; only the log line fires | — |

## 5. step / step_all semantics

**`step(action_str, agentId)`** (`:505-688`) — turn-based: the acting agent gets its action, every live agent gets NoOp, then the underlying env steps (× sustained ticks). An N-agent outer round costs N underlying ticks and later movers observe a world already advanced by earlier ones. Returns `(observations_dict, resolved_action_str)`. Rewards are NOT returned: raw per-step rewards land in `_step_rewards` (`get_step_reward` `:753-756`) and accumulate in `_rewards`, which `get_reward_summary` (`:703-724`) reads-and-resets into the LLM prompt. Despite the docstring (`:515-516`), invalid actions never raise `ValueError` — they are recovered or clamped (Section 3).

**`step_all(actions: dict[int, str])`** (`:411-503`) — simultaneous-move, opt-in via `--simultaneous`: all agents' primitives are resolved through the shared `_resolve_action_for_agent` (`:287-409`, a verbatim lift of `step()`'s guard block; parity checked by `scripts/parity_test_stepping.py`), then ONE batched equip tick (if any agent Digs), then per-agent tick schedules (`Jump -> [Jump, MoveForward]`, `Dig -> [Dig]*10`, else 1 tick) are zipped and the env steps `max_ticks` times with everyone acting on the same `s_t`; finished schedules pad with NoOp (`:460-489`). Returns `(observations, resolved: dict[int, str])`. Terminated agents (absent from `env.agents`) are skipped (`:441-443`); the tick loop breaks early once every live agent is terminated/truncated (`:484-489`). See memory/`07-orchestrator.md` for why turn-based stepping hurt anvil co-op.

**Per-agent error surfacing** — terminations/truncations are cached per agent (`any_done`/`all_done` `:726-740`); a terminated agent's pitch counter resets; invalid actions surface as prompt warnings, never exceptions.

**`episode_over()`** (`:742-751`) — polls for `{world_path}/episode_over.txt`, written by `deaths.lua` on a Ch5 team wipe. Explicit flag because Ch4 force-respawns mean agents are never all "terminated", and a Ch5 permadeath may not map to a persistent PettingZoo termination. Cleared by Lua at episode reset.

**Event drains** (rewards detailed in 06-rewards.md): `poll_milestone_events` (`:1426-1478`), `poll_anvil_coop_events` (`:1493-1535`, no reward — diagnostics only), `poll_death_events` (`:1550-1623`; would-die -10 capped at once per agent per episode in Python via `_woulddie_charged` `:1610-1613`, cleared by `reset_death_offset` `:1625-1639`), `tail_server_log` (`:1651-1683`). All use byte-offset bookkeeping with auto-rewind on file truncation.
> PAPER MISMATCH — would-die scope/frequency: see PAPER_INCONSISTENCIES.md #6

**Warmup & chamber force-teleports** (used by the orchestrator, see 07-orchestrator.md): `warmup_noop()` (`:1159-1165`) keeps TCP channels alive during media load without advancing step counters; `signal_warmup_complete()` (`:1176-1209`) atomically writes `warmup_complete.txt` so Lua's Ch1 timeout counts from warmup end, not server start. `force_ch1_teleport()`..`force_ch4_teleport()` (`:1211-1257`) atomically write `ch{N}_force_teleport.txt` via `_write_force_teleport_flag` (`:1259-1284`); `doors.lua` polls and teleports all agents to the next chamber's fallback spawn. Fired by the Python step loop at 40% / 60% / 80% of `max_steps` for Ch2/3/4 (Ch1 is Python-driven because the Lua tick-based timeout was observed never to fire). Teleports relocate only — no milestone is credited, preserving honest cooperation measurement (see 09-metrics-and-evaluation.md).

**Chamber localisation** — `get_chamber()` maps agent Z-position to ch1..ch5 via `_CHAMBER_BOUNDS` (`:1401-1424`); per-chamber `{chamber_state}` prompt strings (door/anvil/cell status read from Lua-written world files) are produced by `get_chamber_state` (`:758-945`); see 05-five-chambers-world.md.
