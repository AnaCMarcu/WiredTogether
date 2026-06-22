# 05 — Five Chambers World (Lua mod)

**Source files:** `src/marl_craftium/craftium-envs/five-chambers/mods/five_chambers/` — `init.lua`, `config.lua`, `util.lua`, `world_gen.lua`, `milestones.lua`, `anvil.lua`, `switches.lua`, `doors.lua`, `gear.lua`, `mobs.lua`, `deaths.lua`, `player_state.lua`, `state_files.lua`
**Paper sections:** §A (environment), Table 2 (milestone ladder), Table 3 (penalties), §A.2–A.4
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

The world is a sequential gauntlet of five bedrock boxes connected south→north (increasing z).
Progression is one-way: doors unlock forward only, and per-chamber timeout teleports
(flag files written by Python) force the team onward if it stalls. Python-side counterpart:
`src/mindforge/custom_environment_craftium.py` (see 04-environment-interface.md).

## 1. World layout

All geometry derives from `config.lua` at load time. `FLOOR_Y=10`, `CEIL_Y=15` (config.lua:41-42);
walls are unbreakable `mcl_core:bedrock`. Chamber membership is decided by
`get_chamber_for_pos` (util.lua:5-31), mirrored by `_CHAMBER_BOUNDS` in
custom_environment_craftium.py:1401.

| Zone | x-range | z-band | Source | Notes |
|---|---|---|---|---|
| Ch1 solo_learning (16×16) | 0–15 | 0–15 | config.lua:55 | Extra dirt layer at y=11; agents stand at y=12; taller ceiling y=16 (config.lua:51-52) |
| Door 1 + corridor | x=6±1 (3-wide) | z=15 wall, corridor z=16 | config.lua:101; world_gen.lua:156-167, 237 | Stays locked; opened by first m2–m7 milestone or bypassed by teleport |
| Ch2 anvil_coop (9×9) | 2–10 | 17–25 | config.lua:123 | Two anvil pillars at (6,·,19) and (6,·,22) |
| Door 2 | x=6 | z=26 | config.lua:127 | Opens `DOOR2_DELAY=20` env-steps after both anvils break |
| Ch3 switch_puzzle (width 4N+1) | 0–4N (0–12 for N=3) | 27–45 | config.lua:136-146 | Cells z=28–30, front wall z=31, communal z=32–44, Door 3 at (2N, 45) |
| Ch4 combat (11×11) | 1–11 | 47–57 | config.lua:150 | 3 zombies on entry |
| Door 4 | x=6 | z=58 | config.lua:151 | Opens when all Ch4 mobs dead |
| Ch5 boss (9×9) | 2–10 | 59–67 | config.lua:155 | No exit; episode ends on boss death or team wipe |

Spawns: on join each `agentN` is teleported to its Ch1 spawn (`CH1_SPAWNS_3`,
config.lua:69-73; canonical lookup `ch1_spawn_pos`, util.lua:70-82) — positions chosen
3+ blocks from walls with a diggable target in view. Ch1 is a *time-gated* practice
window: after `CH1_TIMEOUT_TICKS` (default 1200 Lua ticks, env-var overridable,
config.lua:108-109) or on Python's `ch1_force_teleport.txt` flag, agents are teleported
across the still-locked Door 1 to `CH2_FALLBACK_SPAWNS_3` (doors.lua:425-584) and all
unearned Ch1 milestones are forfeited (`forfeit_track_milestones`, milestones.lua:211-220).
Equivalent flag-driven teleports exist for Ch2→Ch3, Ch3→Ch4, Ch4→Ch5
(doors.lua:598-659); each suppresses the next chamber's entry milestone
(m16/m20/m24) so a rescued team earns nothing for being relocated.

## 2. Per-file responsibilities

| File | Responsibility |
|---|---|
| `init.lua` | Module loader (init.lua:20-31); server-start init via `on_mods_loaded` (37-57); HUD/hudbars hiding + formspec force-close every tick (72-179); join/respawn handlers (97-141); episode-reset handler on `craftium_channel` — rebuilds world, relocks doors, wipes inventories, re-teleports with retry (185-328) |
| `config.lua` | All tunables: agent count, chamber bounds, spawns, anvil/boss constants, `DEBUG_SINGLE` solo-walkthrough overrides (see §7) |
| `util.lua` | `get_chamber_for_pos` (5-31), `agent_index` accepting `agent0`/`agent_0` (43-50), cell/fallback spawn helpers (53-133) |
| `world_gen.lua` | Builds all five bedrock boxes via VoxelManip (bypasses mcl_observers' set_node hook); places trees/stone pillars/dirt (Ch1), anvil pillars (Ch2), cells+switches (Ch3), doors, ceiling glowstone; `build_all_chambers` (451-457), re-run on every episode reset |
| `milestones.lua` | `MILESTONE_DEFS` table (18-60), `fire_milestone` with per-agent once-per-episode dedup (137-187), dig/pickup/movement trackers (240-365), animal-kill counter (223-236) |
| `anvil.lua` | Purple anvil node + gray pedestal registration (61-110); HP accumulation/decay globalstep, coop event logging, break handling (142-249) |
| `switches.lua` | Cell switch node; rotational (i+1)%N door wiring; fires m17/m18 (37-81) |
| `doors.lua` | Door open/relock primitives via VoxelManip (198-212), Door 1–4 + cell-door unlock logic, door-state IPC files, Ch2→Ch3 cell teleport (363-404), all timeout teleports |
| `gear.lua` | `give_gear_to_all`: auto-equip diamond sword (wield slot 1) / chestplate (armor slot 2) to every agent on anvil break; fires m14/m15 (52-82) |
| `mobs.lua` | Entity patching for damage attribution (15-43); chicken egg-laying disabled (53-62); Ch1 animal spawns + kill detection; Ch4 zombie spawn/kills (m20–m23); Ch5 boss lifecycle (m24–m28); reset-time mob despawn sweep (194-231) |
| `deaths.lua` | Virtual-HP invincibility for Ch1–Ch4, Ch5 permadeath, `episode_over.txt` flag (see §4) |
| `player_state.lua` | Per-agent health/hunger/inventory state files every 20 ticks; hunger drain neutralised by monkey-patching `mcl_hunger.exhaust` to a no-op (34-45); `anvils.txt` HP/punchers file (109-131) |
| `state_files.lua` | JSONL emitters for milestones/deaths/switches + `clear_state_files` at reset (see §6) |

## 3. Canonical milestone table (renumbering rosetta stone)

> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #2 (paper Table 2 renumbers contiguously
> M1–M24+M_door1; paper *text* quotes the code ids below). Reward values themselves are
> CONSISTENT with Table 2 (#12). Pinned by `tests/test_lua_spec.py`.

All definitions: milestones.lua:18-60. Every entry is `once=true` — at most once per agent
per episode, deduped in `fire_milestone` (milestones.lua:137-187). Code ids m10–m13 do not
exist (legacy 6-anvil design removed; `m9_anvil_B1` was renamed from `m11`).

| Code id | Chamber/track | Trigger | Reward | Credit goes to | Paper Table-2 |
|---|---|---|---|---|---|
| `m1_move_5` | ch1_solo | move >5 blocks (XZ) from spawn (milestones.lua:317-327) | 10 | each agent | M1 |
| `m2_dig_3_any` | ch1_solo | dig 3 blocks of any type (240-281) | 30 | each agent | M2 |
| `m3_pickup_3` | ch1_solo | collect 3 items attributed to own digs (329-361) | 30 | each agent | M3 |
| `m4_dig_5_wood` | ch1_solo | dig 5 `tree`-group blocks | 50 | each agent | M4 |
| `m5_kill_1_animal` | ch1_solo | kill 1 Ch1 animal (223-236) | 50 | killer | M5 |
| `m6_kill_2_animals` | ch1_solo | kill 2 Ch1 animals | 80 | killer | M6 |
| `m7_dig_3_stone` | ch1_solo | dig 3 `stone`-group blocks | 60 | each agent | M7 |
| `m_door1_open` | ch1_solo | be the first agent whose m2–m7 unlocks Door 1 (174-186) | 50 | one agent (the unlocker) | M_door1 |
| `m8_anvil_A1` | ch2_anvils | break sword anvil, row A (anvil.lua:208-214) | 40 | active diggers | M8 |
| `m9_anvil_B1` | ch2_anvils | break chestplate anvil, row B | 40 | active diggers | M9 |
| `m14_sword_equipped` | ch2_gear | sword auto-equipped on row-A break (gear.lua:52-78) | 50 | every agent | M10 |
| `m15_chestplate_equipped` | ch2_gear | chestplate auto-equipped on row-B break | 30 | every agent | M11 |
| `m16_enter_cell` | ch3_switch | teleported into own isolation cell via Door 2 (doors.lua:363-404) | 20 | each agent | M12 |
| `m17_switch_pressed` | ch3_switch | punch own cell's switch (switches.lua:61) | 40 | presser | M13 |
| `m18_door_opened` | ch3_switch | own cell door opened by a teammate's switch (switches.lua:63-74) | 60 | freed agent | M14 |
| `m19_all_in_communal` | ch3_switch | all N agents simultaneously in communal room (doors.lua:317-340) | 100 | team (all present) | M15 |
| `m20_enter_ch4` | ch4_combat | enter Ch4 (mobs.lua:261-289) | 30 | each agent | M16 |
| `m21_first_mob_kill` | ch4_combat | land the killing blow on a Ch4 zombie (mobs.lua:329-334) | 60 | killer | M17 |
| `m22_all_mobs_killed` | ch4_combat | all Ch4 zombies dead (mobs.lua:348-359) | 150 | contributors with damage ≥ `MIN_DAMAGE_FOR_CREDIT` | M18 |
| `m23_all_alive_ch4` | ch4_combat | all zombies dead AND zero Ch4 would-die events for any agent (mobs.lua:361-381) | 100 | team | M19 |
| `m24_enter_ch5` | ch5_boss | enter Ch5 (mobs.lua:492-531) | 50 | each agent | M20 |
| `m25_first_boss_dmg` | ch5_boss | first agent reaches ≥5 cumulative dmg on boss (mobs.lua:563-577) | 80 | qualifying contributors | M21 |
| `m26_boss_half_hp` | ch5_boss | boss HP ≤ BOSS_HP/2 = 30 (mobs.lua:579-591) | 120 | qualifying contributors | M22 |
| `m27_boss_defeated` | ch5_boss | boss HP = 0 (mobs.lua:392-409) | 300 | qualifying contributors | M23 |
| `m28_all_alive_bonus` | ch5_boss | boss dead, all N agents alive and each dealt ≥5 dmg (mobs.lua:411-425) | 250 | team | M24 |

Free-rider deterrent: `MIN_DAMAGE_FOR_CREDIT = 5` HP (config.lua:196) filters m22/m25/m26/m27/m28
contributor lists. Communication milestones (`m_comm_ch*`) are Python-side — see 06-rewards.md.

## 4. Deaths (deaths.lua)

Design driver: a headless bot never clicks the engine "You died" respawn formspec, and a dead
Craftium client freezes (deaths.lua:1-22). So Ch1–Ch4 use *virtual-HP invincibility* and only
Ch5 has real death. `minetest.show_death_screen` is no-op'd entirely (deaths.lua:49-51).

| Chamber | Mechanism | Penalty | Anchor |
|---|---|---|---|
| Ch1–Ch3 | hpchange **modifier** returns 0 (no real HP loss); per-agent virtual pool `_virtual_hp` (start 20) drains by the would-be damage; on lethal drain it silently refills — **no penalty, no event** | none | deaths.lua:92-154 (registered with `true` = modifier) |
| Ch4 | same invincibility, but a lethal drain emits a `woulddie` event: −10 via `emit_death_event`, `would_die_count`/`would_die_count_ch4` counters increment, pool refills — **fires per event, repeatable** | −10 per would-die event | deaths.lua:121-150 |
| Ch5 | hpchange passes through untouched (deaths.lua:103); real death → `register_on_dieplayer` emits −50, sets `player_dead[name]`, agent stays down (permadeath, no respawn) | −50 once (terminal) | deaths.lua:157-192 |

> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #6 (paper says −10 "once per episode, Ch1–4";
> code is per-event and Ch4-only).

When every connected agent is `player_dead`, `_signal_episode_over` writes
`{worldpath}/episode_over.txt` (deaths.lua:55-69), polled by
`CraftiumEnvironmentInterface.episode_over` (custom_environment_craftium.py:742-751) to end the
episode. Both penalties reach Python ONLY via the `death_events.jsonl` drain — the in-Lua
`craftium.reward()` calls are a non-functional backup in multi-agent mode (deaths.lua:127-133;
see §6 and 06-rewards.md). Ch4 near-death state feeds m23; agents are full-healed once on Ch5
entry so the boss fight starts at 20 HP (mobs.lua:503-512).

## 5. Mechanics

**Anvils (anvil.lua).** Exactly 2 anvils in Ch2 (`anvil_positions`, anvil.lua:30-57 — single
source of truth shared with world_gen.lua): row A at (6,12,19) drops swords, row B at (6,12,22)
drops chestplates; each sits on a cosmetic non-interactive pedestal. Punching an anvil stamps
the puncher's tick (anvil.lua:73-88). A globalstep counts punchers active within
`ACTIVE_WINDOW=30` ticks (~10 env steps) and advances HP by `dig_rate − DECAY_RATE`:
n=1 → 1−1 = **net 0** (solo is unproductive, not punished); n=2 → 4−1 = **+3/tick**;
n≥3 → 8−1 = **+7/tick** (anvil.lua:199-206; rates config.lua:177-183). At
`ANVIL_MAX_HP=20` the anvil breaks: milestone fires for the active diggers, gear is
distributed to all (gear.lua), and the node is destroyed for the episode (anvil.lua:208-247).
`DIGGER_RADIUS=3` exists in config but proximity is enforced by punching itself, not a radius
check. ≥2 simultaneous punchers also append a no-reward diagnostic row to
`anvil_coop_events.jsonl` (anvil.lua:165-197) for the cooperation metric
(see 09-metrics-and-evaluation.md). Both anvils broken → `start_door2_countdown()`.

**Switches (switches.lua).** Ch3 locks each agent i in cell i. The switch in cell i opens the
door of cell `(i+1) % NUM_AGENTS` (switches.lua:23-25) — nobody can free themselves. Pressing
(one-shot per switch): opens the target cell door, emits a `switch_events.jsonl` row, fires
m17 for the presser and m18 for the freed agent (switches.lua:41-79). Door 3 then opens when
all N agents stand in the communal room simultaneously (m19, doors.lua:317-340).

**Mobs (mobs.lua).** Ch1: 5 chickens + 3 sheep at fixed positions, pinned against despawn
(92-133); egg-laying disabled (53-62). Kill attribution patches `on_punch` of
chicken/sheep/zombie to record last-puncher and per-agent cumulative damage (15-43).
Ch4: `min(NUM_AGENTS, 3)` zombies spawn on first agent entry (235-257) — 1 in solo mode.
Known issue: `CH4_SPAWN_POSITIONS` (mobs.lua:81-85: z=54/57/60) predate the z-shift of Ch4 to
47–57, so zombie 2 spawns in the Door-4 passage opening and zombie 3 spawns *inside Ch5*
(z=60), making m22/Door 4 unreachable by normal play until the Ch4→Ch5 force teleport.
Ch5: boss is a `mobs_mc:zombie` with HP overridden to `BOSS_HP=60` (459-488); in
`DEBUG_SINGLE` it is 20 (config.lua:207-211). A boss that vanishes with HP>0 (reset/unload)
is NOT a defeat (`boss_vanished_handler`, 446-457). Boss death writes `episode_done.txt`
and calls `craftium.terminate()` (392-437).

**Gear (gear.lua).** On anvil break, `give_gear_to_all` gives every agent a diamond sword
(main inventory + wield slot 1) or diamond chestplate (armor list slot 2) directly — the
dropped-item pickup mechanic was removed as untrainable — and fires m14/m15 per recipient
(gear.lua:52-82).

**Doors (doors.lua).** Doors are red glowing `five_chambers:door_locked` blocks swapped to
air via VoxelManip (198-212). Unlock conditions: Door 1 — first m2–m7 milestone
(`open_door1`, 238-249; or bypassed-not-opened by timeout); Door 2 — both anvils broken
+ 20-env-step countdown (252-279); cell doors — teammate's switch (293-313); Door 3 — all
agents in communal (317-340); Door 4 — all Ch4 mobs dead (349-356). Crossing Door 2
teleports each agent into its own cell (m16) and relocks Door 2 after all N transit
(363-404). All doors relock and state files clear at episode reset (133-153, init.lua:207-214).

## 6. Lua→Python IPC (state_files.lua, player_state.lua, doors.lua)

All files live in `{worldpath}/`. JSONL is built by hand (no JSON lib). Python reads JSONL by
byte-offset each step; one-shot flags by existence. Polling lives in
`CraftiumEnvironmentInterface` (custom_environment_craftium.py) and the orchestrator drains
rewards in `multi_agent_craftium.py:1979-2030` — see 04-environment-interface.md and
07-orchestrator.md. `clear_state_files()` (state_files.lua:106-140) deletes everything at reset.

| File | Writer | Schema / format | Python reader |
|---|---|---|---|
| `milestone_events.jsonl` | `emit_milestone` (state_files.lua:8-61) | `{"step":int,"milestone":id,"contributors":[names],"reward":int}` | `poll_milestone_events` (:1426) — authoritative milestone reward channel |
| `death_events.jsonl` | `emit_death_event` (state_files.lua:74-89) | `{"step":int,"kind":"death"\|"woulddie","agent":name,"chamber":str,"reward":int}` | `poll_death_events` (:1550) — authoritative −10/−50 channel; rate-limits would-die to 1/step |
| `switch_events.jsonl` | `emit_switch_event` (state_files.lua:92-102) | `{"step":int,"switch":"A","door_opened":"B","presser":name}` | switch metric |
| `anvil_coop_events.jsonl` | anvil.lua:177-196 | `{"step":int,"anvil":pos,"row":"A"\|"B","n_active":int,"active":[names]}` | `poll_anvil_coop_events` (:1493), no reward |
| `episode_over.txt` | `_signal_episode_over` (deaths.lua:55-69) | flag, content `all_dead` | `episode_over` (:742) — team-wipe episode end |
| `episode_done.txt` | `fire_boss_death` (mobs.lua:428-433) | flag, content = step counter | episode end on boss kill |
| `door{1..4}_state.txt`, `cell_doors_state.txt` | doors.lua:222-233 + open_* fns | `open\n`; cell file `i:open` per line | chamber-state composer → LLM prompt text (see 08-cognitive-agent.md) |
| `health_agent{i}.txt`, `hunger_agent{i}.txt`, `inv_agent{i}.txt` | player_state.lua:47-101, every 20 ticks | `"{hp}/20"`; `"20/20"` (hunger always full); `"{wield_idx}\|{slot}..."` | per-agent observation fields |
| `anvils.txt` | player_state.lua:109-131 | `sword\|<hp>/<max>\|<active names>` per anvil | LLM chamber-state |
| `ch{1..4}_force_teleport.txt` | **Python** (chamber budget timer) | flag | consumed+deleted by doors.lua timeouts |
| `warmup_complete.txt` | **Python** | flag | anchors Lua's Ch1 timeout countdown (doors.lua:459-484) |

> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #1 (Python-side comm-milestone reward values)
> for the only reward channel not emitted by this mod.

## 7. config.lua tunables

| Constant | Value | Anchor | Meaning |
|---|---|---|---|
| `NUM_AGENTS` | 3 | config.lua:5 | Drives Ch3 width (4N+1), cell/switch count, fallback spawns, m19/m28 quorum |
| `DEBUG_SINGLE` | false | :19-22 | Solo human walkthrough: N=1, any player = agent_0, Door 2 pre-opened, SOLO_DIG_RATE=4, BOSS_HP=20 (:207-211) |
| `CHAMBERS[n].enabled` | all true | :27-33 | Disable to skip a chamber (left void, door open) |
| `FLOOR_Y` / `CEIL_Y` | 10 / 15 | :41-42 | Ch1: dirt at 11, ceiling 16 (:51-52) |
| `CH1_TIMEOUT_TICKS` | 1200 (env-var overridable) | :108-109 | Lua-side Ch1 safety-net timer; 20 Hz ticks, 3 ticks = 1 env step |
| `DOOR2_DELAY` | 20 | :128 | Env steps between 2nd anvil break and Door 2 opening |
| `ANVIL_MAX_HP` | 20 | :177 | Was 30; ~7 ticks of pair digging |
| `SOLO/PAIR/TRIO_DIG_RATE` | 1 / 4 / 8 | :178-180 | HP per tick by simultaneous puncher count |
| `DECAY_RATE` | 1 | :181 | Subtracted every tick (solo = net 0) |
| `ACTIVE_WINDOW` | 30 ticks | :182 | Punch recency window (~10 env steps); band-aid for round-robin stepping |
| `DIGGER_RADIUS` | 3 | :183 | Declared but unused by the punch-based mechanic |
| `BOSS_HP` / `BOSS_DMG` | 60 / 3 | :186-187 | BOSS_DMG not wired to the entity (VoxeLibre default melee applies) |
| `MIN_DAMAGE_FOR_CREDIT` | 5 | :196 | Contributor threshold for m22/m25–m28 |
