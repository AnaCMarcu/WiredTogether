# The Five Chambers — Implementation Plan

> A cooperative multi-agent Craftium environment designed to generate strong Hebbian
> social plasticity signal through **physical cooperation** (heavy anvils) and
> **verbal cooperation** (switch-room communication puzzle).
>
> **Status:** Design locked. Ready for implementation.
>
> **Previously known as "The Four Chambers"** — we added a communication puzzle
> chamber between the anvil room and the combat room.

---

## 1. Purpose

Replace the sparse `voxel-libre2` environment with a dense, progression-based world
designed to:

- Generate **frequent, unambiguous cooperation events** (for RL gradient signal)
- Force both **physical co-activity** (Hebbian `cij` from proximity) and **verbal
  coordination** (communication-forced cooperation)
- Produce **shared positive rewards** (for LTP via `m_ltp * cij * (1 - W)`)
- Stay **RL-learnable** within 30-50 episodes

---

## 2. World Layout

Five chambers in a linear sequence. Floor Y=10, ceiling Y=15. Walls are
`mcl_core:bedrock` (unbreakable, prevents digging out).

```
                      [EPISODE END]
                            │
                            ▼
                    ┌───────────────┐
                    │   CHAMBER 5   │   Boss fight (5×5)
                    │  [Boss: 60HP] │
                    └───────▲───────┘
                     ═══ no door ═══
                    ┌───────────────┐
                    │   CHAMBER 4   │   Combat practice (7×7)
                    │  [3 zombies]  │
                    └───────▲───────┘
                     ═══ Door 4 (mob-gated) ═══
                    ┌───────────────┐
                    │   CHAMBER 3   │   ★ NEW — Communication puzzle
                    │   [3 isolation│   • 3 cells + 1 communal room
                    │    cells +    │   • Rotational switch mapping
                    │    communal]  │   • Forces verbal coordination
                    └───────▲───────┘
                     ═══ Door 3 (agents-regrouped) ═══
                    ┌───────────────┐
                    │   CHAMBER 2   │   Cooperative gear (10×10)
                    │  [6 anvils]   │   • Heavy anvils, require 2+ agents
                    └───────▲───────┘   • Produces swords + chestplates
                     ═══ Door 2 (anvil-gated) ═══
                    ┌───────────────┐
                    │   CHAMBER 1   │   Solo learning (12×12)
                    │[trees/stone/  │   • Dig + kill basic mobs
                    │  animals]     │
                    │ [SPAWN x3]    │   • Door 1 always open
                    └───────────────┘
                        [START]
```

### Chamber coordinates

| Chamber | X range | Z range | Size | Notes |
|---------|---------|---------|------|-------|
| Ch1 | 0–11 | 0–11 | 12×12 | Spawn points at 3 corners |
| Door 1 | x=6 | z=12 | 1×2 | Always open |
| Ch2 | 2–11 | 13–22 | 10×10 | 6 anvils |
| Door 2 | x=6 | z=23 | 1×2 | Opens 20 steps after 6th anvil broken |
| **Ch3** | **0–12** | **24–38** | **13×15** | **Switch puzzle (new, widened)** |
| Door 3 | x=6 | z=38 (in Ch3 north wall) | 1×2 | Opens when all 3 agents in communal |
| Ch4 | 3–9 | 40–46 | 7×7 | 3 weak zombies |
| Door 4 | x=6 | z=47 | 1×2 | Opens when all zombies dead |
| Ch5 | 4–8 | 48–52 | 5×5 | Boss |

Spawn points (Ch1 corners): `(1,11,1)`, `(10,11,1)`, `(5,11,10)`.

### 2.3 Resource placement (Chamber 1)

Trees (`mcl_core:tree` + leaves) at 8 positions:
`(2,11,2) (5,11,3) (8,11,2) (3,11,7) (9,11,6) (7,11,9) (2,11,8) (10,11,4)`

Stone blocks (`mcl_core:stone`) at 8 positions:
`(4,11,4) (3,11,5) (6,11,6) (8,11,7) (5,11,8) (4,11,2) (9,11,3) (7,11,5)`

Animals (spawned once on server start, free-roaming within Ch1):
- 5 chickens (`mobs_mc:chicken`) at random Ch1 positions
- 3 sheep (`mobs_mc:sheep`) at random Ch1 positions

Total animals = 8. This supports M5 (kill 1) and M6 (kill 2 — thresholds well
below supply). No animals respawn during an episode.

---

## 3. Chamber 3: The Switch Room (NEW)

### 3.1 Concept

When Door 2 unlocks, each agent is **teleported** into a separate isolation cell.
Each cell has one switch. The switches are wired so that **no switch opens the
door of the agent pressing it** — instead, each switch opens a specific teammate's
door. All three doors exit into a shared communal room.

Because agents cannot see into other cells, they must **communicate verbally** to:
1. Announce which cell they are in
2. Coordinate who should press their switch and when
3. Confirm whose door opened
4. Regroup in the communal room

### 3.2 Layout

Chamber 3 occupies a 13×15 footprint (X:0–12, Z:24–38). Cells are 3×3 each with
bedrock walls between them (no shared walls). All walls, floors, and ceilings
use `mcl_core:bedrock`. Door barriers are also bedrock (replaced with air when
unlocked).

```
                           Z=38 ─ North wall with Door 3 opening at X=6
                           │
                           ▼
   X:  0  1  2  3  4  5  6  7  8  9  10 11 12
      ┌──────────────────────────────────────┐
      │   COMMUNAL ROOM  (13 wide × 9 deep)  │   Z: 29 to 37
      │   all 3 agents must regroup here     │
      │                                      │
      │                                      │
      └──┬───────┬───────┬───────┬───────┬───┘   Z = 28  (front wall of cells)
         │Door A │       │Door B │       │Door C│       (barriers at X=2, 6, 10)
      ┌──┴───────┴───────┴───────┴───────┴───┐   Z = 27  (cell front edge, open inside)
      │ Cell A  ║  Cell B  ║  Cell C        │
      │ X:1-3   ║  X:5-7   ║  X:9-11        │   Z: 25 to 27  (3x3 cell interior)
      │ [sw A]  ║  [sw B]  ║  [sw C]        │
      │         ║          ║                │
      └─────────┴──────────┴────────────────┘   Z = 24  (back wall of cells)
             ║ X=4 ║    ║ X=8 ║                (inter-cell bedrock walls)
             teleport targets: (2,11,26), (6,11,26), (10,11,26)
```

Exact positions:

| Element | Position | Notes |
|---------|----------|-------|
| Cell A interior | X:1–3, Z:25–27 | 3×3 floor, ceiling at Y=15 |
| Cell B interior | X:5–7, Z:25–27 | 3×3 |
| Cell C interior | X:9–11, Z:25–27 | 3×3 |
| Inter-cell walls | X=4, X=8 (full height Y:11–15, Z:24–28) | Bedrock |
| Back wall of cells | Z=24 | Bedrock, full width X:0–12 |
| Front wall of cells | Z=28 | Bedrock with 3 door openings |
| Switch A | `(2, 11, 25)` | On back wall, center of Cell A |
| Switch B | `(6, 11, 25)` | On back wall, center of Cell B |
| Switch C | `(10, 11, 25)` | On back wall, center of Cell C |
| Door A barrier (locked) | `(2, 10, 28)` and `(2, 11, 28)` | Bedrock |
| Door B barrier (locked) | `(6, 10, 28)` and `(6, 11, 28)` | Bedrock |
| Door C barrier (locked) | `(10, 10, 28)` and `(10, 11, 28)` | Bedrock |
| Communal room interior | X:0–12, Z:29–37 | 13×9 |
| North wall (with Door 3 opening) | Z=38, solid except at X=6 | Bedrock |
| Door 3 barrier (locked) | `(6, 10, 38)` and `(6, 11, 38)` | Bedrock |
| Teleport target A | `(2, 11, 26)` | Center of Cell A |
| Teleport target B | `(6, 11, 26)` | Center of Cell B |
| Teleport target C | `(10, 11, 26)` | Center of Cell C |

### 3.3 Switch wiring (rotational mapping)

```
Switch A  →  opens Door B
Switch B  →  opens Door C
Switch C  →  opens Door A
```

This is symmetric (no "special" agent) and makes the puzzle unambiguous:
**every agent needs exactly one specific teammate to press their switch**.

The mapping is **fixed** across episodes (not randomized) to keep the task
RL-learnable. It is **disclosed in the environment prompt** so LLM agents can
reason about it, but agents still must communicate to coordinate timing and
verify success (because they can't see into other cells).

### 3.4 Switch mechanics

- Switch node: custom `five_chambers:switch` node (one per cell). Registered with
  `on_rightclick` or `on_punch` handler — use whichever the existing `mcl_core:lever`
  registration pattern exposes.
- One-shot: pressing a switch permanently opens the target door. Pressing again
  has no effect.
- Open = bedrock door blocks (both Y=10 and Y=11) replaced with air.
- **Broadcast mechanism**: on every switch press, Lua writes a line to
  `state_files/switch_events.jsonl` under the world directory:
  ```json
  {"step": 842, "switch": "A", "door_opened": "B", "presser": "agent_0"}
  ```
  The Python main loop polls this file each step (same pattern the existing
  `voxel-libre2` env uses for inventory/health) and injects a team-chat message
  into every agent's next observation: `"[SYSTEM] Switch A was pressed."` This
  is the only way agents in sealed cells can verify coordination.

### 3.5 Door 3 unlock condition

Door 3 (Ch3 → Ch4) opens when **all 3 agents are simultaneously in the communal
room** (Z: 29–37, X: 0–12). Detected via globalstep position check. Open = the
bedrock barrier at `(6, 10, 38)` and `(6, 11, 38)` is replaced with air.

### 3.6 Teleportation (20 steps after 6th anvil break)

The 6th anvil break **does not teleport immediately**. It starts a 20-step
countdown. This lets agents collect the 6th anvil's chestplate drop and finish
equipping gear before being separated.

```lua
-- In anvil.lua, after 6th break:
five_chambers.teleport_pending = true
five_chambers.teleport_countdown = 20

-- In globalstep:
if five_chambers.teleport_pending then
    five_chambers.teleport_countdown = five_chambers.teleport_countdown - 1
    if five_chambers.teleport_countdown <= 0 then
        do_teleport_and_lock_ch2()
        five_chambers.teleport_pending = false
    end
end

local function do_teleport_and_lock_ch2()
    -- Deterministic assignment by player name, not ipairs order
    local targets = {
        agent_0 = {x=2,  y=11, z=26},  -- Cell A
        agent_1 = {x=6,  y=11, z=26},  -- Cell B
        agent_2 = {x=10, y=11, z=26},  -- Cell C
    }
    for name, target in pairs(targets) do
        local p = minetest.get_player_by_name(name)
        if p then p:set_pos(target) end
    end
    -- Lock Door 2 behind them so they can't wander back
    five_chambers.doors.relock_door_2()
end
```

Assignment is **by explicit player name** (not connection order) to guarantee
agent 0 → Cell A, agent 1 → Cell B, agent 2 → Cell C deterministically across
runs. Agents retain inventory (Minetest default).

---

## 4. Heavy Anvil Mechanic (Chamber 2)

Parameters:

```
ANVIL_MAX_HP = 30        -- target progress to break
SOLO_DIG_RATE = 1        -- progress/step added while 1 digger is active
PAIR_DIG_RATE = 4        -- progress/step added while 2 diggers are active
TRIO_DIG_RATE = 8        -- progress/step added while 3 diggers are active
DECAY_RATE = 2           -- progress lost/step, ALWAYS APPLIED (even during digging)
ACTIVE_WINDOW = 2        -- digger counts as "active" for 2 env steps after dig
DIGGER_RADIUS = 3        -- agent must be ≤3 blocks from anvil
```

**Timing note:** The Craftium env runs at `frameskip=3` (see `CraftiumEnvironmentInterface`
default). Each `env.step()` = 3 physics ticks. `ACTIVE_WINDOW = 2 env steps` means
**6 physics ticks** — a generous enough window for LLM agents whose inter-decision
latency can span multiple env steps. The anvil globalstep loop runs once per
physics tick in Lua, so the `step_counter` in `anvil.lua` ticks at 3× the
Python-visible step count. When comparing `step_counter - last_diggers[name]`,
the threshold must be `2 × frameskip = 6` in Lua ticks, not 2.

**Core rule: decay is always applied.** The per-step progress delta is
`dig_rate(n) − DECAY_RATE` for all `n`, including `n=1`. This is what makes
solo digging impossible.

Net dynamics (per env step, which is 3 physics ticks but the anvil update
runs once per env step so numbers below hold as-is):

| Active diggers | Raw dig rate | After decay | Outcome |
|---|---|---|---|
| 0 | 0 | −2 | decays toward 0 |
| 1 | +1 | −1 | solo impossible (net negative) |
| 2 | +4 | +2 | breaks in ~15 env steps (~45 physics ticks) |
| 3 | +8 | +6 | breaks in ~5 env steps (~15 physics ticks) |

Pseudocode (Lua-side, called once per env step from `register_globalstep`):

```lua
-- Per-step update per anvil
local active_n = count_active_diggers(anvil)  -- see §8-anvil impl
local dig_delta = ({[0]=0, [1]=1, [2]=4, [3]=8})[math.min(active_n, 3)]
anvil.hp = math.max(0, anvil.hp + dig_delta - DECAY_RATE)
if anvil.hp >= ANVIL_MAX_HP then break_anvil(anvil) end
```

6 anvils placed in Ch2 at fixed positions:

| Milestone | Position | Drops |
|-----------|----------|-------|
| M8  (`m8_anvil_A1`)  | `(3, 11, 15)` — Row A left    | Diamond sword |
| M9  (`m9_anvil_A2`)  | `(6, 11, 15)` — Row A middle  | Diamond sword |
| M10 (`m10_anvil_A3`) | `(9, 11, 15)` — Row A right   | Diamond sword |
| M11 (`m11_anvil_B1`) | `(3, 11, 18)` — Row B left    | Diamond chestplate |
| M12 (`m12_anvil_B2`) | `(6, 11, 18)` — Row B middle  | Diamond chestplate |
| M13 (`m13_anvil_B3`) | `(9, 11, 18)` — Row B right   | Diamond chestplate |

**Milestones are indexed by anvil position, not break order.** This guarantees
determinism and avoids tiebreaker issues when two anvils break on the same step.

Node type: `five_chambers:anvil` (custom node, obsidian-style texture, `groups =
{unbreakable=1}` so the engine refuses default removal — the mechanic removes
the node explicitly via `minetest.set_node(pos, {name="air"})` when `hp >=
ANVIL_MAX_HP`). Dig attempts are detected via `on_dig` hook returning `false`
AND via `on_punch` (registered as a fallback in case the specific Minetest
version ignores `on_dig` on unbreakable nodes — see §10 risks).

Row A anvils (first three by Z-row) drop `mcl_swords:sword_diamond`.
Row B anvils drop `mcl_armor:chestplate_diamond`.

### 4.6 Milestone event mechanism (★ critical — resolves position-indexed detection)

**Problem.** `craftium_metric.py`'s existing `_detect_milestone()` matches reward values to
milestone IDs via the `STAGE_REWARDS` set. With all 6 anvils emitting reward `40`,
it cannot distinguish M8 from M13 by value alone. Same for multi-mob milestones in Ch3/Ch4.

**Solution.** Lua writes a per-event JSONL file that Python polls each step, following
the same state-file pattern already used for inventory (`inv_agent{N}.txt`), health
(`health_{agent}.txt`), and phase (`phase.txt`).

**File:** `{world_path}/milestone_events.jsonl` — append-only, one JSON object per line:

```json
{"step": 42, "milestone": "m8_anvil_A1", "contributors": ["agent_0","agent_1"], "reward": 40}
{"step": 85, "milestone": "m17_switch_pressed", "contributors": ["agent_2"], "reward": 40}
{"step": 120, "milestone": "m18_door_opened", "contributors": ["agent_1"], "reward": 60}
{"step": 200, "milestone": "m22_all_mobs_killed", "contributors": ["agent_0","agent_1","agent_2"], "reward": 150}
```

**Lua side** (in `milestones.lua`, write helper):

```lua
local function emit_milestone(milestone_id, contributors, reward)
    local world_path = minetest.get_worldpath()
    local path = world_path .. "/milestone_events.jsonl"
    local f = io.open(path, "a")
    if f then
        local contrib_json = "[" .. table.concat(
            {("'" .. table.concat(contributors, "','") .. "'")}, ","
        ):gsub("'", '"') .. "]"
        f:write(string.format(
            '{"step":%d,"milestone":"%s","contributors":%s,"reward":%d}\n',
            five_chambers.step_counter, milestone_id, contrib_json, reward
        ))
        f:close()
    end
    -- Still call craftium.reward() so the engine's reward channel carries the
    -- value to Python for RL signal — just keep the value non-unique is fine.
    for _, name in ipairs(contributors) do
        local p = minetest.get_player_by_name(name)
        if p then craftium.reward(p, reward) end
    end
end
```

**Python side** (update `craftium_metric.py` to read the file each step):

```python
class CraftiumMetric:
    def __init__(self, ...):
        ...
        self._milestone_file_offset = 0   # byte offset into milestone_events.jsonl
        self._world_path = None           # resolved lazily on first call

    def poll_milestone_events(self, world_path: str) -> list[dict]:
        """Read any new lines from milestone_events.jsonl since the last poll.

        Call once per step BEFORE record_reward. Returns the list of new
        events; for each event, fire record_milestone() with the parsed ID.
        """
        path = os.path.join(world_path, "milestone_events.jsonl")
        if not os.path.exists(path):
            return []
        new_events = []
        with open(path, "r") as f:
            f.seek(self._milestone_file_offset)
            for line in f:
                try:
                    new_events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self._milestone_file_offset = f.tell()
        return new_events

    def record_milestone_event(self, event: dict):
        """Fire a milestone from a parsed Lua event (authoritative ID, not value-matched)."""
        mid = event["milestone"]
        contributors = event.get("contributors", [])
        step = event.get("step", self.timestep)
        reward = event.get("reward", 0)
        # Record in milestone_log (for CooperationMetric observe_milestone)
        for contrib_name in contributors:
            agent_id = int(contrib_name.split("_")[1])
            # Track which milestones each agent has fired; skip duplicates
            track = self._track_for_milestone(mid)
            if track and (step, mid) not in [(s, m) for s, m in self.milestones[agent_id][track]]:
                self.milestones[agent_id][track].append((step, mid))

    def _track_for_milestone(self, mid: str) -> str | None:
        """Map milestone ID to track (ch1_solo, ch2_anvils, etc.)."""
        for track, tier_list in TRACKS.items():
            if any(name == mid for name, _ in tier_list):
                return track
        return None
```

**Main loop integration** (in `multi_agent_craftium.py` step loop):

```python
# After env.step(actions), before record_reward:
world_path = environment._get_world_path()
new_events = metric.poll_milestone_events(world_path)
for event in new_events:
    metric.record_milestone_event(event)
    coop_metric.observe_milestone(
        step=event["step"],
        milestone_id=event["milestone"],
        contributors=event["contributors"],
    )
    episode_logger.log_event({"type": "milestone", **event})
```

This mechanism is used for **all multi-agent milestones** where value-matching
fails: M8–M13 (anvils), M17/M18 (switches/doors), M22 (mob kills), M25–M27 (boss).
Solo milestones M1–M7, M14–M16, M20, M24 can still use the existing reward-value
path since each emits a unique value.

---

## 4.5 Communication Rewards ★ NEW

To directly reinforce the verbal-coordination behavior that Hebbian plasticity
should amplify, the environment rewards inter-agent communication on top of the
milestone system. This runs **Python-side** (the LLM chat channel is in Python,
not in Lua) and is added to each agent's step reward before metrics.

### 4.5.1 Two-tier design

**Tier 1 — Per-message base reward (shaped, always-on)**

+2 reward per valid message sent, capped at 10 valid messages per agent per
episode → **max 20 per agent per episode**.

A message is "valid" if:
- Length ≥ 5 characters (discards "ok", "y", ".", etc.)
- Not identical to this agent's previous message (simple dedup)
- Sent while the agent is still alive and the episode is active

Purpose: combat silent policies; give RL a small gradient toward *any* chat.

**Tier 2 — Chamber communication milestones (discrete, contextual)**

These fire once per agent per chamber, tied to the chamber's cooperation
difficulty. They count messages sent **while the agent was physically in that
chamber**.

| Milestone | Trigger | Reward | Notes |
|-----------|---------|--------|-------|
| `m_comm_ch2` | ≥3 valid messages sent while in Ch2 | 20 | Per agent |
| `m_comm_ch3` | ≥2 valid messages sent while in Ch3 | 30 | Per agent — Ch3 is the pure comm chamber |
| `m_comm_ch4` | ≥2 valid messages sent while in Ch4 | 15 | Per agent |
| `m_comm_ch5` | ≥2 valid messages sent while in Ch5 | 20 | Per agent |

Ch3 gets the largest bonus because the switch puzzle is the only chamber where
communication is *literally* required for progress — we want to reward the
behavior precisely where it matters most.

### 4.5.2 Maximum communication bonus per agent

20 (Tier 1) + 20 + 30 + 15 + 20 (Tier 2) = **105 per agent per episode**

This is ~5% of the total reward budget (~2,000) — significant enough to shape
behavior but small enough not to dominate over task completion.

### 4.5.3 Anti-farming rules

Applied in order before a message counts toward either tier:

1. **Length check**: `len(message.strip()) >= 5`
2. **Dedup check**: `message != agent_previous_message`
3. **Rate limit**: no more than 1 valid message per 2 environment steps per
   agent (prevents within-turn burst spam if the LLM is called multiple times
   per step)
4. **Episode cap**: Tier 1 stops rewarding after 10 valid messages per agent

### 4.5.4 Implementation location

Communication is tracked in the Python multi-agent loop, not in Lua. Pseudocode:

```python
# In multi_agent_craftium.py step loop
for agent_id, message in chat_messages_this_step.items():
    if _is_valid_message(agent_id, message):
        # Tier 1
        if agent_comm_counts[agent_id] < 10:
            step_rewards[agent_id] += 2.0
            agent_comm_counts[agent_id] += 1

        # Track for Tier 2 per-chamber counter
        current_chamber = get_chamber(agent_positions[agent_id])
        if current_chamber in {"ch2", "ch3", "ch4", "ch5"}:
            chamber_msg_counts[agent_id][current_chamber] += 1

        # Fire Tier 2 milestone if threshold crossed
        _check_chamber_comm_milestones(agent_id, chamber_msg_counts)

        agent_previous_message[agent_id] = message
```

The `_is_valid_message` helper enforces the anti-farming rules.

`get_chamber` uses the same coordinate ranges as the Lua `get_chamber_for_pos`
helper — keep a Python mirror of the chamber bounds in `config.py`.

---

## 5. Milestone Specification

**32 milestones total** (28 task + 4 chamber communication), organized by chamber.

### Phase 1 — Chamber 1 (solo learning)

Chamber 1 contains: 8 trees, 8 stone blocks, 5 chickens, 3 sheep (values from §2.3).
Animals spawn once on server start.

| # | ID | Trigger | Reward | Recipient |
|---|-----|---------|--------|-----------|
| M1 | `m1_move_5` | Move >5 blocks from spawn | 10 | Each agent |
| M2 | `m2_dig_3_any` | Dig 3 blocks | 30 | Each agent |
| M3 | `m3_pickup_3` | Pick up 3 items | 30 | Each agent |
| M4 | `m4_dig_5_wood` | Dig 5 wood blocks | 50 | Each agent |
| M5 | `m5_kill_1_animal` | Kill 1 animal (chicken or sheep) | 50 | Each agent |
| M6 | `m6_kill_2_animals` | Kill 2 animals | 80 | Each agent |
| M7 | `m7_dig_3_stone` | Dig 3 stone | 60 | Each agent |

### Phase 2 — Chamber 2 (anvil cooperation)

Anvil milestones are indexed by **position**, not break order (see §4). This
means M8 is always the anvil at `(3,11,15)`, M9 at `(6,11,15)`, etc. An agent
is a "contributor" to a break if they used Dig within `ACTIVE_WINDOW` (2) steps
and within `DIGGER_RADIUS` (3) blocks at the moment HP reached 30.

| # | ID | Trigger | Reward | Recipient |
|---|-----|---------|--------|-----------|
| M8  | `m8_anvil_A1`  | Anvil at (3,11,15) broken | 40 | Each contributor |
| M9  | `m9_anvil_A2`  | Anvil at (6,11,15) broken | 40 | Each contributor |
| M10 | `m10_anvil_A3` | Anvil at (9,11,15) broken | 40 | Each contributor |
| M11 | `m11_anvil_B1` | Anvil at (3,11,18) broken | 40 | Each contributor |
| M12 | `m12_anvil_B2` | Anvil at (6,11,18) broken | 40 | Each contributor |
| M13 | `m13_anvil_B3` | Anvil at (9,11,18) broken | 40 | Each contributor |
| M14 | `m14_sword_equipped` | Diamond sword wielded | 50 | Each agent |
| M15 | `m15_chestplate_equipped` | Diamond chestplate equipped | 30 | Each agent |
| — | `m_comm_ch2` | ≥3 valid chat messages in Ch2 | 20 | Each agent (Python-side) |

### Phase 3 — Chamber 3 (switch puzzle) ★ NEW

| # | ID | Trigger | Reward | Recipient |
|---|-----|---------|--------|-----------|
| M16 | `m16_enter_cell` | Agent teleported into isolation cell | 20 | Each agent |
| M17 | `m17_switch_pressed` | Agent presses their switch | 40 | Each agent who presses |
| M18 | `m18_door_opened` | Agent's door opens (teammate helped) | 60 | Each agent freed |
| M19 | `m19_all_in_communal` | All 3 agents in communal room | 100 | Each agent |
| — | `m_comm_ch3` | ≥2 valid chat messages in Ch3 | 30 | Each agent (Python-side) |

### Phase 4 — Chamber 4 (combat)

| # | ID | Trigger | Reward | Recipient |
|---|-----|---------|--------|-----------|
| M20 | `m20_enter_ch4` | Enter Chamber 4 | 30 | Each agent |
| M21 | `m21_first_mob_kill` | First mob kill per agent | 60 | Each killer |
| M22 | `m22_all_mobs_killed` | All 3 zombies dead | 150 | Each contributor |
| M23 | `m23_all_alive_ch4` | Ch4 cleared with all alive | 100 | Each alive agent |
| — | `m_comm_ch4` | ≥2 valid chat messages in Ch4 | 15 | Each agent (Python-side) |

### Phase 5 — Chamber 5 (boss)

| # | ID | Trigger | Reward | Recipient |
|---|-----|---------|--------|-----------|
| M24 | `m24_enter_ch5` | Enter Chamber 5 | 50 | Each agent |
| M25 | `m25_first_boss_dmg` | Boss takes first damage | 80 | Each contributor |
| M26 | `m26_boss_half_hp` | Boss HP <50% | 120 | Each contributor |
| M27 | `m27_boss_defeated` | Boss killed | 300 | Each contributor |
| M28 | `m28_all_alive_bonus` | Boss killed, all alive | +250 | Each alive agent |
| — | `m_comm_ch5` | ≥2 valid chat messages in Ch5 | 20 | Each agent (Python-side) |
| — | `m_comm_base` | +2 per valid message (capped 10/episode) | up to 20 | Per agent (shaped, Python-side) |

### Reward budget per agent (full cooperation)

| Phase | Task reward | Comm reward | Total |
|-------|-------------|-------------|-------|
| Ch1 | 310 | — | 310 |
| Ch2 | 320 | 20 | 340 |
| Ch3 | 220 | 30 | 250 |
| Ch4 | 340 | 15 | 355 |
| Ch5 | 800 | 20 | 820 |
| Base chat (episode-wide) | — | 20 | 20 |
| **Total** | **~1,990** | **~105** | **~2,095** |

Communication rewards are roughly 5% of the total budget — enough to shape
policy behavior toward talking, not enough to let an agent win by spamming
alone.

---

## 6. File Structure

### 6.1 New Lua environment

Create under `src/craftium/craftium-envs/`:

```
five-chambers/
├── minetest.conf
├── init.lua                     # Craftium entry (reward hooks, episode termination)
├── world/                       # Auto-generated on first run
└── mods/
    ├── five_chambers/
    │   ├── mod.conf
    │   ├── init.lua             # Module loader
    │   ├── config.lua           # All tunable constants
    │   ├── util.lua             # Helpers (pos hashing, chamber lookup)
    │   ├── world_gen.lua        # Chamber geometry + resources
    │   ├── milestones.lua       # M1–M28 state machine
    │   ├── anvil.lua            # Heavy anvil mechanic
    │   ├── switches.lua         # ★ NEW: switch + door wiring + teleport
    │   ├── doors.lua            # Door 2, 3, 4 unlock logic
    │   ├── gear.lua             # Gear drops + equip detection
    │   ├── mobs.lua             # Ch4 mobs + Ch5 boss
    │   └── state_files.lua      # Shared file IO with Python
    ├── mcl_core/                # (copy from voxel-libre2)
    ├── mcl_swords/              # (copy from voxel-libre2)
    ├── mcl_armor/               # (copy from voxel-libre2)
    └── mobs_mc/                 # (copy from voxel-libre2)
```

Copy only the VoxeLibre mods actually needed. Skip hunger, weather, nether, etc.

### 6.2 New Python modules

Create under `src/mindforge/`:

```
src/mindforge/
├── communication_rewards.py     # ★ NEW: CommunicationTracker class (§8.4)
├── cooperation_metric.py        # ★ NEW: CooperationMetric class (§9.1)
├── episode_logger.py            # ★ NEW: EpisodeLogger class (§9.2)
├── multi_agent_craftium.py      # MODIFIED: wire trackers into step loop (§8.6)
├── custom_environment_craftium.py  # MODIFIED: env_dir, chamber lookup (§8.5)
├── agent_modules/
│   └── craftium_metric.py       # MODIFIED: new TRACKS + STAGE_REWARDS (§8.1)
├── rl_layer/
│   ├── rl_layer.py              # MODIFIED: reward decomposition (§8.7)
│   └── hebbian_graph.py         # MODIFIED: communication-driven c_ij (§8.8)
└── prompts/
    ├── environment_prompt.txt   # REWRITTEN: 5 chambers + comm (§8.2)
    └── curriculum_prompt.txt    # SIMPLIFIED: chamber-driven
```

Also under `scripts/`:

```
scripts/
└── analyze_runs.py              # ★ NEW: post-hoc analysis + plots (§9.4)
```

---

## 7. Implementation Deliverables

Nine deliverables, in order. Each is independently testable.

### D1 — Environment skeleton (½ day)
Create folder structure, minetest.conf, mod.conf, stub init files. Craftium
loads without errors. Agents spawn in an empty void. **Test:** `env.reset()`
returns observations.

### D2 — Chamber 1 + Phase 1 milestones (1–2 days)
World generation for Ch1 (floor, walls, trees, stone, animals).
Implement M1–M7 via dig hooks + globalstep checks + mob death hook.
**Test:** Scripted agents dig and fight animals for 300 steps; each agent earns ~250–310 reward.

### D3 — Chamber 2 + anvil mechanic (2–3 days)
Register `five_chambers:anvil` node with `on_dig` and `on_punch` hooks (§4).
Implement per-anvil progress loop in globalstep with **always-applied decay**
(§4 pseudocode). Break anvil at HP=30. Drop gear per position (Row A → swords,
Row B → chestplates). Fire position-indexed milestones M8–M13.
**Test:**
1. Solo dig: one agent repeatedly digs anvil A1 for 50 steps. Verify HP stays
   at 0 (net delta is -1/step, clamped at 0). Anvil is not removed. No milestone
   fires.
2. Pair dig: two agents dig A1 simultaneously within 3 blocks. Verify HP rises
   at +2/step, anvil breaks at step ~15, diamond sword drops at the anvil
   position, and M8 fires for both contributors (40 each).
3. Trio dig: three agents dig B2. Verify breaks at step ~5 and M12 fires for all
   three.

### D4 — Door 2 + gear equip + Ch2→Ch3 teleport (1–2 days)
Implement 20-step post-6th-anvil countdown (§3.6) before teleport. Deterministic
teleport by player name (`agent_0` → Cell A, `agent_1` → Cell B, `agent_2` → Cell C).
Gear equip detection for M14 (sword wielded) and M15 (chestplate equipped).
Relock Door 2 after teleport so agents can't wander back.
**Test:** Use admin commands to break the 6 anvils in sequence; after each
break, verify the correct drop appears (diamond sword for A1/A2/A3, chestplate
for B1/B2/B3). Walk each agent over the drops to collect and auto-equip. Verify:
1. M14 fires for each agent once they pick up and wield a diamond sword.
2. M15 fires for each agent once they have a diamond chestplate equipped.
3. 20 steps after the 6th anvil break, all 3 agents are teleported — agent 0
   to `(2,11,26)`, agent 1 to `(6,11,26)`, agent 2 to `(10,11,26)`.
4. Door 2 barrier is reinstated (bedrock at `(6,10-11,23)`) after teleport.
5. Inventory is preserved across teleport (agents still have gear).

### D5 — Switch Room (Chamber 3) ★ NEW (2 days)
Build isolation cell walls + communal room (§3.2 layout). Register switch nodes
with rotational mapping (A→B, B→C, C→A). Implement switch-press Lua→Python
broadcast via `state_files/switch_events.jsonl` (§3.4). Door unlock on switch
press. Door 3 opens when all 3 agents in communal room. M16–M19 fire.
**Test:** Scripted full-trio unlock:
1. Teleport agents 0, 1, 2 into cells A, B, C respectively (manually via admin
   command or by triggering the Ch2→Ch3 teleport from Deliverable 4).
2. Verify M16 fires for all 3 agents on teleport.
3. Have agent 0 press Switch A → verify Door B opens (bedrock → air) and
   `switch_events.jsonl` gains a new line.
4. Have agent 1 press Switch B → verify Door C opens.
5. Have agent 2 press Switch C → verify Door A opens.
6. All agents walk into the communal room.
7. Verify M17 fires 3 times (once per switch press), M18 fires 3 times (once per
   door open), and M19 fires for all 3 agents when all reach the communal room.
8. Verify Door 3 barrier is removed from `(6,10-11,38)`.

### D6 — Chamber 4 combat (1–2 days)
Spawn 3 weak zombies on Ch4 entry. Hook `mobs_mc:zombie` on_die and on_punch
callbacks. M20–M23 fire. Door 4 opens on all mobs dead.
**Test:** Agents with diamond swords kill all 3 zombies, M22 fires, door 4 opens.

### D7 — Chamber 5 boss (1–2 days)
Spawn modified zombie (60 HP, 3 dmg) on Ch5 entry. Track per-agent damage.
M24–M28 fire. Episode ends on boss death.
**Test:** Full scripted playthrough reaches and defeats boss; episode terminates cleanly.

### D8 — Python-side integration + communication rewards (2 days)
Update `env_dir` path. Replace `TRACKS`, `STAGE_REWARDS`, and delete
`ROLE_STAGE_MULTIPLIERS` in `craftium_metric.py` (§8.1). Add
`communication_rewards.py` module (§8.4). Wire `CommunicationTracker` into
`multi_agent_craftium.py` step loop. Extend `custom_environment_craftium.py`
with `get_agent_position()` + chamber-lookup helpers (§8.5) and extend the
`_LOG_TAGS` tuple from:
```python
_LOG_TAGS = ("[TOOLS]", "[INVENTORY]", "[TRACK STATUS]", "[DIG]", "[HUNT]", "[DEFEND]", "[PHASE]")
```
to include the new Five-Chambers tags:
```python
_LOG_TAGS = (
    "[TOOLS]", "[INVENTORY]", "[TRACK STATUS]",
    "[DIG]", "[HUNT]", "[DEFEND]", "[PHASE]",
    "[ANVIL]", "[SWITCH]", "[DOOR]", "[MOB]", "[BOSS]", "[MILESTONE]",
)
```
Rewrite `environment_prompt.txt` (full §8.2 content, includes macros and comm
section). Replace `curriculum_prompt.txt` with §8.2b content. Make all three
`role_*.txt` files identical (see §8.3 item 1). Do NOT delete the role files
— `load_prompts()` requires all three. Use `--team-mode homogeneous-gatherer`
on the command line for clean no-role runs.
**Test:** Full LLM-driven episode completes without errors; metrics show both
task milestones AND communication milestones. Verify anti-farming rules reject
repeated/short messages. Confirm `[ANVIL]`/`[SWITCH]`/`[DOOR]` lines appear
in the stdout stream when those events occur in Lua.

### D9 — Cooperation metrics + Hebbian extension + logging (2–3 days)
Add `cooperation_metric.py` (§9.1) and `episode_logger.py` (§9.2). Extend
`hebbian_graph.py` to add communication-driven `c_ij` term (§8.8). Add reward
decomposition to `rl_layer.py` (§8.7). Wire `CooperationMetric` and
`EpisodeLogger` into the step loop alongside `CommunicationTracker`. Add
`scripts/analyze_runs.py` for post-hoc analysis.
**Test:**
1. Run a 3-episode training run and verify each episode produces a
   `step_log.csv`, `event_log.jsonl`, `episode_summary.json`, and a line in
   `hebbian_snapshots.jsonl`.
2. Verify `cooperation_score`, `communication_efficacy`, and Hebbian weight
   matrix are populated in the summary.
3. Confirm `c_ij` remains nonzero during Chamber 3 (agents in isolated cells,
   spatial term = 0, comm term = 1 when both are chatting).
4. Run `analyze_runs.py` on the 3-episode directory and verify it produces all
   expected plots.

**Total timeline: ~15–18 working days.**

---

## 8. Python-Side Integration

### 8.1 `craftium_metric.py` replacement

**Note on reward scale.** The existing `voxel-libre2` env emits stage rewards
of `128.0 / 256.0 / 1024.0 / 2048.0`. The Five Chambers plan uses a much
smaller range (`10–300`). This is **intentional** — do not "fix" it back to
larger values:

- `RLLayer._reward_rms` (class `RunningMeanStd` in `rl_layer.py`) normalizes
  every reward by running std before the value head sees it, so absolute scale
  is irrelevant to PPO learning.
- `CraftiumMetric._detect_milestone()` only cares about unique-value matching
  (§4.6 adds an explicit event file for duplicates). Any value set works as
  long as values are distinct within a milestone track.
- Smaller numbers make the budget table (§5) easier to reason about and keep
  per-step rewards on the same order of magnitude as the existing exploration
  shaping (`+0.1 * dist`) and the `+0.3/pair/step` proximity bonus.

Claude Code: when editing `craftium_metric.py`, replace the entire
`TRACKS`, `STAGE_REWARDS`, and `ROLE_STAGE_MULTIPLIERS` blocks at once
with the content below. Do not keep any of the old tools/hunt/defend entries.

```python
TRACKS = {
    "ch1_solo": [
        ("m1_move_5", 10.0), ("m2_dig_3_any", 30.0), ("m3_pickup_3", 30.0),
        ("m4_dig_5_wood", 50.0), ("m5_kill_1_animal", 50.0),
        ("m6_kill_2_animals", 80.0), ("m7_dig_3_stone", 60.0),
    ],
    "ch2_anvils": [
        ("m8_anvil_A1",  40.0), ("m9_anvil_A2",  40.0), ("m10_anvil_A3", 40.0),
        ("m11_anvil_B1", 40.0), ("m12_anvil_B2", 40.0), ("m13_anvil_B3", 40.0),
        ("m14_sword_equipped", 50.0), ("m15_chestplate_equipped", 30.0),
    ],
    "ch3_switches": [
        ("m16_enter_cell", 20.0), ("m17_switch_pressed", 40.0),
        ("m18_door_opened", 60.0), ("m19_all_in_communal", 100.0),
    ],
    "ch4_combat": [
        ("m20_enter_ch4", 30.0), ("m21_first_mob_kill", 60.0),
        ("m22_all_mobs_killed", 150.0), ("m23_all_alive_ch4", 100.0),
    ],
    "ch5_boss": [
        ("m24_enter_ch5", 50.0), ("m25_first_boss_dmg", 80.0),
        ("m26_boss_half_hp", 120.0), ("m27_boss_defeated", 300.0),
        ("m28_all_alive_bonus", 250.0),
    ],
    "communication": [
        ("m_comm_ch2", 20.0), ("m_comm_ch3", 30.0),
        ("m_comm_ch4", 15.0), ("m_comm_ch5", 20.0),
    ],
}

# IMPORTANT: STAGE_REWARDS is used by craftium_metric.py to identify milestones
# from RAW reward values emitted by Lua. The +2 per-message base reward (Tier 1)
# is added Python-side AFTER this detection step, so 2.0 must NOT be in this set.
STAGE_REWARDS = {10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 250.0, 300.0}
```

### 8.2 Environment prompt (full rewrite of `environment_prompt.txt`)

```
You are one of three agents in a five-chamber cooperative dungeon. Your shared
goal is to progress through all chambers and defeat the boss in Chamber 5.

ACTIONS — choose EXACTLY ONE by name each step:
  Primitive: NoOp, MoveForward, MoveBackward, MoveLeft, MoveRight, Jump, Sneak,
  Dig, Place, Drop, Slot1–Slot8, TurnRight, TurnLeft, LookDown, LookUp.
  Macros (multi-step sequences — you are NOT called again until they complete):
    TurnAround     — ~180° rotation (9 turn steps)
    ScanArea       — full 360° scan (18 turn steps)
    ApproachTarget — turn slightly + walk forward 8 steps (11 steps total)
    Escape         — jump + turn + walk, for when physically stuck (23 steps)
    MineStairs     — dig a 3-step staircase forward-down (9 steps)
  To attack a mob: face it and use Dig (there is no separate "attack" action).
  To pick up a dropped item: walk over it (items auto-collect within 1 block).

CHAMBER 1 — Solo learning (12×12 room, always-open door to Chamber 2 at north):
• Practice basic skills. Each reward is individual (not shared).
• Tasks: move around, dig trees/stone, collect dropped items, kill chickens/sheep.
• Door to Chamber 2 is always open — move north (positive Z) when ready.

CHAMBER 2 — Cooperative gear production (10×10 room, 6 heavy anvils):
• 6 obsidian-colored anvils require MULTIPLE AGENTS to break.
• Solo digging an anvil is IMPOSSIBLE (the block decays faster than one agent
  can chip it). Two agents digging simultaneously (within 3 blocks of each other,
  both using Dig) make progress; three agents are even faster.
• Breaking an anvil in Row A (front row, Z=15) drops a DIAMOND SWORD.
• Breaking an anvil in Row B (back row, Z=18) drops a DIAMOND CHESTPLATE.
• Pick up drops by walking over them. Your sword auto-wields; chestplate auto-equips.
• You need BOTH sword and chestplate to survive Chambers 4 and 5.
• 20 steps after the 6th anvil breaks, you will be teleported into sealed cells
  (Chamber 3). Make sure you have gear before then.

CHAMBER 3 — Switch Room (communication puzzle):
• You are teleported into a separate SEALED CELL (A, B, or C) based on your ID:
  agent_0 → Cell A, agent_1 → Cell B, agent_2 → Cell C.
• You cannot see your teammates. Each cell has ONE switch.
• Switches are wired rotationally:
    - Switch in Cell A opens Cell B's door
    - Switch in Cell B opens Cell C's door
    - Switch in Cell C opens Cell A's door
• YOU CANNOT OPEN YOUR OWN DOOR. A teammate must press their switch to free you.
• To escape: (1) announce which cell you are in via team chat, (2) press your
  switch to free the teammate whose door you control, (3) confirm when your own
  door opens (you will see a "[SYSTEM] Switch X was pressed." broadcast).
• All three agents must regroup in the communal room (north side) to unlock the
  door to Chamber 4.

CHAMBER 4 — Combat practice (7×7 room, 3 zombies):
• Attack zombies with your diamond sword (must be wielded).
• Your diamond chestplate reduces incoming damage.
• Door to Chamber 5 opens when all 3 zombies are dead.
• Team bonus if all 3 agents are still alive when the chamber is cleared.

CHAMBER 5 — Boss fight (5×5 room, 1 strong zombie):
• Boss has 60 HP and deals 3 damage per hit. All 3 agents should attack together.
• Episode ends when the boss is defeated.
• Large bonus if all 3 agents are still alive at boss defeat.

COMMUNICATION (rewarded in Ch2, Ch3, Ch4, Ch5):
• You earn reward for valid team-chat messages. Use chat to coordinate.
• Short throwaway messages (under 5 chars) or repeated identical messages do NOT count.
• Talk about: what you're doing, what you see, what you need from teammates.
• In Chamber 3 specifically, communication is the ONLY way to coordinate, since
  you can't see into other cells.
```

### 8.2b Curriculum prompt (simplified `curriculum_prompt.txt`)

The curriculum system becomes chamber-driven — the right task depends almost
entirely on which chamber the agent is in and what progress has been made.

```
You are an assistant suggesting the next immediate task for an agent in the
Five Chambers environment.

Current chamber: {current_chamber}
Completed milestones for this agent: {completed_milestones}
Inventory (wielded + main slots): {inventory}
Position: {position_text}

Suggest ONE specific next task appropriate to the current chamber and progress.
Respond in JSON: {{"reasoning": "...", "task": "..."}}

Guidance by chamber (use only what applies):
- Ch1, no digs yet:                "Dig a tree (move to nearest wood block and use Dig)"
- Ch1, no animals killed:          "Find and attack a chicken or sheep"
- Ch1, all solo tasks done:        "Move north through the open door into Chamber 2"
- Ch2, no anvils broken:           "Stand next to a teammate near an anvil and both use Dig"
- Ch2, sword in inventory:         "Use Slot1 to wield the sword"
- Ch2, ≥6 anvils broken:           "Wait — you'll be teleported to Chamber 3 shortly"
- Ch3, in isolation cell:          "Announce your cell in chat, then press the switch"
- Ch3, door opened, in cell:       "Walk out of your cell into the communal room"
- Ch3, in communal, teammates not: "Wait near the north door for teammates"
- Ch4, zombies present:            "Attack the nearest zombie with your sword"
- Ch4, zombies dead:               "Move north into Chamber 5"
- Ch5, boss alive:                 "Attack the boss together with your teammates"
```

### 8.3 No role specialization

Keep the role *labels* (gatherer / hunter / defender) for Hebbian analysis —
`HebbianSocialGraph` uses `agent_roles` for the modularity-proxy metric, and
`build_role_configs()` / `build_agents()` still need role strings to assign
agent names (`agent_0_gatherer`, etc.). **Do not remove the role label
machinery** — just make the prompts identical.

Concrete changes to the existing codebase:

1. **Replace the three `role_*.txt` prompts** with identical content. The
   simplest approach: make `role_gatherer.txt`, `role_hunter.txt`, and
   `role_defender.txt` all be the same text (a generic curriculum prompt
   that directs agents toward the nearest unfinished chamber milestone).
   Do NOT delete the files — `load_prompts()` expects all three to exist
   at `src/mindforge/prompts/role_{name}.txt`.

2. **Recommended default (to drop into all three files):**
   ```
   You are a helpful assistant suggesting the next immediate task for an
   agent in the Five Chambers environment. All three agents have identical
   capabilities — there is no role specialization. Use the chamber-by-chamber
   task guidance from §8.2b to pick the single most useful next action.
   ```
   Or simply have all three files contain the exact body of §8.2b's
   curriculum prompt. Claude Code can choose whichever is cleaner.

3. **Use `--team-mode homogeneous-gatherer`** (or any homogeneous option) on
   the command line. The code path is already there (see
   `build_role_configs`) and assigns the same role to every agent without
   any code changes. This is the recommended run mode for this environment.

4. **The three `ROLE_STAGE_MULTIPLIERS` in `craftium_metric.py` are dead code**
   once `TRACKS` is replaced (§8.1 already removes the tools/hunt/defend
   tracks). Delete `ROLE_STAGE_MULTIPLIERS` along with the old TRACKS block.

5. **Role-name-in-log-tags:** `multi_agent_craftium.py` uses `role_names` to
   pretty-print per-agent summaries (`[SRV] agent_0 GAT dig...`). Leave
   those labels as-is; they're purely cosmetic.

### 8.4 Communication reward implementation

Add a new module `src/mindforge/communication_rewards.py`:

```python
"""Communication reward tracker for the Five Chambers environment.

Tracks chat messages per agent, applies validity rules, and emits rewards for
both the per-message base (Tier 1) and per-chamber communication milestones (Tier 2).
"""

from collections import defaultdict

CHAMBER_BOUNDS = {
    # Mirror of Lua get_chamber_for_pos
    "ch1": lambda p: 0 <= p[2] <= 11,
    "ch2": lambda p: 13 <= p[2] <= 22,
    "ch3": lambda p: 24 <= p[2] <= 38,
    "ch4": lambda p: 40 <= p[2] <= 46,
    "ch5": lambda p: 48 <= p[2] <= 52,
}

BASE_MSG_REWARD = 2.0
BASE_MSG_CAP = 10  # Max rewarded messages per agent per episode
MIN_MSG_LEN = 5
RATE_LIMIT_STEPS = 2  # Min steps between valid messages per agent

CHAMBER_COMM_THRESHOLDS = {
    "ch2": (3, 20.0, "m_comm_ch2"),
    "ch3": (2, 30.0, "m_comm_ch3"),
    "ch4": (2, 15.0, "m_comm_ch4"),
    "ch5": (2, 20.0, "m_comm_ch5"),
}


class CommunicationTracker:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.total_valid_msgs = defaultdict(int)
        self.chamber_msg_counts = defaultdict(lambda: defaultdict(int))
        self.last_msg = defaultdict(str)
        self.last_msg_step = defaultdict(lambda: -999)
        self.fired_milestones = defaultdict(set)

    def _chamber_for(self, pos):
        for name, fn in CHAMBER_BOUNDS.items():
            if fn(pos):
                return name
        return None

    def _is_valid(self, agent_id, message, step):
        if not message or len(message.strip()) < MIN_MSG_LEN:
            return False
        if message == self.last_msg[agent_id]:
            return False
        if step - self.last_msg_step[agent_id] < RATE_LIMIT_STEPS:
            return False
        return True

    def process_step(self, step, agent_messages, agent_positions):
        """Returns (rewards, milestones_fired, valid_speakers) for this step.

        - rewards: {agent_id: extra_reward}
        - milestones_fired: list of (agent_id, milestone_id, reward)
        - valid_speakers: set of agent_ids whose message passed validity checks
                          (used by HebbianGraph for social c_ij)
        """
        rewards = defaultdict(float)
        milestones_fired = []
        valid_speakers = set()

        for agent_id, message in agent_messages.items():
            if not self._is_valid(agent_id, message, step):
                continue

            valid_speakers.add(agent_id)

            # Tier 1: base per-message reward
            if self.total_valid_msgs[agent_id] < BASE_MSG_CAP:
                rewards[agent_id] += BASE_MSG_REWARD
                self.total_valid_msgs[agent_id] += 1

            # Track for Tier 2
            chamber = self._chamber_for(agent_positions[agent_id])
            if chamber in CHAMBER_COMM_THRESHOLDS:
                self.chamber_msg_counts[agent_id][chamber] += 1
                threshold, reward, mid = CHAMBER_COMM_THRESHOLDS[chamber]
                if (self.chamber_msg_counts[agent_id][chamber] >= threshold
                        and mid not in self.fired_milestones[agent_id]):
                    rewards[agent_id] += reward
                    self.fired_milestones[agent_id].add(mid)
                    milestones_fired.append((agent_id, mid, reward))

            self.last_msg[agent_id] = message
            self.last_msg_step[agent_id] = step

        return dict(rewards), milestones_fired, valid_speakers
```

Wire it into the multi-agent loop in `multi_agent_craftium.py` (see §8.6 for the
full step-loop integration — the snippet below is just the comm-specific calls):

```python
from src.mindforge.communication_rewards import CommunicationTracker

# Before episode loop:
comm_tracker = CommunicationTracker(agent_ids=list(range(num_agents)))

# Inside step loop:
chat_this_step = {i: agents[i].last_message for i in range(num_agents)}
positions = {i: env.get_agent_position(i) for i in range(num_agents)}
extra_rewards, comm_milestones, valid_speakers = comm_tracker.process_step(
    step, chat_this_step, positions
)

for agent_id, bonus in extra_rewards.items():
    step_rewards[agent_id] += bonus

# Log fired milestones. These comm milestones originate Python-side, so we
# go through the same record_milestone_event path as Lua-emitted milestones
# (§4.6). This keeps ONE code path for all milestone bookkeeping. Note:
# metric.record_milestone(...) does NOT exist in the current codebase — the
# only existing method is metric.record_communication(source, message, target)
# which is for individual chat events, not milestone firings.
for agent_id, mid, rw in comm_milestones:
    event = {
        "step": step,
        "milestone": mid,
        "contributors": [f"agent_{agent_id}"],
        "reward": rw,
    }
    metric.record_milestone_event(event)
    coop_metric.observe_milestone(
        step=step, milestone_id=mid, contributors=[f"agent_{agent_id}"]
    )

# Feed valid-speaker set into Hebbian (for social c_ij — see §8.8)
for agent_id in valid_speakers:
    hebbian.register_message(agent_id, step)
```

### 8.5 Env interface (`src/mindforge/custom_environment_craftium.py`)

The existing env already exposes most of what we need — a few additions only:

1. **Update `env_dir`** to `craftium-envs/five-chambers` (was `voxel-libre2`). The
   env-dir fallback logic in `openworld_multi_agents.py` uses `CRAFTIUM_ENV_DIR`
   as an override; update the default or set the env var.

2. **Existing position access pattern:** `environment.env.env._positions[agentId]`
   returns a tuple `(x, y, z)` or `None`. Already used in `multi_agent_craftium.py`
   and `test_scripted_agent.py`. Add a cleaner wrapper on `CraftiumEnvironmentInterface`:

   ```python
   def get_agent_position(self, agentId: int):
       """Return current (x, y, z) tuple for agent, or None if unavailable."""
       try:
           return self.env.env._positions[agentId]
       except (AttributeError, IndexError, TypeError):
           return None
   ```

   This mirrors the existing `get_position_text(agentId)` pattern but returns
   the raw tuple for the `CommunicationTracker` and `CooperationMetric`.

3. **Add chamber-lookup helper** that mirrors the Lua `get_chamber_for_pos`:

   ```python
   # Python mirror of Lua chamber bounds (keep in sync with config.lua)
   CHAMBER_BOUNDS = {
       "ch1": lambda p: 0 <= p[2] <= 11,
       "ch2": lambda p: 13 <= p[2] <= 22,
       "ch3": lambda p: 24 <= p[2] <= 38,
       "ch4": lambda p: 40 <= p[2] <= 46,
       "ch5": lambda p: 48 <= p[2] <= 52,
   }

   def get_chamber(self, agentId: int) -> str | None:
       pos = self.get_agent_position(agentId)
       if pos is None:
           return None
       for name, fn in CHAMBER_BOUNDS.items():
           if fn(pos):
               return name
       return None
   ```

4. **State file pattern (already used):** The existing env writes several files
   to `{world_path}/` each Lua globalstep: `inv_agent{N}.txt`, `health_agent{N}.txt`,
   `hunger_agent{N}.txt`, `timeofday.txt`, `phase.txt`. All our new state sharing
   follows this same pattern — see §4.6 (milestone events) and §3.4 (switch events).
   **Do not invent new IPC mechanisms** — use append-only JSONL files polled from Python.

5. **Server-log tags (existing):** `CraftiumEnvironmentInterface._LOG_TAGS` currently
   lists `"[TOOLS]", "[INVENTORY]", "[TRACK STATUS]", "[DIG]", "[HUNT]", "[DEFEND]", "[PHASE]"`.
   Add `"[ANVIL]", "[SWITCH]", "[DOOR]", "[MOB]", "[BOSS]", "[MILESTONE]"` for the new env.
   These tags are optional but extremely useful for debugging — any line starting
   with them is mirrored to stdout from `tail_server_log()`.

6. **Keep `ACTION_MAP` unchanged.** The existing 23-entry action map plus the
   `_MACRO_ACTIONS` table in `CraftiumEnvironmentInterface` already covers every
   action we need. No new action types are introduced.

7. **Drop role-specific inventory reads** — not applicable in this env; all agents
   are functionally identical. The `_pretty_item_name` helper still works for the
   diamond sword / diamond chestplate drops.

### 8.6 Main loop (`src/mindforge/multi_agent_craftium.py`)

The main loop changes are:

> **Real-code context.** The existing `multi_agent_craftium.py` `run(args)` is
> `async` and iterates agents one at a time inside each step via
> `await agent_do_action(...)`, which returns a `content` dict with keys
> `action`, `thoughts`, `communication`, `communication_target`. The dict is
> stored in `step_contents[agent_id]` and `environment.step(action, agentId=i)`
> fires inside `agent_do_action`. The simplified pseudocode below collapses
> this per-agent loop for readability — in the real integration, build
> `messages = {}`, `actions = {}`, `positions = {}` by iterating
> `step_contents` AFTER the per-agent block, not before. See
> `multi_agent_craftium.py` Phase 1/2/3 comments for the canonical
> ordering that must be preserved.

1. **Instantiate trackers** once per episode:
   ```python
   from src.mindforge.communication_rewards import CommunicationTracker
   from src.mindforge.cooperation_metric import CooperationMetric

   comm_tracker = CommunicationTracker(agent_ids=list(range(num_agents)))
   coop_metric = CooperationMetric(agent_ids=list(range(num_agents)))
   ```

2. **Wire trackers into the step loop:**
   ```python
   for step in range(max_steps):
       # 1. Get LLM actions + messages
       actions, messages = get_agent_decisions(...)

       # 2. Step the env
       obs, task_rewards, dones, infos = env.step(actions)
       positions = {i: env.get_agent_position(i) for i in range(num_agents)}

       # 3. Apply communication rewards (Python-side).
       #    Returns extra rewards AND the set of agents who sent VALID messages.
       comm_rewards, comm_milestones, valid_speakers = comm_tracker.process_step(
           step, messages, positions
       )

       # 4. Aggregate rewards
       total_rewards = {
           i: task_rewards[i] + comm_rewards.get(i, 0.0)
           for i in range(num_agents)
       }

       # 5. Update cooperation metrics (observation only — no reward effect)
       coop_metric.observe_step(step, positions, actions, messages,
                                task_rewards, infos)

       # 6. Feed into Hebbian + RL
       #    Register valid message events BEFORE computing c_ij so the comm term
       #    reflects this step's messages.
       for agent_id in valid_speakers:
           hebbian.register_message(agent_id, step)
       acted_flags = {i: actions[i] != "NoOp" for i in range(num_agents)}
       cij = hebbian.compute_cij(step, positions, acted_flags)
       hebbian.update_weights(cij)   # applies the LTP/LTD rule

       rl_layer.step(obs, total_rewards, actions, positions, task_rewards, comm_rewards)

       # 7. Log
       episode_logger.log_step(step, positions, actions, messages,
                               task_rewards, comm_rewards, infos)

       if all(dones.values()) or infos.get("episode_complete"):
           break
   ```

   > Note: `comm_tracker.process_step` is extended to return
   > `(rewards, milestones_fired, valid_speakers)` where `valid_speakers` is the
   > set of agent IDs whose messages passed all anti-farming checks this step.
   > See §8.4 for the updated signature.

3. **Simplify `build_role_configs`** — roles no longer affect gameplay, so all three
   agents can get the same config. Keep role labels (`"gatherer"`, `"hunter"`,
   `"defender"`) for Hebbian analysis bookkeeping but don't differentiate prompts.

4. **Update episode termination check** — episode ends on boss death (received from
   Lua via the existing termination channel) or `max_steps`, whichever comes first.

### 8.7 RL layer (`src/mindforge/rl_layer/rl_layer.py`)

Action space is preserved, so the policy network does not need retraining from
scratch. Minimal additions:

1. **Reward decomposition bookkeeping.** When RL stores experiences, split the
   reward into `(task, comm)` components so post-hoc analysis can separate what
   the policy learned from task completion vs. from chat:
   ```python
   experience = {
       "obs": obs, "action": action, "reward_total": total_reward,
       "reward_task": task_reward, "reward_comm": comm_reward,
       "next_obs": next_obs, "done": done,
   }
   ```
   The training loss still uses `reward_total`; the split is purely for analysis.

2. **Cooperative-action tagging (optional, for metrics).** Tag actions that
   occurred "in cooperation context" — e.g., `Dig` within 3 blocks of a teammate,
   or any action in Ch3 (where cooperation is forced). Store alongside the
   experience:
   ```python
   experience["coop_context"] = _is_coop_context(position, teammates_positions, chamber)
   ```
   Used only for metric reporting (§9); does not affect training.

3. **No architecture change.** The policy head, value head, and embedding layer
   are all untouched.

### 8.8 Hebbian learning (`src/mindforge/rl_layer/hebbian_graph.py`)

**Good news: the core machinery already exists.** The current `HebbianConfig`
already exposes `communication_coactivity_bonus` (default 0.5, the δ_comm term),
and `HebbianSocialGraph._compute_coactivity` already accepts a `comm_events`
list `[(sender, receiver), ...]` and adds a communication bonus `cij_comm` when
agents are not spatially co-active. The rule implemented there is:

```
cij(t) = clip(cij_spatial + cij_comm, 0, 1)
   where cij_comm fires only when spatial_gate[i,j] = 0 (they are far apart)
```

This is exactly what Chamber 3 needs — agents in sealed cells are far apart in
X/Z, so `spatial_gate = 0`, and `cij_comm` picks up the slack when they chat.
**No code changes are strictly required** for the Hebbian graph to handle the
Five Chambers environment.

The changes below are tuning tweaks and one integration fix, not new logic:

1. **Tune `communication_coactivity_bonus` higher for this env.** Default 0.5 in
   the open world may be too low when Ch3 is the only place agents are separated.
   Consider raising the CLI flag to `--hebbian-no-comm-bond`'s opposite (e.g. 0.7)
   so Ch3 bonds develop robustly. This is a flag change, not a code change.

2. **Wire `comm_events` from the Five Chambers comm channel.** The existing main
   loop already populates `comm_events` when `targeted_communication=True`. With
   our Ch3 broadcast design (switch-press team-chat message from §3.4), the
   broadcast should NOT populate `comm_events` — system broadcasts are not
   inter-agent co-activation. Only LLM-generated agent messages should count.
   Code location: `multi_agent_craftium.py` lines 951–975 (existing); ensure our
   system `[SYSTEM]` messages are filtered out.

3. **Verify `HebbianSocialGraph.update` is still called after teleport.** When
   agents are teleported into Ch3 cells, their positions jump by ~15 blocks in a
   single step. The existing code uses `np.linalg.norm` for distance, which
   handles the jump correctly — no special-case logic needed. But worth noting
   for the tester: in the step immediately after teleport, spatial cij drops
   sharply for all pairs; subsequent steps rely entirely on cij_comm for bond
   maintenance until agents regroup.

4. **Add one method for evaluation snapshots (for `hebbian_snapshots.jsonl`):**

```python
def snapshot(self) -> dict:
    """Serializable per-episode snapshot for post-hoc analysis."""
    return {
        "W": self.W.copy().tolist(),
        "step": self._step_count,
        "max_reward_seen": self._max_reward_seen,
        "num_agents": self.config.num_agents,
    }
```

Already covered by the existing `to_dict()` method — just call that.

**Summary:** §8.8 is mostly a verification/tuning exercise, not new implementation.
The real Hebbian work for this environment is in §9 (metrics + logging), not here.

---

## 9. Evaluation Metrics & Logging

Training and evaluation need metrics that go beyond raw reward. The goal is to
answer: *is the policy actually cooperating, or is it just getting lucky?* Raw
reward can't distinguish a team that split anvil work 50/50 from one where a
single agent got carried by teammates.

### 9.1 Cooperation metrics (`src/mindforge/cooperation_metric.py`)

Create a new module that observes the step loop (without affecting rewards) and
emits per-episode and per-chamber cooperation statistics.

**Per-step observations** (cheap, running counters):

| Metric | Definition |
|--------|-----------|
| `proximity_events` | Count of agent *pairs* within 4 blocks this step |
| `co_action_events` | Count of steps where ≥2 agents performed the same action |
| `joint_dig_events` | Count of steps where ≥2 agents used Dig within 3 blocks of each other |
| `message_count` | Total valid messages emitted per agent |
| `message_unique` | Unique messages per agent (content-dedup) |

**Per-milestone observations** (fired when Lua reports a milestone):

| Metric | Definition |
|--------|-----------|
| `contributor_count` | How many agents contributed to each multi-agent milestone |
| `contribution_entropy` | Shannon entropy over contributor distribution; higher = more even split |
| `comm_before_coop` | Boolean: was there a valid message in the 10 steps before this milestone? |

**Per-chamber observations** (aggregated on chamber exit):

| Metric | Definition |
|--------|-----------|
| `ch2_pair_efficiency` | Mean steps between first joint dig and anvil break, per anvil |
| `ch3_completion_time` | Steps from Ch3 entry (teleport) to `m19_all_in_communal` |
| `ch3_cell_silence` | Per-cell count of steps with no messages from that agent |
| `ch4_kill_distribution` | Gini coefficient of damage-dealt-per-agent in Ch4 |
| `ch5_damage_distribution` | Gini coefficient of damage-dealt-per-agent to boss |
| `ch5_alive_at_victory` | Count of agents alive when boss dies |

**Episode-level derived metrics:**

| Metric | Definition |
|--------|-----------|
| `cooperation_score` | Weighted combination of the above (0-1 normalized) |
| `communication_efficacy` | Fraction of cooperation milestones preceded by a message |
| `carry_imbalance` | Max(contribution_share) − Min(contribution_share) across agents |

Module skeleton:

```python
# src/mindforge/cooperation_metric.py

from collections import defaultdict
import math
import numpy as np

# Same chamber bounds as CommunicationTracker — keep these two in sync
CHAMBER_BOUNDS = {
    "ch1": lambda p: 0 <= p[2] <= 11,
    "ch2": lambda p: 13 <= p[2] <= 22,
    "ch3": lambda p: 24 <= p[2] <= 38,
    "ch4": lambda p: 40 <= p[2] <= 46,
    "ch5": lambda p: 48 <= p[2] <= 52,
}


class CooperationMetric:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.reset()

    def reset(self):
        self.proximity_events = 0
        self.co_action_events = 0
        self.joint_dig_events = 0
        self.messages_per_agent = defaultdict(int)
        self.milestone_log = []           # list of dicts per milestone
        self.chamber_entry_step = {}      # chamber -> step of first entry by any agent
        self.ch4_damage = defaultdict(float)
        self.ch5_damage = defaultdict(float)
        self.recent_messages = []         # rolling buffer of (step, agent_id, message)
                                          # used by observe_milestone for comm_before_coop

    def _chamber_for(self, pos):
        for name, fn in CHAMBER_BOUNDS.items():
            if fn(pos):
                return name
        return None

    def observe_step(self, step, positions, actions, messages, task_rewards, infos):
        # Proximity + co-action
        for i in self.agent_ids:
            for j in self.agent_ids:
                if i < j:
                    if np.linalg.norm(np.array(positions[i]) - np.array(positions[j])) < 4.0:
                        self.proximity_events += 1
        # Co-action: same action by 2+ agents
        action_counts = defaultdict(int)
        for a in actions.values():
            action_counts[a] += 1
        if any(c >= 2 for c in action_counts.values()):
            self.co_action_events += 1
        # Joint dig
        digging_agents = [i for i, a in actions.items() if a == "Dig"]
        if len(digging_agents) >= 2:
            for i in digging_agents:
                for j in digging_agents:
                    if i < j and np.linalg.norm(
                            np.array(positions[i]) - np.array(positions[j])) < 3.0:
                        self.joint_dig_events += 1
                        break
        # Message counts + rolling buffer
        for agent_id, msg in messages.items():
            if msg and len(msg.strip()) >= 5:
                self.messages_per_agent[agent_id] += 1
                self.recent_messages.append((step, agent_id, msg))
        # Prune rolling buffer to 10 steps
        self.recent_messages = [(s, a, m) for (s, a, m) in self.recent_messages
                                 if step - s <= 10]
        # Chamber entry tracking (for timing metrics)
        for i in self.agent_ids:
            chamber = self._chamber_for(positions[i])
            if chamber and chamber not in self.chamber_entry_step:
                self.chamber_entry_step[chamber] = step
        # Damage tracking (from infos)
        for dmg_event in infos.get("damage_events", []):
            if dmg_event["target"] == "ch4_zombie":
                self.ch4_damage[dmg_event["attacker"]] += dmg_event["amount"]
            elif dmg_event["target"] == "boss":
                self.ch5_damage[dmg_event["attacker"]] += dmg_event["amount"]

    def observe_milestone(self, step, milestone_id, contributors):
        recent_msgs = [m for (s, a, m) in self.recent_messages
                        if a in contributors]
        self.milestone_log.append({
            "step": step,
            "milestone": milestone_id,
            "contributors": list(contributors),
            "contributor_count": len(contributors),
            "contribution_entropy": self._entropy(contributors),
            "comm_before_coop": len(recent_msgs) > 0,
        })

    @staticmethod
    def _entropy(contributors):
        """Shannon entropy over contributor counts."""
        if not contributors:
            return 0.0
        counts = defaultdict(int)
        for c in contributors:
            counts[c] += 1
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * math.log(p) for p in probs if p > 0)

    @staticmethod
    def _gini(value_dict):
        """Gini coefficient (0 = perfectly equal, 1 = maximally unequal)."""
        values = sorted(value_dict.values())
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0.0
        cum = sum((i + 1) * v for i, v in enumerate(values))
        return (2 * cum) / (n * sum(values)) - (n + 1) / n

    def _comm_efficacy(self):
        """Fraction of multi-agent milestones preceded by a valid message."""
        multi = [m for m in self.milestone_log if m["contributor_count"] >= 2]
        if not multi:
            return 0.0
        return sum(1 for m in multi if m["comm_before_coop"]) / len(multi)

    def _carry_imbalance(self):
        """Max − min of per-agent milestone counts."""
        per_agent = defaultdict(int)
        for m in self.milestone_log:
            for c in m["contributors"]:
                per_agent[c] += 1
        if not per_agent:
            return 0.0
        return max(per_agent.values()) - min(per_agent.values())

    def _cooperation_score(self):
        """Weighted 0-1 score combining the key cooperation signals.

        Weights chosen so each component contributes ~0.2 to the total when at
        its typical "good cooperation" value. Tune as needed.
        """
        joint_dig_norm = min(self.joint_dig_events / 50.0, 1.0)
        proximity_norm = min(self.proximity_events / 300.0, 1.0)
        comm_eff = self._comm_efficacy()
        ch5_fairness = 1.0 - self._gini(self.ch5_damage)
        balance = 1.0 - min(self._carry_imbalance() / 10.0, 1.0)
        return 0.2 * joint_dig_norm + 0.2 * proximity_norm + \
               0.2 * comm_eff + 0.2 * ch5_fairness + 0.2 * balance

    def episode_summary(self, final_step, hebbian_weights=None) -> dict:
        return {
            "final_step": final_step,
            "proximity_events": self.proximity_events,
            "co_action_events": self.co_action_events,
            "joint_dig_events": self.joint_dig_events,
            "messages_per_agent": dict(self.messages_per_agent),
            "chamber_entry_steps": dict(self.chamber_entry_step),
            "ch4_damage_gini": self._gini(self.ch4_damage),
            "ch5_damage_gini": self._gini(self.ch5_damage),
            "milestone_log": self.milestone_log,
            "communication_efficacy": self._comm_efficacy(),
            "carry_imbalance": self._carry_imbalance(),
            "cooperation_score": self._cooperation_score(),
            "hebbian_W": hebbian_weights,
        }
```

### 9.2 Logging infrastructure (`src/mindforge/episode_logger.py`)

Structured logs, not print statements. Everything for one run is collected under
`run_metrics/{run_id}/` (same pattern the existing `CraftiumMetric._mkdir_metrics()`
uses — we extend it with per-episode subdirs):

```
run_metrics/
└── {run_id}/                              ← top-level run dir (existing pattern)
    ├── data.json                          ← existing CraftiumMetric dump
    ├── summary.txt                        ← existing human-readable summary
    ├── communication_log.json             ← existing
    ├── cumulative_returns.png             ← existing plot
    ├── milestones.png                     ← existing plot
    ├── log.txt                            ← existing Metric.log() output
    │
    ├── hebbian_snapshots.jsonl            ← ★ NEW: one line per Hebbian snapshot
    │                                         (covers the whole run, not per-episode)
    │
    └── episodes/                          ← ★ NEW: one subdir per episode
        ├── ep_0001/
        │   ├── step_log.csv               ← ★ NEW: per-(step, agent) row
        │   ├── event_log.jsonl            ← ★ NEW: milestone/break/switch/kill events
        │   └── episode_summary.json       ← ★ NEW: CooperationMetric.episode_summary()
        ├── ep_0002/
        │   └── ...
        └── ep_NNNN/
            └── ...
```

Three log files per episode:

**`run_{episode}/step_log.csv`** — one row per (step, agent):
```
step, agent_id, chamber, pos_x, pos_y, pos_z, action, reward_task, reward_comm, wielded_item, hp, message
```

**`run_{episode}/event_log.jsonl`** — one JSON object per notable event:
```json
{"step": 120, "type": "anvil_break", "anvil_pos": [3,11,15], "contributors": ["agent_0","agent_1"], "rewards": {"agent_0": 40, "agent_1": 40}}
{"step": 425, "type": "switch_press", "agent": "agent_0", "switch": "A", "door_opened": "B"}
{"step": 426, "type": "broadcast", "message": "Switch A was pressed"}
{"step": 540, "type": "mob_kill", "killer": "agent_2", "victim": "ch4_zombie_1", "weapon": "diamond_sword"}
{"step": 680, "type": "boss_damage", "attacker": "agent_1", "amount": 2, "boss_hp_after": 58}
{"step": 720, "type": "milestone", "id": "m27_boss_defeated", "contributors": ["agent_0","agent_1","agent_2"]}
```

**`run_{episode}/episode_summary.json`** — emitted once, at episode end:
```json
{
  "episode": 42,
  "final_step": 847,
  "total_reward_per_agent": {"agent_0": 1920.0, "agent_1": 1895.0, "agent_2": 1870.0},
  "reward_decomposition": {
    "agent_0": {"task": 1820.0, "comm": 100.0},
    "agent_1": {"task": 1810.0, "comm": 85.0},
    "agent_2": {"task": 1795.0, "comm": 75.0}
  },
  "milestones_fired_per_agent": {"agent_0": 28, "agent_1": 27, "agent_2": 26},
  "cooperation_metrics": { ... (from CooperationMetric.episode_summary) },
  "hebbian_W": [[0.0, 0.42, 0.38], [0.42, 0.0, 0.35], [0.38, 0.35, 0.0]]
}
```

Module skeleton:

```python
# src/mindforge/episode_logger.py

import csv
import json
from pathlib import Path

class EpisodeLogger:
    def __init__(self, run_dir: Path, episode: int):
        self.dir = Path(run_dir) / f"ep_{episode:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episode = episode
        self.step_csv = open(self.dir / "step_log.csv", "w", newline="")
        self.step_writer = csv.DictWriter(self.step_csv, fieldnames=[
            "step","agent_id","chamber","pos_x","pos_y","pos_z",
            "action","reward_task","reward_comm","wielded_item","hp","message"
        ])
        self.step_writer.writeheader()
        self.event_jsonl = open(self.dir / "event_log.jsonl", "w")

    def log_step(self, step, positions, actions, messages,
                 task_rewards, comm_rewards, infos):
        for agent_id in positions:
            self.step_writer.writerow({
                "step": step,
                "agent_id": agent_id,
                "chamber": infos.get("chambers", {}).get(agent_id),
                "pos_x": positions[agent_id][0],
                "pos_y": positions[agent_id][1],
                "pos_z": positions[agent_id][2],
                "action": actions.get(agent_id),
                "reward_task": task_rewards.get(agent_id, 0.0),
                "reward_comm": comm_rewards.get(agent_id, 0.0),
                "wielded_item": infos.get("wielded", {}).get(agent_id),
                "hp": infos.get("hp", {}).get(agent_id),
                "message": messages.get(agent_id, ""),
            })

    def log_event(self, event: dict):
        self.event_jsonl.write(json.dumps(event) + "\n")

    def finalize(self, summary: dict):
        with open(self.dir / "episode_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self.step_csv.close()
        self.event_jsonl.close()
```

### 9.3 Hebbian snapshot stream (`run_dir/hebbian_snapshots.jsonl`)

At the end of each episode, append one line with the full weight matrix plus
episode-level metrics:

```json
{"episode": 42, "final_step": 847, "W": [[...], [...], [...]], "cooperation_score": 0.73, "reward_total": 5685.0}
```

This gives a per-episode time series for post-hoc analysis of weight evolution,
without blowing up disk usage (one snapshot per episode, not per step).

### 9.4 Aggregation script (`scripts/analyze_runs.py`)

A standalone script that reads `hebbian_snapshots.jsonl` and all
`episode_summary.json` files in a run directory and produces:

- `W_evolution.png` — line plot of each `W_ij` over episodes
- `cooperation_vs_reward.png` — scatter of cooperation_score vs. total reward
- `chamber_timings.png` — distributions of chamber completion times across episodes
- `milestone_rate_by_episode.png` — how milestone-firing rates evolve during training
- `comm_efficacy_vs_coop.png` — did communication become more predictive of
  cooperation over training?
- `summary_stats.csv` — one row per episode with all key metrics

Not needed for training — purely for thesis analysis. Can be run after training
completes.

---

## 10. Risks & Open Questions

| Risk | Severity | Mitigation |
|------|----------|------------|
| Anvil `on_dig` returning false may not prevent default removal on all Minetest versions | High | Fallback: register as `unbreakable=1` group and detect dig attempts via `on_punch` instead. |
| `mobs_mc` API varies across VoxeLibre versions (on_die / on_punch signatures) | High | Inspect the actual mod source copied from voxel-libre2; adapt the wrapper to match. |
| LLM agents may never learn to use Slot1 to wield swords | Medium | Auto-wield on pickup when nothing is wielded. Add explicit prompt instruction. |
| Agents may never discover anvil cooperation → stuck in Ch2 forever | Medium | After step 500 with 0 anvils broken, log warning. Consider fallback: inject hint broadcast at step 300. |
| Agents may not communicate in switch room → stuck in cells | Medium | Prompt explicitly instructs "announce your cell and press your switch." Emit team-chat broadcast on any switch press so inaction is visible. |
| Episode may exceed `max_steps=1000` with 5 chambers | Medium | Trim Phase 1 milestones first (7 → 5). Or raise to `max_steps=1500`. |
| Teleportation on Door 2 unlock may be jarring for agents (lost context) | Low | Prompt prepares them: "When all 6 anvils are broken, you will be transported to a sealed cell…" |
| Agents may spam chat to farm communication rewards | Medium | Anti-farming rules: min 5 chars, no identical consecutive messages, rate-limit 1/2 steps, episode cap at 10 Tier-1 messages. Tier 2 capped per chamber. Max total comm reward ≈ 5% of budget. |
| Communication tracker may misattribute messages to the wrong chamber (agent crosses boundary mid-step) | Low | Use position at start of step, not end. Document the convention in `communication_rewards.py`. |

### Open design questions (flag before implementation if you want to change)

1. **Switch mapping: fixed rotational vs. randomized per episode.** Current plan: fixed.
   Randomized would force stronger communication but is harder for RL to learn.
2. **Can a single agent press multiple switches?** Current plan: no — each cell has
   exactly one switch, each agent only has access to one cell. This is enforced by physics.
3. **Should switch presses be broadcast?** Current plan: yes — emit team-chat message
   "Switch X was pressed" so agents in sealed cells can verify.
4. **Teleport assignment: by agent index (deterministic) vs. random.** Current plan: deterministic.
5. **Communication reward magnitudes.** Current plan: +2 per message (cap 10/ep), +20/30/15/20
   per chamber milestone. Max 105 per agent (~5% of budget). Tune up if agents don't talk
   enough, tune down if they spam. The `CommunicationTracker` exposes all constants at the
   top of the file for easy adjustment.
6. **Should messages sent during Ch1 count for anything?** Current plan: no — Ch1 is solo
   learning, communication isn't cooperative there. The base chat reward (Tier 1) still
   applies episode-wide, but no chamber milestone exists for Ch1.

---

## 11. Reference Tables

### 11.1 All coordinates at a glance

| Object | Position(s) |
|--------|-------------|
| Spawn 0/1/2 | (1,11,1) / (10,11,1) / (5,11,10) |
| Anvils Row A (M8/M9/M10) | (3,11,15) (6,11,15) (9,11,15) — swords |
| Anvils Row B (M11/M12/M13) | (3,11,18) (6,11,18) (9,11,18) — chestplates |
| Ch3 cells (teleport targets) | A:(2,11,26) / B:(6,11,26) / C:(10,11,26) |
| Ch3 switches | A:(2,11,25) / B:(6,11,25) / C:(10,11,25) |
| Ch3 cell doors (barriers) | A:(2,10-11,28) / B:(6,10-11,28) / C:(10,10-11,28) |
| Ch3 inter-cell walls | X=4, X=8 (spanning Z:24–28, Y:11–15) |
| Ch3 communal room | X:0–12, Z:29–37 |
| Door 3 (Ch3→Ch4, barrier) | (6,10-11,38) |
| Ch4 mob spawns | (4,11,42) (6,11,43) (8,11,42) |
| Door 4 (Ch4→Ch5, barrier) | (6,10-11,47) |
| Boss spawn | (6,11,50) |

### 11.2 Key parameters

| Param | Value | Used in |
|-------|-------|---------|
| Anvil max HP | 30 | §4 |
| Anvil solo rate (after decay) | -1 / step | §4 |
| Anvil pair rate (after decay) | +2 / step | §4 |
| Anvil trio rate (after decay) | +6 / step | §4 |
| Anvil decay (always applied) | -2 / step | §4 |
| Post-6th-anvil teleport delay | 20 steps | §3.6 |
| Ch4 zombie HP | 8 | §7, D6 |
| Ch4 zombie damage | 2 | §7, D6 |
| Boss HP | 60 | §7, D7 |
| Boss damage | 3 | §7, D7 |
| Hebbian spatial radius `R_coact` | 4.0 blocks | §8.8 |
| Hebbian comm window `K_comm` | 5 steps | §8.8 |
| Episode max steps | 1500 (recommended) | §10 |

### 11.3 Reward tiers (unique Lua-emitted values for milestone detection)

`{10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 250, 300}`

The `+2` per-message base reward is Python-side only and is NOT in this set
(would otherwise cause false milestone detection).

### 11.4 Milestone phase→count summary

| Phase | Task milestones | Comm milestone | Task reward | Comm reward | Total |
|-------|----------------|----------------|-------------|-------------|-------|
| Ch1 solo | 7 | — | 310 | — | 310 |
| Ch2 anvils | 8 | 1 | 320 | 20 | 340 |
| Ch3 switches | 4 | 1 | 220 | 30 | 250 |
| Ch4 combat | 4 | 1 | 340 | 15 | 355 |
| Ch5 boss | 5 | 1 | 800 | 20 | 820 |
| Base chat (shaped) | — | — | — | 20 | 20 |
| **Total** | **28** | **4** | **~1,990** | **~105** | **~2,095** |
