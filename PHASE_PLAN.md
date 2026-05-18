# Phase Plan: Agent Homogeneity/Heterogeneity + Phased Difficulty

**Status:** Draft — awaiting confirmation before any implementation code is written.

---

## 1. Files Touched and Why

| File | Change type | Reason |
|------|-------------|--------|
| `src/mindforge/multi_agent_craftium.py` | Additive | New CLI args, `build_role_configs()` team-mode branch, phase detection loop, `_should_transition_to_survival()`, phase injection into instruction prompt, checkpoint save/restore of phase state |
| `src/mindforge/custom_environment_craftium.py` | Additive | `_write_phase_file()` helper; add `"[PHASE]"` to `_LOG_TAGS` (line 630) |
| `src/mindforge/agent_modules/craftium_metric.py` | Additive | `phase_transitions` field, `record_phase_transition()`, phase marker on cumulative-returns plot, phase note in `_save_text_summary()` |
| `craftium/craftium-envs/voxel-libre2/mods/craftium_env/init.lua` | Additive | Phase-polling globalstep timer block; `ENABLE_MOBS` / `ENABLE_HUNGER` already exist as runtime locals — the new block mutates them live |
| `scripts/run_first.sh` | Additive | New optional flags documented in comments (no behavior change by default) |
| `scripts/run_continue.sh` | Additive | Pass through `current_phase` from `run_state.json` on resume |

**No files are deleted. No existing logic is modified — only new `if/else` branches and new fields are added.**

---

## 2. New CLI Flags

All flags default to current behavior. A run with none of the new flags produces identical output to the current codebase.

### 2a. Team composition (`--team-mode`)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--team-mode` | str (choices) | `heterogeneous` | Role assignment strategy |
| `--homogeneous-role` | str | `gatherer` | Role to use when `--team-mode homogeneous-auto` |

**`--team-mode` choices:**

| Value | Behavior |
|-------|----------|
| `heterogeneous` | Current behavior: `ROLE_NAMES[i % 3]` cycling. No change to any existing path. |
| `homogeneous-gatherer` | All agents get `"gatherer"` role, prompt, curriculum. |
| `homogeneous-hunter` | All agents get `"hunter"`. |
| `homogeneous-defender` | All agents get `"defender"`. |
| `homogeneous-auto` | All agents get the role named by `--homogeneous-role` (default: `gatherer`). Allows overriding without spelling out the full `homogeneous-gatherer` form. |

### 2b. Phased difficulty

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--survival-mode` | bool flag | off | Master switch. Without this flag nothing changes. |
| `--survival-episode` | int | `1` | Switch to survival at start of this episode (1-indexed). Ignored unless `--survival-mode`. |
| `--survival-step` | int | `None` | Switch mid-episode at this global step count. If both set, first threshold wins. Ignored unless `--survival-mode`. |
| `--survival-gradual` | bool flag | off | Ramp: mobs first, hunger 500 steps later. Ignored unless `--survival-mode`. |
| `--survival-gradual-delay` | int | `500` | Steps after first gradual trigger before hunger also enables. Ignored unless `--survival-gradual`. |

---

## 3. Lua-side Signaling Mechanism

### 3a. Why file-based IPC

Python cannot call Lua functions in a running Minetest process directly. The server and client are separate processes communicating over TCP (the Craftium `mod_channel`). File-based IPC in the world directory is already the established pattern (inventory files, health files, hunger files). A `phase.txt` file follows the same contract.

### 3b. Python side: `_write_phase_file()`

**Location:** `CraftiumEnvironmentInterface` in `custom_environment_craftium.py`

```python
def _write_phase_file(self, phase: str) -> None:
    """Atomically write the current phase to {world_path}/phase.txt.

    Uses write-to-tmp + os.replace so Lua never reads a partial file.
    Safe to call before reset() — returns early if world path not resolved.
    """
    try:
        world_path = self._get_world_path()
    except AttributeError:
        return  # called before reset(), env not yet initialised
    tmp = os.path.join(world_path, "phase.tmp")
    dst = os.path.join(world_path, "phase.txt")
    with open(tmp, "w") as f:
        f.write(phase)
    os.replace(tmp, dst)  # POSIX-atomic rename
```

**Called from `multi_agent_craftium.py`:**

1. In `environment.reset()` call block — write current phase so a resumed job re-signals a freshly started server.
2. In the main step loop when `_should_transition_to_survival()` returns `True`.

### 3c. Phase strings

| String | Meaning |
|--------|---------|
| `"exploration"` | Default. Mobs frozen/cleared, hunger frozen. Current behavior. |
| `"survival_mobs_only"` | Mobs hostile. Hunger still frozen. Gradual ramp step 1. |
| `"survival"` | Full survival. Mobs hostile, hunger drains. |

### 3d. Lua side: phase polling block

**Location:** `craftium/craftium-envs/voxel-libre2/mods/craftium_env/init.lua`, added as a new `minetest.register_globalstep` call after the existing ones.

```lua
-- ── Phase file polling ────────────────────────────────────────────────────
-- Python writes {world_path}/phase.txt to signal phase transitions.
-- We poll every 5 real-seconds (accumulated dtime) to avoid file I/O
-- on every physics tick. ENABLE_MOBS / ENABLE_HUNGER are already declared
-- as locals above — mutating them here takes effect on the NEXT globalstep.
local _phase_timer  = 0.0
local _current_phase = "exploration"
local HOSTILE_MOBS = {
    "mobs_mc:zombie", "mobs_mc:skeleton", "mobs_mc:spider",
    "mobs_mc:cave_spider", "mobs_mc:creeper",
}

local function _clear_hostile_mobs()
    local all = minetest.get_objects_in_area(
        {x=-2000, y=-2000, z=-2000}, {x=2000, y=2000, z=2000}
    )
    for _, obj in ipairs(all) do
        local ent = obj:get_luaentity()
        if ent then
            for _, hostile in ipairs(HOSTILE_MOBS) do
                if ent.name == hostile then
                    obj:remove()
                    break
                end
            end
        end
    end
end

local function _apply_phase(phase)
    if phase == "exploration" then
        ENABLE_MOBS   = false
        ENABLE_HUNGER = false
        minetest.settings:set("enable_damage", "false")
        _clear_hostile_mobs()
    elseif phase == "survival_mobs_only" then
        ENABLE_MOBS   = true
        ENABLE_HUNGER = false
        minetest.settings:set("enable_damage", "true")
    elseif phase == "survival" then
        ENABLE_MOBS   = true
        ENABLE_HUNGER = true
        minetest.settings:set("enable_damage", "true")
    end
    minetest.log("action", "[PHASE] switched to " .. phase)
    -- Broadcast phase to clients so client mods can adjust reward shaping
    channel:send_all(minetest.serialize({event = "phase_change", phase = phase}))
end

minetest.register_globalstep(function(dtime)
    _phase_timer = _phase_timer + dtime
    if _phase_timer < 5.0 then return end
    _phase_timer = 0.0

    local path = minetest.get_worldpath() .. "/phase.txt"
    local ok, result = pcall(function()
        local f = io.open(path, "r")
        if not f then return nil end  -- file absent → stay in current phase
        local s = f:read("*l")
        f:close()
        return s
    end)
    if not ok or result == nil then return end
    local phase = result:match("^%s*(.-)%s*$")  -- trim whitespace
    if phase ~= "" and phase ~= _current_phase then
        _current_phase = phase
        _apply_phase(phase)
    end
end)
-- ─────────────────────────────────────────────────────────────────────────
```

**Key design choices:**

- `pcall` wraps all file I/O — a missing, locked, or malformed file is silently ignored.
- `_phase_timer` accumulates real `dtime` seconds, not game ticks — correct even when `pmul` (time multiplier) varies.
- `ENABLE_MOBS` / `ENABLE_HUNGER` mutation takes effect for all code that reads them on the next physics tick (the HP-clamp globalstep at line 172, the hunger-freeze loop at line 191 — those check `ENABLE_MOBS` / `ENABLE_HUNGER` at registration time, so **this approach has a limitation**: the `on_joinplayer` callbacks that register the HP-clamp and hunger-freeze loops fire once at join and use closures that capture the value of `ENABLE_MOBS/HUNGER` at that moment. Mutating the locals afterward does NOT affect already-registered loops.

### 3e. Lua limitation and mitigation

**Problem:** The HP-clamp globalstep and hunger-freeze `minetest.after` loop are registered inside `on_joinplayer`, which fires once at player join. They capture `ENABLE_MOBS` and `ENABLE_HUNGER` by value at join time. Mutating the locals later has no effect on those already-running loops.

**Mitigation:** The `_apply_phase()` function must also directly:
- For `exploration` → `minetest.settings:set("enable_damage", "false")` (prevents all damage), which is redundant with but independent of the HP-clamp loop.
- For `survival` → `minetest.settings:set("enable_damage", "true")` and **cancel** the hunger-freeze loop by setting a shared flag that the loop checks. The simplest approach: add a module-level `local _hunger_frozen = true` flag, and change the existing hunger loop inside `on_joinplayer` to check `_hunger_frozen` before calling `_freeze_hunger()`. The phase handler then sets `_hunger_frozen = false` for survival.

**Change to existing `on_joinplayer` hunger block (lines 179-196):**

```lua
-- Replace the existing freeze_hunger block:
if not ENABLE_HUNGER then
    local pname2 = player:get_player_name()
    local function _freeze_hunger()
        if not _hunger_frozen then return end  -- ← new guard
        local p = minetest.get_player_by_name(pname2)
        if p then
            p:get_meta():set_float("mcl_hunger:hunger", 20)
            p:get_meta():set_float("mcl_hunger:saturation", 5)
            p:get_meta():set_float("mcl_hunger:exhaustion", 0)
        end
    end
    ...
```

And in `_apply_phase()`:
```lua
if phase == "exploration" or phase == "survival_mobs_only" then
    _hunger_frozen = true
elseif phase == "survival" then
    _hunger_frozen = false
end
```

Similarly for HP-clamp: set `minetest.settings:set("enable_damage", "false")` which prevents HP loss from all sources regardless of the HP-clamp loop. When switching to survival, set `"enable_damage", "true"` and the HP-clamp globalstep (line 172) will fight VoxeLibre's damage system. The HP-clamp loop must also check a flag:

```lua
-- In on_joinplayer, replace the HP-clamp globalstep:
minetest.register_globalstep(function(_dtime)
    if _hp_frozen then  -- ← new module-level flag
        local p = minetest.get_player_by_name(pname3)
        if p then p:set_hp(20) end
    end
end)
```

`_hp_frozen` starts `true` (exploration mode) and `_apply_phase("survival")` sets it `false`.

**These are the only two lines in `on_joinplayer` that change.** Both changes are guards on existing behavior — no behavior change when flags are in their default state.

---

## 4. Python Phase Logic

### 4a. `_should_transition_to_survival()`

```python
def _should_transition_to_survival(episode: int, global_step: int, args) -> bool:
    """episode is 0-indexed. args.survival_episode is 1-indexed."""
    if not args.survival_mode:
        return False
    if args.survival_step is not None and global_step >= args.survival_step:
        return True
    if episode + 1 >= args.survival_episode:
        return True
    return False
```

### 4b. Main loop integration

```python
current_phase = "exploration"
global_step = 0      # cumulative across all episodes

for episode in range(resume_episode, num_episodes):
    # Write phase file at episode start (handles resume case)
    environment._write_phase_file(current_phase)

    for step in range(max_steps):
        global_step += 1

        # Phase transition check
        if current_phase == "exploration" and _should_transition_to_survival(
            episode, global_step, args
        ):
            new_phase = "survival_mobs_only" if args.survival_gradual else "survival"
            current_phase = new_phase
            environment._write_phase_file(current_phase)
            metric.record_phase_transition(global_step, episode + 1, current_phase)
            logging.info(
                "[PHASE TRANSITION] → %s at ep=%d step=%d",
                current_phase, episode + 1, global_step,
            )
            print(f"\n{'!'*60}")
            print(f"[PHASE TRANSITION] Switching to {current_phase} at ep={episode+1} step={global_step}")
            print(f"{'!'*60}\n")

        elif (
            current_phase == "survival_mobs_only"
            and args.survival_gradual
            and global_step >= _gradual_trigger_step + args.survival_gradual_delay
        ):
            current_phase = "survival"
            environment._write_phase_file(current_phase)
            metric.record_phase_transition(global_step, episode + 1, current_phase)
            ...
```

`_gradual_trigger_step` is set when `survival_mobs_only` is first written.

### 4c. Survival phase instruction injection

When `current_phase != "exploration"`, prepend a one-liner to the prompt text **at call site only** — do not modify `instruction_prompt` in place:

```python
_phase_prefix = (
    "[SURVIVAL MODE ACTIVE: hostile mobs now spawn, hunger drains. "
    "Prioritize safety alongside your role tasks.]\n\n"
    if current_phase != "exploration" else ""
)
content, last_action, error_count = await agent_do_action(
    ...,
    instruction_prompt=_phase_prefix + instruction_prompt,
    ...
)
```

This is a local variable, never assigned back to `instruction_prompt` — perfectly additive.

---

## 5. `build_role_configs()` Changes

Current signature (line 190):
```python
def build_role_configs(num_agents: int, role_prompts: dict) -> list:
```

New signature:
```python
def build_role_configs(
    num_agents: int,
    role_prompts: dict,
    team_mode: str = "heterogeneous",
    homogeneous_role: str = "gatherer",
) -> list:
```

**New branch (inserted before the existing loop, which becomes the `else`):**

```python
if team_mode != "heterogeneous":
    # Resolve the target role name
    if team_mode == "homogeneous-auto":
        role_name = homogeneous_role
    else:
        role_name = team_mode.removeprefix("homogeneous-")  # "gatherer", "hunter", "defender"
    return [
        {
            "name": role_name,
            "agent_name": f"agent_{i}_{role_name}",
            "curriculum_prompt": role_prompts[f"role_{role_name}"].format(
                num_agents=num_agents
            ),
        }
        for i in range(num_agents)
    ]
else:
    # Existing cycling logic — unchanged
    ...
```

**`CraftiumMetric` note:** `_save_text_summary()` (line 522) already uses
`["gatherer", "hunter", "defender"][i % 3]` for display labels — this is correct as long
as `role_configs[i]["name"]` is passed. In fact the summary does not use `role_configs` at all; it labels by agent index. In homogeneous mode the specialization index will be uniform across agents (all agents score equally on the same track). This is correct and expected. A note will be added to `_save_text_summary()`:

```python
if len(set(role for role, _ in zip(ROLE_NAMES * 10, range(self.num_agents)))) == 1:
    lines.append("NOTE: Homogeneous team — specialization index expected to be uniform.")
```

More simply: detect homogeneous by checking if all agents have the same role in `run_state`, and add the note then.

---

## 6. `CraftiumMetric` Additions

### 6a. New fields in `__init__`
```python
self.phase_transitions = []   # list of {"step": int, "episode": int, "phase": str}
self.team_mode = "heterogeneous"          # set from args after build
self.homogeneous_role = "gatherer"        # set from args after build
```

### 6b. New method
```python
def record_phase_transition(self, step: int, episode: int, phase: str) -> None:
    self.phase_transitions.append({"step": step, "episode": episode, "phase": phase})
```

### 6c. `save_run_metrics()` — add to serialized dict
```python
"phase_transitions": self.phase_transitions,
"team_mode": getattr(self, "team_mode", "heterogeneous"),
"homogeneous_role": getattr(self, "homogeneous_role", "gatherer"),
```

### 6d. Cumulative returns plot — vertical phase marker

In `_save_plots()` after drawing the cumulative returns lines (before `plt.legend()`):

```python
for pt in self.phase_transitions:
    plt.axvline(x=pt["step"], color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    plt.text(pt["step"], plt.ylim()[1] * 0.95, f"→ {pt['phase']}",
             fontsize=8, color="red", rotation=90, va="top")
```

### 6e. `_save_text_summary()` — phase section

Appended after the Hebbian section:
```
PHASE TRANSITIONS
  Total transitions: N
  [ep=E step=S] → survival_mobs_only
  [ep=E step=S] → survival
```

And if `team_mode != "heterogeneous"`:
```
TEAM MODE: homogeneous-gatherer
  (All agents share the gatherer role. Specialization index is expected uniform.)
```

---

## 7. Checkpoint Integration

In `save_checkpoint()`, add to the `run_state` dict:
```python
"current_phase": current_phase,
"phase_transitions": metric.phase_transitions,
"team_mode": getattr(metric, "team_mode", "heterogeneous"),
"homogeneous_role": getattr(metric, "homogeneous_role", "gatherer"),
```

In `load_checkpoint()` / `run()` resume block:
```python
current_phase = restored_state.get("current_phase", "exploration")
metric.phase_transitions = restored_state.get("phase_transitions", [])
# Re-signal the freshly started Minetest server
environment._write_phase_file(current_phase)
```

In `CraftiumMetric.restore_from_dict()`:
```python
metric.phase_transitions = d.get("phase_transitions", [])
metric.team_mode = d.get("team_mode", "heterogeneous")
metric.homogeneous_role = d.get("homogeneous_role", "gatherer")
```

---

## 8. `_LOG_TAGS` Addition

In `custom_environment_craftium.py` at line 630:
```python
_LOG_TAGS = (
    "[TOOLS]", "[INVENTORY]", "[TRACK STATUS]",
    "[DIG]", "[HUNT]", "[DEFEND]",
    "[PHASE]",   # ← new: picks up Lua minetest.log("[PHASE] switched to ...") lines
)
```

---

## 9. Risk Surface

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Lua `_apply_phase()` fires but HP-clamp loop still runs (captured-value closure) | High | HP freezes in survival mode | Add `_hp_frozen` module flag checked inside the loop (§3e) |
| Hunger-freeze `minetest.after` loop still runs after transition | High | Hunger never drains in survival | Add `_hunger_frozen` flag checked inside `_freeze_hunger()` (§3e) |
| `phase.txt` read mid-write (Lua reads partial file) | Low | Invalid phase string, ignored | `pcall` + `os.replace` atomic rename on Python side; Lua trims whitespace and only acts on known strings |
| `_write_phase_file()` called before `reset()` (world path not resolved) | Medium | AttributeError crash | `try/except AttributeError` early-return guard |
| `minetest.get_objects_in_area` with huge coordinates hangs server | Low | Server freeze | Use player positions ± 500 blocks instead of ±2000; or skip clearance on survival transition (mobs don't exist yet) |
| Gradual trigger step logic incorrect on resume | Medium | Hunger never enables | Save `_gradual_trigger_step` in `run_state.json` and restore it on resume |
| Homogeneous mode breaks Hebbian role-index lookup (`agent_roles = [ROLE_NAMES.index(rc["name"]) for rc in role_configs]`) | Low | All agents get same role index → `HebbianSocialGraph` treats them as same role, which is correct | No change needed; this is the expected outcome |
| `removeprefix()` not available on Python < 3.9 | Low | AttributeError | Replace with `team_mode[len("homogeneous-"):]` or `team_mode.split("-", 1)[1]` |
| Phase marker in plot crashes on empty `phase_transitions` | Low | Plot fails to save | Guard: `if self.phase_transitions: ...` |

---

## 10. Rollback Strategy

Every new flag has a default that exactly reproduces current behavior:

| New flag | Default | Current behavior preserved when |
|----------|---------|----------------------------------|
| `--team-mode` | `heterogeneous` | Always, unless flag is set |
| `--homogeneous-role` | `gatherer` | Only used with `homogeneous-auto` |
| `--survival-mode` | off | `_should_transition_to_survival()` returns False immediately |
| `--survival-episode` | `1` | Irrelevant unless `--survival-mode` |
| `--survival-step` | `None` | Irrelevant unless `--survival-mode` |
| `--survival-gradual` | off | Irrelevant unless `--survival-mode` |
| `--survival-gradual-delay` | `500` | Irrelevant unless `--survival-gradual` |

**Lua rollback:** The new globalstep block reads `phase.txt` which does not exist in current runs. The `pcall` returns `nil` on `io.open` failure → nothing happens. The `ENABLE_MOBS = false` / `ENABLE_HUNGER = false` defaults remain unchanged. The `_hp_frozen` and `_hunger_frozen` flags start as `true`, so the existing HP-clamp and hunger-freeze loops behave identically to today.

**If the implementation breaks something:** Removing the `--survival-mode` flag from any SLURM script is sufficient to revert to current behavior. The Lua `phase.txt` polling block can be disabled by removing just the one `minetest.register_globalstep` call and the two flag guards in `on_joinplayer`.

---

## 11. Test Coverage (new)

Add to `scripts/test_scripted_agent.py` (create if it doesn't exist):

```
--test-phase-transition   flag
```

When set: run 100 NoOp steps, then call `environment._write_phase_file("survival")`, run 20 more NoOp steps, then call `tail_server_log()` and assert that a line containing `[PHASE] switched to survival` appears. This validates the full Python→file→Lua→log round-trip.

---

## 12. Implementation Order

Once this plan is confirmed, implementation proceeds in this order to minimize risk at each step:

| # | Task | File | Reversible unit |
|---|------|------|-----------------|
| 1 | Add `_hp_frozen` / `_hunger_frozen` flags to Lua `on_joinplayer` | `init.lua` | Lua-only; no Python change |
| 2 | Add phase-polling globalstep block to Lua | `init.lua` | No effect until `phase.txt` exists |
| 3 | Add `_write_phase_file()` to `CraftiumEnvironmentInterface` | `custom_environment_craftium.py` | Standalone method, not called yet |
| 4 | Add `"[PHASE]"` to `_LOG_TAGS` | `custom_environment_craftium.py` | 1-word addition |
| 5 | Add `phase_transitions` field + `record_phase_transition()` to `CraftiumMetric` | `craftium_metric.py` | New field only |
| 6 | Add phase marker to cumulative returns plot | `craftium_metric.py` | Guard on empty list |
| 7 | Add phase section to `_save_text_summary()` | `craftium_metric.py` | No effect until transitions recorded |
| 8 | Add team-mode branch to `build_role_configs()` | `multi_agent_craftium.py` | Defaulting to `heterogeneous` = no change |
| 9 | Add all new CLI args to `parse_args()` | `multi_agent_craftium.py` | Flags parse silently with defaults |
| 10 | Wire phase detection + `_write_phase_file()` into `run()` | `multi_agent_craftium.py` | Gated behind `args.survival_mode` |
| 11 | Wire `_phase_prefix` injection into `agent_do_action()` call | `multi_agent_craftium.py` | Empty string when exploration |
| 12 | Checkpoint save/restore for phase fields | `multi_agent_craftium.py` | Additive dict keys |
| 13 | Test script | `scripts/test_scripted_agent.py` | New file, no prod impact |
