-- doors.lua: door unlock logic for doors 2, 3, and 4 (plan §2).
-- Doors are bedrock blocks that get replaced with air when conditions are met.
-- Door positions (2 blocks tall: y=FLOOR_Y and y=FLOOR_Y+1):
--   Door 2: x=DOOR2_POS.x, z=DOOR2_POS.z  — opens 20 steps after 6th anvil
--   Door 3: x=DOOR3_X,     z=CH3_NORTH_WALL_Z — opens when all agents in communal
--   Door 4: x=DOOR4_POS.x, z=DOOR4_POS.z  — opens when all Ch4 mobs dead
--   Cell doors: x=cell_x_center(i), z=CH3_FRONT_WALL_Z — opened by switches

-- Visible "locked door" block: distinct red glowing texture so players /
-- agents can see exactly where doors are. Solid + unbreakable: behaves
-- like bedrock for collision and digging, but is recognisable on sight.
minetest.register_node("five_chambers:door_locked", {
    description = "Locked Door",
    tiles = {
        "mcl_core_stonebrick.png^[colorize:#bb1f1f:200",  -- top
        "mcl_core_stonebrick.png^[colorize:#bb1f1f:200",  -- bottom
        "mcl_core_stonebrick.png^[colorize:#ff4444:220",  -- sides (brighter)
    },
    paramtype = "light",
    light_source = 7,
    is_ground_content = false,
    groups = {unbreakable = 1, not_in_creative_inventory = 1},
    sounds = nil,
})

five_chambers.door_state = {
    door1_open = false,
    door1_force_teleported = false,  -- did the Ch1 timeout already fire?
    door2_open = false,
    door2_countdown = -1,
    door2_force_teleported = false,  -- did the Ch2 timeout already fire?
    door3_open = false,
    door3_force_teleported = false,  -- did the Ch3 timeout already fire?
    door4_open = false,
    door4_force_teleported = false,  -- did the Ch4 timeout already fire?
    cell_doors = {},   -- cell_doors[i] = true when cell i door is open
}

-- ch2_transitioned[i] = true once agent_i has been teleported to Ch3 cell i.
five_chambers.ch2_transitioned       = {}
five_chambers.ch2_transitioned_count = 0

-- Tick at which all NUM_AGENTS players were first connected. The Ch1
-- timeout countdown measures from THIS tick, not from server start —
-- otherwise the warmup phase (server boots minutes before clients join)
-- burns the whole timeout budget and the teleport fires the instant the
-- last agent joins, dropping them in Ch2 with zero Ch1 experience.
-- Reset to nil in init_doors() at episode boundaries.
five_chambers.all_connected_tick = nil

-- Tick at which Python signals warmup-complete (preferred anchor over
-- all_connected_tick — see the stamping logic in the timeout globalstep).
-- Reset to nil in init_doors() so each episode re-anchors at episode start;
-- the warmup_complete.txt file persists on disk, so the next globalstep
-- after a reset will immediately re-stamp this to the current sc.
five_chambers.warmup_complete_tick = nil

-- Re-place the visible door_locked block at a door position. Inverse of
-- open_door_at(). Used by relock_all_doors() at episode reset to undo any
-- doors that were swapped to air during the previous episode.
local function lock_door_at(x, z)
    local y = five_chambers.FLOOR_Y
    minetest.set_node({x=x, y=y+1, z=z}, {name="five_chambers:door_locked"})
    minetest.set_node({x=x, y=y+2, z=z}, {name="five_chambers:door_locked"})
end

-- Re-places every door_locked block in the world. Called from the reset
-- handler so that doors opened during episode N are re-locked before
-- episode N+1 begins. Without this, only door_state flags reset but the
-- physical air gap from the previous open_door_at() persists.
-- Door 1 spans 3 X-blocks centred on DOOR1_X at z = CH1.z1. Helpers below
-- iterate over those three columns so the open/relock semantics match the
-- 1-wide doors used elsewhere.
local function _door1_columns()
    local dx = five_chambers.DOOR1_X
    local z  = five_chambers.CH1.z1
    return {
        {x = dx - 1, z = z},
        {x = dx,     z = z},
        {x = dx + 1, z = z},
    }
end

-- Ch1's floor is one block higher than chambers 2-5 (dirt layer on top of
-- bedrock subfloor), so Door 1's opening sits one block higher than the
-- other doors. lock_door_at / open_door_at bake in y=FLOOR_Y+1,+2 — we
-- can't reuse them. The door blocks live at y=FLOOR_Y+2,+3 (which equals
-- CH1_DIRT_Y+1, CH1_DIRT_Y+2 — agent height in Ch1).
local function _door1_y_pair()
    local y_dirt = five_chambers.CH1_DIRT_Y or (five_chambers.FLOOR_Y + 1)
    return y_dirt + 1, y_dirt + 2
end

local function lock_door1()
    local y_lo, y_hi = _door1_y_pair()
    for _, p in ipairs(_door1_columns()) do
        minetest.set_node({x=p.x, y=y_lo, z=p.z}, {name="five_chambers:door_locked"})
        minetest.set_node({x=p.x, y=y_hi, z=p.z}, {name="five_chambers:door_locked"})
    end
end

local function open_door1_blocks()
    local y_lo, y_hi = _door1_y_pair()
    local cols = _door1_columns()
    -- Use VoxelManip directly — same technique world_gen.lua uses to
    -- PLACE the door_locked blocks. mcl_observers monkey-patches
    -- minetest.set_node and minetest.swap_node; on partially-loaded
    -- chunks its hook calls get_node() which can crash or silently
    -- swallow the write, leaving the red door_locked block visible to
    -- clients even though door_state.door1_open is true. Writing
    -- through VoxelManip bypasses the hook entirely and write_to_map
    -- with light=true forces lighting recalc to clear the locked door's
    -- light_source=7 emission.
    local air_cid = minetest.get_content_id("air")
    local x0 = cols[1].x
    local x1 = cols[#cols].x
    local z  = cols[1].z
    local vm = minetest.get_voxel_manip(
        {x = x0, y = y_lo, z = z},
        {x = x1, y = y_hi, z = z}
    )
    local emin, emax = vm:get_emerged_area()
    local data = vm:get_data()
    local va   = VoxelArea:new{MinEdge = emin, MaxEdge = emax}
    for _, p in ipairs(cols) do
        data[va:index(p.x, y_lo, z)] = air_cid
        data[va:index(p.x, y_hi, z)] = air_cid
    end
    vm:set_data(data)
    vm:write_to_map(true)  -- true = update lighting
end

function five_chambers.relock_all_doors()
    -- Door 1 (Ch1 → Ch2)
    lock_door1()

    -- Door 2 (Ch2 → Ch3)
    local d2 = five_chambers.DOOR2_POS
    lock_door_at(d2.x, d2.z)

    -- Door 3 (Ch3 communal → Ch4 corridor)
    lock_door_at(five_chambers.DOOR3_X, five_chambers.CH3_NORTH_WALL_Z)

    -- Door 4 (Ch4 → Ch5)
    local d4 = five_chambers.DOOR4_POS
    lock_door_at(d4.x, d4.z)

    -- Cell doors (Ch3 front wall, one per agent)
    local front_z = five_chambers.CH3_FRONT_WALL_Z
    for i = 0, five_chambers.NUM_AGENTS - 1 do
        lock_door_at(five_chambers.cell_x_center(i), front_z)
    end
end

function five_chambers.init_doors()
    five_chambers.door_state.door1_open             = false
    five_chambers.door_state.door1_force_teleported = false
    five_chambers.door_state.door2_open             = false
    five_chambers.door_state.door2_countdown        = -1
    five_chambers.door_state.door2_force_teleported = false
    five_chambers.door_state.door3_open             = false
    five_chambers.door_state.door3_force_teleported = false
    five_chambers.door_state.door4_open             = false
    five_chambers.door_state.door4_force_teleported = false
    five_chambers.door_state.cell_doors             = {}
    five_chambers.ch2_transitioned                  = {}
    five_chambers.ch2_transitioned_count            = 0
    five_chambers.all_connected_tick                = nil
    five_chambers.warmup_complete_tick              = nil
    for i = 0, five_chambers.NUM_AGENTS - 1 do
        five_chambers.door_state.cell_doors[i] = false
        five_chambers.ch2_transitioned[i]      = false
    end

    -- DEBUG_SINGLE: skip the anvil-coop mechanic and leave Door 2 open from
    -- the start so a solo human player can walk Ch2 → Ch3 directly. The
    -- Ch2→Ch3 teleport globalstep then catches them at z >= DOOR2_POS.z and
    -- drops them into cell 0, so the switch puzzle still gets exercised.
    if five_chambers.DEBUG_SINGLE then
        five_chambers.door_state.door2_open = true
        local d2 = five_chambers.DOOR2_POS
        five_chambers.open_door_at(d2.x, d2.z)
    end
end

-- Replaces the 2-block door opening at (x, FLOOR_Y+1, z) and (x, FLOOR_Y+2, z)
-- with air. Idempotent. Used by Door 2 (Ch2→Ch3), Door 3 (Ch3 communal),
-- Door 4 (Ch4→Ch5), and the per-cell doors in Ch3.
--
-- Uses VoxelManip directly — same technique world_gen.lua's place_node
-- uses to PLACE these blocks at world build time. mcl_observers
-- monkey-patches minetest.set_node and minetest.swap_node and its hook
-- can crash or silently swallow writes on partially-loaded chunks,
-- leaving the red door_locked block visible to clients even though Lua
-- thinks the door is open. Going through VoxelManip avoids the hook;
-- write_to_map(true) forces lighting recalc to clear door_locked's
-- light_source=7 emission so the area properly darkens.
function five_chambers.open_door_at(x, z)
    local y = five_chambers.FLOOR_Y
    local air_cid = minetest.get_content_id("air")
    local vm = minetest.get_voxel_manip(
        {x = x, y = y + 1, z = z},
        {x = x, y = y + 2, z = z}
    )
    local emin, emax = vm:get_emerged_area()
    local data = vm:get_data()
    local va   = VoxelArea:new{MinEdge = emin, MaxEdge = emax}
    data[va:index(x, y + 1, z)] = air_cid
    data[va:index(x, y + 2, z)] = air_cid
    vm:set_data(data)
    vm:write_to_map(true)  -- true = update lighting
end

-- Atomically write `content` to `{world_path}/<filename>`. Used to
-- surface door-open events to Python via state-file IPC — the chamber-
-- state composer on the Python side reads these files and appends a
-- "Door X: OPEN — walk north" line to the LLM prompt so the agent has
-- a text signal of the unlock event, not just the visual delta in the
-- rendered frame. Generic helper so doors 1-4 + cell doors all use the
-- same path (state_files.lua's clear_state_files() must drop every
-- filename written by callers of this helper).
local function _write_door_state_file(filename, content)
    local wp = minetest.get_worldpath()
    if not wp then return end
    local tmp = wp .. "/" .. filename .. ".tmp"
    local dst = wp .. "/" .. filename
    local f = io.open(tmp, "w")
    if f then
        f:write(content)
        f:close()
        os.rename(tmp, dst)
    end
end

-- Opens Door 1 (Ch1 → Ch2). Idempotent. Called by milestones.lua when a
-- "real" Ch1 milestone fires (M2/M3/M4/M5/M6/M7) and by the Ch1 timeout
-- globalstep below as a fallback.
function five_chambers.open_door1()
    if five_chambers.door_state.door1_open then return end
    five_chambers.door_state.door1_open = true
    open_door1_blocks()
    minetest.log("action", "[five_chambers] Door 1 opened.")
    -- Surface to Python via the state-file IPC so agents still in Ch1
    -- can see "Door 1: OPEN — walk north to enter Chamber 2" in their
    -- chamber_state field on the next prompt. Without this they have
    -- no signal that the door unlocked (the visual delta is subtle —
    -- a red bedrock-textured block becomes air at the same coords).
    _write_door_state_file("door1_state.txt", "open\n")
end

-- Called when all anvils have been broken at least once; starts the countdown.
function five_chambers.start_door2_countdown()
    if five_chambers.door_state.door2_countdown < 0 then
        five_chambers.door_state.door2_countdown = five_chambers.DOOR2_DELAY
    end
end

-- Ticked each globalstep. Counts down by 1 env-step (3 Lua ticks) and opens
-- Door 2 when the counter reaches zero.
function five_chambers.tick_door2()
    local ds = five_chambers.door_state
    if ds.door2_open or ds.door2_countdown < 0 then return end

    ds.door2_countdown = ds.door2_countdown - 1
    if ds.door2_countdown <= 0 then
        ds.door2_open = true
        local d2 = five_chambers.DOOR2_POS
        five_chambers.open_door_at(d2.x, d2.z)
        minetest.log("action", "[five_chambers] Door 2 opened.")
        -- Surface to Python (same pattern as door1_state.txt).
        _write_door_state_file("door2_state.txt", "open\n")
    end
end

minetest.register_globalstep(function(dtime)
    -- Only tick every 3 Lua ticks (1 env step) using step_counter.
    if five_chambers.step_counter % 3 ~= 0 then return end
    five_chambers.tick_door2()
end)

-- Reinstates Door 2 (visible locked-door block) after all agents have been
-- teleported to Ch3.
function five_chambers.relock_door_2()
    local d2 = five_chambers.DOOR2_POS
    local y  = five_chambers.FLOOR_Y
    minetest.set_node({x=d2.x, y=y+1, z=d2.z}, {name="five_chambers:door_locked"})
    minetest.set_node({x=d2.x, y=y+2, z=d2.z}, {name="five_chambers:door_locked"})
    minetest.log("action", "[five_chambers] Door 2 relocked.")
end

-- Opens a specific cell door (0-indexed). Called by switches.lua.
-- Door for cell i is at (cell_x_center(i), FLOOR_Y+1, CH3_FRONT_WALL_Z).
function five_chambers.open_cell_door(cell_i)
    if five_chambers.door_state.cell_doors[cell_i] then return end
    five_chambers.door_state.cell_doors[cell_i] = true
    five_chambers.open_door_at(
        five_chambers.cell_x_center(cell_i),
        five_chambers.CH3_FRONT_WALL_Z)

    -- Surface to Python: rewrite the full per-cell map each time a door
    -- opens. File format is one line per agent index, "<i>:open" sorted
    -- numerically. Python reads this and tells each agent "Your cell
    -- door is OPEN/LOCKED" in chamber_state.
    local lines = {}
    for i = 0, (five_chambers.NUM_AGENTS - 1) do
        if five_chambers.door_state.cell_doors[i] then
            table.insert(lines, tostring(i) .. ":open")
        end
    end
    table.sort(lines)
    _write_door_state_file("cell_doors_state.txt",
        table.concat(lines, "\n") .. "\n")
end

-- Checks if all NUM_AGENTS agents are simultaneously in the communal room.
-- When true: fires M19 for all agents in communal and opens Door 3.
function five_chambers.check_door3()
    if five_chambers.door_state.door3_open then return end

    local names = {}
    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        if five_chambers.agent_index(name) >= 0 then
            local pos = player:get_pos()
            if pos and five_chambers.get_chamber_for_pos(pos) == "ch3_communal" then
                table.insert(names, name)
            end
        end
    end

    if #names >= five_chambers.NUM_AGENTS then
        five_chambers.door_state.door3_open = true
        five_chambers.open_door_at(
            five_chambers.DOOR3_X, five_chambers.CH3_NORTH_WALL_Z)
        five_chambers.fire_milestone("m19_all_in_communal", names)
        minetest.log("action",
            "[five_chambers] All agents in communal — Door 3 opened.")
        _write_door_state_file("door3_state.txt", "open\n")
    end
end

minetest.register_globalstep(function(dtime)
    if not five_chambers.CHAMBERS[3].enabled then return end
    if five_chambers.step_counter % 3 ~= 0 then return end
    five_chambers.check_door3()
end)

-- Opens Door 4 after all Ch4 mobs are dead. Called by mobs.lua.
function five_chambers.open_door4()
    if five_chambers.door_state.door4_open then return end
    five_chambers.door_state.door4_open = true
    local d4 = five_chambers.DOOR4_POS
    five_chambers.open_door_at(d4.x, d4.z)
    minetest.log("action", "[five_chambers] Door 4 opened.")
    _write_door_state_file("door4_state.txt", "open\n")
end

-- ── Ch2→Ch3 transition globalstep ────────────────────────────────
-- When Door 2 is open and an agent steps onto z >= DOOR2_POS.z,
-- teleport them directly into their isolation cell and fire M16.
-- After all NUM_AGENTS agents have transitioned, relock Door 2.

minetest.register_globalstep(function(dtime)
    if not five_chambers.door_state.door2_open then return end
    if not five_chambers.CHAMBERS[3].enabled then return end

    local d2z = five_chambers.DOOR2_POS.z  -- 23

    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        local idx  = five_chambers.agent_index(name)
        if idx >= 0 and not five_chambers.ch2_transitioned[idx] then
            local pos = player:get_pos()
            if pos and pos.z >= d2z then
                local dest = five_chambers.cell_teleport_pos(idx)
                player:set_pos(dest)
                five_chambers.ch2_transitioned[idx] = true
                five_chambers.ch2_transitioned_count =
                    five_chambers.ch2_transitioned_count + 1
                -- Suppress m16_enter_cell when the Ch2→Ch3 timeout
                -- teleport already fired this episode: the agents were
                -- force-relocated rather than entering via Door 2 on
                -- their own, so this is not an honest entry milestone.
                -- The teleport itself remains for layout continuity;
                -- the milestone reward is what we withhold.
                if five_chambers.door_state
                   and five_chambers.door_state.door2_force_teleported then
                    minetest.log("action",
                        "[five_chambers] m16_enter_cell suppressed for "
                        .. name .. " (Ch2 timeout teleport fired this episode)")
                else
                    five_chambers.fire_milestone("m16_enter_cell", {name})
                end
                minetest.log("action",
                    "[five_chambers] " .. name .. " teleported to Ch3 cell " .. idx)

                if five_chambers.ch2_transitioned_count >= five_chambers.NUM_AGENTS then
                    five_chambers.relock_door_2()
                    five_chambers.door_state.door2_open = false
                end
            end
        end
    end
end)

-- ── Ch1 timeout teleport ─────────────────────────────────────────
-- Door 1 stays locked for the entire Ch1 phase. The teleport is fired
-- by EITHER (a) Python writing the ch1_force_teleport.txt flag once it has
-- counted --ch1-timeout-steps env steps in this episode (primary path; see
-- CraftiumEnvironmentInterface.force_ch1_teleport in
-- src/mindforge/custom_environment_craftium.py), OR (b) the lua-side
-- step_counter crossing CH1_TIMEOUT_TICKS (fallback). Door 1 itself stays
-- locked — agents are teleported *across* it, not *through* it, so the
-- "Ch1 is over, no going back" framing holds. Fires at most once per
-- episode (gated by door1_force_teleported, reset in init_doors()).

-- Diagnostic: log step_counter / timeout / connected player count every
-- 600 ticks (~30s wall, ~67 env steps with 9 ticks/step). Lets us see
-- in the trainer stderr whether the timeout teleport globalstep is even
-- being entered, what step_counter it sees, and whether
-- get_connected_players() returns the agents (a frequent silent failure
-- on HPC where the clients reconnect under different names).
local _CH1_DIAG_INTERVAL = 600

minetest.register_globalstep(function(dtime)
    if not five_chambers.CHAMBERS[2].enabled then return end

    local sc = five_chambers.step_counter or 0
    local n_required = five_chambers.NUM_AGENTS or 1
    local connected_now = minetest.get_connected_players()
    local n_connected_now = #connected_now

    -- Stamp the tick at which all agents first appeared. Subsequent
    -- timeout math is relative to THIS tick, not server start, so the
    -- warmup phase (server up several minutes before clients finish
    -- joining) doesn't burn the Ch1 budget. Once stamped, doesn't move.
    if (not five_chambers.all_connected_tick)
       and n_connected_now >= n_required then
        five_chambers.all_connected_tick = sc
        if io and io.stderr then
            io.stderr:write(string.format(
                "[CH1_TIMEOUT] all %d/%d players connected at tick=%d "
                .. "— Ch1 timeout countdown starts now (target=%d ticks "
                .. "from this point)\n",
                n_connected_now, n_required, sc,
                five_chambers.CH1_TIMEOUT_TICKS or -1))
            io.stderr:flush()
        end
    end

    -- Stamp the tick at which Python signals warmup-complete. The
    -- all_connected_tick stamp above isn't enough because clients join
    -- early in warmup; Python then sits in warmup_noop() for the full
    -- warmup_time (e.g. 300 s) while Lua's globalstep keeps ticking. With
    -- CH1_TIMEOUT_TICKS=400 (~20 s) the timeout would fire mid-warmup and
    -- agents start in Ch2. Warmup-complete is signalled by Python writing
    -- {worldpath}/warmup_complete.txt after the media-load poll loop ends
    -- (see CraftiumEnvironmentInterface.signal_warmup_complete).
    if not five_chambers.warmup_complete_tick then
        local wc_path = (minetest.get_worldpath() or "") .. "/warmup_complete.txt"
        local f = io.open(wc_path, "r")
        if f then
            f:close()
            five_chambers.warmup_complete_tick = sc
            if io and io.stderr then
                io.stderr:write(string.format(
                    "[CH1_TIMEOUT] warmup_complete signalled at tick=%d "
                    .. "— Ch1 timeout countdown anchored here\n", sc))
                io.stderr:flush()
            end
        end
    end

    -- Reference tick for the countdown: ONLY warmup_complete_tick. The
    -- previous "warmup_complete OR all_connected" fallback fired the
    -- timeout mid-warmup because clients connect ~5 min before Python
    -- finishes its media-load poll loop, so ticks_in_ch1 (anchored at
    -- all_connected_tick) crossed target before warmup_complete arrived.
    -- Python's force-flag (force_ch1_teleport.txt at step >=
    -- ch1_timeout_steps) remains the primary trigger; this lua-side timer
    -- is only the safety net for runaway Python — and that safety net
    -- correctly stays disarmed until Python signals it's actually using
    -- the env.
    local ref_tick = five_chambers.warmup_complete_tick
    local ticks_in_ch1 = (ref_tick and (sc - ref_tick)) or -1

    if sc > 0 and sc % _CH1_DIAG_INTERVAL == 0
       and not five_chambers.door_state.door1_force_teleported then
        if io and io.stderr then
            local anchor = five_chambers.warmup_complete_tick
                and "warmup_complete"
                or (five_chambers.all_connected_tick and "all_connected_only_no_countdown" or "none")
            io.stderr:write(string.format(
                "[CH1_TIMEOUT_DIAG] step_counter=%d ticks_in_ch1=%d "
                .. "target=%d anchor=%s force_teleported=%s players=%d\n",
                sc, ticks_in_ch1,
                five_chambers.CH1_TIMEOUT_TICKS or -1,
                anchor,
                tostring(five_chambers.door_state.door1_force_teleported),
                n_connected_now))
            io.stderr:flush()
        end
    end

    if five_chambers.door_state.door1_force_teleported then return end
    if sc % 3 ~= 0 then return end

    -- Two paths fire the teleport: (1) Python writes a force-flag file once
    -- it has counted ch1_timeout_steps env steps in this episode (robust);
    -- (2) lua step_counter crosses CH1_TIMEOUT_TICKS (legacy fallback in
    -- case the file isn't being written for some reason).
    local force_flag_path = (minetest.get_worldpath() or "") .. "/ch1_force_teleport.txt"
    local force_fired = false
    do
        local f = io.open(force_flag_path, "r")
        if f then
            f:close()
            force_fired = true
            os.remove(force_flag_path)
        end
    end
    -- Timer compares ticks-since-anchor to the target. Anchor is
    -- warmup_complete_tick ONLY. Until Python signals warmup-complete,
    -- ticks_in_ch1 stays -1 and the lua-side timer is disarmed — Python's
    -- force-flag is the only Ch1->Ch2 path during warmup.
    -- Failure modes this avoids:
    --   1. Pre-join race: warmup ticks accumulate while clients load;
    --      raw counter crosses target before any agent connects; teleport
    --      fires with 0 players and burns the flag for the episode.
    --   2. Post-join instant-fire: if clients join slower than target,
    --      teleport fires within seconds — agents land in Ch2 with zero
    --      Ch1 experience.
    --   3. Warmup-poll race: clients join EARLY in warmup, then Python
    --      sits in warmup_noop() for the full warmup_time (e.g. 300 s).
    --      With all_connected_tick as the anchor, ticks_in_ch1 crosses
    --      target mid-warmup and agents start step 0 in Ch2.
    -- Python's force-flag still bypasses the timer for the explicit-trigger
    -- path so a manual --ch1-timeout-steps still works.
    if (not force_fired)
       and (ticks_in_ch1 < 0
            or ticks_in_ch1 < (five_chambers.CH1_TIMEOUT_TICKS or 3000)) then
        return
    end

    local connected = connected_now
    local n_connected = n_connected_now

    -- Door 1 stays locked. Agents are teleported across, not through —
    -- leaving it visibly closed is consistent with "Ch1 is over, no
    -- going back" and stops agents wasting actions trying to dig through.
    five_chambers.door_state.door1_force_teleported = true
    minetest.log("action",
        "[five_chambers] Ch1 timeout fired at tick "
        .. tostring(sc)
        .. " — teleporting agents to Ch2.")
    if io and io.stderr then
        io.stderr:write("[CH1_TIMEOUT] tick="
            .. tostring(sc)
            .. " forced teleport to Ch2 — players="
            .. tostring(n_connected) .. "\n")
        io.stderr:flush()
    end
    for _, player in ipairs(connected) do
        local name = player:get_player_name()
        local idx  = five_chambers.agent_index(name)
        local dest = (idx >= 0)
            and five_chambers.ch2_fallback_spawn_pos(idx)
            or  {x = 7, y = five_chambers.FLOOR_Y + 1,
                 z = (five_chambers.CH2 and five_chambers.CH2.z0 or 17) + 2}
        player:set_pos(dest)
        -- Forfeit any unearned Ch1 milestones: the team is being RESCUED out of
        -- Ch1, not clearing it, so it should not collect Ch1 rewards. Critically
        -- this stops m1_move_5 from firing off this teleport's position jump —
        -- the agent did not move, it was teleported. (Already-earned Ch1
        -- milestones stay earned; this only marks the not-yet-fired ones.)
        five_chambers.forfeit_track_milestones(name, "ch1_solo")
        if io and io.stderr then
            io.stderr:write(string.format(
                "[CH1_TIMEOUT] %s idx=%d -> (%d,%d,%d)\n",
                name, idx, dest.x, dest.y, dest.z))
            io.stderr:flush()
        end
    end
end)

-- ── Per-chamber force-teleport globalstep (Ch2/Ch3/Ch4 -> next chamber) ──
--
-- Each chamber gets 20% of the episode (see _CHAMBER_PCT in
-- multi_agent_craftium.py). When Python decides a chamber's budget is up,
-- it writes ch{N}_force_teleport.txt to the world dir. Lua polls for the
-- file, teleports every connected agent to the next chamber's spawn, and
-- removes the flag. One-shot per episode (gated by door{N}_force_teleported,
-- which init_doors() resets each reset).
--
-- Door state is NOT changed by this teleport — we just relocate the players.
-- That preserves milestone honesty: doors only count as "open" when they
-- were opened by the agents' actions, not by the timer.
local _per_chamber_timeout_specs = {
    {
        from         = 2,
        flag_name    = "ch2_force_teleport.txt",
        spawn_fn     = function(i) return five_chambers.cell_teleport_pos(i) end,
        log_tag      = "CH2_TIMEOUT",
        dest_label   = "Ch3 cell",
    },
    {
        from         = 3,
        flag_name    = "ch3_force_teleport.txt",
        spawn_fn     = function(i) return five_chambers.ch4_fallback_spawn_pos(i) end,
        log_tag      = "CH3_TIMEOUT",
        dest_label   = "Ch4",
    },
    {
        from         = 4,
        flag_name    = "ch4_force_teleport.txt",
        spawn_fn     = function(i) return five_chambers.ch5_fallback_spawn_pos(i) end,
        log_tag      = "CH4_TIMEOUT",
        dest_label   = "Ch5",
    },
}

minetest.register_globalstep(function(_dtime)
    local world_path = minetest.get_worldpath() or ""
    if world_path == "" then return end
    for _, spec in ipairs(_per_chamber_timeout_specs) do
        local already_key = "door" .. spec.from .. "_force_teleported"
        if not five_chambers.door_state[already_key] then
            local flag_path = world_path .. "/" .. spec.flag_name
            local f = io.open(flag_path, "r")
            if f then
                f:close()
                os.remove(flag_path)
                five_chambers.door_state[already_key] = true
                local connected = minetest.get_connected_players() or {}
                if io and io.stderr then
                    io.stderr:write(string.format(
                        "[%s] forced teleport to %s -- players=%d\n",
                        spec.log_tag, spec.dest_label, #connected))
                    io.stderr:flush()
                end
                for _, player in ipairs(connected) do
                    local name = player:get_player_name()
                    local idx  = five_chambers.agent_index(name)
                    if idx >= 0 then
                        local dest = spec.spawn_fn(idx)
                        player:set_pos(dest)
                        if io and io.stderr then
                            io.stderr:write(string.format(
                                "[%s] %s idx=%d -> (%d,%d,%d)\n",
                                spec.log_tag, name, idx,
                                dest.x, dest.y, dest.z))
                            io.stderr:flush()
                        end
                    end
                end
            end
        end
    end
end)
