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
    door3_open = false,
    door4_open = false,
    cell_doors = {},   -- cell_doors[i] = true when cell i door is open
}

-- ch2_transitioned[i] = true once agent_i has been teleported to Ch3 cell i.
five_chambers.ch2_transitioned       = {}
five_chambers.ch2_transitioned_count = 0

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
    for _, p in ipairs(_door1_columns()) do
        minetest.set_node({x=p.x, y=y_lo, z=p.z}, {name="air"})
        minetest.set_node({x=p.x, y=y_hi, z=p.z}, {name="air"})
    end
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
    five_chambers.door_state.door3_open             = false
    five_chambers.door_state.door4_open             = false
    five_chambers.door_state.cell_doors             = {}
    five_chambers.ch2_transitioned                  = {}
    five_chambers.ch2_transitioned_count            = 0
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
-- with air. Idempotent.
function five_chambers.open_door_at(x, z)
    local y = five_chambers.FLOOR_Y
    minetest.set_node({x=x, y=y+1, z=z}, {name="air"})
    minetest.set_node({x=x, y=y+2, z=z}, {name="air"})
end

-- Opens Door 1 (Ch1 → Ch2). Idempotent. Called by milestones.lua when a
-- "real" Ch1 milestone fires (M2/M3/M4/M5/M6/M7) and by the Ch1 timeout
-- globalstep below as a fallback.
function five_chambers.open_door1()
    if five_chambers.door_state.door1_open then return end
    five_chambers.door_state.door1_open = true
    open_door1_blocks()
    minetest.log("action", "[five_chambers] Door 1 opened.")
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
                five_chambers.fire_milestone("m16_enter_cell", {name})
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
    if sc > 0 and sc % _CH1_DIAG_INTERVAL == 0
       and not five_chambers.door_state.door1_force_teleported then
        if io and io.stderr then
            io.stderr:write(string.format(
                "[CH1_TIMEOUT_DIAG] step_counter=%d target=%d "
                .. "force_teleported=%s players=%d\n",
                sc,
                five_chambers.CH1_TIMEOUT_TICKS or -1,
                tostring(five_chambers.door_state.door1_force_teleported),
                #minetest.get_connected_players()))
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
    if (not force_fired)
       and sc < (five_chambers.CH1_TIMEOUT_TICKS or 3000) then
        return
    end

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
            .. tostring(#minetest.get_connected_players()) .. "\n")
        io.stderr:flush()
    end
    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        local idx  = five_chambers.agent_index(name)
        local dest = (idx >= 0)
            and five_chambers.ch2_fallback_spawn_pos(idx)
            or  {x = 7, y = five_chambers.FLOOR_Y + 1,
                 z = (five_chambers.CH2 and five_chambers.CH2.z0 or 17) + 2}
        player:set_pos(dest)
        if io and io.stderr then
            io.stderr:write(string.format(
                "[CH1_TIMEOUT] %s idx=%d -> (%d,%d,%d)\n",
                name, idx, dest.x, dest.y, dest.z))
            io.stderr:flush()
        end
    end
end)
