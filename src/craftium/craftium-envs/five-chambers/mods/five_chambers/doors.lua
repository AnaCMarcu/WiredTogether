-- doors.lua: door unlock logic for doors 2, 3, and 4 (plan §2).
-- Doors are bedrock blocks that get replaced with air when conditions are met.
-- Door positions (2 blocks tall: y=FLOOR_Y and y=FLOOR_Y+1):
--   Door 2: x=DOOR2_POS.x, z=DOOR2_POS.z  — opens 20 steps after 6th anvil
--   Door 3: x=DOOR3_X,     z=CH3_NORTH_WALL_Z — opens when all agents in communal
--   Door 4: x=DOOR4_POS.x, z=DOOR4_POS.z  — opens when all Ch4 mobs dead
--   Cell doors: x=cell_x_center(i), z=CH3_FRONT_WALL_Z — opened by switches

five_chambers.door_state = {
    door2_open = false,
    door2_countdown = -1,
    door3_open = false,
    door4_open = false,
    cell_doors = {},   -- cell_doors[i] = true when cell i door is open
}

-- ch2_transitioned[i] = true once agent_i has been teleported to Ch3 cell i.
five_chambers.ch2_transitioned       = {}
five_chambers.ch2_transitioned_count = 0

function five_chambers.init_doors()
    five_chambers.door_state.door2_open      = false
    five_chambers.door_state.door2_countdown = -1
    five_chambers.door_state.door3_open      = false
    five_chambers.door_state.door4_open      = false
    five_chambers.door_state.cell_doors      = {}
    five_chambers.ch2_transitioned           = {}
    five_chambers.ch2_transitioned_count     = 0
    for i = 0, five_chambers.NUM_AGENTS - 1 do
        five_chambers.door_state.cell_doors[i] = false
        five_chambers.ch2_transitioned[i]      = false
    end
end

-- Replaces the 2-block door opening at (x, FLOOR_Y+1, z) and (x, FLOOR_Y+2, z)
-- with air. Idempotent.
function five_chambers.open_door_at(x, z)
    local y = five_chambers.FLOOR_Y
    minetest.set_node({x=x, y=y+1, z=z}, {name="air"})
    minetest.set_node({x=x, y=y+2, z=z}, {name="air"})
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

-- Reinstates Door 2 bedrock after all agents have been teleported to Ch3.
function five_chambers.relock_door_2()
    local d2 = five_chambers.DOOR2_POS
    local y  = five_chambers.FLOOR_Y
    minetest.set_node({x=d2.x, y=y+1, z=d2.z}, {name=five_chambers.WALL_NODE})
    minetest.set_node({x=d2.x, y=y+2, z=d2.z}, {name=five_chambers.WALL_NODE})
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
