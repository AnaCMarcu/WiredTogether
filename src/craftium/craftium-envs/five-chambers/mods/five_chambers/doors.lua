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

function five_chambers.init_doors()
    five_chambers.door_state.door2_open = false
    five_chambers.door_state.door2_countdown = -1
    five_chambers.door_state.door3_open = false
    five_chambers.door_state.door4_open = false
    five_chambers.door_state.cell_doors = {}
    for i = 0, five_chambers.NUM_AGENTS - 1 do
        five_chambers.door_state.cell_doors[i] = false
    end
end

-- Replaces the 2-block door opening at (x, FLOOR_Y, z) and (x, FLOOR_Y+1, z)
-- with air. Idempotent.
function five_chambers.open_door_at(x, z)
    -- stub: D4/D5/D6 will call minetest.set_node for both blocks
end

-- Called when 6th anvil breaks; starts the 20-step countdown.
function five_chambers.start_door2_countdown()
    five_chambers.door_state.door2_countdown = five_chambers.DOOR2_DELAY
end

-- Called from globalstep to tick the door 2 countdown.
function five_chambers.tick_door2(dt)
    -- stub: D4
end

-- Reinstates Door 2 bedrock after Ch2→Ch3 teleport.
function five_chambers.relock_door_2()
    -- stub: D4
end

-- Opens a specific cell door (0-indexed). Called by switches.lua.
function five_chambers.open_cell_door(cell_i)
    -- stub: D5
end

-- Checks if all agents are in the communal room; if so, opens Door 3.
function five_chambers.check_door3()
    -- stub: D5
end

-- Opens Door 4 after all Ch4 mobs are dead.
function five_chambers.open_door4()
    -- stub: D6
end
