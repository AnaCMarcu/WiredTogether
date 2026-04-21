-- switches.lua: isolation cell switch nodes + rotational door wiring (plan §3).
-- Each agent is assigned cell i (agent_i → cell i).
-- Switch i opens the door of cell (i+1) mod NUM_AGENTS (rotational mapping).
-- Switch positions: x = cell_x_center(i), y = FLOOR_Y+1, z = CH3_CELL_Z0 (z=25)
--
-- On punch:
--   1. Replace target door bedrock blocks with air (doors.lua helper).
--   2. Emit switch event to switch_events.jsonl.
--   3. Fire M17 (switch_pressed) for the presser.
--   4. Fire M18 (door_opened) for the freed agent.
--
-- One-shot: switch_pressed[i] = true permanently after first press.

five_chambers.switch_pressed = {}

function five_chambers.init_switches()
    for i = 0, five_chambers.NUM_AGENTS - 1 do
        five_chambers.switch_pressed[i] = false
    end
end

-- Returns the cell index whose door is opened by switch i (rotational).
function five_chambers.switch_target_cell(i)
    return (i + 1) % five_chambers.NUM_AGENTS
end

-- Returns the switch index (0-based) for a node at pos, or -1 if not a switch.
local function switch_index_at(pos)
    local N = five_chambers.NUM_AGENTS
    if pos.z ~= five_chambers.CH3_CELL_Z0 then return -1 end
    for i = 0, N - 1 do
        if pos.x == five_chambers.cell_x_center(i) then return i end
    end
    return -1
end

minetest.register_node("five_chambers:switch", {
    description = "Cell Switch",
    tiles  = {"default_stone.png^[colorize:#3090ff:128"},
    groups = {unbreakable = 1},
    on_punch = function(pos, node, puncher, pointed_thing)
        if not puncher or not puncher:is_player() then return end
        local presser = puncher:get_player_name()
        local sw_i    = switch_index_at(pos)
        if sw_i < 0 then return end
        if five_chambers.switch_pressed[sw_i] then return end  -- one-shot

        five_chambers.switch_pressed[sw_i] = true

        local target = five_chambers.switch_target_cell(sw_i)

        -- 1. Open target cell door.
        five_chambers.open_cell_door(target)

        -- 2. Emit switch event for Python polling.
        local sw_label   = string.char(65 + sw_i)    -- "A", "B", "C"
        local door_label = string.char(65 + target)  -- "B", "C", "A"
        five_chambers.emit_switch_event(sw_label, door_label, presser)

        -- 3. M17: switch pressed (for the presser).
        five_chambers.fire_milestone("m17_switch_pressed", {presser})

        -- 4. M18: door opened (for the freed agent in the target cell).
        local freed_name   = "agent_" .. target
        local freed_player = minetest.get_player_by_name(freed_name)
        if freed_player then
            five_chambers.fire_milestone("m18_door_opened", {freed_name})
        end

        minetest.log("action",
            "[five_chambers] Switch " .. sw_label .. " pressed by " .. presser
            .. " — Door " .. door_label .. " opened.")
    end,
})
