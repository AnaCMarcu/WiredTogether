-- switches.lua: isolation cell switch nodes + rotational door wiring (plan §3).
-- Each agent is assigned cell i (agent_i → cell i).
-- Switch i opens the door of cell (i+1) mod NUM_AGENTS (rotational mapping).
-- Switch positions: x = cell_x_center(i), y = FLOOR_Y+1, z = CH3_CELL_Z0
--
-- On press:
--   1. Replace target door bedrock blocks with air (doors.lua helper).
--   2. Emit switch event to state_files/switch_events.jsonl.
--   3. Fire milestone M17 (switch pressed) for the presser.
--   4. Fire milestone M18 (door opened) for the agent freed.
--
-- One-shot: switch state is stored in five_chambers.switch_pressed[i].
-- Pressing again after the door is open has no effect.

five_chambers.switch_pressed = {}

function five_chambers.register_switch_node()
    -- stub: D5 will register five_chambers:switch with on_rightclick + on_punch
end

function five_chambers.init_switches()
    for i = 0, five_chambers.NUM_AGENTS - 1 do
        five_chambers.switch_pressed[i] = false
    end
end

-- Returns the cell index whose door is opened by switch i (rotational).
function five_chambers.switch_target_cell(i)
    return (i + 1) % five_chambers.NUM_AGENTS
end
