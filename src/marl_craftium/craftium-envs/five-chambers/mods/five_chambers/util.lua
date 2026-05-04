-- util.lua: shared helpers used across sub-modules.

-- Returns which chamber a position belongs to, or "unknown".
-- Keep in sync with the CHAMBER_BOUNDS dict in custom_environment_craftium.py.
function five_chambers.get_chamber_for_pos(pos)
    local x, z = pos.x, pos.z
    local N = five_chambers.NUM_AGENTS
    local ch3_x1 = 4 * N

    if x >= five_chambers.CH1.x0 and x <= five_chambers.CH1.x1
       and z >= five_chambers.CH1.z0 and z <= five_chambers.CH1.z1 then
        return "ch1"
    elseif x >= five_chambers.CH2.x0 and x <= five_chambers.CH2.x1
       and z >= five_chambers.CH2.z0 and z <= five_chambers.CH2.z1 then
        return "ch2"
    elseif x >= 0 and x <= ch3_x1
       and z >= five_chambers.CH3_Z0 and z <= five_chambers.CH3_NORTH_WALL_Z then
        if z >= five_chambers.CH3_COMMUNAL_Z0 then
            return "ch3_communal"
        else
            return "ch3_cell"
        end
    elseif x >= five_chambers.CH4.x0 and x <= five_chambers.CH4.x1
       and z >= five_chambers.CH4.z0 and z <= five_chambers.CH4.z1 then
        return "ch4"
    elseif x >= five_chambers.CH5.x0 and x <= five_chambers.CH5.x1
       and z >= five_chambers.CH5.z0 and z <= five_chambers.CH5.z1 then
        return "ch5"
    end
    return "unknown"
end

-- Returns the 0-based agent index from a player name ("agent_0" → 0).
-- Returns -1 if the name is not a recognised agent.
-- In DEBUG_SINGLE mode any connected player is treated as agent_0 so the
-- standalone-Luanti "singleplayer" name still triggers switch / milestone
-- / door-3 logic during a manual walkthrough.
function five_chambers.agent_index(name)
    if five_chambers.DEBUG_SINGLE then return 0 end
    local idx = tonumber(name:match("^agent_(%d+)$"))
    if idx and idx >= 0 and idx < five_chambers.NUM_AGENTS then
        return idx
    end
    return -1
end

-- Returns the X-center of cell i (0-based) in Chamber 3.
function five_chambers.cell_x_center(i)
    return i * 4 + 2
end

-- Returns the teleport target position for agent index i (into Cell i).
-- Drops the agent in the middle of its 3-deep cell (cell_z0..cell_z1).
function five_chambers.cell_teleport_pos(i)
    return {
        x = five_chambers.cell_x_center(i),
        y = five_chambers.FLOOR_Y + 1,
        z = math.floor((five_chambers.CH3_CELL_Z0 + five_chambers.CH3_CELL_Z1) / 2),
    }
end

-- Returns the Ch1 spawn position for agent index i.
-- Uses the plan-specified corner spawns for N=3; distributes linearly otherwise.
function five_chambers.ch1_spawn_pos(i)
    local N = five_chambers.NUM_AGENTS
    if N == 3 and five_chambers.CH1_SPAWNS_3 and five_chambers.CH1_SPAWNS_3[i] then
        return five_chambers.CH1_SPAWNS_3[i]
    end
    -- Generic: spread along Z=5, X:1-10
    local frac = (N == 1) and 0.5 or (i / (N - 1))
    return { x = math.floor(1 + frac * 9 + 0.5), y = 11, z = 5 }
end

-- Safe node-set: only replaces a block if it is currently air or the
-- target node (idempotent). Useful for leaf/drop placement that must
-- not overwrite bedrock containment walls.
function five_chambers.safe_set_node(pos, node_def)
    local existing = minetest.get_node(pos).name
    if existing == five_chambers.WALL_NODE then return end
    minetest.set_node(pos, node_def)
end
