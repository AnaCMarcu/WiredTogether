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

-- Returns the 0-based agent index from a player name ("agent_0" or "agent0" → 0).
-- Returns -1 if the name is not a recognised agent.
-- Underscore is optional because upstream craftium spawns clients with
-- ``client_name=f"agent{i}"`` (no underscore) while our own scripts and docs
-- often use ``agent_0``. Accepting both keeps Ch1-timeout teleport,
-- ch2_fallback_spawn_pos, Ch3 cell assignment, and milestone attribution
-- working regardless of which convention is in play.
-- In DEBUG_SINGLE mode any connected player is treated as agent_0 so the
-- standalone-Luanti "singleplayer" name still triggers switch / milestone
-- / door-3 logic during a manual walkthrough.
function five_chambers.agent_index(name)
    if five_chambers.DEBUG_SINGLE then return 0 end
    local idx = tonumber(name:match("^agent_?(%d+)$"))
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
-- Y is CH1_DIRT_Y + 1 — agents stand on the dirt layer, not the bedrock subfloor.
function five_chambers.ch1_spawn_pos(i)
    local N = five_chambers.NUM_AGENTS
    if N == 3 and five_chambers.CH1_SPAWNS_3 and five_chambers.CH1_SPAWNS_3[i] then
        return five_chambers.CH1_SPAWNS_3[i]
    end
    -- Generic (N ~= 3) spawn row, gated by TEAM_SCALING (config.lua /
    -- WT_TEAM_SCALING) so legacy suites are bit-for-bit unchanged.
    local frac = (N == 1) and 0.5 or (i / (N - 1))
    local y = (five_chambers.CH1_DIRT_Y or 11) + 1
    if five_chambers.TEAM_SCALING then
        -- Scaling suite: z=10, x in [2,10]. This row satisfies the same
        -- three criteria CH1_SPAWNS_3 was hand-tuned for:
        --   1. No solid tile underfoot. All trunks/stone pillars sit at
        --      z<=9, so z=10 is clear (the legacy z=5 row wedges agents
        --      inside the pillars at (3,5)/(7,5) for N=4,5,6,9).
        --   2. >=2 blocks from every bedrock wall. x starts at 2, not 1:
        --      a wall 1 block ahead fills the agent's first frame with
        --      grey bedrock, which the LLM reliably misreads as
        --      mcl_core:stone and Digs -- no break, no milestone. (An
        --      earlier z=12 row failed BOTH this and criterion 3: it put
        --      every agent 3 blocks off the north wall with no resource
        --      in reach, and the N=2 smoke run reported exactly that --
        --      "solid brick wall directly in front of me".)
        --   3. A breakable target within ~3 blocks. z=10 hugs the top of
        --      the resource band, so trees (7,9)/(2,8) and stones
        --      (5,8)/(8,7) are 1.0-3.2 away for every agent.
        -- x spans [2,10] rather than [3,10] so the N=9 spread stays
        -- collision-free (2..10 is exactly 9 distinct columns).
        return {
            x = math.floor(2 + frac * 8 + 0.5),
            y = y,
            z = 10,
        }
    end
    -- Legacy: the original Z=5 row, spread along X:1-10. Reproduces the
    -- N=2 pair-bonding and N=6 transplant runs exactly.
    return {
        x = math.floor(1 + frac * 9 + 0.5),
        y = y,
        z = 5,
    }
end

-- Returns the Ch2 fallback spawn position for agent index i.
-- Used by the Ch1 timeout teleport (see CH1_TIMEOUT_TICKS).
function five_chambers.ch2_fallback_spawn_pos(i)
    local N = five_chambers.NUM_AGENTS
    if N == 3 and five_chambers.CH2_FALLBACK_SPAWNS_3
       and five_chambers.CH2_FALLBACK_SPAWNS_3[i] then
        return five_chambers.CH2_FALLBACK_SPAWNS_3[i]
    end
    -- Generic: spread along z=CH2.z0+2, X across the chamber width.
    local x_min = five_chambers.CH2.x0 + 2
    local x_max = five_chambers.CH2.x1 - 2
    local frac  = (N == 1) and 0.5 or (i / (N - 1))
    return {
        x = math.floor(x_min + frac * (x_max - x_min) + 0.5),
        y = five_chambers.FLOOR_Y + 1,
        z = five_chambers.CH2.z0 + 2,
    }
end

-- Returns the Ch4 fallback spawn position for agent index i.
-- Used by the Ch3→Ch4 timeout teleport. Spread agents along the south
-- edge of Ch4 (z = CH4.z0 + 2) so they appear just inside the door.
function five_chambers.ch4_fallback_spawn_pos(i)
    local N = five_chambers.NUM_AGENTS
    local c = five_chambers.CH4
    local x_min = c.x0 + 2
    local x_max = c.x1 - 2
    local frac  = (N == 1) and 0.5 or (i / (N - 1))
    return {
        x = math.floor(x_min + frac * (x_max - x_min) + 0.5),
        y = five_chambers.FLOOR_Y + 1,
        z = c.z0 + 2,
    }
end

-- Returns the Ch5 fallback spawn position for agent index i.
-- Used by the Ch4→Ch5 timeout teleport. Spread agents along the south
-- edge of Ch5 (z = CH5.z0 + 2).
function five_chambers.ch5_fallback_spawn_pos(i)
    local N = five_chambers.NUM_AGENTS
    local c = five_chambers.CH5
    local x_min = c.x0 + 2
    local x_max = c.x1 - 2
    local frac  = (N == 1) and 0.5 or (i / (N - 1))
    return {
        x = math.floor(x_min + frac * (x_max - x_min) + 0.5),
        y = five_chambers.FLOOR_Y + 1,
        z = c.z0 + 2,
    }
end

-- Safe node-set: only replaces a block if it is currently air or the
-- target node (idempotent). Useful for leaf/drop placement that must
-- not overwrite bedrock containment walls.
function five_chambers.safe_set_node(pos, node_def)
    local existing = minetest.get_node(pos).name
    if existing == five_chambers.WALL_NODE then return end
    minetest.set_node(pos, node_def)
end
