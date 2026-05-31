-- state_files.lua: Lua→Python state file IPC (plan §4.6).
-- All files are written to {world_path}/ and polled by Python each step.
-- Every function builds JSON manually — no LuaJIT JSON dependency.

-- Appends one milestone event line to milestone_events.jsonl.
-- Called by fire_milestone() in milestones.lua whenever a milestone fires.
-- Python polls this file via CraftiumEnvironmentInterface.poll_milestone_events().
function five_chambers.emit_milestone(milestone_id, contributors, reward)
    local world_path = minetest.get_worldpath()
    local path = world_path .. "/milestone_events.jsonl"

    -- Build contributors JSON array (e.g. ["agent_0","agent_1"])
    local parts = {}
    for _, name in ipairs(contributors) do
        table.insert(parts, '"' .. name .. '"')
    end
    local contrib_json = "[" .. table.concat(parts, ",") .. "]"

    local json_line = string.format(
        '{"step":%d,"milestone":"%s","contributors":%s,"reward":%d}\n',
        five_chambers.step_counter or 0,
        milestone_id,
        contrib_json,
        reward
    )

    local f = io.open(path, "a")
    if f then
        f:write(json_line)
        f:close()
    else
        minetest.log("error", "[five_chambers] emit_milestone: cannot open " .. path)
        return
    end

    -- Issue craftium reward to each contributor so the RL signal reaches Python.
    for _, name in ipairs(contributors) do
        local player = minetest.get_player_by_name(name)
        if player then
            if craftium and craftium.reward then
                craftium.reward(player, reward)
            end
        end
    end

    -- Diagnostic line picked up by Python's log tailer (looks for "[MILESTONE]"
    -- in stderr.txt). Craftium disables Luanti's mod-security sandbox, so
    -- io.stderr is available there. Vanilla Luanti enables the sandbox and
    -- io.stderr is nil — guard the write and also log via minetest.log so
    -- debug.txt has the line in both environments.
    local contrib_str = table.concat(contributors, ",")
    local line = "[MILESTONE] " .. milestone_id
        .. " contributors=" .. contrib_str
        .. " reward=" .. tostring(reward)
        .. " step=" .. tostring(five_chambers.step_counter or 0)
    minetest.log("action", line)
    if io and io.stderr then
        io.stderr:write(line .. "\n")
        io.stderr:flush()
    end
end

-- Appends one switch event line to switch_events.jsonl (D5 stub).
function five_chambers.emit_switch_event(switch_id, door_opened, presser_name)
    local world_path = minetest.get_worldpath()
    local path = world_path .. "/switch_events.jsonl"
    local json_line = string.format(
        '{"step":%d,"switch":"%s","door_opened":"%s","presser":"%s"}\n',
        five_chambers.step_counter or 0,
        switch_id, door_opened, presser_name
    )
    local f = io.open(path, "a")
    if f then f:write(json_line); f:close() end
end

-- Deletes all state files at episode start so Python sees a clean slate.
-- Called from the reset handler in init.lua.
function five_chambers.clear_state_files()
    local world_path = minetest.get_worldpath()
    os.remove(world_path .. "/milestone_events.jsonl")
    os.remove(world_path .. "/switch_events.jsonl")
    os.remove(world_path .. "/episode_done.txt")
    -- Door 1 unlock state — written by doors.lua's open_door1() when the
    -- door is unlocked via an m2..m7 milestone (or the timeout fallback).
    -- Must be cleared at episode reset because the door re-locks via
    -- relock_all_doors() and agents shouldn't see a stale "open" flag.
    os.remove(world_path .. "/door1_state.txt")
    -- Doors 2-4 + per-cell doors — same lifecycle as door1_state.txt.
    -- All are written by the corresponding open_doorN() / open_cell_door()
    -- calls in doors.lua and re-locked by relock_all_doors() at episode
    -- start. Clearing the state files here matches that lifecycle so
    -- agents don't see stale "open" flags from the previous episode.
    os.remove(world_path .. "/door2_state.txt")
    os.remove(world_path .. "/door3_state.txt")
    os.remove(world_path .. "/door4_state.txt")
    os.remove(world_path .. "/cell_doors_state.txt")
    -- Anvil coop-detected diagnostic JSONL (anvil.lua globalstep). No
    -- reward attached — purely for post-hoc analysis of "did the team
    -- ever try to coordinate?". Cleared per-episode so each episode's
    -- file contains only that episode's events.
    os.remove(world_path .. "/anvil_coop_events.jsonl")
end
