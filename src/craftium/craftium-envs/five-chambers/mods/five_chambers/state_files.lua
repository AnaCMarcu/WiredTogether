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

    -- Diagnostic line picked up by Python log tailer once [MILESTONE] is in _LOG_TAGS.
    local contrib_str = table.concat(contributors, ",")
    io.stderr:write("[MILESTONE] " .. milestone_id
        .. " contributors=" .. contrib_str
        .. " reward=" .. tostring(reward)
        .. " step=" .. tostring(five_chambers.step_counter or 0) .. "\n")
    io.stderr:flush()
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
end
