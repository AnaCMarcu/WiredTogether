-- deaths.lua: player-death handling for Five Chambers.
--
-- A headless RL bot never clicks the engine's "You died" respawn button, so
-- without this a dead agent gets stuck behind the gray death formspec for the
-- rest of the episode. On death we:
--   (1) apply a -10 RL penalty, delivered through craftium.reward (the same
--       channel milestones use), so it lands in the agent's reward signal;
--   (2) branch on the chamber the agent died in:
--        * Ch5 (boss): PERMADEATH. The agent is NOT respawned; it stays down.
--          Ch5 is the last chamber, so once EVERY agent is dead we write an
--          episode_over flag the Python loop polls to end the episode.
--        * anywhere else (Ch4 combat practice, or a stray Ch1-3 death):
--          force-respawn the bot so the death screen clears and it keeps
--          playing; a Ch4 death is placed back inside Ch4 (not the Ch1 spawn).

five_chambers.player_dead = five_chambers.player_dead or {}
local DEATH_PENALTY = -10

-- Write the episode_over flag (polled by CraftiumEnvironmentInterface.episode_over)
-- once all connected agents are permanently dead. Cleared on episode reset.
local function _signal_episode_over()
    for _, p in ipairs(minetest.get_connected_players()) do
        if not five_chambers.player_dead[p:get_player_name()] then
            return  -- someone is still alive
        end
    end
    local path = minetest.get_worldpath() .. "/episode_over.txt"
    local f = io.open(path, "w")
    if f then f:write("all_dead\n"); f:close() end
    minetest.log("action", "[DEATH] all agents dead in Ch5 — episode_over signalled")
    if io and io.stderr then
        io.stderr:write("[DEATH] all agents dead in Ch5 — episode_over\n")
        io.stderr:flush()
    end
end

minetest.register_on_dieplayer(function(player, reason)
    if not (player and player.is_player and player:is_player()) then return end
    local name    = player:get_player_name()
    local idx     = five_chambers.agent_index(name)
    local pos     = player:get_pos()
    local chamber = pos and five_chambers.get_chamber_for_pos(pos) or nil

    -- (1) -10 death penalty into the RL reward signal.
    if craftium and craftium.reward then
        craftium.reward(player, DEATH_PENALTY)
    end
    minetest.log("action", string.format(
        "[DEATH] %s died in %s (penalty %d)", name, tostring(chamber), DEATH_PENALTY))
    if io and io.stderr then
        io.stderr:write(string.format(
            "[DEATH] %s died in %s penalty=%d\n", name, tostring(chamber), DEATH_PENALTY))
        io.stderr:flush()
    end

    if chamber == "ch5" then
        -- (2a) Boss room: permadeath. Leaving the agent down is what lets the
        -- team-wipe end the episode (Ch5 is the last chamber — nowhere to go).
        five_chambers.player_dead[name] = true
        _signal_episode_over()
        return
    end

    -- (2b) Forgiving chambers: force-respawn the bot (it never clicks respawn
    -- itself), deferred one tick so the engine has finished the death. A Ch4
    -- death is placed back inside Ch4 so combat practice continues there rather
    -- than dumping the agent at the distant Ch1 world spawn.
    local back_in_ch4 = (chamber == "ch4")
    minetest.after(0, function()
        local pl = minetest.get_player_by_name(name)
        if not pl then return end
        pl:respawn()
        if back_in_ch4 and idx >= 0 then
            -- Re-place + heal one more tick later, after respawn placement has
            -- run (mcl_spawn would otherwise drop them at the world spawn).
            minetest.after(0, function()
                local p2 = minetest.get_player_by_name(name)
                if not p2 then return end
                local dest = five_chambers.ch4_fallback_spawn_pos(idx)
                if dest then
                    p2:set_pos(dest)
                    p2:set_hp(20, {type = "set_hp", from = "mod"})
                end
            end)
        end
    end)
end)
