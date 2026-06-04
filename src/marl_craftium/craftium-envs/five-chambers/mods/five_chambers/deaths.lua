-- deaths.lua: player-death handling for Five Chambers.
--
-- A headless RL bot never clicks the engine's "You died" respawn button, and
-- the Craftium client FREEZES a player (stops sending movement) for as long as
-- it is dead — HP 0 — (see client.cpp isDead() gating). Relying on a post-death
-- server-side respawn to un-freeze the client proved fragile, so in the four
-- FORGIVING chambers we never let a player die at all:
--   * Ch1-Ch4: the killing blow is intercepted in register_on_player_hpchange
--     and clamped so HP stays >= 1 (a "virtual death"). The agent still takes
--     the -10 RL penalty through craftium.reward, then is healed + (in Ch4)
--     repositioned next tick. No engine death => no "You died" formspec, no
--     frozen dead-client, no stuck agent — the team keeps progressing to Ch5.
--   * Ch5 (boss): the lethal hit is LET THROUGH to a real PERMADEATH. The agent
--     stays down; once EVERY agent is permanently dead we write an episode_over
--     flag the Python loop polls to end the episode.
-- register_on_dieplayer below still runs: in Ch5 it records the permadeath, and
-- anywhere else it is a defensive fallback for a real death that somehow bypass-
-- ed the hpchange clamp (force-respawn so the agent is never left stuck dead).

five_chambers.player_dead = five_chambers.player_dead or {}
-- Names currently mid-"virtual death" (downed this tick, heal pending). Guards
-- against a multi-hit burst charging the -10 penalty more than once per downing.
five_chambers._dying = five_chambers._dying or {}
local DEATH_PENALTY = -10

-- Suppress the engine "You died / Respawn" formspec entirely. The builtin
-- (builtin/game/death_screen.lua) opens it via core.show_death_screen on every
-- death AND on join-with-0-HP, occluding the whole observation behind a gray
-- menu that a headless RL bot never dismisses. Both triggers look up
-- core.show_death_screen dynamically at call time, and mods load after builtin,
-- so replacing it with a no-op here neutralises both. Respawn/penalty handling
-- below is unaffected (it runs from register_on_dieplayer, not the formspec).
if minetest.show_death_screen then
    minetest.show_death_screen = function() end
end

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

-- Revive an agent one tick after a virtual death: heal to full and, in Ch4,
-- move it clear of the mob that downed it (its fallback spawn on the south
-- edge). Deferred so it runs after the damage event has fully resolved.
local function _revive_after_virtual_death(name, chamber, idx)
    local pl = minetest.get_player_by_name(name)
    if not pl then
        five_chambers._dying[name] = nil
        return
    end
    if chamber == "ch4" and idx >= 0 then
        local dest = five_chambers.ch4_fallback_spawn_pos(idx)
        if dest then pl:set_pos(dest) end
    end
    pl:set_hp(20, {type = "set_hp", from = "mod"})
    five_chambers._dying[name] = nil
end

-- Forgiving-chamber damage interception (the primary death-avoidance path).
-- Registered as a MODIFIER so its return value replaces hp_change before the
-- engine applies it. A would-be-lethal hit in Ch1-Ch4 is clamped to leave the
-- agent at 1 HP (no death), charged the -10 penalty once, and revived next tick.
-- Ch5 lethal hits are passed through untouched so the boss room keeps its real
-- permadeath. Healing (hp_change >= 0), including our own set_hp revive, is
-- ignored so it can never loop.
minetest.register_on_player_hpchange(function(player, hp_change, reason)
    if hp_change >= 0 then return hp_change end
    if not (player and player.is_player and player:is_player()) then return hp_change end
    local hp = player:get_hp()
    if hp + hp_change > 0 then return hp_change end  -- survivable hit, pass through

    local name    = player:get_player_name()
    local pos     = player:get_pos()
    local chamber = pos and five_chambers.get_chamber_for_pos(pos) or nil

    -- Boss room: let the lethal hit through to a real permadeath.
    if chamber == "ch5" then return hp_change end

    -- Forgiving chamber: convert into a non-fatal virtual death.
    if not five_chambers._dying[name] then
        five_chambers._dying[name] = true
        local idx = five_chambers.agent_index(name)
        if craftium and craftium.reward then
            craftium.reward(player, DEATH_PENALTY)
        end
        minetest.log("action", string.format(
            "[VDEATH] %s downed in %s (penalty %d) — clamped + revived",
            name, tostring(chamber), DEATH_PENALTY))
        if io and io.stderr then
            io.stderr:write(string.format(
                "[VDEATH] %s downed in %s penalty=%d\n", name, tostring(chamber), DEATH_PENALTY))
            io.stderr:flush()
        end
        minetest.after(0, function()
            _revive_after_virtual_death(name, chamber, idx)
        end)
    end

    -- Clamp this tick's damage so HP lands at 1 (alive). Armor/other modifiers
    -- only ever reduce damage further, so HP stays >= 1 regardless of ordering.
    return 1 - hp
end, true)  -- true => modifier callback (may change hp_change)

minetest.register_on_dieplayer(function(player, reason)
    if not (player and player.is_player and player:is_player()) then return end
    local name    = player:get_player_name()
    local idx     = five_chambers.agent_index(name)
    local pos     = player:get_pos()
    local chamber = pos and five_chambers.get_chamber_for_pos(pos) or nil

    -- (1) -10 death penalty into the RL reward signal. Skip it when a virtual
    -- death is already mid-flight for this name (the hpchange clamp charged it
    -- this same tick) so a real death that slipped past the clamp can't
    -- double-charge the penalty.
    if not five_chambers._dying[name] then
        if craftium and craftium.reward then
            craftium.reward(player, DEATH_PENALTY)
        end
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

    -- (2b) Forgiving chambers — DEFENSIVE FALLBACK ONLY. The hpchange clamp
    -- above should prevent any Ch1-Ch4 death, so this normally never runs; it
    -- exists so a real death that bypassed the clamp still force-respawns the
    -- bot (it never clicks respawn itself) rather than leaving it stuck dead.
    -- Deferred one tick so the engine has finished the death; a Ch4 death is
    -- placed back inside Ch4 rather than at the distant Ch1 world spawn.
    five_chambers._dying[name] = nil
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
