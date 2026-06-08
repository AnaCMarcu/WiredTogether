-- deaths.lua: player-death handling for Five Chambers.
--
-- A headless RL bot never clicks the engine's "You died" respawn button, and
-- the Craftium client FREEZES a player (stops sending movement) while it is
-- dead — HP 0 (see client.cpp isDead() gating). Both the engine death and any
-- respawn/teleport that follows caused trouble (stuck dead-client, agents
-- launched into the air by mob-collision push-out after a reposition, etc.).
--
-- So in the FORGIVING chambers we make the agent INVINCIBLE and never move it:
--   * Ch1-Ch4: the agent takes NO real damage. register_on_player_hpchange
--     intercepts every incoming hit, drains a per-agent "virtual HP" pool by
--     what the hit WOULD have done, and returns 0 so real HP never changes.
--     When that pool would have hit 0 we RECORD a would-have-died (the -10 RL
--     penalty + a [WOULDDIE] log + a counter), then refill the pool. No engine
--     death, no respawn, no teleport, no heal — the agent just keeps playing
--     exactly where it is. Zombies become harmless; only their kill-milestones
--     and the recorded would-deaths matter.
--   * Ch5 (boss): damage is REAL and a lethal hit is a real PERMADEATH. The
--     agent stays down; once EVERY agent is permanently dead we write an
--     episode_over flag the Python loop polls to end the episode.

five_chambers.player_dead = five_chambers.player_dead or {}
-- Per-agent virtual HP pool for the forgiving chambers (Ch1-Ch4). Drained by
-- would-be damage; a would-have-died fires when it reaches 0, then it refills.
five_chambers._virtual_hp = five_chambers._virtual_hp or {}
-- Per-agent count of would-have-died events this episode (for metrics/logging).
five_chambers.would_die_count = five_chambers.would_die_count or {}
-- Per-agent count of would-have-died events that occurred specifically in Ch4,
-- so the M23 "all survived" bonus can require that NO agent had a near-death in
-- the combat chamber (see mobs.lua). Reset each episode in init.lua.
five_chambers.would_die_count_ch4 = five_chambers.would_die_count_ch4 or {}
-- Real (terminal) death in the lethal boss chamber (Ch5) — the only place a
-- death actually ends the agent's episode.
local DEATH_PENALTY = -50
-- Would-have-died near-miss in the FORGIVING chambers (Ch1-Ch4): the agent is
-- invincible and keeps playing, so this is a smaller graded penalty, not a
-- termination. Real death and a near-death are deliberately different
-- magnitudes (-50 vs -10).
local WOULD_DIE_PENALTY = -10
local MAX_HP = 20

-- Suppress the engine "You died / Respawn" formspec entirely. Even though the
-- forgiving chambers no longer produce deaths, Ch5 permadeaths still do, and a
-- headless bot never dismisses the gray menu. The builtin opens it via
-- core.show_death_screen (on death AND on join-with-0-HP); both look the
-- function up dynamically and mods load after builtin, so a no-op kills both.
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

-- Human-readable description of what dealt a hit, from the hpchange reason
-- (mob entity name for a punch, else the damage type: fall / drown / ...).
local function _damage_source(reason)
    if not reason then return "unknown" end
    if reason.object then
        local le = reason.object:get_luaentity()
        if le and le.name then return le.name end
        if reason.object:is_player and reason.object:is_player() then
            return "player:" .. reason.object:get_player_name()
        end
        return "object"
    end
    return reason.type or "unknown"
end

-- Forgiving-chamber invincibility + would-have-died recording.
-- Registered as a MODIFIER so its return replaces hp_change before the engine
-- applies it. Ch1-Ch4: drain the virtual-HP pool by the would-be damage and
-- return 0 (no real HP loss); on a lethal drain, record the would-death and
-- refill. Ch5: pass the hit through untouched (real damage / permadeath).
-- Healing (hp_change >= 0) always passes through.
minetest.register_on_player_hpchange(function(player, hp_change, reason)
    if hp_change >= 0 then return hp_change end  -- healing: leave alone
    if not (player and player.is_player and player:is_player()) then return hp_change end

    local name = player:get_player_name()
    if five_chambers.agent_index(name) < 0 then return hp_change end  -- non-agent

    local pos     = player:get_pos()
    local chamber = pos and five_chambers.get_chamber_for_pos(pos) or nil

    -- Boss room: real damage, real (perma)death — handled by on_dieplayer.
    if chamber == "ch5" then return hp_change end

    -- Forgiving chambers: invincible. Drain the virtual pool by what this hit
    -- would have dealt; record a would-have-died when it would have been lethal.
    local prev = five_chambers._virtual_hp[name] or MAX_HP
    local vhp  = prev + hp_change  -- hp_change < 0

    -- Debug: log every absorbed hit (damage, source, reason type, virtual HP).
    local hit_msg = string.format(
        "[HIT] %s in %s took %d (absorbed) from %s [%s] vHP %d->%d",
        name, tostring(chamber), -hp_change, _damage_source(reason),
        (reason and reason.type) or "?", prev, math.max(0, vhp))
    minetest.log("action", hit_msg)
    if io and io.stderr then
        io.stderr:write(hit_msg .. "\n")
        io.stderr:flush()
    end

    if vhp <= 0 then
        if craftium and craftium.reward then
            craftium.reward(player, WOULD_DIE_PENALTY)  -- record in the RL signal
        end
        local n = (five_chambers.would_die_count[name] or 0) + 1
        five_chambers.would_die_count[name] = n
        -- Track Ch4 near-deaths separately so M23 can reward a clean run.
        if chamber == "ch4" then
            five_chambers.would_die_count_ch4[name] =
                (five_chambers.would_die_count_ch4[name] or 0) + 1
        end
        minetest.log("action", string.format(
            "[WOULDDIE] %s would have died in %s (#%d, penalty %d) — no respawn",
            name, tostring(chamber), n, WOULD_DIE_PENALTY))
        if io and io.stderr then
            io.stderr:write(string.format(
                "[WOULDDIE] %s would have died in %s #%d penalty=%d\n",
                name, tostring(chamber), n, WOULD_DIE_PENALTY))
            io.stderr:flush()
        end
        vhp = MAX_HP  -- refill so the next accumulated lethal damage records again
    end
    five_chambers._virtual_hp[name] = vhp

    return 0  -- absorb all real damage: the agent never actually loses HP
end, true)  -- true => modifier callback (may change hp_change)

-- Real deaths only happen in Ch5 now (the forgiving chambers are invincible).
minetest.register_on_dieplayer(function(player, reason)
    if not (player and player.is_player and player:is_player()) then return end
    local name    = player:get_player_name()
    local pos     = player:get_pos()
    local chamber = pos and five_chambers.get_chamber_for_pos(pos) or nil

    -- -50 (terminal) death penalty into the RL reward signal.
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
        -- Boss room: permadeath. Leaving the agent down is what lets the
        -- team-wipe end the episode (Ch5 is the last chamber — nowhere to go).
        five_chambers.player_dead[name] = true
        _signal_episode_over()
        return
    end

    -- A non-Ch5 death should be impossible (forgiving chambers are invincible).
    -- If one ever happens, log it loudly rather than silently respawning — the
    -- design is "no respawn, no nothing", so investigate the source instead.
    minetest.log("warning", string.format(
        "[DEATH] UNEXPECTED non-Ch5 death for %s in %s — invincibility was bypassed",
        name, tostring(chamber)))
end)
