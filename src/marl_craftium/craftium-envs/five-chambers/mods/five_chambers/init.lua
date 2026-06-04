-- init.lua: module loader and Craftium entry hooks for Five Chambers.

-- Fast-fail API check: report missing minetest functions before any dofile.
local _top_level_apis = {
    "swap_node", "load_area", "get_node", "register_node", "register_entity",
    "register_on_joinplayer", "register_on_mods_loaded", "register_globalstep",
    "mod_channel_join", "get_modpath", "get_worldpath", "after",
}
for _, _n in ipairs(_top_level_apis) do
    if not minetest[_n] then
        error("[five_chambers] init.lua: minetest." .. _n .. " is nil — " ..
              "Luanti build is missing required API.")
    end
end

local modpath = minetest.get_modpath("five_chambers")
five_chambers = {}
five_chambers.step_counter = 0

dofile(modpath .. "/config.lua")
dofile(modpath .. "/util.lua")
dofile(modpath .. "/state_files.lua")
dofile(modpath .. "/world_gen.lua")
dofile(modpath .. "/milestones.lua")
dofile(modpath .. "/anvil.lua")
dofile(modpath .. "/switches.lua")
dofile(modpath .. "/doors.lua")
dofile(modpath .. "/gear.lua")
dofile(modpath .. "/mobs.lua")
dofile(modpath .. "/player_state.lua")
dofile(modpath .. "/deaths.lua")

-- ── Server-start initialisation ───────────────────────────────────
-- Runs after all mods (including mobs_mc) have loaded, so entity
-- definitions exist and the world database is ready for set_node calls.

minetest.register_on_mods_loaded(function()
    -- Patch mob entities for kill-tracking BEFORE any entities spawn.
    -- Safe to do here because it only modifies entity definition tables.
    five_chambers.patch_mobs_for_kill_tracking()

    -- Defer everything that touches the map (world_gen, doors, anvils,
    -- mob spawns) by one tick. During on_mods_loaded the map subsystem
    -- is not yet fully initialised — minetest.get_voxel_manip() returns
    -- nil and the mcl_observers monkey-patch on set_node/swap_node calls
    -- get_node() on unloaded chunks, which crashes.
    minetest.after(0, function()
        five_chambers.build_all_chambers()
        five_chambers.init_doors()
        five_chambers.init_switches()
        five_chambers.init_anvils()
        five_chambers.reset_mob_state()
        five_chambers.spawn_ch1_animals()
        five_chambers.clear_state_files()
        minetest.log("action", "[five_chambers] Server-start init complete.")
    end)
end)

-- ── HUD hiding (shared by join / respawn / global-step safety net) ──
-- Two DISTINCT HUD systems occlude the agent's observation and must both be
-- hidden:
--   1. Engine HUD flags (hotbar, crosshair, healthbar, chat) — hud_set_flags.
--      These only reset on respawn.
--   2. VoxeLibre "hudbars" (health/breath/armor/hunger/...) — separate HUD
--      elements NOT controlled by hud_set_flags. hudbars' own global step
--      AUTO-SHOWS the health bar whenever HP < max (i.e. the instant an agent
--      takes its first hit in Ch4) and keeps re-showing it every tick, which is
--      why the gray bar appears on Ch4 ENTRY (not death) and lingers into ep2.
-- hb.hide_hudbar is idempotent (early-returns when already hidden) and
-- pcall-guarded here, so re-hiding every tick is cheap and crash-safe even
-- before a bar is initialised for the player.
local _HUDBAR_IDS = {
    "health", "breath", "armor", "hunger",
    "saturation", "exhaustion", "absorption",
}
local function hide_player_hudbars(player)
    if not (player and hb and hb.hide_hudbar) then return end
    for _, id in ipairs(_HUDBAR_IDS) do
        pcall(hb.hide_hudbar, player, id)
    end
end

local function hide_player_hud(player)
    if not player then return end
    player:hud_set_flags({
        hotbar    = false,
        crosshair = false,
        healthbar = false,
        chat      = false,
    })
    player:set_nametag_attributes({color = {a = 0, r = 0, g = 0, b = 0}})
    hide_player_hudbars(player)
end

-- ── Player join ───────────────────────────────────────────────────

minetest.register_on_joinplayer(function(player)
    local name = player:get_player_name()
    local idx  = five_chambers.agent_index(name)

    -- Teleport to designated Ch1 spawn corner.
    local spawn_pos
    if idx >= 0 then
        spawn_pos = five_chambers.ch1_spawn_pos(idx)
    else
        spawn_pos = {x=5, y=five_chambers.FLOOR_Y + 1, z=5}
    end
    player:set_pos(spawn_pos)
    player:set_hp(20, {type="set_hp", from="mod"})

    -- HUD: hide hotbar/crosshair/healthbar/nametag — keep what the RL obs needs.
    hide_player_hud(player)

    -- Milestone tracking: record initial position for M1 and inventory for M3.
    five_chambers.init_player_milestone_state(name)
    five_chambers.record_spawn_pos(name, spawn_pos)
    -- Inventory is empty on first join; prev_inv_total defaults to 0.
end)

-- ── Player respawn ────────────────────────────────────────────────
-- Re-apply the HUD-hide on respawn. Minetest RESETS hud flags to their
-- defaults on death/respawn, so without this the hotbar/crosshair/healthbar
-- reappear after the FIRST death (Ch4 zombies / Ch5 boss) and stay visible —
-- a gray HUD panel occluding the bottom of the agent's observation for the
-- rest of the episode. register_on_joinplayer only covers the initial spawn,
-- which is why the box first shows up exactly when an agent enters Ch4 (the
-- first combat chamber) and dies.
minetest.register_on_respawnplayer(function(player)
    -- Apply immediately AND again on the next server step. Minetest/VoxeLibre can
    -- apply its default HUD state slightly AFTER on_respawnplayer callbacks run,
    -- which would override a synchronous-only re-hide and leave the gray
    -- hotbar/health panel occluding the view for the rest of the episode. The
    -- deferred minetest.after(0) re-apply lands after the engine's reset, so the
    -- hide sticks regardless of ordering.
    hide_player_hud(player)
    local pname = player:get_player_name()
    minetest.after(0, function()
        hide_player_hud(minetest.get_player_by_name(pname))
    end)
    return false  -- let the default respawn placement proceed
end)

-- ── Global step ───────────────────────────────────────────────────

minetest.register_globalstep(function(dtime)
    -- Keep world at midday (no day/night cycle).
    minetest.set_timeofday(0.5)
    -- Increment Lua-tick counter (runs at 20Hz; Python step = 3 Lua ticks).
    five_chambers.step_counter = five_chambers.step_counter + 1

    local _players = minetest.get_connected_players()

    -- EVERY tick: re-hide the VoxeLibre hudbars. hudbars' own global step
    -- re-shows the health bar whenever HP < max — every combat hit in Ch4/Ch5 —
    -- so this must run as often as theirs to keep it gone, and it must keep
    -- running across episodes (a bar shown after ep1 combat otherwise lingers
    -- into ep2). The hb.hide_hudbar calls are idempotent no-ops once hidden, so
    -- this is cheap; real work happens only on the tick a bar was re-shown.
    for _, p in ipairs(_players) do
        hide_player_hudbars(p)
    end

    -- Once per second: re-apply the engine HUD flags (hotbar/crosshair/nametag).
    -- These only reset on respawn, and hud_set_flags sends a packet on every
    -- call, so it is throttled rather than run every tick. (hide_player_hud also
    -- re-hides the hudbars, harmlessly.)
    if five_chambers.step_counter % 20 == 0 then
        for _, p in ipairs(_players) do
            hide_player_hud(p)
        end
    end
end)

-- ── Episode reset (Craftium channel) ─────────────────────────────

local channel = minetest.mod_channel_join("craftium_channel")

minetest.register_on_modchannel_message(function(ch, sender, raw)
    if ch ~= "craftium_channel" then return end
    local msg = minetest.deserialize(raw)
    if not (msg and msg.agent == "server" and msg.reset == true) then return end

    -- Clear all per-episode state.
    five_chambers.step_counter = 0
    five_chambers.reset_milestone_state()
    -- Clear death bookkeeping + the episode_over flag (see deaths.lua) so the
    -- new episode starts with every agent alive and the loop doesn't see a
    -- stale all-dead signal.
    five_chambers.player_dead = {}
    os.remove(minetest.get_worldpath() .. "/episode_over.txt")
    -- Rebuild the world geometry: re-places trees, stone blocks, ceiling
    -- glowstones, and the anvils (which got DESTROYED on break in the
    -- previous episode). VoxelManip writes directly so re-placing nodes
    -- where they originally were is safe and idempotent. Without this,
    -- ep 2+ would have an empty Ch1 (trees/stones dug) and an anvil-less
    -- Ch2, blocking progression to Ch3.
    five_chambers.build_all_chambers()
    -- Re-place door_locked blocks at every door position. Must run BEFORE
    -- init_doors() so that init_doors's DEBUG_SINGLE branch (which opens
    -- Door 2) wins over the relock for that one door.
    five_chambers.relock_all_doors()
    five_chambers.init_doors()
    five_chambers.init_switches()
    five_chambers.init_anvils()
    five_chambers.reset_mob_state()
    -- Respawn the Ch1 animals (chickens + sheep) too — the previous
    -- episode may have killed them, and reset_mob_state only clears
    -- the bookkeeping table, not the entities.
    five_chambers.spawn_ch1_animals()
    five_chambers.clear_state_files()

    -- Teleport all connected agents back to Ch1 spawns AND fully wipe
    -- per-player state so each episode starts with identical conditions.
    -- (Reproducibility: variance across episodes should come from the
    --  policy, not from ep N's leftover inventory / hunger drift.)
    for _, p in ipairs(minetest.get_connected_players()) do
        local name = p:get_player_name()
        local i    = five_chambers.agent_index(name)
        local pos  = (i >= 0)
            and five_chambers.ch1_spawn_pos(i)
            or  {x=5, y=five_chambers.FLOOR_Y + 1, z=5}

        -- Robust teleport-back-to-Ch1. Observed failure mode in exp15:
        -- chamber_entry_steps['ch2'] = 0 for ep2 AND ep3 → none of the
        -- 3 agents were teleported back to Ch1 between episodes; they
        -- stayed at their ep-end Ch2 positions. With agents stuck in
        -- Ch2, no Ch1 milestone (M1..M7, M_door1_open) can fire — the
        -- ep2/ep3 milestone log is empty except for the spurious step-1
        -- M1 fires that come from the distance check finding the agent
        -- 13+ blocks from the recorded spawn_pos. So set_pos has been
        -- silently failing entirely, not just timing-asymmetric.
        --
        -- Suspected cause: the Ch1 spawn chunk isn't fully loaded at
        -- the moment of set_pos. Fix is belt-and-braces:
        --   1. load_area over a generous radius around the target.
        --   2. call set_pos once.
        --   3. log pre/post positions so debug.txt shows whether the
        --      teleport actually took effect this episode.
        --   4. schedule a verification on the next tick; if the player
        --      drifted >2 blocks from the target, re-set_pos. Handles
        --      both the chunk-load race AND any client-side prediction
        --      that fights the server position.
        local pre = p:get_pos()
        minetest.load_area(
            {x = pos.x - 5, y = pos.y - 5, z = pos.z - 5},
            {x = pos.x + 5, y = pos.y + 5, z = pos.z + 5}
        )
        p:set_pos(pos)
        local post = p:get_pos()
        local pre_s  = pre  and string.format("(%.1f,%.1f,%.1f)", pre.x,  pre.y,  pre.z)  or "nil"
        local post_s = post and string.format("(%.1f,%.1f,%.1f)", post.x, post.y, post.z) or "nil"
        minetest.log("action", string.format(
            "[five_chambers] reset-teleport %s: pre=%s -> target=(%d,%d,%d) post=%s",
            name, pre_s, pos.x, pos.y, pos.z, post_s))
        if io and io.stderr then
            io.stderr:write(string.format(
                "[RESET_TP] %s pre=%s target=(%d,%d,%d) post=%s\n",
                name, pre_s, pos.x, pos.y, pos.z, post_s))
            io.stderr:flush()
        end
        -- Schedule a verification + retry on the next tick. minetest.after
        -- captures by closure so the target pos and player name survive
        -- across the tick boundary.
        local _target = pos
        local _name   = name
        minetest.after(0.5, function()
            local pl = minetest.get_player_by_name(_name)
            if not pl then return end
            local cur = pl:get_pos()
            if not cur then return end
            local dx = cur.x - _target.x
            local dz = cur.z - _target.z
            if math.sqrt(dx*dx + dz*dz) > 2.0 then
                -- Drifted: re-teleport.
                pl:set_pos(_target)
                if io and io.stderr then
                    io.stderr:write(string.format(
                        "[RESET_TP] %s RETRY (was at (%.1f,%.1f,%.1f))\n",
                        _name, cur.x, cur.y, cur.z))
                    io.stderr:flush()
                end
            end
        end)
        p:set_hp(20, {type="set_hp", from="mod"})

        -- Wipe inventory: main hotbar, armor slots, craft grid, craft preview.
        local inv = p:get_inventory()
        if inv then
            for _, list_name in ipairs({"main", "craft", "craftpreview", "armor"}) do
                if inv:get_size(list_name) > 0 then
                    inv:set_list(list_name, {})
                end
            end
        end
        -- Reset wielded slot to slot 1 (in case the previous episode left
        -- the agent holding gear that no longer exists).
        if p.set_wield_index then
            pcall(p.set_wield_index, p, 1)
        end
        -- Reset breath to max (VoxeLibre water/oxygen).
        local props = p:get_properties()
        local breath_max = (props and props.breath_max) or 10
        pcall(p.set_breath, p, breath_max)
        -- Reset hunger + saturation (VoxeLibre's mcl_hunger).
        if mcl_hunger then
            if mcl_hunger.set_hunger    then pcall(mcl_hunger.set_hunger,    p, 20) end
            if mcl_hunger.set_saturation then pcall(mcl_hunger.set_saturation, p, 5) end
            if mcl_hunger.set_exhaustion then pcall(mcl_hunger.set_exhaustion, p, 0) end
        end

        -- Re-record clean baselines for milestone tracking. Order matters:
        -- prev_inv_total was set in reset_milestone_state() based on the
        -- pre-wipe inventory; now that we've cleared the inventory it
        -- must be 0 so M3 (pick up 3 items) counts only NEW pickups.
        five_chambers.record_spawn_pos(name, pos)
        five_chambers.prev_inv_total[name] = 0
    end
end)
