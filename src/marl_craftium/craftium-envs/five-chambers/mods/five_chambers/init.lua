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

    -- HUD: hide hotbar/crosshair/healthbar — keep what the RL obs needs.
    player:hud_set_flags({
        hotbar    = false,
        crosshair = false,
        healthbar = false,
        chat      = false,
    })
    player:set_nametag_attributes({color={a=0, r=0, g=0, b=0}})

    -- Milestone tracking: record initial position for M1 and inventory for M3.
    five_chambers.init_player_milestone_state(name)
    five_chambers.record_spawn_pos(name, spawn_pos)
    -- Inventory is empty on first join; prev_inv_total defaults to 0.
end)

-- ── Global step ───────────────────────────────────────────────────

minetest.register_globalstep(function(dtime)
    -- Keep world at midday (no day/night cycle).
    minetest.set_timeofday(0.5)
    -- Increment Lua-tick counter (runs at 20Hz; Python step = 3 Lua ticks).
    five_chambers.step_counter = five_chambers.step_counter + 1
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
    -- Re-place door_locked blocks at every door position. Must run BEFORE
    -- init_doors() so that init_doors's DEBUG_SINGLE branch (which opens
    -- Door 2) wins over the relock for that one door.
    five_chambers.relock_all_doors()
    five_chambers.init_doors()
    five_chambers.init_switches()
    five_chambers.init_anvils()
    five_chambers.reset_mob_state()
    five_chambers.clear_state_files()

    -- Teleport all connected agents back to Ch1 spawns.
    for _, p in ipairs(minetest.get_connected_players()) do
        local name = p:get_player_name()
        local i    = five_chambers.agent_index(name)
        local pos  = (i >= 0)
            and five_chambers.ch1_spawn_pos(i)
            or  {x=5, y=five_chambers.FLOOR_Y + 1, z=5}
        p:set_pos(pos)
        p:set_hp(20, {type="set_hp", from="mod"})
        five_chambers.record_spawn_pos(name, pos)
    end
end)
