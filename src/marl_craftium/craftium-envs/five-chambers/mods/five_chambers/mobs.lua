-- mobs.lua: mob spawning and kill-attribution tracking.
-- Ch1: 5 chickens + 3 sheep spawned once at server start.
-- Ch4/Ch5: stubs (D6/D7).
--
-- Kill attribution (M5/M6):
--   We patch mobs_mc:chicken and mobs_mc:sheep entity definitions so each
--   on_punch call records the last puncher in self._fc_last_puncher.
--   A globalstep then checks which tracked mob ObjectRefs are no longer
--   valid; when one disappears, the stored last_puncher gets credit.

-- ── Entity-definition patching ───────────────────────────────────

-- Patches one mob entity so on_punch records both the last puncher and all
-- contributors who dealt damage. Must be called from on_mods_loaded.
local function patch_entity_for_kill_tracking(entity_name)
    local def = minetest.registered_entities[entity_name]
    if not def then
        minetest.log("warning",
            "[five_chambers] patch_entity: not found: " .. entity_name)
        return false
    end

    local orig = def.on_punch
    def.on_punch = function(self, puncher, tflp, tool_caps, dir, damage)
        if puncher and puncher:is_player() then
            local pname = puncher:get_player_name()
            self._fc_last_puncher = pname
            if not self._fc_contributors then self._fc_contributors = {} end
            self._fc_contributors[pname] = true
        end
        if orig then return orig(self, puncher, tflp, tool_caps, dir, damage) end
    end
    return true
end

function five_chambers.patch_mobs_for_kill_tracking()
    patch_entity_for_kill_tracking("mobs_mc:chicken")
    patch_entity_for_kill_tracking("mobs_mc:sheep")
    patch_entity_for_kill_tracking("mobs_mc:zombie")  -- Ch4 combat mobs
end

-- ── Spawn Ch1 animals ────────────────────────────────────────────

-- Initialise mob_state table immediately (before on_mods_loaded / globalstep runs).
five_chambers.mob_state = {
    active_ch1_mobs  = {},
    ch4_mobs         = {},
    ch4_triggered    = false,
    ch4_contributors = {},   -- {[agent_name]=true} union across all Ch4 mobs
    ch5_boss         = nil,
    ch5_triggered    = false,
    ch4_kills        = {},
    boss_damage      = {},
}

-- Ch4 zombie spawn positions (inside 11×11 interior x=2..10, z=53..61).
local CH4_SPAWN_POSITIONS = {
    {x=4, y=11, z=54},
    {x=6, y=11, z=57},
    {x=8, y=11, z=60},
}

-- active_ch1_mobs: list of {obj=ObjectRef, last_puncher=name|nil}
-- Populated by spawn_ch1_animals(); checked each globalstep for deaths.
-- Do NOT clear the list here — reset_mob_state() does that, and is called
-- before spawn_ch1_animals() in on_mods_loaded.

function five_chambers.spawn_ch1_animals()

    local y = five_chambers.FLOOR_Y + 1  -- stand on the grass floor

    local function try_spawn(entity_name, pos)
        -- Ensure the spawn position is air before adding entity.
        local node = minetest.get_node({x=pos.x, y=pos.y, z=pos.z})
        if node.name ~= "air" then
            -- Nudge one step in Z if occupied (trees/stone at same tile).
            pos = {x=pos.x, y=pos.y, z=pos.z + 1}
        end
        local obj = minetest.add_entity(pos, entity_name)
        if obj then
            -- Pin the entity so VoxeLibre's mobs_mc despawn logic
            -- (passive-mob culling, biome/static checks) doesn't remove it.
            local lua_ent = obj:get_luaentity()
            if lua_ent then
                lua_ent.static_save = true
                lua_ent.persistent = true
                lua_ent.despawn_immediately = false
            end
            table.insert(five_chambers.mob_state.active_ch1_mobs,
                         {obj=obj, last_puncher=nil})
        else
            minetest.log("warning",
                "[five_chambers] spawn failed for " .. entity_name
                .. " at " .. minetest.pos_to_string(pos))
        end
    end

    for _, cp in ipairs(five_chambers.CH1_CHICKEN_POSITIONS) do
        try_spawn("mobs_mc:chicken", {x=cp.x, y=y, z=cp.z})
    end
    for _, sp in ipairs(five_chambers.CH1_SHEEP_POSITIONS) do
        try_spawn("mobs_mc:sheep", {x=sp.x, y=y, z=sp.z})
    end

    minetest.log("action",
        "[five_chambers] Spawned " ..
        #five_chambers.mob_state.active_ch1_mobs .. " Ch1 animals.")
end

-- ── Globalstep: kill detection ───────────────────────────────────

minetest.register_globalstep(function(dtime)
    local active = five_chambers.mob_state.active_ch1_mobs
    if not active or #active == 0 then return end

    local still_alive = {}
    for _, entry in ipairs(active) do
        if entry.obj:is_valid() then
            -- Mob is still in the world; sync last_puncher from entity.
            local ent = entry.obj:get_luaentity()
            if ent then
                if ent._fc_last_puncher then
                    entry.last_puncher = ent._fc_last_puncher
                    ent._fc_last_puncher = nil
                end
                -- Mob alive: keep tracking it.
                -- (VoxeLibre mobs store health in self.health or self.hp)
                local hp = ent.health or ent.hp
                if hp and hp <= 0 then
                    -- Dead but obj still momentarily valid.
                    if entry.last_puncher then
                        five_chambers.record_animal_kill(entry.last_puncher)
                    end
                    -- Don't add back to still_alive.
                else
                    table.insert(still_alive, entry)
                end
            else
                -- luaentity gone; treat as death.
                if entry.last_puncher then
                    five_chambers.record_animal_kill(entry.last_puncher)
                end
            end
        else
            -- ObjectRef became invalid → mob was removed (killed or despawned).
            if entry.last_puncher then
                five_chambers.record_animal_kill(entry.last_puncher)
            end
        end
    end
    five_chambers.mob_state.active_ch1_mobs = still_alive
end)

-- ── Public reset ─────────────────────────────────────────────────

function five_chambers.reset_mob_state()
    five_chambers.mob_state = {
        active_ch1_mobs  = {},
        ch4_mobs         = {},
        ch4_triggered    = false,
        ch4_contributors = {},
        ch5_boss         = nil,
        ch5_triggered    = false,
        ch4_kills        = {},
        boss_damage      = {},
    }
end

-- ── Ch4 zombie spawn ─────────────────────────────────────────────

function five_chambers.spawn_ch4_mobs()
    for _, pos in ipairs(CH4_SPAWN_POSITIONS) do
        local obj = minetest.add_entity(pos, "mobs_mc:zombie")
        if obj then
            table.insert(five_chambers.mob_state.ch4_mobs, {
                obj          = obj,
                last_puncher = nil,
                contributors = {},  -- {[name]=true}
            })
        else
            minetest.log("warning",
                "[five_chambers] Ch4 zombie spawn failed at "
                .. minetest.pos_to_string(pos))
        end
    end
    minetest.log("action",
        "[five_chambers] Spawned "
        .. #five_chambers.mob_state.ch4_mobs .. " Ch4 zombies.")
end

-- ── Globalstep: Ch4 entry detection (M20 + spawn trigger) ────────

minetest.register_globalstep(function(dtime)
    if not five_chambers.CHAMBERS[4].enabled then return end
    if five_chambers.step_counter % 3 ~= 0 then return end

    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        if five_chambers.agent_index(name) >= 0 then
            local pos = player:get_pos()
            if pos and five_chambers.get_chamber_for_pos(pos) == "ch4" then
                five_chambers.fire_milestone("m20_enter_ch4", {name})
                if not five_chambers.mob_state.ch4_triggered then
                    five_chambers.mob_state.ch4_triggered = true
                    five_chambers.spawn_ch4_mobs()
                end
            end
        end
    end
end)

-- ── Globalstep: Ch4 kill detection (M21, M22, M23) ───────────────

minetest.register_globalstep(function(dtime)
    local mobs = five_chambers.mob_state.ch4_mobs
    if not five_chambers.mob_state.ch4_triggered then return end
    if not mobs or #mobs == 0 then return end

    local still_alive = {}

    for _, entry in ipairs(mobs) do
        local dead = false

        if not entry.obj:is_valid() then
            dead = true
        else
            local ent = entry.obj:get_luaentity()
            if not ent then
                dead = true
            else
                -- Sync punch attribution from entity fields.
                if ent._fc_last_puncher then
                    entry.last_puncher = ent._fc_last_puncher
                    ent._fc_last_puncher = nil
                end
                if ent._fc_contributors then
                    for pname in pairs(ent._fc_contributors) do
                        entry.contributors[pname] = true
                    end
                    ent._fc_contributors = {}
                end
                local hp = ent.health or ent.hp
                if hp and hp <= 0 then dead = true end
            end
        end

        if dead then
            -- M21: first Ch4 kill per agent (once=true handles dedup).
            if entry.last_puncher then
                local killer = entry.last_puncher
                five_chambers.mob_state.ch4_kills[killer] =
                    (five_chambers.mob_state.ch4_kills[killer] or 0) + 1
                five_chambers.fire_milestone("m21_first_mob_kill", {killer})
            end
            -- Accumulate contributors across all Ch4 mobs for M22.
            for pname in pairs(entry.contributors) do
                five_chambers.mob_state.ch4_contributors[pname] = true
            end
        else
            table.insert(still_alive, entry)
        end
    end

    five_chambers.mob_state.ch4_mobs = still_alive

    -- All Ch4 mobs cleared: fire M22, M23, open Door 4.
    if #still_alive == 0 then
        local contrib_list = {}
        for pname in pairs(five_chambers.mob_state.ch4_contributors) do
            table.insert(contrib_list, pname)
        end
        if #contrib_list > 0 then
            five_chambers.fire_milestone("m22_all_mobs_killed", contrib_list)

            -- M23: bonus if all agents are alive.
            local alive_list = {}
            for _, player in ipairs(minetest.get_connected_players()) do
                local name = player:get_player_name()
                if five_chambers.agent_index(name) >= 0
                   and player:get_hp() > 0 then
                    table.insert(alive_list, name)
                end
            end
            if #alive_list >= five_chambers.NUM_AGENTS then
                five_chambers.fire_milestone("m23_all_alive_ch4", alive_list)
            end

            five_chambers.open_door4()
        end
    end
end)

-- ── Ch5 boss spawn ───────────────────────────────────────────────

-- Fires M27 (boss defeated) and M28 (all alive bonus), then signals
-- episode termination via episode_done.txt and craftium.terminate().
local function fire_boss_death()
    local boss = five_chambers.mob_state.ch5_boss
    if not boss then return end

    local contrib_list = {}
    for pname in pairs(boss.contributors) do
        table.insert(contrib_list, pname)
    end

    if #contrib_list > 0 then
        five_chambers.fire_milestone("m27_boss_defeated", contrib_list)
    end

    -- M28: bonus if every agent is still alive.
    local alive_list = {}
    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        if five_chambers.agent_index(name) >= 0 and player:get_hp() > 0 then
            table.insert(alive_list, name)
        end
    end
    if #alive_list >= five_chambers.NUM_AGENTS then
        five_chambers.fire_milestone("m28_all_alive_bonus", alive_list)
    end

    -- Signal episode termination.
    local world_path = minetest.get_worldpath()
    local f = io.open(world_path .. "/episode_done.txt", "w")
    if f then
        f:write(tostring(five_chambers.step_counter))
        f:close()
    end
    if craftium and craftium.terminate then craftium.terminate() end

    minetest.log("action", "[five_chambers] Boss defeated — episode complete.")
end

function five_chambers.spawn_boss()
    local c   = five_chambers.CH5
    local pos = {x = 6, y = five_chambers.FLOOR_Y + 1, z = math.floor((c.z0 + c.z1) / 2)}

    local obj = minetest.add_entity(pos, "mobs_mc:zombie")
    if not obj then
        minetest.log("error", "[five_chambers] Boss spawn failed.")
        return
    end

    -- Override HP to BOSS_HP (60); set after on_activate has run.
    local ent = obj:get_luaentity()
    if ent then
        ent.health = five_chambers.BOSS_HP
        ent.hp_max = five_chambers.BOSS_HP
    end
    obj:set_hp(five_chambers.BOSS_HP)

    five_chambers.mob_state.ch5_boss = {
        obj           = obj,
        contributors  = {},   -- {[agent_name]=true}
        dmg_fired     = false,
        half_hp_fired = false,
    }

    minetest.log("action", "[five_chambers] Boss spawned at "
        .. minetest.pos_to_string(pos)
        .. " with " .. five_chambers.BOSS_HP .. " HP.")
end

-- ── Globalstep: Ch5 entry detection (M24 + spawn trigger) ────────

minetest.register_globalstep(function(dtime)
    if not five_chambers.CHAMBERS[5].enabled then return end
    if five_chambers.step_counter % 3 ~= 0 then return end

    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        if five_chambers.agent_index(name) >= 0 then
            local pos = player:get_pos()
            if pos and five_chambers.get_chamber_for_pos(pos) == "ch5" then
                five_chambers.fire_milestone("m24_enter_ch5", {name})
                if not five_chambers.mob_state.ch5_triggered then
                    five_chambers.mob_state.ch5_triggered = true
                    five_chambers.spawn_boss()
                end
            end
        end
    end
end)

-- ── Globalstep: boss damage tracking (M25, M26, M27, M28) ────────

minetest.register_globalstep(function(dtime)
    local boss = five_chambers.mob_state.ch5_boss
    if not boss then return end

    local obj = boss.obj
    if not obj:is_valid() then
        fire_boss_death()
        five_chambers.mob_state.ch5_boss = nil
        return
    end

    local ent = obj:get_luaentity()
    if not ent then
        fire_boss_death()
        five_chambers.mob_state.ch5_boss = nil
        return
    end

    -- Sync contributors from entity punch tracking.
    if ent._fc_contributors then
        for pname in pairs(ent._fc_contributors) do
            boss.contributors[pname] = true
        end
        ent._fc_contributors = {}
    end

    local hp = ent.health or ent.hp or obj:get_hp()

    -- M25: first damage landed.
    if not boss.dmg_fired and next(boss.contributors) then
        boss.dmg_fired = true
        local contrib_list = {}
        for pname in pairs(boss.contributors) do
            table.insert(contrib_list, pname)
        end
        five_chambers.fire_milestone("m25_first_boss_dmg", contrib_list)
    end

    -- M26: boss below half HP.
    if not boss.half_hp_fired and hp and hp <= five_chambers.BOSS_HP / 2 then
        boss.half_hp_fired = true
        local contrib_list = {}
        for pname in pairs(boss.contributors) do
            table.insert(contrib_list, pname)
        end
        five_chambers.fire_milestone("m26_boss_half_hp", contrib_list)
    end

    -- Boss dead (VoxeLibre mob HP reaches 0 before removal).
    if hp and hp <= 0 then
        fire_boss_death()
        five_chambers.mob_state.ch5_boss = nil
    end
end)
