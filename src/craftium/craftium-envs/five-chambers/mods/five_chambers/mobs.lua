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

-- Patches one mob entity so on_punch stores the last human puncher.
-- Must be called from register_on_mods_loaded (after mobs_mc has loaded).
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
            self._fc_last_puncher = puncher:get_player_name()
        end
        if orig then return orig(self, puncher, tflp, tool_caps, dir, damage) end
    end
    return true
end

function five_chambers.patch_mobs_for_kill_tracking()
    patch_entity_for_kill_tracking("mobs_mc:chicken")
    patch_entity_for_kill_tracking("mobs_mc:sheep")
end

-- ── Spawn Ch1 animals ────────────────────────────────────────────

-- Initialise mob_state table immediately (before on_mods_loaded / globalstep runs).
five_chambers.mob_state = {
    active_ch1_mobs = {},
    ch4_mobs        = {},
    ch4_triggered   = false,
    ch5_boss        = nil,
    ch5_triggered   = false,
    ch4_kills       = {},
    boss_damage     = {},
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
        active_ch1_mobs = {},
        ch4_mobs        = {},
        ch4_triggered   = false,
        ch5_boss        = nil,
        ch5_triggered   = false,
        ch4_kills       = {},
        boss_damage     = {},
    }
end

-- Stubs for D6/D7.
function five_chambers.spawn_ch4_mobs()  end
function five_chambers.spawn_boss()       end
