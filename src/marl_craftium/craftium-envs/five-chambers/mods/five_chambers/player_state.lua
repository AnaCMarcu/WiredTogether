-- player_state.lua: write per-agent state files for Python polling.
-- Writes health, hunger, and inventory to world_path every 20 ticks (~1s).
-- Files read by custom_environment_craftium.py:
--   health_agent{i}.txt  → "{hp}/20"
--   hunger_agent{i}.txt  → "20/20" (always — see hunger drain neutralisation below)
--   inv_agent{i}.txt     → "{wield_idx}|{slot1}|{slot2}|..."
--     Each slot: "" (empty) or "item_name count"
--
-- Hunger drain neutralisation: VoxeLibre's mcl_hunger.exhaust() is the
-- single function that drains saturation → hunger over wall time. It's
-- called from mcl_hunger/hunger.lua on every Dig/Jump/Sprint/Attack.
-- We unconditionally monkey-patch mcl_hunger.exhaust() to a no-op so
-- hunger never drops. The mod's API stays loaded (set_hunger /
-- get_hunger / HUD updates still work) but the only path that REDUCES
-- hunger is severed. Plus we periodically set_hunger(20) as a belt-and-
-- suspenders for any code path (eating poisonous food, lava, etc.) that
-- directly modifies hunger.
--
-- (The phase-based gating — survival vs exploration — used to live here
-- but the survival mode was removed. Hunger is now always off.)

local _tick = 0
local WRITE_EVERY = 20  -- Lua ticks between writes (~1 second at 20 Hz)

local function write_file(path, content)
    local f = io.open(path, "w")
    if f then f:write(content); f:close() end
end

-- Monkey-patch hunger drain. Done in on_mods_loaded so mcl_hunger is
-- guaranteed initialised. We replace the global function rather than
-- removing it so any caller that does `mcl_hunger.exhaust(...)` still
-- works — it just becomes a no-op.
minetest.register_on_mods_loaded(function()
    if not (mcl_hunger and mcl_hunger.exhaust) then
        minetest.log("warning",
            "[five_chambers] mcl_hunger.exhaust not found; hunger drain "
            .. "patch skipped (agents may starve over long episodes).")
        return
    end
    mcl_hunger.exhaust = function(_playername, _increase) return true end
    minetest.log("action",
        "[five_chambers] mcl_hunger.exhaust patched to no-op "
        .. "(hunger drain unconditionally disabled).")
end)

minetest.register_globalstep(function(dtime)
    _tick = _tick + 1

    if _tick % WRITE_EVERY ~= 0 then return end

    local world_path = minetest.get_worldpath()

    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        local idx  = five_chambers.agent_index(name)
        if idx >= 0 then
            -- Health: player:get_hp() returns 0-20
            local hp = math.floor(player:get_hp())
            write_file(world_path .. "/health_agent" .. idx .. ".txt",
                       hp .. "/20")

            -- Hunger: drain is neutralised at the source via the
            -- mcl_hunger.exhaust monkey-patch above. Belt-and-suspenders:
            -- still call set_hunger(20) here in case any other code path
            -- (eating poisonous food, lava, etc.) directly modified it.
            if mcl_hunger and mcl_hunger.set_hunger then
                pcall(mcl_hunger.set_hunger, player, 20)
                if mcl_hunger.set_saturation then
                    pcall(mcl_hunger.set_saturation, player, 5)
                end
                if mcl_hunger.set_exhaustion then
                    pcall(mcl_hunger.set_exhaustion, player, 0)
                end
            end
            write_file(world_path .. "/hunger_agent" .. idx .. ".txt",
                       "20/20")

            -- Inventory: wielded slot + up to 9 main slots
            local inv     = player:get_inventory()
            local wield   = player:get_wield_index()  -- 1-based hotbar index
            local parts   = { tostring(wield) }

            if inv then
                local main = inv:get_list("main") or {}
                for slot = 1, math.min(9, #main) do
                    local stack = main[slot]
                    if stack and not stack:is_empty() then
                        table.insert(parts,
                            stack:get_name() .. " " .. stack:get_count())
                    else
                        table.insert(parts, "")
                    end
                end
            end

            write_file(world_path .. "/inv_agent" .. idx .. ".txt",
                       table.concat(parts, "|"))
        end
    end

    -- Anvils state file: chamber-state visibility for the LLM prompt
    -- (Tier 5 — exposes hp + active punchers per anvil so the policy can
    -- reason about who's working on what without having to infer from
    -- the visual frame alone).
    -- Format (one anvil per line):
    --   sword|<hp>/<max>|<comma-separated active puncher names or empty>
    --   chestplate|<hp>/<max>|<comma-separated names>
    if five_chambers.anvil_state then
        local lines = {}
        local now_tick = five_chambers.step_counter or 0
        local W = five_chambers.ACTIVE_WINDOW or 18
        for _, state in pairs(five_chambers.anvil_state) do
            local active_names = {}
            if state.punchers then
                for name, last_tick in pairs(state.punchers) do
                    if now_tick - last_tick <= W then
                        table.insert(active_names, name)
                    end
                end
            end
            local kind = (state.row == "A") and "sword" or "chestplate"
            table.insert(lines, string.format(
                "%s|%d/%d|%s",
                kind,
                state.hp or 0,
                five_chambers.ANVIL_MAX_HP or 30,
                table.concat(active_names, ",")))
        end
        write_file(world_path .. "/anvils.txt", table.concat(lines, "\n"))
    end
end)
