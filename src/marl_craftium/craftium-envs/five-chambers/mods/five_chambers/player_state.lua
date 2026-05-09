-- player_state.lua: write per-agent state files for Python polling.
-- Writes health, hunger, and inventory to world_path every 20 ticks (~1s).
-- Files read by custom_environment_craftium.py:
--   health_agent{i}.txt  → "{hp}/20"
--   hunger_agent{i}.txt  → "20/20" while phase is exploration; live value in survival
--   inv_agent{i}.txt     → "{wield_idx}|{slot1}|{slot2}|..."
--     Each slot: "" (empty) or "item_name count"
--
-- Hunger pinning: VoxeLibre's mcl_hunger drains hunger over wall time even
-- when the trainer is in non-survival "exploration" mode. With episodes
-- running ~7000 outer steps (≈ minutes of in-game time), hunger reaches
-- 0 and starvation damage kills agents → the trainer sees a -50 death
-- penalty per kill and the policy learns "everything I do leads to death".
-- We re-pin hunger/saturation/exhaustion EVERY write tick whenever phase
-- is not "survival", so non-survival training never starves out.

local _tick = 0
local WRITE_EVERY = 20  -- Lua ticks between writes (~1 second at 20 Hz)

local function write_file(path, content)
    local f = io.open(path, "w")
    if f then f:write(content); f:close() end
end

local function _read_phase(world_path)
    local f = io.open(world_path .. "/phase.txt", "r")
    if not f then return "exploration" end
    local s = f:read("*a") or ""
    f:close()
    return (s:gsub("%s+", ""))
end

minetest.register_globalstep(function(dtime)
    _tick = _tick + 1
    if _tick % WRITE_EVERY ~= 0 then return end

    local world_path = minetest.get_worldpath()
    local phase = _read_phase(world_path)
    local pin_hunger = (phase ~= "survival")

    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        local idx  = five_chambers.agent_index(name)
        if idx >= 0 then
            -- Health: player:get_hp() returns 0-20
            local hp = math.floor(player:get_hp())
            write_file(world_path .. "/health_agent" .. idx .. ".txt",
                       hp .. "/20")

            -- Hunger: when not in survival, force-pin to max so mcl_hunger's
            -- drain never accumulates into starvation damage.
            if pin_hunger and mcl_hunger then
                if mcl_hunger.set_hunger then
                    pcall(mcl_hunger.set_hunger, player, 20)
                end
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
end)
