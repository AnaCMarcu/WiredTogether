-- player_state.lua: write per-agent state files for Python polling.
-- Writes health, hunger, and inventory to world_path every 20 ticks (~1s).
-- Files read by custom_environment_craftium.py:
--   health_agent{i}.txt  → "{hp}/20"
--   hunger_agent{i}.txt  → "20/20"  (VoxeLibre hunger not tracked here)
--   inv_agent{i}.txt     → "{wield_idx}|{slot1}|{slot2}|..."
--     Each slot: "" (empty) or "item_name count"

local _tick = 0
local WRITE_EVERY = 20  -- Lua ticks between writes (~1 second at 20 Hz)

local function write_file(path, content)
    local f = io.open(path, "w")
    if f then f:write(content); f:close() end
end

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

            -- Hunger: VoxeLibre tracks saturation via mcl_hunger, but we pin
            -- it to full since five-chambers has no hunger drain configured.
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
