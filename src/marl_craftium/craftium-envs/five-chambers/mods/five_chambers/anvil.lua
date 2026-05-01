-- anvil.lua: heavy anvil mechanic (plan §4).
-- Anvils require 2+ simultaneous diggers because solo digging has net negative
-- progress (dig_rate(1)=1 < DECAY_RATE=2). All decay is always applied.
--
-- Anvil positions for i=0..NUM_AGENTS-1:
--   Row A (swords):      x = CH2.x0+1 + i*3,  z = CH2.z0+2
--   Row B (chestplates): x = CH2.x0+1 + i*3,  z = CH2.z0+5
--
-- Milestone mapping:
--   Anvil A_i → m{8+i}_anvil_A{i+1}
--   Anvil B_i → m{8+N+i}_anvil_B{i+1}

five_chambers.anvil_state        = {}  -- key=pos_string → {pos,hp,punchers,milestone_id,row,first_break}
five_chambers.anvil_breaks_total = 0
five_chambers.anvil_first_breaks = 0  -- distinct anvils broken at least once

-- ── Helpers ──────────────────────────────────────────────────────────

local function anvil_positions()
    local N = five_chambers.NUM_AGENTS
    local c = five_chambers.CH2
    local y = five_chambers.FLOOR_Y + 1
    local list = {}
    for i = 0, N - 1 do
        local ax = c.x0 + 1 + (i * 3)
        table.insert(list, {
            pos          = {x = ax, y = y, z = c.z0 + 2},
            milestone_id = "m" .. (8 + i)     .. "_anvil_A" .. (i + 1),
            row          = "A",
        })
        table.insert(list, {
            pos          = {x = ax, y = y, z = c.z0 + 5},
            milestone_id = "m" .. (8 + N + i) .. "_anvil_B" .. (i + 1),
            row          = "B",
        })
    end
    return list
end

-- ── Node registration (module-level, runs when anvil.lua is dofile'd) ──

minetest.register_node("five_chambers:anvil", {
    description = "Heavy Anvil",
    -- Use stone texture as a visual placeholder; anvil uses custom HP logic.
    tiles  = {"default_stone.png"},
    groups = {unbreakable = 1},
    on_punch = function(pos, node, puncher, pointed_thing)
        if not puncher or not puncher:is_player() then return end
        local key   = minetest.pos_to_string(pos)
        local state = five_chambers.anvil_state[key]
        if not state then return end
        state.punchers[puncher:get_player_name()] = five_chambers.step_counter
    end,
})

-- Keep as no-op so init.lua call (legacy) does nothing harmful.
function five_chambers.register_anvil_node() end

-- ── Public init (called from init.lua on_mods_loaded + reset handler) ──

function five_chambers.init_anvils()
    five_chambers.anvil_state        = {}
    five_chambers.anvil_breaks_total = 0
    five_chambers.anvil_first_breaks = 0
    for _, info in ipairs(anvil_positions()) do
        local key = minetest.pos_to_string(info.pos)
        five_chambers.anvil_state[key] = {
            pos          = info.pos,
            hp           = 0,
            punchers     = {},
            milestone_id = info.milestone_id,
            row          = info.row,
            first_break  = false,
        }
    end
end

-- ── Globalstep: HP decay + break detection ───────────────────────────

minetest.register_globalstep(function(dtime)
    if not five_chambers.CHAMBERS[2].enabled then return end
    if not next(five_chambers.anvil_state) then return end

    local tick = five_chambers.step_counter
    local W    = five_chambers.ACTIVE_WINDOW

    for key, state in pairs(five_chambers.anvil_state) do
        -- Count diggers active within the last W ticks.
        local active = {}
        for name, last_tick in pairs(state.punchers) do
            if tick - last_tick <= W then
                table.insert(active, name)
            end
        end
        local n = #active

        local dig_rate
        if     n == 0 then dig_rate = 0
        elseif n == 1 then dig_rate = five_chambers.SOLO_DIG_RATE
        elseif n == 2 then dig_rate = five_chambers.PAIR_DIG_RATE
        else               dig_rate = five_chambers.TRIO_DIG_RATE
        end

        state.hp = math.max(0, state.hp + dig_rate - five_chambers.DECAY_RATE)

        if state.hp >= five_chambers.ANVIL_MAX_HP then
            state.hp = 0
            five_chambers.anvil_breaks_total = five_chambers.anvil_breaks_total + 1

            -- Fire milestone for all active diggers.
            if #active > 0 then
                five_chambers.fire_milestone(state.milestone_id, active)
            end

            -- Drop gear at anvil position.
            five_chambers.drop_gear(state.pos,
                state.row == "A" and "sword" or "chestplate")

            -- Track first-break per distinct anvil; unlock door when all done.
            if not state.first_break then
                state.first_break = true
                five_chambers.anvil_first_breaks = five_chambers.anvil_first_breaks + 1
                if five_chambers.anvil_first_breaks >= five_chambers.NUM_AGENTS * 2 then
                    five_chambers.start_door2_countdown()
                    minetest.log("action",
                        "[five_chambers] All anvils broken — Door 2 countdown started.")
                end
            end
        end
    end
end)
