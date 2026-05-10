-- anvil.lua: heavy anvil mechanic (plan §4, simplified for RL tractability).
-- Two anvils total in Ch2: one drops swords (row A → m8), one drops
-- chestplates (row B → m11). Solo digging is net 0 (DECAY=1, SOLO=1) —
-- not punished, just unproductive. Pair digging is +3/tick, trio +7/tick.
-- When BOTH anvils have been broken once, gear is given directly to all
-- agents (sword in wield slot, chestplate in armor slot — see gear.lua's
-- give_gear_to_all) AND Door 2 opens.
--
-- Why 2 anvils and not 6: with the original 2×N=6 anvils and net-negative
-- solo digging, MAPPO would need to discover synchronised punching within
-- a 6-tick window for 6 separate anvils with zero learning signal until
-- the first break — observed never to happen in practice. Reducing to one
-- anvil per gear type (one round of cooperation needed) keeps the coop
-- requirement central without the exponential discovery cost.

five_chambers.anvil_state        = {}  -- key=pos_string → {pos,hp,punchers,milestone_id,row,first_break}
five_chambers.anvil_breaks_total = 0
five_chambers.anvil_first_breaks = 0  -- distinct anvils broken at least once
five_chambers.total_anvils       = 0  -- set in init_anvils(); door 2 fires when first_breaks >= total_anvils

-- ── Helpers ──────────────────────────────────────────────────────────

local function anvil_positions()
    -- Two anvils: sword (row A) and chestplate (row B). Centred along x in
    -- Ch2 so all 3 agents can reach either from the south-side spawn.
    local c  = five_chambers.CH2
    local y  = five_chambers.FLOOR_Y + 1
    local cx = math.floor((c.x0 + c.x1) / 2)  -- 6 for default Ch2 bounds
    return {
        {
            pos          = {x = cx, y = y, z = c.z0 + 2},
            milestone_id = "m8_anvil_A1",
            row          = "A",  -- drops swords
        },
        {
            pos          = {x = cx, y = y, z = c.z0 + 5},
            milestone_id = "m11_anvil_B1",
            row          = "B",  -- drops chestplates
        },
    }
end

-- ── Node registration (module-level, runs when anvil.lua is dofile'd) ──

minetest.register_node("five_chambers:anvil", {
    description = "Heavy Anvil (Purple)",
    -- Vivid purple tint over a stone base so agents can recognise the
    -- coop-anvil at a glance and tell it apart from regular stone, the
    -- red locked-door blocks, and the bedrock walls. The colour is
    -- referenced by name in the agent prompts ("purple anvils").
    tiles  = {
        "default_stone.png^[colorize:#7a00ff:200",  -- top
        "default_stone.png^[colorize:#7a00ff:200",  -- bottom
        "default_stone.png^[colorize:#9a3aff:220",  -- sides (slightly brighter)
    },
    groups = {unbreakable = 1},
    on_punch = function(pos, node, puncher, pointed_thing)
        if not puncher or not puncher:is_player() then return end
        local key   = minetest.pos_to_string(pos)
        local state = five_chambers.anvil_state[key]
        if not state then return end
        state.punchers[puncher:get_player_name()] = five_chambers.step_counter

        -- DEBUG_SINGLE: directly advance HP per click. The tick-based dig
        -- rate is calibrated for AI agents that punch every 3 Lua ticks
        -- (one env step); a human's ~1 click/sec is too slow to stack hits
        -- inside the 6-tick ACTIVE_WINDOW before decay clears progress.
        -- 10 HP/click × ANVIL_MAX_HP=30 → ~3 clicks per anvil.
        if five_chambers.DEBUG_SINGLE then
            state.hp = state.hp + 10
        end
    end,
})

-- Keep as no-op so init.lua call (legacy) does nothing harmful.
function five_chambers.register_anvil_node() end

-- ── Public init (called from init.lua on_mods_loaded + reset handler) ──

function five_chambers.init_anvils()
    five_chambers.anvil_state        = {}
    five_chambers.anvil_breaks_total = 0
    five_chambers.anvil_first_breaks = 0
    local positions = anvil_positions()
    five_chambers.total_anvils = #positions
    for _, info in ipairs(positions) do
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

            -- Distribute gear DIRECTLY to every connected agent's inventory
            -- (sword → wield slot; chestplate → armor slot). Auto-equip means
            -- agents don't need to walk over a dropped item to gain combat
            -- ability — Ch4/Ch5 fights become tractable even without
            -- dedicated pickup behaviour.
            five_chambers.give_gear_to_all(
                state.row == "A" and "sword" or "chestplate")

            -- Track first-break per distinct anvil; open Door 2 when ALL
            -- anvils (currently 2) have broken at least once.
            if not state.first_break then
                state.first_break = true
                five_chambers.anvil_first_breaks = five_chambers.anvil_first_breaks + 1
                if five_chambers.anvil_first_breaks >= (five_chambers.total_anvils or 2) then
                    five_chambers.start_door2_countdown()
                    minetest.log("action",
                        "[five_chambers] Both anvils broken — Door 2 countdown started.")
                end
            end

            -- Destroy the anvil so it cannot be broken again this episode.
            -- Replaces the node with air (the floor below at y0 stays intact)
            -- and drops the entry from anvil_state so the globalstep loop
            -- stops ticking it. Lua's `pairs` allows clearing the current
            -- key during iteration, so this is safe inside the for loop.
            minetest.set_node(state.pos, {name = "air"})
            five_chambers.anvil_state[key] = nil
            minetest.log("action",
                "[five_chambers] Anvil " .. key .. " destroyed after break.")
        end
    end
end)
