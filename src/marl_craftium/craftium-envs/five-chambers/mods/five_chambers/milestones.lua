-- milestones.lua: M1–M28 state machine.
-- Each milestone fires at most once per agent per episode (for once=true milestones)
-- via emit_milestone() which handles file IO and RL reward delivery.

-- Milestone definitions: id → { track, reward, once }
-- once=true  → fires at most once per agent per episode
-- once=true (every entry below): each agent can be credited for a milestone
-- AT MOST ONCE PER EPISODE. Per-agent dedup happens in fire_milestone() via
-- milestone_fired[name][milestone_id]; that table resets in
-- reset_milestone_state() so the next episode is a fresh slate.
--
-- Why all-once: free-rider deterrent + anti-farming. Without once=true, an
-- agent could (a) repeatedly break the same Ch2 anvil for unlimited 40 HP
-- of reward, (b) walk in/out of the Ch3 communal room to farm M19, or (c)
-- collect M22/M23/M25-M28 each time the trigger condition is re-evaluated.
-- With once=true, agents are incentivised to make new contributions rather
-- than re-trigger old ones.
five_chambers.MILESTONE_DEFS = {
    -- Ch1 solo learning
    m1_move_5            = { track="ch1_solo",  reward=10,  once=true },
    m2_dig_3_any         = { track="ch1_solo",  reward=30,  once=true },
    m3_pickup_3          = { track="ch1_solo",  reward=30,  once=true },
    m4_dig_5_wood        = { track="ch1_solo",  reward=50,  once=true },
    m5_kill_1_animal     = { track="ch1_solo",  reward=50,  once=true },
    m6_kill_2_animals    = { track="ch1_solo",  reward=80,  once=true },
    m7_dig_3_stone       = { track="ch1_solo",  reward=60,  once=true },
    -- Ch2 anvil cooperation
    m8_anvil_A1          = { track="ch2_anvils", reward=40, once=true },
    m9_anvil_A2          = { track="ch2_anvils", reward=40, once=true },
    m10_anvil_A3         = { track="ch2_anvils", reward=40, once=true },
    m11_anvil_B1         = { track="ch2_anvils", reward=40, once=true },
    m12_anvil_B2         = { track="ch2_anvils", reward=40, once=true },
    m13_anvil_B3         = { track="ch2_anvils", reward=40, once=true },
    m14_sword_equipped   = { track="ch2_gear",  reward=50,  once=true },
    m15_chestplate_equipped = { track="ch2_gear", reward=30, once=true },
    -- Ch3 switch puzzle
    m16_enter_cell       = { track="ch3_switch", reward=20, once=true },
    m17_switch_pressed   = { track="ch3_switch", reward=40, once=true },
    m18_door_opened      = { track="ch3_switch", reward=60, once=true },
    m19_all_in_communal  = { track="ch3_switch", reward=100, once=true },
    -- Ch4 combat
    m20_enter_ch4        = { track="ch4_combat", reward=30, once=true },
    m21_first_mob_kill   = { track="ch4_combat", reward=60, once=true },
    m22_all_mobs_killed  = { track="ch4_combat", reward=150, once=true },
    m23_all_alive_ch4    = { track="ch4_combat", reward=100, once=true },
    -- Ch5 boss
    m24_enter_ch5        = { track="ch5_boss", reward=50,  once=true },
    m25_first_boss_dmg   = { track="ch5_boss", reward=80,  once=true },
    m26_boss_half_hp     = { track="ch5_boss", reward=120, once=true },
    m27_boss_defeated    = { track="ch5_boss", reward=300, once=true },
    m28_all_alive_bonus  = { track="ch5_boss", reward=250, once=true },
}

-- Per-episode state (reset by reset_milestone_state()).
five_chambers.milestone_fired    = {}  -- [name][milestone_id] = true
five_chambers.dig_counts         = {}  -- [name] = {any=N, wood=N, stone=N}
five_chambers.pickup_counts      = {}  -- [name] = N (only dig-attributed pickups)
five_chambers.kill_counts        = {}  -- [name] = N  (Ch1 animal kills)
five_chambers.spawn_pos          = {}  -- [name] = {x, z} initial position for M1
five_chambers.prev_inv_total     = {}  -- [name] = N  for inventory-diff pickup tracking
five_chambers.pending_dig_drops  = {}  -- [name] = FIFO list of step_counter ticks
                                       --          recording when this agent dug a
                                       --          node with non-empty drops. An
                                       --          inventory increase only counts
                                       --          toward M3 if there is a pending
                                       --          credit, so passive sources
                                       --          (chicken eggs, etc.) are ignored.

-- Window (in step_counter ticks) during which a pending dig credit stays valid.
-- ~150 Lua ticks ≈ 50 env steps. If the dropped item falls somewhere unreachable
-- (lava, void) the credit silently expires instead of being collected by an
-- unrelated future pickup.
five_chambers.PENDING_DIG_TTL = 150

-- Initialise or re-initialise per-player tracking for one agent.
function five_chambers.init_player_milestone_state(name)
    five_chambers.milestone_fired[name]    = {}
    five_chambers.dig_counts[name]         = {any=0, wood=0, stone=0}
    five_chambers.pickup_counts[name]      = 0
    five_chambers.kill_counts[name]        = 0
    five_chambers.prev_inv_total[name]     = 0
    five_chambers.pending_dig_drops[name]  = {}
    five_chambers.spawn_pos[name]          = nil  -- filled on joinplayer
end

-- Reset all milestone tracking. Called at episode start (reset handler in init.lua).
function five_chambers.reset_milestone_state()
    five_chambers.milestone_fired    = {}
    five_chambers.dig_counts         = {}
    five_chambers.pickup_counts      = {}
    five_chambers.kill_counts        = {}
    five_chambers.prev_inv_total     = {}
    five_chambers.pending_dig_drops  = {}
    -- Re-record current position as spawn reference so M1 resets correctly each episode.
    five_chambers.step_counter = 0

    for _, p in ipairs(minetest.get_connected_players()) do
        five_chambers.init_player_milestone_state(p:get_player_name())
        -- Re-record current position as spawn reference for M1.
        local pos = p:get_pos()
        if pos then
            five_chambers.spawn_pos[p:get_player_name()] = {x=pos.x, z=pos.z}
        end
        -- Re-record current inventory total so we don't count gear already held.
        local inv = p:get_inventory()
        if inv then
            local total = 0
            for _, stack in ipairs(inv:get_list("main") or {}) do
                total = total + stack:get_count()
            end
            five_chambers.prev_inv_total[p:get_player_name()] = total
        end
    end
end

-- Fire a milestone for a list of contributor player names.
-- Skips contributors who already fired this milestone (when once=true).
function five_chambers.fire_milestone(milestone_id, contributors)
    local def = five_chambers.MILESTONE_DEFS[milestone_id]
    if not def then
        minetest.log("warning", "[five_chambers] Unknown milestone: " .. milestone_id)
        return
    end

    local actual = {}
    for _, name in ipairs(contributors) do
        if not five_chambers.milestone_fired[name] then
            five_chambers.milestone_fired[name] = {}
        end
        if def.once and five_chambers.milestone_fired[name][milestone_id] then
            -- Already fired for this agent; skip.
        else
            if def.once then
                five_chambers.milestone_fired[name][milestone_id] = true
            end
            table.insert(actual, name)
        end
    end

    if #actual == 0 then return end

    five_chambers.emit_milestone(milestone_id, actual, def.reward)
end

-- ──────────────────────────────────────────────────────────────────
-- M1 — move >5 blocks from spawn (detected in globalstep)
-- M2 — dig 3 any blocks     (on_dignode)
-- M3 — pick up 3 items      (globalstep inventory diff)
-- M4 — dig 5 wood blocks    (on_dignode)
-- M5 — kill 1 animal        (mobs.lua → record_animal_kill)
-- M6 — kill 2 animals       (mobs.lua → record_animal_kill)
-- M7 — dig 3 stone blocks   (on_dignode)
-- ──────────────────────────────────────────────────────────────────

-- Called by joinplayer and reset handler to capture initial position.
function five_chambers.record_spawn_pos(name, pos)
    five_chambers.spawn_pos[name] = {x=pos.x, z=pos.z}
end

-- Called by mobs.lua when a Ch1 animal dies and the killer is known.
function five_chambers.record_animal_kill(killer_name)
    if not five_chambers.kill_counts[killer_name] then
        five_chambers.kill_counts[killer_name] = 0
    end
    five_chambers.kill_counts[killer_name] = five_chambers.kill_counts[killer_name] + 1
    local n = five_chambers.kill_counts[killer_name]

    if n >= 1 then
        five_chambers.fire_milestone("m5_kill_1_animal", {killer_name})
    end
    if n >= 2 then
        five_chambers.fire_milestone("m6_kill_2_animals", {killer_name})
    end
end

-- ── Dig hook (M2, M4, M7) ───────────────────────────────────────

minetest.register_on_dignode(function(pos, oldnode, digger)
    if not digger or not digger:is_player() then return end
    local name = digger:get_player_name()

    -- Ensure per-player state exists (might have joined mid-episode).
    if not five_chambers.dig_counts[name] then
        five_chambers.dig_counts[name] = {any=0, wood=0, stone=0}
    end
    if not five_chambers.pending_dig_drops[name] then
        five_chambers.pending_dig_drops[name] = {}
    end

    local node_name = oldnode.name
    local counts    = five_chambers.dig_counts[name]

    counts.any = counts.any + 1
    if minetest.get_item_group(node_name, "tree") > 0 then
        counts.wood = counts.wood + 1
    end
    if minetest.get_item_group(node_name, "stone") > 0 then
        counts.stone = counts.stone + 1
    end

    if counts.any   >= 3 then five_chambers.fire_milestone("m2_dig_3_any",   {name}) end
    if counts.wood  >= 5 then five_chambers.fire_milestone("m4_dig_5_wood",  {name}) end
    if counts.stone >= 3 then five_chambers.fire_milestone("m7_dig_3_stone", {name}) end

    -- M3 attribution: only push a pickup credit if the dug node actually
    -- drops an item. Without this filter, digging a no-drop node would let a
    -- subsequent passive drop (e.g. a chicken egg) consume the credit.
    local drops = minetest.get_node_drops(node_name, "")
    if drops and #drops > 0 then
        table.insert(
            five_chambers.pending_dig_drops[name],
            five_chambers.step_counter
        )
    end
end)

-- ── Globalstep: M14/M15 gear equip (Ch2) ────────────────────────
-- check_equip is defined in gear.lua (loaded after milestones.lua).

minetest.register_globalstep(function(dtime)
    if not five_chambers.CHAMBERS[2].enabled then return end
    for _, player in ipairs(minetest.get_connected_players()) do
        five_chambers.check_equip(player)
    end
end)

-- ── Globalstep: M1 (movement) + M3 (pickup) ─────────────────────

minetest.register_globalstep(function(dtime)
    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()

        -- Ensure state is initialised (first globalstep after join).
        if not five_chambers.milestone_fired[name] then
            five_chambers.init_player_milestone_state(name)
        end

        -- M1: distance >5 from initial spawn position (Y-plane only).
        local sp = five_chambers.spawn_pos[name]
        if sp and not five_chambers.milestone_fired[name]["m1_move_5"] then
            local cur = player:get_pos()
            if cur then
                local dx = cur.x - sp.x
                local dz = cur.z - sp.z
                if math.sqrt(dx*dx + dz*dz) > 5 then
                    five_chambers.fire_milestone("m1_move_5", {name})
                end
            end
        end

        -- M3: inventory total increased AND there is a pending dig credit
        -- from this same agent. Inventory growth without a recent dig
        -- (e.g. walking over a chicken egg) does NOT count toward M3.
        local inv = player:get_inventory()
        if inv then
            local total = 0
            for _, stack in ipairs(inv:get_list("main") or {}) do
                total = total + stack:get_count()
            end
            local prev  = five_chambers.prev_inv_total[name] or 0
            local delta = total - prev
            if delta > 0 then
                local pending = five_chambers.pending_dig_drops[name] or {}
                -- Expire stale credits whose drop was never collected.
                local cutoff = (five_chambers.step_counter or 0)
                               - five_chambers.PENDING_DIG_TTL
                while #pending > 0 and pending[1] < cutoff do
                    table.remove(pending, 1)
                end
                -- Consume up to `delta` credits from the FIFO queue.
                local credits = math.min(delta, #pending)
                for _ = 1, credits do
                    table.remove(pending, 1)
                end
                five_chambers.pending_dig_drops[name] = pending
                if credits > 0 then
                    five_chambers.pickup_counts[name] =
                        (five_chambers.pickup_counts[name] or 0) + credits
                    if five_chambers.pickup_counts[name] >= 3 then
                        five_chambers.fire_milestone("m3_pickup_3", {name})
                    end
                end
            end
            five_chambers.prev_inv_total[name] = total
        end
    end
end)
