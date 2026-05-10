-- gear.lua: gear distribution from anvil breaks (plan §4, D3, simplified).
-- Both anvils (1 sword + 1 chestplate) distribute their gear DIRECTLY to
-- every connected agent's inventory on break, with auto-equip:
--   * sword → first hotbar slot, set as wielded
--   * chestplate → armor inventory's chestplate slot
-- Removed dropped-item pickup mechanic: in the original design, agents had
-- to physically walk over a dropped sword to gain it, which never
-- consistently happened in policy training. Auto-equip removes that
-- bottleneck so Ch4/Ch5 combat is reachable as soon as Ch2 cooperation
-- succeeds.

-- Internal: place item directly in player's main inventory (slot 1 if
-- empty, else first empty slot, else dropped at feet as fallback).
local function _give_to_main(player, itemstring)
    local inv = player:get_inventory()
    if not inv then return false end
    local stack = ItemStack(itemstring)
    if inv:room_for_item("main", stack) then
        inv:add_item("main", stack)
        return true
    end
    -- Fallback: drop at feet so the item isn't lost
    minetest.add_item(player:get_pos(), stack)
    return false
end

-- Internal: equip chestplate by writing directly to the armor inventory.
-- VoxeLibre stores armor as `inv:set_list("armor", {head, torso, legs, feet})`.
-- Slot 2 is the torso/chestplate slot. We zero out any existing torso item
-- first to keep the slot exactly as the new chestplate.
local function _equip_chestplate(player, itemstring)
    local inv = player:get_inventory()
    if not inv then return false end
    -- VoxeLibre's mcl_armor uses _mcl_armor lists or `armor` depending on
    -- version. Try both to be tolerant.
    local stack = ItemStack(itemstring)
    if inv:get_size("armor") > 0 then
        local list = inv:get_list("armor") or {ItemStack(""), ItemStack(""),
                                                 ItemStack(""), ItemStack("")}
        list[2] = stack  -- torso slot
        inv:set_list("armor", list)
        return true
    end
    -- Fallback: put it in main inventory so the agent at least has it
    return _give_to_main(player, itemstring)
end

-- Called by anvil.lua when an anvil breaks. Distributes the corresponding
-- gear to every connected agent and fires the equip milestone (M14/M15)
-- for each agent who received it.
-- drop_type: "sword" | "chestplate"
function five_chambers.give_gear_to_all(drop_type)
    local item, milestone_id, equip_fn
    if drop_type == "sword" then
        item         = "mcl_tools:sword_diamond"
        milestone_id = "m14_sword_equipped"
        equip_fn = function(player)
            local ok = _give_to_main(player, item)
            if ok and player.set_wield_index then
                pcall(player.set_wield_index, player, 1)
            end
            return ok
        end
    else
        item         = "mcl_armor:chestplate_diamond"
        milestone_id = "m15_chestplate_equipped"
        equip_fn = function(player) return _equip_chestplate(player, item) end
    end

    for _, player in ipairs(minetest.get_connected_players()) do
        local name = player:get_player_name()
        if five_chambers.agent_index(name) >= 0 then
            local ok = equip_fn(player)
            if ok then
                five_chambers.fire_milestone(milestone_id, {name})
            end
        end
    end
    minetest.log("action",
        "[five_chambers] Distributed " .. drop_type
        .. " to all agents on anvil break.")
end

-- Backwards-compat shim: any code still calling drop_gear(pos, kind) now
-- routes through give_gear_to_all (kind matters; pos is ignored since
-- agents receive directly).
function five_chambers.drop_gear(_pos, drop_type)
    five_chambers.give_gear_to_all(drop_type)
end

-- check_equip is no longer needed for milestone firing (M14/M15 are fired
-- in give_gear_to_all). Kept as a no-op so any legacy caller doesn't break.
function five_chambers.check_equip(_player) end
