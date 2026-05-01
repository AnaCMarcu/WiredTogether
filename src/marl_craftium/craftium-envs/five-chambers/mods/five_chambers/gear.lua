-- gear.lua: gear drops from anvils and equip detection (plan §4, D3).
-- Row A anvils drop mcl_tools:sword_diamond.
-- Row B anvils drop mcl_armor:chestplate_diamond.
-- Equip detection fires M14 (sword wielded) and M15 (chestplate on armor slot).

-- Called by anvil.lua when an anvil breaks.
-- drop_type: "sword" | "chestplate"
function five_chambers.drop_gear(pos, drop_type)
    local item = drop_type == "sword"
        and "mcl_tools:sword_diamond"
        or  "mcl_armor:chestplate_diamond"
    minetest.add_item(pos, item)
end

-- Check if player has a sword wielded or a chestplate equipped.
-- Fires M14 / M15 once per agent per episode (once=true in MILESTONE_DEFS).
function five_chambers.check_equip(player)
    local name = player:get_player_name()

    -- M14: sword in wield slot.
    local wielded = player:get_wielded_item()
    if minetest.get_item_group(wielded:get_name(), "sword") > 0 then
        five_chambers.fire_milestone("m14_sword_equipped", {name})
    end

    -- M15: chestplate in armor inventory.
    local inv = player:get_inventory()
    if inv then
        for _, stack in ipairs(inv:get_list("armor") or {}) do
            if not stack:is_empty() and stack:get_name():find("chestplate") then
                five_chambers.fire_milestone("m15_chestplate_equipped", {name})
                break
            end
        end
    end
end
