-- gear.lua: gear drops from anvils and equip detection (plan §4, D4).
-- Row A anvils (swords) drop mcl_swords:sword_diamond at the anvil position.
-- Row B anvils (chestplates) drop mcl_armor:chestplate_diamond.
-- Equip detection fires milestones M14 (sword wielded) and M15 (chestplate on).

-- Called by anvil.lua when an anvil breaks.
-- drop_type: "sword" | "chestplate"
function five_chambers.drop_gear(pos, drop_type)
    -- stub: D4 will call minetest.add_item(pos, itemstack)
end

-- Called each globalstep or on inventory change to check equipped gear.
-- Fires M14 / M15 once per agent once the item is in the right slot.
function five_chambers.check_equip(player)
    -- stub: D4 will inspect player:get_wielded_item() and armor API
end
