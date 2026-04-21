-- anvil.lua: heavy anvil mechanic (plan §4).
-- Registers the five_chambers:anvil node and the per-step progress loop.
-- Anvils require 2+ simultaneous diggers because solo digging has net negative
-- progress (dig_rate(1)=1 < DECAY_RATE=2). All decay is always applied.
--
-- Anvil positions are computed from NUM_AGENTS:
--   Row A (swords):      x = CH2.x0 + 1 + (i*3), z = CH2.z0 + 2
--   Row B (chestplates): x = CH2.x0 + 1 + (i*3), z = CH2.z0 + 5
-- for i = 0..NUM_AGENTS-1.
--
-- Milestone mapping (position-indexed, not break-order):
--   Anvil A_i → m{8+i}_anvil_A{i+1}
--   Anvil B_i → m{8+NUM_AGENTS+i}_anvil_B{i+1}

five_chambers.anvil_state = {}  -- keyed by minetest.pos_to_string(pos)

function five_chambers.register_anvil_node()
    -- stub: D3 will register five_chambers:anvil with on_punch + on_dig hooks
    -- and add the globalstep update loop
end

function five_chambers.init_anvils()
    -- stub: D3 will populate anvil_state with initial hp=0 for each anvil position
end
