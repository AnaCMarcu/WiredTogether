-- config.lua: all tunable constants for Five Chambers.
-- Change NUM_AGENTS to run with more or fewer agents; all geometry
-- and wiring is derived from it at load time.

five_chambers.NUM_AGENTS = 3

-- DEBUG_SINGLE: solo human walkthrough mode. When true:
--   * NUM_AGENTS is forced to 1 — one cell, one switch, one Ch4 spawn group.
--   * Any connected player is treated as agent_0, so switches, milestones,
--     Door 2 → cell teleport, Door 3 communal check, etc. all fire for the
--     standalone-Luanti "singleplayer" name.
--   * The cell switch's rotational mapping (i+1)%N becomes (0+1)%1 = 0,
--     i.e. switch 0 opens cell 0's own door — the player lets themselves out.
-- The flow then is: spawn in Ch1 → Door 1 (always open) → Ch2 anvils →
-- Door 2 opens after countdown → teleport into cell 0 → press switch →
-- walk into communal → Door 3 opens (1 agent suffices) → Ch4 → kill mob →
-- Door 4 opens → Ch5 boss.
-- Set back to false before training runs.
five_chambers.DEBUG_SINGLE = false
if five_chambers.DEBUG_SINGLE then
    five_chambers.NUM_AGENTS = 1
end

-- Per-chamber enable flags. Set enabled=false to skip a chamber
-- entirely; world_gen will leave its space void and open the connecting
-- door so the sequence still connects.
five_chambers.CHAMBERS = {
    [1] = { enabled = true,  name = "solo_learning" },
    [2] = { enabled = true,  name = "anvil_coop" },
    [3] = { enabled = true,  name = "switch_puzzle" },
    [4] = { enabled = true,  name = "combat" },
    [5] = { enabled = true,  name = "boss" },
}

-- World geometry (plan §2)
-- Floor block is at FLOOR_Y; agents stand at FLOOR_Y+1.
five_chambers.FLOOR_Y   = 10
five_chambers.CEIL_Y    = 15
five_chambers.WALL_NODE = "mcl_core:bedrock"
five_chambers.AIR_NODE  = "air"

-- Chamber 1 bounds (solo learning, 16×16)
five_chambers.CH1 = { x0=0, x1=15, z0=0, z1=15 }

-- Ch1 spawn points (plan §2.3; exact corners for N=3)
five_chambers.CH1_SPAWNS_3 = {
    [0] = {x=1,  y=11, z=1},
    [1] = {x=10, y=11, z=1},
    [2] = {x=5,  y=11, z=10},
}

-- Ch1 resource positions (plan §2.3) — all at Y=FLOOR_Y+1=11
five_chambers.CH1_TREE_POSITIONS = {
    {x=2,z=2},{x=5,z=3},{x=8,z=2},{x=3,z=7},
    {x=9,z=6},{x=7,z=9},{x=2,z=8},{x=10,z=4},
}
five_chambers.CH1_STONE_POSITIONS = {
    {x=4,z=4},{x=3,z=5},{x=6,z=6},{x=8,z=7},
    {x=5,z=8},{x=4,z=2},{x=9,z=3},{x=7,z=5},
}

-- Ch1 animal spawn positions (5 chickens + 3 sheep, away from trees/stone)
five_chambers.CH1_CHICKEN_POSITIONS = {
    {x=1,z=1},{x=6,z=1},{x=10,z=2},{x=1,z=5},{x=9,z=9},
}
five_chambers.CH1_SHEEP_POSITIONS = {
    {x=6,z=5},{x=3,z=9},{x=8,z=3},
}

-- Door 1: always open (no bedrock placed here); gap in Ch1 north wall
five_chambers.DOOR1_X = 7

-- Chamber 2 bounds (anvil coop, 14×14)
five_chambers.CH2 = { x0=0, x1=13, z0=17, z1=30 }

-- Door 2: opens 20 steps after 6th anvil break
five_chambers.DOOR2_POS    = { x=7, z=31 }
five_chambers.DOOR2_DELAY  = 20

-- Chamber 3 (switch puzzle) — width scales with NUM_AGENTS
-- Width = 4*N+1 blocks; X: 0..(4N)
-- Cells: Z:33–35; communal room: Z:37–49; north wall at Z=50
five_chambers.CH3_Z0           = 32
five_chambers.CH3_CELL_Z0      = 33
five_chambers.CH3_CELL_Z1      = 35
five_chambers.CH3_FRONT_WALL_Z = 36
five_chambers.CH3_COMMUNAL_Z0  = 37
five_chambers.CH3_COMMUNAL_Z1  = 49
five_chambers.CH3_NORTH_WALL_Z = 50
-- Door 3 sits at the middle of Ch3's north wall. Ch3 width = 4*N+1 (x: 0..4N),
-- so the centre is at x = 2*N. This keeps the door inside Ch3 for any
-- NUM_AGENTS up to 5 (Ch4 spans x=1..11, so 2*N must stay <= 11).
five_chambers.DOOR3_X          = 2 * five_chambers.NUM_AGENTS

-- Chamber 4 (combat, 11×11)
five_chambers.CH4     = { x0=1, x1=11, z0=52, z1=62 }
five_chambers.DOOR4_POS = { x=6, z=63 }

-- Chamber 5 (boss, 9×9)
five_chambers.CH5 = { x0=2, x1=10, z0=64, z1=72 }

-- Anvil mechanic (plan §4)
five_chambers.ANVIL_MAX_HP  = 30
five_chambers.SOLO_DIG_RATE = 1
five_chambers.PAIR_DIG_RATE = 4
five_chambers.TRIO_DIG_RATE = 8
five_chambers.DECAY_RATE    = 2
five_chambers.ACTIVE_WINDOW = 6  -- Lua ticks (2 env steps × frameskip=3)
five_chambers.DIGGER_RADIUS = 3

-- Boss (plan §5)
five_chambers.BOSS_HP  = 60
five_chambers.BOSS_DMG = 3

-- DEBUG_SINGLE balance overrides. The production env is tuned for 3 agents
-- cooperating; a solo human walkthrough has to clear the same content alone
-- so we soften every coop-gated mechanic:
--   * Anvils: SOLO=1, DECAY=2 → solo digging is net negative (impossible).
--     Bumping SOLO to 4 gives net +2 → ~15 ticks per anvil.
--   * Ch4: zombie spawn count is min(NUM_AGENTS, len(positions)) → 1 solo.
--   * Boss: lower HP from 60 → 20 so an unarmed player can punch it to death.
--     (BOSS_DMG isn't wired to the entity yet — VoxeLibre's mobs_mc:zombie
--     default melee applies; keep the override anyway as a marker.)
if five_chambers.DEBUG_SINGLE then
    five_chambers.SOLO_DIG_RATE = 4
    five_chambers.BOSS_HP       = 20
    five_chambers.BOSS_DMG      = 1
end
