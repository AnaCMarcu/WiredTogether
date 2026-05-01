-- config.lua: all tunable constants for Five Chambers.
-- Change NUM_AGENTS to run with more or fewer agents; all geometry
-- and wiring is derived from it at load time.

five_chambers.NUM_AGENTS = 3

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

-- Chamber 1 bounds (solo learning, fixed 12×12)
five_chambers.CH1 = { x0=0, x1=11, z0=0, z1=11 }

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
five_chambers.DOOR1_X = 6

-- Chamber 2 bounds (anvil coop, fixed 10×10)
five_chambers.CH2 = { x0=2, x1=11, z0=13, z1=22 }

-- Door 2: opens 20 steps after 6th anvil break
five_chambers.DOOR2_POS    = { x=6, z=23 }
five_chambers.DOOR2_DELAY  = 20

-- Chamber 3 (switch puzzle) — width scales with NUM_AGENTS
-- Width = 4*N+1 blocks; X: 0..(4N)
-- Cells: Z:25–27; communal room: Z:29–37; north wall at Z=38
five_chambers.CH3_Z0           = 24
five_chambers.CH3_CELL_Z0      = 25
five_chambers.CH3_CELL_Z1      = 27
five_chambers.CH3_FRONT_WALL_Z = 28
five_chambers.CH3_COMMUNAL_Z0  = 29
five_chambers.CH3_COMMUNAL_Z1  = 37
five_chambers.CH3_NORTH_WALL_Z = 38
five_chambers.DOOR3_X          = 6

-- Chamber 4 (combat, fixed 7×7)
five_chambers.CH4     = { x0=3, x1=9, z0=40, z1=46 }
five_chambers.DOOR4_POS = { x=6, z=47 }

-- Chamber 5 (boss, fixed 5×5)
five_chambers.CH5 = { x0=4, x1=8, z0=48, z1=52 }

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
