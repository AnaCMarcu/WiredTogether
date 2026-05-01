-- world_gen.lua: chamber geometry and resource placement.
-- Each chamber is a fully enclosed bedrock box (floor + ceiling + 4 walls).
-- Doors start as bedrock and are replaced with air by doors.lua when unlocked.
-- A chamber whose CHAMBERS[n].enabled = false is left as void.

-- API compatibility check — fail fast with a clear message.
local required_apis = {
    "swap_node", "load_area", "get_node", "add_entity", "get_worldpath",
    "get_voxel_manip", "get_content_id",
}
for _, name in ipairs(required_apis) do
    if not minetest[name] then
        error("[five_chambers] world_gen.lua: required API minetest." .. name .. " is nil. " ..
              "This Luanti build is too old or missing required functions.")
    end
end

-- Fill a rectangular prism using VoxelManip — writes directly to map data
-- and bypasses ALL per-node callbacks. This is required because mcl_observers
-- monkey-patches both set_node and swap_node, and its hook calls get_node()
-- on unloaded map areas during on_mods_loaded, which crashes with
-- "bad argument #1 to 'old_get_name_from_content_id' (number expected, got nil)".
local function fill_box(x0, y0, z0, x1, y1, z1, node_name)
    local pos1 = {x=x0, y=y0, z=z0}
    local pos2 = {x=x1, y=y1, z=z1}
    local cid  = minetest.get_content_id(node_name)

    -- get_voxel_manip(pos1, pos2) reads the area in one call (documented form).
    local vm = minetest.get_voxel_manip(pos1, pos2)
    local emin, emax = vm:get_emerged_area()
    local data = vm:get_data()
    local va = VoxelArea:new{MinEdge=emin, MaxEdge=emax}

    for z = z0, z1 do
        for y = y0, y1 do
            for x = x0, x1 do
                data[va:index(x, y, z)] = cid
            end
        end
    end

    vm:set_data(data)
    vm:write_to_map(true)  -- true = update lighting
end

-- Single-node placement that bypasses mcl_observers' swap_node hook by
-- writing through VoxelManip. Drop-in replacement for place_node:
-- accepts the same (pos, {name=...}) signature.
local function place_node(pos, node)
    local cid = minetest.get_content_id(node.name)
    local vm = minetest.get_voxel_manip(pos, pos)
    local emin, emax = vm:get_emerged_area()
    local data = vm:get_data()
    local va = VoxelArea:new{MinEdge=emin, MaxEdge=emax}
    data[va:index(pos.x, pos.y, pos.z)] = cid
    vm:set_data(data)
    vm:write_to_map(true)
end

-- Build Chamber 1: 12×12 solo-learning room (plan §2, §2.3).
-- Floor Y=10, ceiling Y=15, walls bedrock. Interior is dirt_with_grass
-- floor + air. Door 1 (always open) is a 2-block gap in the north wall
-- at X=DOOR1_X, Y=11–12.
local function build_chamber_1()
    if not five_chambers.CHAMBERS[1].enabled then return end

    local c  = five_chambers.CH1
    local y0 = five_chambers.FLOOR_Y      -- 10
    local y1 = five_chambers.CEIL_Y       -- 15
    local wall = five_chambers.WALL_NODE
    local door_x = five_chambers.DOOR1_X  -- 6

    -- Force-load map chunks so set_node calls succeed on first run.
    minetest.load_area({x=c.x0, y=y0-1, z=c.z0}, {x=c.x1, y=y1+1, z=c.z1+2})

    -- 1. Fill entire volume with bedrock (shell + interior overwrite).
    fill_box(c.x0, y0, c.z0, c.x1, y1, c.z1, wall)

    -- 2. Carve interior to air (Y:11–14, X:1–10, Z:1–10).
    fill_box(c.x0+1, y0+1, c.z0+1, c.x1-1, y1-1, c.z1-1, "air")

    -- 3. Replace interior floor with grass.
    for x = c.x0+1, c.x1-1 do
        for z = c.z0+1, c.z1-1 do
            place_node({x=x, y=y0, z=z}, {name="mcl_core:dirt_with_grass"})
        end
    end

    -- 4. Open Door 1 in north wall (Z=z1=11) at X=door_x, Y=11–12.
    --    1×2 opening: 1 block wide, 2 blocks tall (player height ~1.75).
    place_node({x=door_x, y=y0+1, z=c.z1}, {name="air"})  -- y=11
    place_node({x=door_x, y=y0+2, z=c.z1}, {name="air"})  -- y=12
    -- Floor at door position: leave as dirt_with_grass for walkability.
    place_node({x=door_x, y=y0, z=c.z1}, {name="mcl_core:dirt_with_grass"})

    -- 5. Place trees (trunk + simple leaf crown) at plan §2.3 positions.
    --    All positions are interior and verified clear of walls.
    for _, tp in ipairs(five_chambers.CH1_TREE_POSITIONS) do
        local tx, tz = tp.x, tp.z
        -- Two-block trunk
        place_node({x=tx, y=y0+1, z=tz}, {name="mcl_core:tree"})
        place_node({x=tx, y=y0+2, z=tz}, {name="mcl_core:tree"})
        -- Simple cross-shaped leaf crown at Y=13 — clipped to interior.
        local leaf_offsets = {{0,0},{-1,0},{1,0},{0,-1},{0,1}}
        for _, off in ipairs(leaf_offsets) do
            local lx = tx + off[1]
            local lz = tz + off[2]
            if lx >= c.x0+1 and lx <= c.x1-1
               and lz >= c.z0+1 and lz <= c.z1-1 then
                place_node({x=lx, y=y0+3, z=lz},
                    {name="mcl_core:leaves"})
            end
        end
    end

    -- 6. Place stone blocks (single solid block at Y=11).
    for _, sp in ipairs(five_chambers.CH1_STONE_POSITIONS) do
        place_node({x=sp.x, y=y0+1, z=sp.z}, {name="mcl_core:stone"})
    end

    minetest.log("action", "[five_chambers] Chamber 1 built.")
end

local function build_chamber_2()
    if not five_chambers.CHAMBERS[2].enabled then return end

    local c    = five_chambers.CH2         -- {x0=2,x1=11,z0=13,z1=22}
    local y0   = five_chambers.FLOOR_Y     -- 10
    local y1   = five_chambers.CEIL_Y      -- 15
    local wall = five_chambers.WALL_NODE
    local N    = five_chambers.NUM_AGENTS

    minetest.load_area({x=c.x0, y=y0-1, z=c.z0-1}, {x=c.x1, y=y1+1, z=c.z1+2})

    -- 1. Fill entire Ch2 volume with bedrock.
    fill_box(c.x0, y0, c.z0, c.x1, y1, c.z1, wall)

    -- 2. Carve interior to air.
    fill_box(c.x0+1, y0+1, c.z0+1, c.x1-1, y1-1, c.z1-1, "air")

    -- 3. Grass floor.
    for x = c.x0+1, c.x1-1 do
        for z = c.z0+1, c.z1-1 do
            place_node({x=x, y=y0, z=z}, {name="mcl_core:dirt_with_grass"})
        end
    end

    -- 4. South entrance (aligns with Door 1 gap at x=DOOR1_X=6, z=CH1.z1=11).
    --    Ch2 south wall is at z=CH2.z0=13. Open 2-block tall gap at x=6.
    local door_x = five_chambers.DOOR1_X
    place_node({x=door_x, y=y0+1, z=c.z0}, {name="air"})
    place_node({x=door_x, y=y0+2, z=c.z0}, {name="air"})
    place_node({x=door_x, y=y0,   z=c.z0}, {name="mcl_core:dirt_with_grass"})

    -- 5. Corridor floor at z=12 (gap between Ch1 z=11 and Ch2 z=13).
    place_node({x=door_x, y=y0, z=12}, {name="mcl_core:dirt_with_grass"})

    -- 6. North wall exit at x=door_x, z=CH2.z1=22 (towards Door 2 at z=23).
    place_node({x=door_x, y=y0+1, z=c.z1}, {name="air"})
    place_node({x=door_x, y=y0+2, z=c.z1}, {name="air"})
    place_node({x=door_x, y=y0,   z=c.z1}, {name="mcl_core:dirt_with_grass"})

    -- 7. Door 2: single bedrock block at DOOR2_POS (z=23) — opened later by doors.lua.
    local d2 = five_chambers.DOOR2_POS
    place_node({x=d2.x, y=y0+1, z=d2.z}, {name=wall})
    place_node({x=d2.x, y=y0+2, z=d2.z}, {name=wall})
    -- Floor tile so agents can stand in front of it.
    place_node({x=d2.x, y=y0, z=d2.z}, {name="mcl_core:dirt_with_grass"})

    -- 8. Place anvils: Row A (z=z0+2=15) and Row B (z=z0+5=18).
    --    x = x0+1 + i*3 = 3, 6, 9 for N=3.
    for i = 0, N - 1 do
        local ax = c.x0 + 1 + (i * 3)
        place_node({x=ax, y=y0+1, z=c.z0+2}, {name="five_chambers:anvil"})
        place_node({x=ax, y=y0+1, z=c.z0+5}, {name="five_chambers:anvil"})
    end

    minetest.log("action", "[five_chambers] Chamber 2 built.")
end

local function build_chamber_3()
    if not five_chambers.CHAMBERS[3].enabled then return end

    local N    = five_chambers.NUM_AGENTS
    local y0   = five_chambers.FLOOR_Y               -- 10
    local y1   = five_chambers.CEIL_Y                -- 15
    local wall = five_chambers.WALL_NODE
    local x0   = 0
    local x1   = 4 * N                               -- 12 for N=3

    local z0      = five_chambers.CH3_Z0             -- 24
    local z1      = five_chambers.CH3_NORTH_WALL_Z   -- 38
    local cell_z0 = five_chambers.CH3_CELL_Z0        -- 25
    local cell_z1 = five_chambers.CH3_CELL_Z1        -- 27
    local comm_z0 = five_chambers.CH3_COMMUNAL_Z0    -- 29
    local comm_z1 = five_chambers.CH3_COMMUNAL_Z1    -- 37

    minetest.load_area({x=x0, y=y0-1, z=z0}, {x=x1, y=y1+1, z=z1+1})

    -- 1. Fill entire Ch3 volume with bedrock.
    --    This places south back wall (z=24), inter-cell walls (x=4,8 for N=3),
    --    front wall (z=28, with locked cell-door positions), north wall (z=38).
    fill_box(x0, y0, z0, x1, y1, z1, wall)

    -- 2. Carve N isolation cells.
    --    Cell i interior: x = 1+4i .. 3+4i, z = cell_z0 .. cell_z1.
    --    D5 places switch nodes at (cell_x_center(i), y0+1, cell_z0).
    for i = 0, N - 1 do
        local cx0 = 1 + i * 4
        local cx1 = cx0 + 2
        fill_box(cx0, y0+1, cell_z0, cx1, y1-1, cell_z1, "air")
        for x = cx0, cx1 do
            for z = cell_z0, cell_z1 do
                place_node({x=x, y=y0, z=z}, {name="mcl_core:dirt_with_grass"})
            end
        end
    end
    -- 2b. Place switch nodes on the south-facing wall of each cell (z=cell_z0).
    --     five_chambers:switch is registered by switches.lua (already dofile'd).
    for i = 0, N - 1 do
        local sx = five_chambers.cell_x_center(i)
        place_node({x=sx, y=y0+1, z=cell_z0}, {name="five_chambers:switch"})
    end
    -- z=28 (front wall) stays full bedrock; open_cell_door() clears doors there.

    -- 3. Carve communal room (interior x=1..x1-1, z=comm_z0..comm_z1).
    fill_box(x0+1, y0+1, comm_z0, x1-1, y1-1, comm_z1, "air")
    for x = x0+1, x1-1 do
        for z = comm_z0, comm_z1 do
            place_node({x=x, y=y0, z=z}, {name="mcl_core:dirt_with_grass"})
        end
    end

    -- 4. Corridor floor at z=39 (gap between Ch3 north wall and Ch4 south wall).
    place_node({x=five_chambers.DOOR3_X, y=y0, z=z1+1},
                      {name="mcl_core:dirt_with_grass"})

    minetest.log("action", "[five_chambers] Chamber 3 built.")
end

local function build_chamber_4()
    if not five_chambers.CHAMBERS[4].enabled then return end

    local c    = five_chambers.CH4      -- {x0=3, x1=9, z0=40, z1=46}
    local y0   = five_chambers.FLOOR_Y  -- 10
    local y1   = five_chambers.CEIL_Y   -- 15
    local wall = five_chambers.WALL_NODE
    local dx   = five_chambers.DOOR3_X  -- 6

    minetest.load_area({x=c.x0, y=y0-1, z=c.z0}, {x=c.x1, y=y1+1, z=c.z1+2})

    -- 1. Fill entire Ch4 volume with bedrock.
    fill_box(c.x0, y0, c.z0, c.x1, y1, c.z1, wall)

    -- 2. Carve interior.
    fill_box(c.x0+1, y0+1, c.z0+1, c.x1-1, y1-1, c.z1-1, "air")

    -- 3. Grass floor.
    for x = c.x0+1, c.x1-1 do
        for z = c.z0+1, c.z1-1 do
            place_node({x=x, y=y0, z=z}, {name="mcl_core:dirt_with_grass"})
        end
    end

    -- 4. South entrance at x=6, z=40 (aligns with Door 3 corridor at z=39).
    place_node({x=dx, y=y0+1, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0+2, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0,   z=c.z0}, {name="mcl_core:dirt_with_grass"})

    -- 5. North passage at x=6, z=46 (exits toward Door 4 at z=47).
    place_node({x=dx, y=y0+1, z=c.z1}, {name="air"})
    place_node({x=dx, y=y0+2, z=c.z1}, {name="air"})
    place_node({x=dx, y=y0,   z=c.z1}, {name="mcl_core:dirt_with_grass"})

    -- 6. Door 4: bedrock at DOOR4_POS z=47; stays locked until all Ch4 mobs die.
    local d4 = five_chambers.DOOR4_POS  -- {x=6, z=47}
    place_node({x=d4.x, y=y0+1, z=d4.z}, {name=wall})
    place_node({x=d4.x, y=y0+2, z=d4.z}, {name=wall})
    place_node({x=d4.x, y=y0,   z=d4.z}, {name="mcl_core:dirt_with_grass"})

    minetest.log("action", "[five_chambers] Chamber 4 built.")
end

local function build_chamber_5()
    if not five_chambers.CHAMBERS[5].enabled then return end

    local c    = five_chambers.CH5      -- {x0=4, x1=8, z0=48, z1=52}
    local y0   = five_chambers.FLOOR_Y  -- 10
    local y1   = five_chambers.CEIL_Y   -- 15
    local wall = five_chambers.WALL_NODE
    local dx   = five_chambers.DOOR3_X  -- 6  (shared corridor x)

    minetest.load_area({x=c.x0, y=y0-1, z=c.z0}, {x=c.x1, y=y1+1, z=c.z1})

    -- 1. Fill entire Ch5 volume with bedrock.
    fill_box(c.x0, y0, c.z0, c.x1, y1, c.z1, wall)

    -- 2. Carve interior (x=5..7, z=49..51).
    fill_box(c.x0+1, y0+1, c.z0+1, c.x1-1, y1-1, c.z1-1, "air")

    -- 3. Grass floor.
    for x = c.x0+1, c.x1-1 do
        for z = c.z0+1, c.z1-1 do
            place_node({x=x, y=y0, z=z}, {name="mcl_core:dirt_with_grass"})
        end
    end

    -- 4. South entrance at x=6, z=48 (from Door 4 corridor at z=47).
    place_node({x=dx, y=y0+1, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0+2, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0,   z=c.z0}, {name="mcl_core:dirt_with_grass"})
    -- No exit — episode ends on boss death.

    minetest.log("action", "[five_chambers] Chamber 5 built.")
end

-- Public entry-point called from init.lua's on_mods_loaded callback.
function five_chambers.build_all_chambers()
    build_chamber_1()
    build_chamber_2()
    build_chamber_3()
    build_chamber_4()
    build_chamber_5()
end
