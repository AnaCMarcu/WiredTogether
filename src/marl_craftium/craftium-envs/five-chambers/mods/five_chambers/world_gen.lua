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

-- Build an enclosed corridor segment at (x, z), full height, `width` blocks wide
-- (centered on x). Floor + ceiling + both side walls are bedrock; the walking
-- columns (x-half..x+half, y0+1..y1-1, z) are air so the player can pass.
-- Default width=1 (single-block tube, used for locked doors). For always-open
-- corridors use width=3 so LLM agents don't have to align perfectly with x.
local function build_corridor(x, z, y0, y1, wall, width)
    width = width or 1
    local half = math.floor(width / 2)
    local x_lo = x - half
    local x_hi = x + half
    -- Floor + ceiling across the full corridor span (including side-wall columns).
    for cx = x_lo - 1, x_hi + 1 do
        place_node({x=cx, y=y0, z=z}, {name=wall})
        place_node({x=cx, y=y1, z=z}, {name=wall})
    end
    -- Side walls + air interior.
    for y = y0+1, y1-1 do
        place_node({x=x_lo - 1, y=y, z=z}, {name=wall})
        place_node({x=x_hi + 1, y=y, z=z}, {name=wall})
        for cx = x_lo, x_hi do
            place_node({x=cx, y=y, z=z}, {name="air"})
        end
    end
end

-- Carve a `width`-wide, 2-block-tall doorway in a wall plane at (x, z).
-- Centered on x (so width=3 opens x-1, x, x+1). Floor below stays bedrock.
local function carve_doorway(x, z, y0, width)
    width = width or 1
    local half = math.floor(width / 2)
    for cx = x - half, x + half do
        place_node({x=cx, y=y0+1, z=z}, {name="air"})
        place_node({x=cx, y=y0+2, z=z}, {name="air"})
        place_node({x=cx, y=y0,   z=z}, {name="mcl_core:bedrock"})
    end
end

-- Add evenly-spaced light sources at ceiling level so rooms aren't pitch-dark.
-- Replaces 4 corner-ish ceiling tiles + 1 centre tile with glowstone (light 14).
local function add_ceiling_lights(x0, x1, z0, z1, y_ceiling)
    local light = "mcl_nether:glowstone"
    local cx = math.floor((x0 + x1) / 2)
    local cz = math.floor((z0 + z1) / 2)
    -- Inset by 2 from walls so the light reaches the floor centres of corner regions.
    local positions = {
        {x=x0 + 2, z=z0 + 2},
        {x=x1 - 2, z=z0 + 2},
        {x=x0 + 2, z=z1 - 2},
        {x=x1 - 2, z=z1 - 2},
        {x=cx,     z=cz    },
    }
    for _, p in ipairs(positions) do
        place_node({x=p.x, y=y_ceiling, z=p.z}, {name=light})
    end
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
            place_node({x=x, y=y0, z=z}, {name="mcl_core:bedrock"})
        end
    end

    -- 4. Open Door 1 in north wall (Z=c.z1) at X=door_x.
    --    3-wide × 2-tall opening so an LLM agent doesn't have to align
    --    perfectly with door_x to pass through.
    carve_doorway(door_x, c.z1, y0, 3)

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

    -- 7. Ceiling lights so the room isn't pitch-dark.
    add_ceiling_lights(c.x0, c.x1, c.z0, c.z1, y1)

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
            place_node({x=x, y=y0, z=z}, {name="mcl_core:bedrock"})
        end
    end

    -- 4. South entrance (aligns with Ch1 north opening at x=DOOR1_X).
    local door_x = five_chambers.DOOR1_X
    carve_doorway(door_x, c.z0, y0, 3)

    -- 5. Enclosed 3-wide always-open corridor between Ch1 (z=CH1.z1) and Ch2 (z=c.z0).
    build_corridor(door_x, c.z0 - 1, y0, y1, wall, 3)

    -- 6. North wall exit at x=door_x, z=c.z1 (towards Door 2).
    carve_doorway(door_x, c.z1, y0, 1)

    -- 7. Door 2: enclosed corridor with the y0+1..y0+2 column starting as
    --    bedrock (= closed). doors.lua replaces those two blocks with air to
    --    open. Floor / ceiling / side walls / above-door blocks stay bedrock
    --    so agents can't slip past the door or jump over it.
    local d2 = five_chambers.DOOR2_POS
    build_corridor(d2.x, d2.z, y0, y1, wall)
    -- Re-close the door column over the air the corridor builder put there.
    place_node({x=d2.x, y=y0+1, z=d2.z}, {name=wall})
    place_node({x=d2.x, y=y0+2, z=d2.z}, {name=wall})
    for y = y0+3, y1-1 do
        place_node({x=d2.x, y=y, z=d2.z}, {name=wall})  -- block jumping over
    end

    -- 8. Place anvils: Row A (z=z0+2=15) and Row B (z=z0+5=18).
    --    x = x0+1 + i*3 = 3, 6, 9 for N=3.
    for i = 0, N - 1 do
        local ax = c.x0 + 1 + (i * 3)
        place_node({x=ax, y=y0+1, z=c.z0+2}, {name="five_chambers:anvil"})
        place_node({x=ax, y=y0+1, z=c.z0+5}, {name="five_chambers:anvil"})
    end

    add_ceiling_lights(c.x0, c.x1, c.z0, c.z1, y1)

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
                place_node({x=x, y=y0, z=z}, {name="mcl_core:bedrock"})
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
            place_node({x=x, y=y0, z=z}, {name="mcl_core:bedrock"})
        end
    end

    -- 4. North wall opening + enclosed corridor between Ch3 (z=z1) and Ch4.
    --    Without the opening at z=z1 the communal room is sealed northward.
    local dx3 = five_chambers.DOOR3_X
    place_node({x=dx3, y=y0+1, z=z1}, {name="air"})
    place_node({x=dx3, y=y0+2, z=z1}, {name="air"})
    build_corridor(dx3, z1 + 1, y0, y1, wall)

    -- 5. Lighting: one glowstone per cell ceiling + spread across communal ceiling.
    for i = 0, N - 1 do
        local cx = five_chambers.cell_x_center(i)
        local mz = math.floor((cell_z0 + cell_z1) / 2)
        place_node({x=cx, y=y1, z=mz}, {name="mcl_nether:glowstone"})
    end
    add_ceiling_lights(x0, x1, comm_z0, comm_z1, y1)

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
            place_node({x=x, y=y0, z=z}, {name="mcl_core:bedrock"})
        end
    end

    -- 4. South entrance at x=6, z=40 (aligns with Door 3 corridor at z=39).
    place_node({x=dx, y=y0+1, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0+2, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0,   z=c.z0}, {name="mcl_core:bedrock"})

    -- 5. North passage at x=6, z=46 (exits toward Door 4 at z=47).
    place_node({x=dx, y=y0+1, z=c.z1}, {name="air"})
    place_node({x=dx, y=y0+2, z=c.z1}, {name="air"})
    place_node({x=dx, y=y0,   z=c.z1}, {name="mcl_core:bedrock"})

    -- 6. Door 4: enclosed corridor with the y0+1..y0+2 column starting as
    --    bedrock (= closed). doors.lua replaces those two blocks with air to
    --    open. Floor / ceiling / side walls / above-door blocks stay bedrock
    --    so agents can't slip past the door or jump over it.
    local d4 = five_chambers.DOOR4_POS
    build_corridor(d4.x, d4.z, y0, y1, wall)
    place_node({x=d4.x, y=y0+1, z=d4.z}, {name=wall})
    place_node({x=d4.x, y=y0+2, z=d4.z}, {name=wall})
    for y = y0+3, y1-1 do
        place_node({x=d4.x, y=y, z=d4.z}, {name=wall})  -- block jumping over
    end

    add_ceiling_lights(c.x0, c.x1, c.z0, c.z1, y1)

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
            place_node({x=x, y=y0, z=z}, {name="mcl_core:bedrock"})
        end
    end

    -- 4. South entrance at x=6, z=48 (from Door 4 corridor at z=47).
    place_node({x=dx, y=y0+1, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0+2, z=c.z0}, {name="air"})
    place_node({x=dx, y=y0,   z=c.z0}, {name="mcl_core:bedrock"})
    -- No exit — episode ends on boss death.

    add_ceiling_lights(c.x0, c.x1, c.z0, c.z1, y1)

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
