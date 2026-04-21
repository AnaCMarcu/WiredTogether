-- world_gen.lua: chamber geometry and resource placement.
-- Each chamber is a fully enclosed bedrock box (floor + ceiling + 4 walls).
-- Doors start as bedrock and are replaced with air by doors.lua when unlocked.
-- A chamber whose CHAMBERS[n].enabled = false is left as void.

-- Fills a rectangular prism entirely with one node type.
local function fill_box(x0, y0, z0, x1, y1, z1, node_name)
    for x = x0, x1 do
        for y = y0, y1 do
            for z = z0, z1 do
                minetest.set_node({x=x, y=y, z=z}, {name=node_name})
            end
        end
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
            minetest.set_node({x=x, y=y0, z=z}, {name="mcl_core:dirt_with_grass"})
        end
    end

    -- 4. Open Door 1 in north wall (Z=z1=11) at X=door_x, Y=11–12.
    --    1×2 opening: 1 block wide, 2 blocks tall (player height ~1.75).
    minetest.set_node({x=door_x, y=y0+1, z=c.z1}, {name="air"})  -- y=11
    minetest.set_node({x=door_x, y=y0+2, z=c.z1}, {name="air"})  -- y=12
    -- Floor at door position: leave as dirt_with_grass for walkability.
    minetest.set_node({x=door_x, y=y0, z=c.z1}, {name="mcl_core:dirt_with_grass"})

    -- 5. Place trees (trunk + simple leaf crown) at plan §2.3 positions.
    --    All positions are interior and verified clear of walls.
    for _, tp in ipairs(five_chambers.CH1_TREE_POSITIONS) do
        local tx, tz = tp.x, tp.z
        -- Two-block trunk
        minetest.set_node({x=tx, y=y0+1, z=tz}, {name="mcl_core:tree"})
        minetest.set_node({x=tx, y=y0+2, z=tz}, {name="mcl_core:tree"})
        -- Simple cross-shaped leaf crown at Y=13 — clipped to interior.
        local leaf_offsets = {{0,0},{-1,0},{1,0},{0,-1},{0,1}}
        for _, off in ipairs(leaf_offsets) do
            local lx = tx + off[1]
            local lz = tz + off[2]
            if lx >= c.x0+1 and lx <= c.x1-1
               and lz >= c.z0+1 and lz <= c.z1-1 then
                minetest.set_node({x=lx, y=y0+3, z=lz},
                    {name="mcl_core:leaves"})
            end
        end
    end

    -- 6. Place stone blocks (single solid block at Y=11).
    for _, sp in ipairs(five_chambers.CH1_STONE_POSITIONS) do
        minetest.set_node({x=sp.x, y=y0+1, z=sp.z}, {name="mcl_core:stone"})
    end

    minetest.log("action", "[five_chambers] Chamber 1 built.")
end

local function build_chamber_2()
    if not five_chambers.CHAMBERS[2].enabled then return end
    -- D3: 10×10 bedrock shell; 2*NUM_AGENTS anvils in two rows.
end

local function build_chamber_3()
    if not five_chambers.CHAMBERS[3].enabled then return end
    -- D5: N cells + inter-cell bedrock + communal room.
end

local function build_chamber_4()
    if not five_chambers.CHAMBERS[4].enabled then return end
    -- D6: 7×7 bedrock shell; NUM_AGENTS weak zombies spawned on entry.
end

local function build_chamber_5()
    if not five_chambers.CHAMBERS[5].enabled then return end
    -- D7: 5×5 bedrock shell; boss spawned on entry.
end

-- Public entry-point called from init.lua's on_mods_loaded callback.
function five_chambers.build_all_chambers()
    build_chamber_1()
    build_chamber_2()
    build_chamber_3()
    build_chamber_4()
    build_chamber_5()
end
