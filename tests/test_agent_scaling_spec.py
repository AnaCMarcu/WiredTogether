"""Lua-spec pins for the agent-count scaling suite (N ∈ {2,...,9}).

Same style as test_lua_spec.py: parse the five_chambers Lua sources and
assert the contracts the scaling experiment depends on —

1. config.lua exposes the FC_CH4_MOB_COUNT pin (nil when unset → legacy
   one-zombie-per-agent) and the WT_TEAM_SCALING master switch (false when
   unset), so the Ch4 environment can be held constant across team sizes
   while every legacy suite stays bit-for-bit unchanged.
2. mobs.lua spawns min(CH4_MOB_COUNT or NUM_AGENTS, #positions) zombies.
3. util.lua's generic Ch1 spawn row (N ≠ 3) is gated: legacy x∈[1,10]/z=5
   unchanged, while scaling mode uses x∈[2,10]/z=10 and satisfies the three
   criteria CH1_SPAWNS_3 was hand-tuned for — no solid tile underfoot (the
   z=5 row wedges agents inside the stone pillars at (3,5)/(7,5) for
   N=4,5,6,9), ≥2 blocks from every wall, and a breakable within ~3 blocks.
"""

import math
import re

import pytest


def _lua(lua_root, name: str) -> str:
    return (lua_root / name).read_text(encoding="utf-8")


# ── 1. config.lua: FC_CH4_MOB_COUNT pin ──────────────────────────────────

def test_config_ch4_mob_count_env_override(lua_root):
    text = _lua(lua_root, "config.lua")
    assert re.search(
        r'^local _env_ch4_mobs = tonumber\(os and os\.getenv and '
        r'os\.getenv\("FC_CH4_MOB_COUNT"\) or ""\)$',
        text, re.MULTILINE,
    )
    # No integer-literal default: unset must stay nil so mobs.lua falls back
    # to the legacy per-agent count.
    assert re.search(
        r"^five_chambers\.CH4_MOB_COUNT = _env_ch4_mobs$",
        text, re.MULTILINE,
    )


def test_config_team_scaling_master_switch(lua_root):
    """WT_TEAM_SCALING gates the Lua-side scaling tweaks; unset → false so
    legacy suites are bit-for-bit unchanged."""
    text = _lua(lua_root, "config.lua")
    assert re.search(
        r'five_chambers\.TEAM_SCALING =\s*\n?\s*'
        r'\(\(os and os\.getenv and os\.getenv\("WT_TEAM_SCALING"\)\) or ""\)'
        r' == "1"',
        text,
    )


# ── 2. mobs.lua: spawn count honors the pin ──────────────────────────────

def test_spawn_ch4_mobs_uses_pin_with_legacy_fallback(lua_root):
    text = _lua(lua_root, "mobs.lua")
    assert re.search(
        r"local want = five_chambers\.CH4_MOB_COUNT or five_chambers\.NUM_AGENTS",
        text,
    )
    assert re.search(r"math\.min\(want, #CH4_SPAWN_POSITIONS\)", text)


# ── 3. util.lua: generic Ch1 spawn row is collision-free ────────────────

def _positions(config_text: str, table_name: str) -> list[tuple[int, int]]:
    # Tables span multiple lines; grab up to the table's closing brace
    # (first "\n}" after the opening line).
    start = config_text.index(f"five_chambers.{table_name} = {{")
    end = config_text.index("\n}", start)
    block = config_text[start:end]
    return [(int(x), int(z))
            for x, z in re.findall(r"\{x=(\d+),z=(\d+)\}", block)]


def _spawn_branch(util_text: str, scaling: bool) -> tuple[int, int, int]:
    """(x_base, x_span, z_row) for one branch of ch1_spawn_pos's generic
    (N != 3) path: `x = math.floor(<base> + frac * <span> + 0.5)`, `z = <row>`.

    The scaling branch is the one guarded by `if five_chambers.TEAM_SCALING`;
    the legacy branch is the fall-through after it.
    """
    fn = util_text[util_text.index("function five_chambers.ch1_spawn_pos"):]
    guard = fn.index("if five_chambers.TEAM_SCALING then")
    block = fn[guard:fn.index("\n    end", guard)] if scaling else fn[fn.index("\n    end", guard):]
    m = re.search(
        r"x = math\.floor\((\d+) \+ frac \* (\d+) \+ 0\.5\),\s*\n"
        r"\s*y = y,\s*\n\s*z = (\d+),", block)
    assert m, f"generic ch1_spawn_pos {'scaling' if scaling else 'legacy'} branch not parsed"
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _generic_spawn_x(i: int, n: int, base: int, span: int) -> int:
    # Mirrors util.lua: math.floor(base + frac * span + 0.5), frac = i/(N-1).
    frac = 0.5 if n == 1 else i / (n - 1)
    return math.floor(base + frac * span + 0.5)


def test_generic_ch1_spawn_branches_gated(lua_root):
    util = _lua(lua_root, "util.lua")
    # Scaling branch: the resource-adjacent, wall-clear row.
    assert _spawn_branch(util, scaling=True) == (2, 8, 10)
    # Legacy pin: the pre-scaling suites (incl. the N=2 pair-bonding and
    # N=6 transplant runs) must keep their original x∈[1,10], z=5 row.
    assert _spawn_branch(util, scaling=False) == (1, 9, 5)


@pytest.mark.parametrize("n", [2, 4, 5, 6, 9])
def test_generic_ch1_spawns_avoid_solid_resources(lua_root, n):
    """In scaling mode, no generic spawn tile may hold a tree trunk or a
    2-high stone pillar (both are solid at agent height). Animals are
    entities — not checked. (The legacy z=5 row DOES collide for some N —
    that is the historical behavior the gate preserves.)"""
    config = _lua(lua_root, "config.lua")
    base, span, z_row = _spawn_branch(_lua(lua_root, "util.lua"), scaling=True)

    trees = _positions(config, "CH1_TREE_POSITIONS")
    stones = _positions(config, "CH1_STONE_POSITIONS")
    assert len(trees) == 8 and len(stones) == 8  # parse sanity (plan §2.3)
    solid = set(trees) | set(stones)

    for i in range(n):
        tile = (_generic_spawn_x(i, n, base, span), z_row)
        assert tile not in solid, (
            f"N={n}: agent_{i} generic spawn {tile} sits inside a solid "
            f"Ch1 resource")


@pytest.mark.parametrize("n", [2, 4, 5, 6, 9])
def test_generic_ch1_spawns_are_distinct(lua_root, n):
    """Two agents must never share a spawn tile (the engine would have to
    push them apart, and one starts inside the other)."""
    base, span, _ = _spawn_branch(_lua(lua_root, "util.lua"), scaling=True)
    xs = [_generic_spawn_x(i, n, base, span) for i in range(n)]
    assert len(set(xs)) == n, f"N={n}: duplicate spawn columns {xs}"


@pytest.mark.parametrize("n", [2, 4, 5, 6, 9])
def test_generic_ch1_spawns_clear_of_walls(lua_root, n):
    """>=2 blocks from every bedrock wall (walls at x/z = 0 and 15).

    A wall 1 block ahead fills the agent's first frame with grey bedrock,
    which the LLM reliably misreads as mcl_core:stone and Digs — no break,
    no milestone. This is the criterion the earlier z=12 row violated (and
    the N=2 smoke run reported it verbatim).
    """
    base, span, z_row = _spawn_branch(_lua(lua_root, "util.lua"), scaling=True)
    assert 2 <= z_row <= 13
    for i in range(n):
        x = _generic_spawn_x(i, n, base, span)
        assert 2 <= x <= 13, f"N={n}: agent_{i} spawns at x={x}, <2 from a wall"


def _grid_slot(i, x0, x1, z0, z1, blocked):
    """Mirrors util.lua's grid_slot: i-th free tile, z-rows then x-columns."""
    n = 0
    for z in range(z0, z1 + 1):
        for x in range(x0, x1 + 1):
            if (x, z) in blocked:
                continue
            if n == i:
                return (x, z)
            n += 1
    return (x0, z0)


def _chamber_bounds(config_text: str, name: str) -> dict:
    m = re.search(
        rf"five_chambers\.{name}\s*=\s*\{{\s*x0=(-?\d+),\s*x1=(-?\d+),"
        rf"\s*z0=(-?\d+),\s*z1=(-?\d+)\s*\}}", config_text)
    assert m, f"{name} bounds not parsed"
    return dict(zip(("x0", "x1", "z0", "z1"), map(int, m.groups())))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 9])
def test_rescue_teleports_never_stack_agents(lua_root, n):
    """The Ch1→Ch2 / Ch3→Ch4 / Ch4→Ch5 rescue teleports fire unconditionally
    once per chamber per episode. Two agents materialising on ONE tile makes
    the engine resolve the overlap by push-out — the documented cause of
    agents being launched into the air after a reposition.

    The legacy single-row spread collapses onto duplicate columns once N
    exceeds the (narrow) chamber width: at N=9 Ch2/Ch5 stacked four pairs
    each and Ch4 stacked two. Scaling mode uses a grid instead.
    """
    config = _lua(lua_root, "config.lua")
    ch2 = _chamber_bounds(config, "CH2")
    ch4 = _chamber_bounds(config, "CH4")
    ch5 = _chamber_bounds(config, "CH5")

    # Ch2: anvil pedestals are solid at agent height (FLOOR_Y+1).
    anvil_x = (ch2["x0"] + ch2["x1"]) // 2
    ch2_blocked = {(anvil_x, ch2["z0"] + 2), (anvil_x, ch2["z0"] + 5)}
    boss_tile = (6, (ch5["z0"] + ch5["z1"]) // 2)

    for label, (c, blocked) in {
        "ch2": (ch2, ch2_blocked),
        "ch4": (ch4, set()),
        "ch5": (ch5, {boss_tile}),
    }.items():
        tiles = [
            _grid_slot(i, c["x0"] + 1, c["x1"] - 1, c["z0"] + 2, c["z0"] + 4,
                       blocked)
            for i in range(n)
        ]
        assert len(set(tiles)) == n, (
            f"N={n} {label}: rescue teleport stacks agents — {tiles}")
        for t in tiles:
            assert t not in blocked, (
                f"N={n} {label}: agent teleported onto solid tile {t}")
            assert c["x0"] < t[0] < c["x1"], f"N={n} {label}: {t} outside x"
            assert c["z0"] < t[1] < c["z1"], f"N={n} {label}: {t} outside z"


def test_rescue_teleports_gated_by_team_scaling(lua_root):
    """Each fallback keeps its legacy single-row branch behind the gate, so
    pre-scaling suites reproduce exactly."""
    util = _lua(lua_root, "util.lua")
    for fn in ("ch2_fallback_spawn_pos", "ch4_fallback_spawn_pos",
               "ch5_fallback_spawn_pos"):
        body = util[util.index(f"function five_chambers.{fn}"):]
        body = body[:body.index("\nend")]
        assert "five_chambers.TEAM_SCALING" in body, f"{fn} not gated"
        assert "grid_slot(" in body, f"{fn} missing grid placement"
        # Legacy formula still present for the switch-off path.
        assert "math.floor(x_min + frac * (x_max - x_min) + 0.5)" in body, (
            f"{fn} lost its legacy linear spread")


@pytest.mark.parametrize("n", [2, 4, 5, 6, 9])
def test_generic_ch1_spawns_near_a_breakable(lua_root, n):
    """Every agent starts within ~3.5 blocks of a tree or stone, so the
    first 'scan the room' actually has a dig target in it (the third
    CH1_SPAWNS_3 criterion)."""
    config = _lua(lua_root, "config.lua")
    base, span, z_row = _spawn_branch(_lua(lua_root, "util.lua"), scaling=True)
    solid = set(_positions(config, "CH1_TREE_POSITIONS")) \
        | set(_positions(config, "CH1_STONE_POSITIONS"))

    for i in range(n):
        x = _generic_spawn_x(i, n, base, span)
        nearest = min(math.dist((x, z_row), s) for s in solid)
        assert nearest <= 3.5, (
            f"N={n}: agent_{i} at ({x},{z_row}) has no breakable within "
            f"3.5 blocks (nearest {nearest:.1f})")
