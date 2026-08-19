"""Lua-spec pins for the agent-count scaling suite (N ∈ {2,...,9}).

Same style as test_lua_spec.py: parse the five_chambers Lua sources and
assert the contracts the scaling experiment depends on —

1. config.lua exposes the FC_CH4_MOB_COUNT pin (nil when unset → legacy
   one-zombie-per-agent) and the WT_TEAM_SCALING master switch (false when
   unset), so the Ch4 environment can be held constant across team sizes
   while every legacy suite stays bit-for-bit unchanged.
2. mobs.lua spawns min(CH4_MOB_COUNT or NUM_AGENTS, #positions) zombies.
3. util.lua's generic Ch1 spawn row (N ≠ 3) is gated: legacy z=5 unchanged,
   scaling-mode z=12 collides with no solid resource (the z=5 row puts
   spawns inside the 2-high stone pillars at (3,5)/(7,5) for several N).
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


def _generic_spawn_rows(util_text: str) -> tuple[int, int]:
    """(scaling_row, legacy_row) used by the generic (N != 3) branch of
    ch1_spawn_pos: `local z_row = five_chambers.TEAM_SCALING and <s> or <l>`."""
    fn = util_text[util_text.index("function five_chambers.ch1_spawn_pos"):]
    fn = fn[:fn.index("\nend")]
    m = re.search(
        r"local z_row = five_chambers\.TEAM_SCALING and (\d+) or (\d+)", fn)
    assert m, "generic ch1_spawn_pos row must be gated by TEAM_SCALING"
    return int(m.group(1)), int(m.group(2))


def _generic_spawn_x(i: int, n: int) -> int:
    # Mirrors util.lua: math.floor(1 + frac * 9 + 0.5), frac = i/(N-1).
    frac = 0.5 if n == 1 else i / (n - 1)
    return math.floor(1 + frac * 9 + 0.5)


def test_generic_ch1_spawn_rows_gated(lua_root):
    scaling_row, legacy_row = _generic_spawn_rows(_lua(lua_root, "util.lua"))
    assert scaling_row == 12
    # Legacy pin: the pre-scaling suites (incl. the N=2 pair and N=6
    # transplant runs) must keep their original z=5 row.
    assert legacy_row == 5


@pytest.mark.parametrize("n", [2, 4, 5, 6, 9])
def test_generic_ch1_spawns_avoid_solid_resources(lua_root, n):
    """In scaling mode, no generic spawn tile may hold a tree trunk or a
    2-high stone pillar (both are solid at agent height). Animals are
    entities — not checked. (The legacy z=5 row DOES collide for some N —
    that is the historical behavior the gate preserves.)"""
    config = _lua(lua_root, "config.lua")
    util = _lua(lua_root, "util.lua")
    z_row, _legacy = _generic_spawn_rows(util)

    trees = _positions(config, "CH1_TREE_POSITIONS")
    stones = _positions(config, "CH1_STONE_POSITIONS")
    assert len(trees) == 8 and len(stones) == 8  # parse sanity (plan §2.3)
    solid = set(trees) | set(stones)

    for i in range(n):
        tile = (_generic_spawn_x(i, n), z_row)
        assert tile not in solid, (
            f"N={n}: agent_{i} generic spawn {tile} sits inside a solid "
            f"Ch1 resource")


def test_generic_ch1_spawns_inside_chamber(lua_root):
    """Spawn tiles stay inside Ch1's interior (walls at x/z = 0 and 15)."""
    util = _lua(lua_root, "util.lua")
    for z_row in _generic_spawn_rows(util):
        assert 1 <= z_row <= 14
    for n in (2, 4, 5, 6, 9):
        for i in range(n):
            assert 1 <= _generic_spawn_x(i, n) <= 14
