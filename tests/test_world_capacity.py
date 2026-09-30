"""WIRE world capacity for large teams (agent-count scaling past N=9).

Pure text parsing plus a Python model of util.lua's grid_slot — no Lua
runtime. Pins:
  * every chamber gives each agent its own spawn tile up to N=33 under
    WT_TEAM_SCALING, and N<=9 (every existing suite) keeps its tiles;
  * cell/switch labels stay distinct past 26 agents and agree between Lua
    (wire.cell_label) and Python (chamber_facts._cell_letter);
  * the 34 extra Ch4 zombie spawns sit inside Ch4, clear of the agents'
    landing rows, and the Python mirror's cap matches the Lua table.
"""

import re

import pytest

from mindforge.agent_modules import chamber_facts
from mindforge.agent_modules.chamber_facts import _cell_letter, ch4_zombie_count

# Chamber rectangles from config.lua (walls on the bounds).
CH1 = dict(x0=0, x1=15, z0=0, z1=15)
CH2 = dict(x0=2, x1=10, z0=17, z1=25)
CH4 = dict(x0=1, x1=11, z0=47, z1=57)
CH5 = dict(x0=2, x1=10, z0=59, z1=67)
ANVIL_TILES = [(6, CH2["z0"] + 2), (6, CH2["z0"] + 5)]      # anvil.lua pedestals
BOSS_TILE = [(6, (CH5["z0"] + CH5["z1"]) // 2)]

MAX_N = 33   # the smallest capacity below (Ch2)


def grid_slot(i, x_min, x_max, z_min, z_max, blocked=()):
    """util.lua grid_slot: the i-th free tile scanning z rows, then x."""
    n_free = 0
    for z in range(z_min, z_max + 1):
        for x in range(x_min, x_max + 1):
            if (x, z) in blocked:
                continue
            if n_free == i:
                return (x, z)
            n_free += 1
    return None   # Lua falls back to the first tile; here: out of capacity


def ch1(i, n):
    if n > 9:
        return grid_slot(i, 2, 13, 10, 13)
    frac = 0.5 if n == 1 else i / (n - 1)
    return (int(2 + frac * 8 + 0.5), 10)


def ch2(i, n):
    z_max = CH2["z1"] - 2 if n > 20 else CH2["z0"] + 4
    return grid_slot(i, CH2["x0"] + 1, CH2["x1"] - 1, CH2["z0"] + 2, z_max, ANVIL_TILES)


def ch4(i, n):
    z_max = CH4["z0"] + 5 if n > 27 else CH4["z0"] + 4
    return grid_slot(i, CH4["x0"] + 1, CH4["x1"] - 1, CH4["z0"] + 2, z_max)


def ch5(i, n):
    z_max = CH5["z1"] - 2 if n > 20 else CH5["z0"] + 4
    return grid_slot(i, CH5["x0"] + 1, CH5["x1"] - 1, CH5["z0"] + 2, z_max, BOSS_TILE)


def _util(lua_root) -> str:
    return (lua_root / "util.lua").read_text(encoding="utf-8")


def test_util_lua_thresholds_match_the_model(lua_root):
    """The model above encodes these exact branches of util.lua."""
    text = _util(lua_root)
    assert "if wire.TEAM_SCALING and N > 9 then" in text
    assert "grid_slot(i, 2, 13, 10, 13, nil)" in text
    assert text.count("(N > 20) and (c.z1 - 2) or (c.z0 + 4)") == 2   # Ch2, Ch5
    assert "(N > 27) and (c.z0 + 5) or (c.z0 + 4)" in text            # Ch4


@pytest.mark.parametrize("room", [ch1, ch2, ch4, ch5])
@pytest.mark.parametrize("n", list(range(2, MAX_N + 1)))
def test_every_agent_gets_its_own_tile(room, n):
    tiles = [room(i, n) for i in range(n)]
    assert None not in tiles, f"{room.__name__}: out of tiles at N={n}"
    assert len(set(tiles)) == n, f"{room.__name__}: shared tile at N={n}"


@pytest.mark.parametrize("room,interior", [
    (ch1, (1, 14, 1, 14)), (ch2, (3, 9, 18, 24)),
    (ch4, (2, 10, 48, 56)), (ch5, (3, 9, 60, 66)),
])
def test_tiles_stay_inside_the_room(room, interior):
    x_lo, x_hi, z_lo, z_hi = interior
    for i in range(MAX_N):
        x, z = room(i, MAX_N)
        assert x_lo <= x <= x_hi and z_lo <= z <= z_hi, (room.__name__, i, x, z)


def test_existing_team_sizes_keep_their_tiles():
    """N<=20 (Ch2/Ch5) and N<=27 (Ch4) use the original 3-row band; the
    first agents keep the same tiles when the band is extended."""
    for n in range(2, 21):
        for room in (ch2, ch5):
            band = [grid_slot(i, *_band(room)) for i in range(n)]
            assert [room(i, n) for i in range(n)] == band
    for n in range(21, MAX_N + 1):
        for room in (ch2, ch5):
            assert [room(i, n) for i in range(20)] == [room(i, 20) for i in range(20)]


def _band(room):
    c, blocked = {ch2: (CH2, ANVIL_TILES), ch5: (CH5, BOSS_TILE)}[room]
    return (c["x0"] + 1, c["x1"] - 1, c["z0"] + 2, c["z0"] + 4, blocked)


def test_cell_labels_distinct_and_backward_compatible():
    labels = [_cell_letter(i) for i in range(702)]
    assert labels[:3] == ["A", "B", "C"] and labels[25] == "Z"
    assert labels[26:29] == ["AA", "AB", "AC"] and labels[52] == "BA"
    assert len(set(labels)) == len(labels)


def test_lua_cell_label_matches_python(lua_root):
    text = _util(lua_root)
    assert "function wire.cell_label(i)" in text
    assert "string.char(65 + math.floor(i / 26) - 1) .. string.char(65 + i % 26)" in text
    switches = (lua_root / "switches.lua").read_text(encoding="utf-8")
    assert "wire.cell_label(sw_i)" in switches and "string.char(65" not in switches


def _ch4_extra_positions(lua_root):
    text = (lua_root / "mobs.lua").read_text(encoding="utf-8")
    start = text.index("local CH4_SPAWN_POSITIONS = {")
    block = text[start:text.index("\n}", start)]
    hand = {(int(x), int(z)) for x, z in re.findall(r"\{x=(\d+),\s*y=\d+,\s*z=(\d+)\}", block)}
    assert "for z = 56, 53, -1 do" in text and "for x = 2, 10 do" in text
    return hand, [(x, z) for z in range(56, 52, -1) for x in range(2, 11) if (x, z) not in hand]


def test_ch4_extra_zombies_inside_and_clear_of_agents(lua_root):
    hand, extra = _ch4_extra_positions(lua_root)
    assert len(hand) + len(extra) == chamber_facts.CH4_MAX_ZOMBIES_SCALING == 40
    agent_rows = {ch4(i, MAX_N)[1] for i in range(MAX_N)}
    for x, z in extra:
        assert 2 <= x <= 10 and 48 <= z <= 56
        assert z > max(agent_rows)


def test_zombie_count_cap_follows_team_scaling(monkeypatch):
    monkeypatch.delenv("FC_CH4_MOB_COUNT", raising=False)
    monkeypatch.delenv("WT_TEAM_SCALING", raising=False)
    assert ch4_zombie_count(32) == 6                      # legacy cap unchanged
    monkeypatch.setenv("WT_TEAM_SCALING", "1")
    assert ch4_zombie_count(32) == 32
    assert ch4_zombie_count(50) == 40
    monkeypatch.setenv("FC_CH4_MOB_COUNT", "3")
    assert ch4_zombie_count(32) == 3                      # the suite's pin still wins
