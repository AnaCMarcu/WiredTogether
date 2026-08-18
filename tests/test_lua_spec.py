"""Spec tests over the five_chambers Lua mod, plus Python<->Lua drift guards.

Pure text parsing — no Lua runtime, no game launch.  Pins:
  * the complete milestone/reward table in milestones.lua (paper Table 2),
  * the Ch3 rotational switch formula,
  * canonical config constants (NUM_AGENTS, BOSS_HP, anvil coop constants),
    explicitly distinguishing them from the DEBUG_SINGLE solo-mode overrides,
  * the death / would-die penalty constants and HOW the -10 penalty is gated
    (per would-death EVENT in Ch4 only — the virtual-HP pool refills, so it is
    NOT once-per-agent-per-episode),
  * cross-file consistency: every milestone id/prefix referenced from Python
    (coop_eval, cooperation_metric, craftium_metric, make_results) resolves to
    an id defined in milestones.lua.

Source-truth notes (divergences from the informal paper-table spec):
  * m14_sword_equipped / m15_chestplate_equipped use Lua track "ch2_gear"
    (Python aggregates them under "ch2_anvils").
  * m16..m19 use Lua track "ch3_switch" (Python: "ch3_switches").
  * m_door1_open is defined in Lua but absent from BOTH Python milestone
    tables (craftium_metric.MILESTONE_TRACK/TRACKS and make_results
    .MILESTONE_TRACK) — pinned below as the single known Lua->Python gap.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ── Lua parsing helpers ──────────────────────────────────────────────────────

# One MILESTONE_DEFS entry: id = { track="...", reward=N, once=true },
_DEF_RE = re.compile(
    r'(m\w+)\s*=\s*\{\s*track\s*=\s*"([^"]+)"\s*,'
    r'\s*reward\s*=\s*(-?\d+)\s*,\s*once\s*=\s*(true|false)\s*,?\s*\}'
)

# Milestone-id-shaped quoted strings in Python sources ("m8_anvil_A1",
# "m_door1_open", "m_comm_ch1", ...). Anchored on the quote + the m<digit>_ /
# m_ shape so ordinary strings ("mean", "milestone_id", ...) never match.
_PY_ID_RE = re.compile(r'["\'](m\d+_\w+|m_\w+)["\']')


def _lua(lua_root, name: str) -> str:
    return (lua_root / name).read_text(encoding="utf-8")


def _milestone_defs(lua_root) -> dict:
    """{id: (track, reward, once_str)} parsed from the MILESTONE_DEFS block."""
    text = _lua(lua_root, "milestones.lua")
    start = text.index("five_chambers.MILESTONE_DEFS = {")
    end = text.index("\n}", start)  # outer table closes at column 0
    block = text[start:end]
    return {
        m.group(1): (m.group(2), int(m.group(3)), m.group(4))
        for m in _DEF_RE.finditer(block)
    }


def _canonical_assignments(text: str, name: str) -> list:
    """Values of column-0 `five_chambers.<name> = <int>` lines, in file order.

    Column-0 anchoring excludes indented assignments inside the
    `if five_chambers.DEBUG_SINGLE then` solo-mode override block.
    """
    pat = re.compile(
        r"^five_chambers\.%s\s*=\s*(-?\d+)" % re.escape(name), re.MULTILINE
    )
    return [int(m.group(1)) for m in pat.finditer(text)]


def _debug_guarded_assignments(text: str, name: str) -> list:
    """(value, inside_DEBUG_SINGLE_guard) for each INDENTED assignment."""
    pat = re.compile(
        r"^[ \t]+five_chambers\.%s\s*=\s*(-?\d+)" % re.escape(name), re.MULTILINE
    )
    out = []
    for m in pat.finditer(text):
        before = text[: m.start()]
        if_pos = before.rfind("if five_chambers.DEBUG_SINGLE then")
        end_pos = before.rfind("\nend")
        out.append((int(m.group(1)), if_pos > end_pos))
    return out


def _py_milestone_track_table(text: str) -> dict:
    """{milestone_id: track} from a Python MILESTONE_TRACK = {...} block."""
    start = text.index("MILESTONE_TRACK = {")
    end = text.index("\n}", start)
    return dict(
        re.findall(r'["\'](m\w+)["\']\s*:\s*["\'](\w+)["\']', text[start:end])
    )


# ── Expected milestone table (SOURCE truth; rewards mirror paper Table 2) ────

EXPECTED_MILESTONES = {
    # Ch1 solo learning
    "m1_move_5":               ("ch1_solo", 10),
    "m2_dig_3_any":            ("ch1_solo", 30),
    "m3_pickup_3":             ("ch1_solo", 30),
    "m4_dig_5_wood":           ("ch1_solo", 50),
    "m5_kill_1_animal":        ("ch1_solo", 50),
    "m6_kill_2_animals":       ("ch1_solo", 80),
    "m7_dig_3_stone":          ("ch1_solo", 60),
    "m_door1_open":            ("ch1_solo", 50),
    # Ch2 anvils + gear (Lua splits gear equips into their own track)
    "m8_anvil_A1":             ("ch2_anvils", 40),
    "m9_anvil_B1":             ("ch2_anvils", 40),
    "m14_sword_equipped":      ("ch2_gear", 50),
    "m15_chestplate_equipped": ("ch2_gear", 30),
    # Ch3 switch puzzle (Lua track is singular "ch3_switch")
    "m16_enter_cell":          ("ch3_switch", 20),
    "m17_switch_pressed":      ("ch3_switch", 40),
    "m18_door_opened":         ("ch3_switch", 60),
    "m19_all_in_communal":     ("ch3_switch", 100),
    # Ch4 combat
    "m20_enter_ch4":           ("ch4_combat", 30),
    "m21_first_mob_kill":      ("ch4_combat", 60),
    "m22_all_mobs_killed":     ("ch4_combat", 150),
    "m23_all_alive_ch4":       ("ch4_combat", 100),
    # Ch5 boss
    "m24_enter_ch5":           ("ch5_boss", 50),
    "m25_first_boss_dmg":      ("ch5_boss", 80),
    "m26_boss_half_hp":        ("ch5_boss", 120),
    "m27_boss_defeated":       ("ch5_boss", 300),
    "m28_all_alive_bonus":     ("ch5_boss", 250),
}


# ── milestones.lua ───────────────────────────────────────────────────────────

def test_milestone_defs_complete(lua_root):
    """milestones.lua defines exactly the 25-entry {id: (track, reward)} table."""
    defs = _milestone_defs(lua_root)
    got = {mid: (track, reward) for mid, (track, reward, _) in defs.items()}
    assert got == EXPECTED_MILESTONES


def test_milestones_all_once_true(lua_root):
    """Every MILESTONE_DEFS entry is once=true (anti-farming / free-rider deterrent)."""
    defs = _milestone_defs(lua_root)
    assert len(defs) == 25
    assert all(once == "true" for (_, _, once) in defs.values())


# ── switches.lua ─────────────────────────────────────────────────────────────

def test_switch_rotation_formula(lua_root):
    """switches.lua: switch i opens cell (i + 1) % NUM_AGENTS, one-shot, fires m17/m18."""
    text = _lua(lua_root, "switches.lua")
    assert re.search(r"\(\s*i\s*\+\s*1\s*\)\s*%\s*five_chambers\.NUM_AGENTS", text)
    # One-shot guard: a pressed switch never re-fires.
    assert "if five_chambers.switch_pressed[sw_i] then return end" in text
    # Milestones fired from this file exist in MILESTONE_DEFS.
    fired = set(re.findall(r'fire_milestone\("(m\w+)"', text))
    assert fired == {"m17_switch_pressed", "m18_door_opened"}
    assert fired <= set(_milestone_defs(lua_root))


# ── config.lua ───────────────────────────────────────────────────────────────

def test_config_num_agents(lua_root):
    """config.lua: NUM_AGENTS = FC_NUM_AGENTS env override, default 3; the =1
    override only inside DEBUG_SINGLE."""
    text = _lua(lua_root, "config.lua")
    # Canonical assignment is the env-var pattern (same shape as
    # CH1_TIMEOUT_TICKS), not an integer literal.
    assert _canonical_assignments(text, "NUM_AGENTS") == []
    assert re.search(
        r'^local _env_agents = tonumber\(os and os\.getenv and '
        r'os\.getenv\("FC_NUM_AGENTS"\) or ""\)$',
        text, re.MULTILINE,
    )
    assert re.search(
        r"^five_chambers\.NUM_AGENTS\s*=\s*_env_agents or 3$",
        text, re.MULTILINE,
    )
    assert _debug_guarded_assignments(text, "NUM_AGENTS") == [(1, True)]
    # DEBUG_SINGLE is off, so the override is dead code in training runs.
    assert re.search(r"^five_chambers\.DEBUG_SINGLE\s*=\s*false", text, re.MULTILINE)


def test_config_door3_x_clamped(lua_root):
    """config.lua: DOOR3_X = min(2*N, 10) — the Ch3→Ch4 doorway must stay
    inside Ch4's fixed x-span (1..11) when NUM_AGENTS >= 6."""
    text = _lua(lua_root, "config.lua")
    assert re.search(
        r"^five_chambers\.DOOR3_X\s*=\s*math\.min\("
        r"2 \* five_chambers\.NUM_AGENTS, 10\)",
        text, re.MULTILINE,
    )


def test_ch4_spawn_positions_inside_ch4(lua_root):
    """mobs.lua: 6 CH4 zombie spawn entries, every one inside CH4's interior
    (x=1..11, z=47..57). Regression pin: the old third entry {x=8,z=60} sat
    in the Ch5 band (z0=59) after the 5-block southward chamber shift, so
    the third zombie spawned in the boss room and the organic all-mobs-dead
    Ch4 clear could never fire. Spawn count is min(NUM_AGENTS, #positions),
    so N=3 uses only the first three entries."""
    text = _lua(lua_root, "mobs.lua")
    start = text.index("local CH4_SPAWN_POSITIONS = {")
    block = text[start:text.index("\n}", start)]
    entries = [
        (int(x), int(y), int(z))
        for x, y, z in re.findall(r"\{x=(\d+),\s*y=(\d+),\s*z=(\d+)\}", block)
    ]
    assert len(entries) == 6
    for x, y, z in entries:
        assert 1 <= x <= 11, (x, y, z)
        assert 47 <= z <= 57, (x, y, z)
        assert y == 11, (x, y, z)


def test_config_boss_hp(lua_root):
    """config.lua: canonical BOSS_HP = 60; the solo-mode 20 only inside DEBUG_SINGLE."""
    text = _lua(lua_root, "config.lua")
    assert _canonical_assignments(text, "BOSS_HP") == [60]
    assert _debug_guarded_assignments(text, "BOSS_HP") == [(20, True)]
    assert _canonical_assignments(text, "BOSS_DMG") == [3]


def test_anvil_coop_constants(lua_root):
    """config.lua pins the cooperative-dig constants; anvil.lua consumes them by name."""
    text = _lua(lua_root, "config.lua")
    expected = {
        "ANVIL_MAX_HP": 20,    # break threshold (lowered from 30)
        "SOLO_DIG_RATE": 1,    # net 0 with DECAY=1: solo digging unproductive, not punished
        "PAIR_DIG_RATE": 4,    # net +3/tick
        "TRIO_DIG_RATE": 8,    # net +7/tick
        "DECAY_RATE": 1,
        "ACTIVE_WINDOW": 30,   # ticks a punch stays "active" (~10 env steps)
        "DIGGER_RADIUS": 3,    # defined but unreferenced elsewhere (vestigial)
    }
    for name, value in expected.items():
        assert _canonical_assignments(text, name) == [value], name
    anvil = _lua(lua_root, "anvil.lua")
    for name in ("SOLO_DIG_RATE", "PAIR_DIG_RATE", "TRIO_DIG_RATE",
                 "DECAY_RATE", "ANVIL_MAX_HP", "ACTIVE_WINDOW"):
        assert f"five_chambers.{name}" in anvil, name


# ── anvil.lua ────────────────────────────────────────────────────────────────

def test_anvil_milestone_ids(lua_root):
    """anvil.lua registers exactly two anvils, m8 (row A) / m9 (row B), both 40 pts."""
    anvil = _lua(lua_root, "anvil.lua")
    ids = set(re.findall(r'milestone_id\s*=\s*"(m\w+)"', anvil))
    assert ids == {"m8_anvil_A1", "m9_anvil_B1"}
    defs = _milestone_defs(lua_root)
    for mid in ids:
        track, reward, once = defs[mid]
        assert track == "ch2_anvils"
        assert reward == 40
        assert once == "true"


# ── deaths.lua ───────────────────────────────────────────────────────────────

def test_death_penalty_constants(lua_root):
    """deaths.lua: DEATH_PENALTY = -50 (Ch5 permadeath), WOULD_DIE_PENALTY = -10, MAX_HP = 20."""
    text = _lua(lua_root, "deaths.lua")
    assert re.search(r"^local DEATH_PENALTY\s*=\s*-50\b", text, re.MULTILINE)
    assert re.search(r"^local WOULD_DIE_PENALTY\s*=\s*-10\b", text, re.MULTILINE)
    assert re.search(r"^local MAX_HP\s*=\s*20\b", text, re.MULTILINE)


def test_would_die_penalty_gating(lua_root):
    """The -10 would-die penalty fires per EVENT in Ch4 only (pool refills; not once/episode).

    Code path: virtual-HP pool drains by would-be damage; when `vhp <= 0`
    AND `chamber == "ch4"`, emit_death_event("woulddie", ..) fires and the
    per-agent counter increments; THEN `vhp = MAX_HP` refills the pool so the
    next 20 accumulated would-be damage triggers ANOTHER -10.  Ch1-Ch3
    near-deaths refill silently with no penalty; Ch5 damage passes through to
    a real permadeath handled by on_dieplayer with the -50 DEATH_PENALTY.
    """
    text = _lua(lua_root, "deaths.lua")
    p_lethal = text.index("if vhp <= 0 then")
    p_gate = text.index('if chamber == "ch4" then')
    p_emit = text.index('emit_death_event("woulddie"')
    p_refill = text.index("vhp = MAX_HP")
    assert p_lethal < p_gate < p_emit < p_refill
    # Per-event counter; no fired-once dedup between the Ch4 gate and the emit.
    assert "(five_chambers.would_die_count[name] or 0) + 1" in text
    assert "would_die_count" not in text[p_gate:p_emit]
    # Ch5: real damage passes through; real death emits the -50 terminal penalty.
    assert 'if chamber == "ch5" then return hp_change end' in text
    assert 'emit_death_event("death", name, chamber, DEATH_PENALTY)' in text


# ── Python <-> Lua milestone consistency (cross-file drift guards) ───────────

def test_cooperation_metric_prefixes(lua_root):
    """cooperation_metric prefixes are ('m8_','m9_')/('m17_',)/('m18_',)/('m19_',) and all resolve in Lua."""
    from mindforge.env.cooperation_metric import CooperationMetric as CM

    assert CM._CH2_ANVIL_PREFIXES == ("m8_", "m9_")  # post-fix value (was stale "m11_")
    assert CM._CH3_PRESS_PREFIXES == ("m17_",)
    assert CM._CH3_DOOR_PREFIXES == ("m18_",)
    assert CM._CH3_REGROUP_PREFIX == ("m19_",)
    lua_ids = set(_milestone_defs(lua_root))
    prefixes = (CM._CH2_ANVIL_PREFIXES + CM._CH3_PRESS_PREFIXES
                + CM._CH3_DOOR_PREFIXES + CM._CH3_REGROUP_PREFIX)
    for prefix in prefixes:
        assert any(mid.startswith(prefix) for mid in lua_ids), prefix


def test_coop_eval_credit_rules(lua_root):
    """coop_eval._CREDIT_RULES keys are exactly the 7 expected ids, all defined in Lua."""
    from mindforge.agent_modules.coop_eval import _CREDIT_RULES

    assert _CREDIT_RULES == {
        "m22_all_mobs_killed": "ch4_damage",
        "m23_all_alive_ch4":   "equal",
        "m25_first_boss_dmg":  "boss_damage",
        "m26_boss_half_hp":    "boss_damage",
        "m27_boss_defeated":   "boss_damage",
        "m28_all_alive_bonus": "equal",
        "m19_all_in_communal": "equal",
    }
    assert set(_CREDIT_RULES) <= set(_milestone_defs(lua_root))


def test_craftium_metric_milestones_match_lua(lua_root):
    """craftium_metric.py milestone ids/rewards mirror Lua with NO gap
    (FIXED BUG: m_door1_open was missing from the Python tables, silently
    dropping its 50 reward from every ch1_solo track aggregate)."""
    text = (REPO / "src" / "mindforge" / "agent_modules"
            / "craftium_metric.py").read_text(encoding="utf-8")
    defs = _milestone_defs(lua_root)
    lua_ids = set(defs)

    # Every Lua-shaped id referenced anywhere in the file exists in Lua
    # (m_comm_* / m_obs_* / m_imit_* are Python-side social-act milestones
    # — comm tracker + act-reward symmetry suite — never emitted by Lua).
    py_ids = set(_PY_ID_RE.findall(text))
    lua_facing = {i for i in py_ids
                  if not i.startswith(("m_comm_", "m_obs_", "m_imit_"))}
    assert lua_facing <= lua_ids
    assert lua_ids - lua_facing == set()  # complete coverage of Lua milestones

    # TRACKS (id, reward) pairs match the Lua rewards exactly.
    start = text.index("\nTRACKS = {")
    end = text.index("\n}", start)
    rewards = {
        mid: float(r)
        for mid, r in re.findall(
            r'\(\s*"(m\w+)"\s*,\s*(\d+(?:\.\d+)?)\s*\)', text[start:end]
        )
    }
    non_comm = {m: r for m, r in rewards.items() if not m.startswith("m_comm_")}
    assert len(non_comm) == 25  # all 25 Lua milestones incl. m_door1_open
    for mid, r in non_comm.items():
        assert r == float(defs[mid][1]), mid


def test_make_results_milestones_match_lua(lua_root):
    """make_results.py mirrors craftium_metric's table; key-milestone ids exist in Lua."""
    mr = (REPO / "make_results.py").read_text(encoding="utf-8")
    cm = (REPO / "src" / "mindforge" / "agent_modules"
          / "craftium_metric.py").read_text(encoding="utf-8")

    # "mirrors craftium_metric.py — keep in sync" header claim, enforced.
    assert _py_milestone_track_table(mr) == _py_milestone_track_table(cm)

    lua_ids = set(_milestone_defs(lua_root))
    py_ids = {i for i in _PY_ID_RE.findall(mr)
              if not i.startswith(("m_comm_", "m_obs_", "m_imit_"))}
    assert py_ids <= lua_ids
    assert lua_ids - py_ids == set()  # complete coverage, same as craftium_metric

    key = re.findall(r'\(\s*"M\d+",\s*"(m\w+)"\s*\)', mr)
    assert set(key) == {"m8_anvil_A1", "m19_all_in_communal",
                        "m22_all_mobs_killed", "m27_boss_defeated"}
    boss = re.search(r'^BOSS_MILESTONE\s*=\s*"(m\w+)"', mr, re.MULTILINE)
    assert boss and boss.group(1) == "m27_boss_defeated"
    assert boss.group(1) in lua_ids
