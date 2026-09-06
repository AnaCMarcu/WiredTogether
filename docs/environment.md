# WIRE

Five chambers connected by one-way transitions, traversed by a fixed team of `N` agents. Each
chamber gates progression on a shared objective, so co-location is enforced rather than left to
emerge — which is the whole point: bonds cannot form if agents never meet.

World: `src/marl_craftium/craftium-envs/wire/`, geometry and mechanics in
`mods/wire/*.lua`, all constants in `config.lua`.

## The chambers

| # | Mechanic | What it demands | Lua |
|---|---|---|---|
| 1 | Solo skill acquisition | Nothing shared. Move, dig, pick up, kill passive animals — so the team enters Ch2 with the primitives already learned. | `world_gen.lua` |
| 2 | Cooperative anvils | An anvil has 20 HP, decays 1/tick, and takes 1/4/8 damage per tick from 1/2/3 simultaneous diggers inside a 30-tick window. Solo digging is exactly net zero: cooperation is required, but trying alone is not punished. | `anvil.lua` |
| 3 | Switch puzzle under partial observability | Agents are teleported into isolated cells wired in a cycle: agent *i*'s switch opens agent *(i mod N)+1*'s door. Nobody can free themselves and nobody can see the cell their own switch governs, so release only propagates through targeted messages. | `switches.lua`, `doors.lua` |
| 4 | Team combat | Zombies (one per agent, or pinned with `--ch4-mob-count`). Non-lethal: a fatal hit costs −10 instead of killing, so the chamber is a rehearsal for Ch5. | `mobs.lua`, `deaths.lua` |
| 5 | Cooperative boss | A powered zombie with 60 HP — three times an ordinary mob, more than one agent can burn through inside the time budget. Death is real (−50) and the episode ends when the boss falls or the whole team is down. Kill and survival credit require a minimum cumulative damage, so surviving alongside the team is not enough. | `mobs.lua` |

If a chamber is not solved within its slice of the step budget (20% each), the team is
rescue-teleported forward *without* the chamber's milestone rewards. `config.lua` can disable a
chamber outright; `world_gen` then leaves the space void and opens the connecting door.

## Rewards

The reward stream is a milestone ladder plus a small set of auxiliary terms. The Python-side table
is `TRACKS` in `mindforge/agent_modules/craftium_metric.py` and mirrors `milestones.lua` *(pinned
by `tests/test_lua_spec.py`)*.

| Track | Milestones | Total |
|---|---|---|
| `ch1_solo` | move, dig 3, pick up 3, dig 5 wood, kill 1, kill 2, dig 3 stone (+ the door unlock) | 360 |
| `ch2_anvils` | first anvil, second anvil, sword equipped, chestplate equipped | 160 |
| `ch3_switches` | enter cell, press switch, door opened by a teammate, all in the communal room | 220 |
| `ch4_combat` | enter, first kill, all mobs killed, all alive having dealt damage | 340 |
| `ch5_boss` | enter, first damage, boss at half HP, boss defeated, all alive at the kill | 800 |
| `communication` | sustained valid targeted messaging, once per chamber (40/20/30/15/20 for Ch1–Ch5) | 125 |

Team-level milestones (both anvils, all-in-communal, all-mobs, survival bonuses, the boss set) fire
once when the global condition holds and are credited to every contributing agent; damage-share
rules apply to the combat ones so credit tracks damage dealt, not presence. Auxiliary terms:
`+0.5` per valid targeted message (capped at 50 per agent per episode), `−1` per pitch-capped
action, `−10` for a would-be-fatal hit in Ch1–4, `−50` for a real death in Ch5.

Milestone ids in code (`m1`–`m28`) predate the paper's `M1`–`M24` numbering; the mapping is the
track table above, read in order. The 25 task milestones are the five chamber tracks; the 17
cooperative ones are Ch2–Ch5. The communication track pays for social acts rather than task
progress, so every comparable milestone count in `analysis/` strips it.

## The Lua ↔ Python interface

`craftium.reward()` does not reach Python in multi-agent mode. Everything therefore crosses in one
of two ways, both in `mindforge/env/`:

- **Event drains** (`lua_events.py`). The mod appends milestone fires, anvil co-op attempts and
  deaths to JSONL files; each poller keeps a byte offset so a line is never read twice, and offsets
  reset per episode. This is the only path by which a Lua-side reward becomes a Python reward.
- **State files** (`chamber_state.py`). Door locks, anvil HP and current diggers, cell and switch
  status are written as flat files and rendered into the one-line "Chamber state" block in the
  action prompt — the block the prompt tells the agent to trust over its own frame.

Above that sits `custom_environment_craftium.py`: action canonicalisation and the idle guard,
`step_all` for simultaneous stepping, frames, positions, inventory, status text, and the forced
chamber teleports.

## Actions

22 discrete primitives — 6 movement, 4 camera, `Dig`/`Place`/`Drop`, 8 hotbar slots, `NoOp`. Three
conventions matter:

- `Dig` is sustained (held for 10 environment steps) and `Jump` expands to jump-then-forward, so
  one action is enough to break a block or clear a ledge.
- Camera pitch is clamped to two `LookDown` and one `LookUp` from level; an action beyond the clamp
  becomes `NoOp` and costs the futility penalty. This killed the sky-staring loops seen in early
  runs.
- Malformed action strings from the LLM are recovered through a synonym map where the intent is
  unambiguous (`Attack`/`Mine` → `Dig`), otherwise clamped to `NoOp`; repeated `NoOp`s are forced
  to `MoveForward` so an agent cannot silently stall. Both counts are logged as run diagnostics.

The 8 hotbar actions are masked from the RL candidate set — the environment auto-equips the best
tool before every `Dig` — leaving 14 candidates for constrained decoding.

## Team size

`--num-agents` sets `FC_NUM_AGENTS`, which the Lua side reads at load time and derives all geometry
from: one cell and one switch per agent in Ch3, one Ch4 spawn per agent. `--ch4-mob-count` pins the
combat chamber to a fixed mob count so team-size sweeps face an identical environment, and
`--team-scaling` templates the N-dependent prompt text (`agent_modules/team_scaling.py`).

## Running against another world

`CRAFTIUM_ENV_DIR` selects the world; the stack above the adapter knows nothing about chambers. A
different Craftium world needs to honour only the two contracts above: a milestone event log for
rewards, and per-chamber state files for the prompt block. Adding a mechanic to WIRE means firing
an event in Lua, appending it to the event log, and giving it a milestone id in `milestones.lua` —
`tests/test_lua_spec.py` fails if the Lua and Python milestone tables drift apart.
