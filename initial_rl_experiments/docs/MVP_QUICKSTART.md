# MVP Role-Based Environment - Quick Start

## What is this?

A **simplified role-based multi-agent environment** that extends Craftium's OpenWorld with soft role incentives. This is the MVP version before implementing hard constraints.

## Changes from Base OpenWorld

| Feature | Base OpenWorldParallelEnv | MVP RoleBasedOpenWorld |
|---------|---------------------------|------------------------|
| Role assignment | None | ✅ Fixed per episode (engineer/hunter/guardian) |
| Role-based rewards | None | ✅ 2x multipliers for correct role actions |
| Hard action gating | No | ❌ Not yet (coming in v2) |
| Team milestones | Basic | ✅ Enhanced (tools, combat, hunting) |
| Cooperation bonus | Optional task_focus | ✅ Proximity-based bonus |

## Role Mechanics (MVP)

### Engineer (Tools Track)
- **In-track actions**: Crafting tools, mining ore, building (1.0x - full reward)
- **Out-of-track actions**: Hunting, combat (0.5x - half reward)
- **No hard blocks**: Can still hunt/fight, just gets half reward

### Hunter (Hunt Track)
- **In-track actions**: Hunting animals (1.0x - full reward)
- **Out-of-track actions**: Crafting, mining, combat (0.5x - half reward)

### Guardian (Defend Track)
- **In-track actions**: Killing monsters, defense (1.0x - full reward)
- **Out-of-track actions**: Crafting, mining, hunting (0.5x - half reward)

## Reward Structure

```python
# Individual shaped reward (from OpenWorld base rewards)
if action_track == agent_track:
    agent_reward = base_reward * 1.0  # Full reward in-track
else:
    agent_reward = base_reward * 0.5  # Half reward out-of-track

# Team reward (shared)
team_reward = sum(shaped_rewards) + milestone_bonus + cooperation_bonus

# All agents receive team_reward
```

**Example:**
- Engineer crafts a pickaxe → gets full OpenWorld reward (1.0x)
- Hunter crafts a pickaxe → gets half OpenWorld reward (0.5x)
- Guardian hunts an animal → gets half OpenWorld reward (0.5x)

### Milestone Bonuses (Sparse)
- Wooden pickaxe: +10
- Stone pickaxe: +20
- Iron pickaxe: +50
- Kill 5 monsters: +15
- Kill 10 monsters: +30
- Hunt 5 animals: +10
- Hunt 10 animals: +20

### Cooperation Bonus
- Agents within 10 blocks: +0.01 per pair (encourages staying together)

## Usage

### Test the Environment

```bash
# Make sure dependencies are installed
poetry install
poetry run pip install https://github.com/mikelma/craftium/releases/download/v0.0.1/craftium-0.0.1-cp312-cp312-manylinux_2_28_x86_64.whl
poetry run pip install stable-baselines3 moviepy

# Run the test script
poetry run python scripts/test_roles_mvp.py
```

Expected output:
```
=============================================================
Testing MVP Role-Based OpenWorld Environment
=============================================================

✓ Environment created
  Agents: ['agent_0', 'agent_1', 'agent_2']
  Roles: ['engineer', 'hunter', 'guardian']

✓ Environment reset
  Active agents: ['agent_0', 'agent_1', 'agent_2']
  Role assignments:
    agent_0: engineer
    agent_1: hunter
    agent_2: guardian

...

  Team Stats:
    Tools crafted: 2
    Ore mined: 5
    Monsters killed: 3
    Animals hunted: 4

  Milestones unlocked:
    ✓ has_wooden_pickaxe
    ✓ hunted_5_animals
```

### Import in Code

```python
from src.envs import RoleBasedOpenWorldMVP

# Create environment
env = RoleBasedOpenWorldMVP(
    num_agents=3,
    roles=["engineer", "hunter", "guardian"],
    max_steps=10000
)

# Use like any PettingZoo ParallelEnv
obs, info = env.reset(seed=42)

for step in range(100):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Check team stats
    stats = env.get_team_stats()
    print(f"Milestones: {stats['milestones']}")

env.close()
```

### Train with Stable-Baselines3 (Simple Wrapper)

Since SB3 doesn't support multi-agent natively, we can create a simple shared-policy wrapper:

```python
# TODO: Create sb3_wrapper.py for parameter-sharing training
# For now, use independent single-agent training or RLlib
```

## Limitations (MVP)

❌ **No hard action gating** - All agents can perform all actions
❌ **No guarded zones** - Uses default OpenWorld map
❌ **Simple milestone detection** - Based on counters, not actual Craftium state
❌ **Placeholder cooperation detection** - Just proximity-based
❌ **No Lua modifications** - Relies on base Craftium environment

## Next Steps (v2)

After validating the MVP:
1. Add hard action gating (crafting/equipment restrictions)
2. Create custom Lua world with guarded zones
3. Implement proper milestone detection from Craftium state
4. Add sophisticated cooperation bonuses
5. Integrate with MAPPO/QMIX for training

## Testing Checklist

- [ ] Environment creates successfully
- [ ] Roles are assigned correctly
- [ ] Role-based reward multipliers apply
- [ ] Team milestones unlock
- [ ] Cooperation bonus triggers
- [ ] Can run for full episode without crashes
- [ ] Compatible with PettingZoo API

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'craftium'`
**Solution**: Install Craftium wheel:
```bash
poetry run pip install https://github.com/mikelma/craftium/releases/download/v0.0.1/craftium-0.0.1-cp312-cp312-manylinux_2_28_x86_64.whl
```

**Issue**: `KeyError: 'position'` in cooperation detection
**Solution**: Craftium might not provide position in info dict. The code handles this gracefully, but cooperation bonus won't trigger.

**Issue**: Milestones not unlocking
**Solution**: MVP uses simple counters. In v2, we'll integrate with actual Craftium state tracking.

## Questions?

See [ROLE_SYSTEM_DESIGN.md](ROLE_SYSTEM_DESIGN.md) for full design spec.
