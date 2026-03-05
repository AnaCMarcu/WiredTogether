# Multi-Agent Role-Based OpenWorld Design

## Executive Summary
Transform Craftium's OpenWorld into a 3-role cooperative Dec-POMDP where **Engineer + Hunter + Guardian must collaborate** to progress through tool tiers (T1→T4). Collaboration is enforced through:
- **Capability gating** (only certain roles can perform certain actions)
- **Milestone gating** (role-specific progression locks)
- **Structural hazards** (monsters guard high-tier resources)
- **Resource dependencies** (food, materials, protection)

---

## 1. Role Capabilities Matrix

| Capability | Engineer | Hunter | Guardian |
|-----------|----------|--------|----------|
| **Craft T3+ tools** | ✅ Only | ❌ | ❌ |
| **Craft T4 diamond gear** | ✅ Only | ❌ | ❌ |
| **Wield T3+ swords** | ❌ | ❌ | ✅ Only |
| **Hunt efficiently** | ❌ 50% yield | ✅ 100% yield | ❌ 50% yield |
| **Mine iron/diamond** | ✅ Normal | ✅ Slow (2x time) | ✅ Slow (2x time) |
| **Combat effectiveness** | ❌ 3x damage taken | ✅ Normal | ✅ 2x damage dealt |
| **Unlock Tools milestones** | ✅ Full rewards | 🔸 10% rewards | 🔸 10% rewards |
| **Unlock Hunt milestones** | 🔸 10% rewards | ✅ Full rewards | 🔸 10% rewards |
| **Unlock Defend milestones** | 🔸 10% rewards | 🔸 10% rewards | ✅ Full rewards |

### Enforcement Strategy
- **Crafting gating**: Lua mod checks agent role before allowing T3/T4 crafting recipes
- **Equipment gating**: High-tier swords can be crafted but only Guardian can equip/use them
- **Efficiency modifiers**: Hunting/mining actions have role-based multipliers
- **Reward gating**: Milestone achievements give 10% reward if wrong role unlocks them

---

## 2. World Structure: Forcing Interdependence

### 2.1 Resource Placement (Guarded Zones)

```
World Layout (50x50 Minetest map):
┌─────────────────────────────────┐
│  Safe Zone (Spawn)              │
│  - Wood, stone, coal            │
│  - Pigs, cows (food)            │
│  - No monsters                  │
│                                 │
│  ┌─────────────────┐            │
│  │ IRON ZONE       │            │
│  │ - Iron nodes    │            │
│  │ - 5+ skeletons  │  ←─ Guardian needed
│  │ - Constant spawn│            │
│  └─────────────────┘            │
│                                 │
│      ┌───────────────────┐      │
│      │ DIAMOND ZONE      │      │
│      │ - Diamond nodes   │      │
│      │ - 10+ spiders     │  ←─ Guardian essential
│      │ - High spawn rate │      │
│      └───────────────────┘      │
└─────────────────────────────────┘
```

### 2.2 Structural Constraints

**Iron Zone** (T3 gate):
- Distance from spawn: 100+ blocks
- Monster density: 5-8 skeletons active
- Respawn rate: 1 skeleton every 30 seconds
- Engineer alone: ~80% chance of death before mining 5 iron
- Guardian + Engineer: Safe extraction

**Diamond Zone** (T4 gate):
- Distance from spawn: 200+ blocks
- Monster density: 10-15 spiders/skeletons
- Respawn rate: 1 monster every 20 seconds
- Engineer alone: ~95% chance of death
- Guardian + Engineer: Manageable with coordination

**Food Scarcity**:
- Pigs/cows in safe zone: 3-5 total
- Natural regen: 1 animal every 60 seconds
- Without Hunter efficiency: Team starves during long expeditions
- Hunter contribution: Sustains 2-3 agents continuously

---

## 3. Reward Design: Team Objective with Role Incentives

### 3.1 Joint Return Formula

**Team reward** (all agents receive same scalar):
```
R_team(t) = R_milestone(t) + R_survival(t) + R_cooperation(t)
```

### 3.2 Reward Components

#### A. Milestone Rewards (Sparse, High-Value)
```python
MILESTONE_REWARDS = {
    # Tools track (Engineer primary)
    "tools_t1": 10.0,   # Wooden pickaxe
    "tools_t2": 20.0,   # Stone pickaxe
    "tools_t3": 50.0,   # Iron pickaxe (ENGINEER ONLY unlocks full)
    "tools_t4": 100.0,  # Diamond pickaxe (ENGINEER ONLY)

    # Hunt track (Hunter primary)
    "hunt_t1": 10.0,    # Kill 5 pigs
    "hunt_t2": 20.0,    # Kill 10 cows
    "hunt_t3": 40.0,    # Sustain team (30+ food delivered)

    # Defend track (Guardian primary)
    "defend_t1": 10.0,  # Kill 3 zombies
    "defend_t2": 25.0,  # Kill 5 skeletons
    "defend_t3": 50.0,  # Clear iron zone (8+ monsters)
    "defend_t4": 80.0,  # Clear diamond zone (15+ monsters)
}

# Role-based multiplier (only for role-matched milestones)
ROLE_MULTIPLIER = {
    "correct_role": 1.0,    # Full reward
    "wrong_role": 0.1,      # Only 10% if wrong agent unlocks
}
```

**Key insight**: Engineer gets 100 pts for T4 unlock, but Guardian only gets 10 pts. This creates **role specialization incentive** while still rewarding team progress.

#### B. Survival Penalty (Dense, Small)
```python
R_survival(t) = -0.01 * num_deaths(t) - 0.001 * hunger_damage(t)
```
- Death: -0.01 per agent death
- Hunger: -0.001 per hunger-damage tick
- Encourages Hunter to provide food, Guardian to protect

#### C. Cooperation Bonus (Event-Triggered)
```python
COOPERATION_EVENTS = {
    "guardian_escort": 5.0,      # Guardian within 10 blocks of Engineer in danger zone
    "hunter_feed": 2.0,          # Hunter gives food to low-health teammate
    "engineer_share_tools": 3.0, # Engineer crafts pickaxe, Guardian mines with it
}
```

### 3.3 Aggregation: Shared Team Reward

**All agents receive identical reward** at each timestep:
```python
def step(actions: Dict[str, int]) -> Tuple[...]:
    # ... execute actions ...

    team_reward = compute_team_reward()  # Scalar

    # All agents get same reward (Dec-POMDP joint return)
    rewards = {agent: team_reward for agent in self.agents}

    return observations, rewards, terminations, truncations, infos
```

**Justification**:
- Pure cooperation (no conflicting objectives)
- Credit assignment handled by value factorization (QMIX, MAPPO, etc.)
- Simpler than hand-designed per-agent rewards

---

## 4. Implementation Plan: Mapping to Craftium

### 4.1 Environment Class Structure

**Extend existing MarlCraftiumEnv**:
```
src/envs/
├── openworld_parallel.py          (existing)
└── openworld_roles.py             (NEW)
    ├── RoleBasedOpenWorldEnv      (main class)
    ├── RoleConfig                 (dataclass)
    └── GuardedZoneGenerator       (world gen)
```

### 4.2 Core Components to Implement

#### Component 1: Role Assignment System
**File**: `src/envs/openworld_roles.py`

**Logic**:
```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class RoleConfig:
    name: str                          # "engineer", "hunter", "guardian"
    track: str                         # "tools", "hunt", "defend"
    crafting_whitelist: List[str]      # ["iron_pick", "diamond_pick"]
    equipment_whitelist: List[str]     # ["diamond_sword"]
    damage_multiplier: float           # 3.0 for engineer (takes 3x damage)
    attack_multiplier: float           # 2.0 for guardian (deals 2x damage)
    mining_speed_multiplier: float     # 0.5 for non-engineer (slower)
    hunting_efficiency: float          # 1.0 for hunter, 0.5 for others
    milestone_reward_multiplier: float # 1.0 for correct track, 0.1 otherwise

ROLES = {
    "engineer": RoleConfig(
        name="engineer",
        track="tools",
        crafting_whitelist=["iron_pick", "diamond_pick", "iron_axe", "diamond_axe"],
        equipment_whitelist=[],
        damage_multiplier=3.0,    # Vulnerable
        attack_multiplier=0.5,    # Weak combat
        mining_speed_multiplier=1.0,
        hunting_efficiency=0.5,
        milestone_reward_multiplier=1.0  # Full reward for tools track
    ),
    "hunter": RoleConfig(...),
    "guardian": RoleConfig(...)
}
```

**Reset logic**:
```python
def reset(self, seed=None, options=None):
    # Assign roles (fixed per episode)
    self.agent_roles = {
        "agent_0": "engineer",
        "agent_1": "hunter",
        "agent_2": "guardian"
    }

    # Pass roles to Lua environment via init_code
    init_lua = self._generate_role_init_lua(self.agent_roles)

    # Create environment with custom world generation
    self.env = MarlCraftiumEnv(
        env_dir_name="voxel-libre2-roles",  # Custom env variant
        num_agents=3,
        init_code=init_lua,
        ...
    )

    return obs, info
```

#### Component 2: Action Gating (Capability Enforcement)
**File**: `src/envs/openworld_roles.py` (in step method)

**Wrapper approach** (cleaner than modifying Lua):
```python
def step(self, actions: Dict[str, int]) -> Tuple[...]:
    # Pre-process actions: block invalid role actions
    filtered_actions = {}
    for agent, action in actions.items():
        role = self.agent_roles[agent]

        # Check if action is valid for this role
        if self._is_action_valid(agent, action, role):
            filtered_actions[agent] = action
        else:
            # Replace with no-op or warning
            filtered_actions[agent] = 0  # No-op action
            self.infos[agent]["invalid_action"] = True

    # Execute filtered actions
    obs, base_rewards, terms, truncs, infos = self.env.step(filtered_actions)

    # Post-process rewards with role multipliers
    team_reward = self._compute_team_reward(base_rewards, infos)
    rewards = {agent: team_reward for agent in self.agents}

    return obs, rewards, terms, truncs, infos

def _is_action_valid(self, agent: str, action: int, role: str) -> bool:
    """Check if action is allowed for agent's role."""
    action_type = self._decode_action(action)

    # Example: Crafting T3+ tools requires engineer role
    if action_type == "craft_iron_pickaxe":
        return role == "engineer"

    # Equipping high-tier weapons requires guardian
    if action_type == "equip_diamond_sword":
        return role == "guardian"

    return True  # Most actions allowed for all roles
```

#### Component 3: Guarded Zone Generation
**File**: `src/envs/world_gen.py` (NEW)

**Approach**: Modify Lua world generation script
```lua
-- craftium-envs/voxel-libre2-roles/worldgen.lua

function generate_guarded_zones()
    local spawn_pos = {x=0, y=0, z=0}

    -- Safe zone: 50 block radius around spawn
    -- (Already done by default, just ensure no monsters)

    -- Iron zone: 100-150 blocks away
    local iron_zone_center = {x=120, y=0, z=120}
    place_ore_cluster("default:stone_with_iron", iron_zone_center, 20)

    -- Spawn monsters around iron zone
    for i = 1, 8 do
        local monster_pos = random_point_in_radius(iron_zone_center, 15)
        spawn_monster("skeleton", monster_pos)
    end

    -- Set continuous spawn
    register_monster_spawner(iron_zone_center, "skeleton", 30)  -- Every 30 sec

    -- Diamond zone: 200+ blocks away, higher monster density
    local diamond_zone_center = {x=220, y=-20, z=220}
    place_ore_cluster("default:stone_with_diamond", diamond_zone_center, 15)

    for i = 1, 15 do
        local monster_pos = random_point_in_radius(diamond_zone_center, 20)
        local monster_type = choose_random({"spider", "skeleton"})
        spawn_monster(monster_type, monster_pos)
    end

    register_monster_spawner(diamond_zone_center, "spider", 20)  -- Every 20 sec
end
```

**Craftium integration**:
- Create new env directory: `craftium-envs/voxel-libre2-roles/`
- Copy from `voxel-libre2/` and modify world generation Lua
- Use `init_code` parameter in MarlCraftiumEnv to inject role-specific logic

#### Component 4: Milestone Gating & Reward Computation
**File**: `src/envs/openworld_roles.py`

```python
def _compute_team_reward(self, base_rewards: Dict, infos: Dict) -> float:
    """Compute shared team reward with role-based multipliers."""
    team_reward = 0.0

    # 1. Milestone rewards (check info for unlocked milestones)
    for agent, info in infos.items():
        role = self.agent_roles[agent]

        if "milestone_unlocked" in info:
            milestone_name = info["milestone_unlocked"]
            base_value = MILESTONE_REWARDS[milestone_name]

            # Apply role multiplier
            if self._is_correct_role_for_milestone(role, milestone_name):
                team_reward += base_value * 1.0  # Full reward
            else:
                team_reward += base_value * 0.1  # 10% penalty

    # 2. Survival penalty
    for agent, info in infos.items():
        if info.get("died", False):
            team_reward -= 0.01

        hunger_damage = info.get("hunger_damage", 0)
        team_reward -= 0.001 * hunger_damage

    # 3. Cooperation bonus (distance-based heuristic)
    if self._detect_guardian_escort(infos):
        team_reward += 5.0

    if self._detect_food_sharing(infos):
        team_reward += 2.0

    return team_reward

def _is_correct_role_for_milestone(self, role: str, milestone: str) -> bool:
    """Check if milestone matches agent's role track."""
    if milestone.startswith("tools_"):
        return role == "engineer"
    elif milestone.startswith("hunt_"):
        return role == "hunter"
    elif milestone.startswith("defend_"):
        return role == "guardian"
    return False

def _detect_guardian_escort(self, infos: Dict) -> bool:
    """Detect if Guardian is protecting Engineer in danger zone."""
    engineer_pos = None
    guardian_pos = None

    for agent, info in infos.items():
        if self.agent_roles[agent] == "engineer":
            engineer_pos = info.get("position")
        elif self.agent_roles[agent] == "guardian":
            guardian_pos = info.get("position")

    if engineer_pos and guardian_pos:
        # Check if both in danger zone and close together
        distance = np.linalg.norm(np.array(engineer_pos) - np.array(guardian_pos))
        in_danger_zone = self._is_danger_zone(engineer_pos)

        return distance < 10.0 and in_danger_zone

    return False
```

---

## 5. Configuration Knobs

Create Hydra config for role environment:

**File**: `configs/env/openworld_roles.yaml`
```yaml
_target_: src.envs.openworld_roles.RoleBasedOpenWorldEnv

num_agents: 3
obs_width: 320
obs_height: 180
max_steps: 10000

# Role assignment (fixed per episode)
roles:
  - engineer
  - hunter
  - guardian

# World generation parameters
world_gen:
  safe_zone_radius: 50
  iron_zone_distance: 120
  iron_zone_monsters: 8
  iron_monster_type: skeleton
  iron_spawn_interval: 30  # seconds

  diamond_zone_distance: 220
  diamond_zone_monsters: 15
  diamond_monster_types: [spider, skeleton]
  diamond_spawn_interval: 20

# Role capability modifiers
role_modifiers:
  engineer:
    damage_taken_multiplier: 3.0
    attack_damage_multiplier: 0.5
    mining_speed_multiplier: 1.0
    hunting_efficiency: 0.5

  hunter:
    damage_taken_multiplier: 1.0
    attack_damage_multiplier: 1.0
    mining_speed_multiplier: 0.5
    hunting_efficiency: 1.0

  guardian:
    damage_taken_multiplier: 0.5
    attack_damage_multiplier: 2.0
    mining_speed_multiplier: 0.5
    hunting_efficiency: 0.5

# Reward weights
rewards:
  milestone_base_values:
    tools_t1: 10.0
    tools_t2: 20.0
    tools_t3: 50.0
    tools_t4: 100.0
    hunt_t1: 10.0
    hunt_t2: 20.0
    hunt_t3: 40.0
    defend_t1: 10.0
    defend_t2: 25.0
    defend_t3: 50.0
    defend_t4: 80.0

  wrong_role_multiplier: 0.1
  death_penalty: 0.01
  hunger_penalty: 0.001

  cooperation_bonuses:
    guardian_escort: 5.0
    food_sharing: 2.0
    tool_sharing: 3.0
```

---

## 6. Edge Cases & Anti-Degeneracy

### 6.1 Preventing Single-Role Dominance

**Problem**: Guardian farms monsters for defend rewards, ignoring team goal

**Solution**:
- Defend milestones only unlock after Tools milestones (dependency chain)
- Monster kills give 0 reward unless team has T2+ tools
```python
if milestone == "defend_t3" and not self.team_milestones["tools_t2"]:
    return 0.0  # No reward for clearing iron zone without stone tools
```

### 6.2 Death/Respawn Handling

**Problem**: Engineers repeatedly die trying to solo iron zone

**Solution**:
- Death penalty: -0.01 per death (accumulates)
- Respawn location: Back at safe zone spawn
- Lost inventory: Dropped items despawn after 60 seconds
- Discourages reckless solo attempts

### 6.3 Kiting/Exploit Prevention

**Problem**: Engineer kites monsters indefinitely to avoid combat

**Solution**:
- Hunger system: Running depletes food faster
- Monsters have leash range (return to spawn if kited too far)
- Timeout: Episode ends after max_steps (10k steps)

### 6.4 Role Swap Prevention

**Problem**: Agents try to swap roles mid-episode

**Solution**:
- Roles assigned at reset(), fixed for entire episode
- No role-swap action available
- Equipment restrictions enforced every step

---

## 7. Pseudocode: Key Implementation Logic

### 7.1 Environment Reset
```python
def reset(self, seed=None, options=None):
    # 1. Assign fixed roles
    self.agent_roles = {
        f"agent_{i}": role
        for i, role in enumerate(self.config.roles)
    }

    # 2. Generate Lua init code with role assignments
    init_lua = f"""
    -- Role assignments
    agent_roles = {{
        agent_0 = "{self.agent_roles['agent_0']}",
        agent_1 = "{self.agent_roles['agent_1']}",
        agent_2 = "{self.agent_roles['agent_2']}"
    }}

    -- Role modifiers
    role_modifiers = {{
        engineer = {{damage_taken = 3.0, attack = 0.5}},
        hunter = {{damage_taken = 1.0, attack = 1.0}},
        guardian = {{damage_taken = 0.5, attack = 2.0}}
    }}

    -- Apply modifiers to agents
    for agent_id, role in pairs(agent_roles) do
        local agent = get_agent(agent_id)
        local mods = role_modifiers[role]
        agent:set_damage_multiplier(mods.damage_taken)
        agent:set_attack_multiplier(mods.attack)
    end

    -- Generate guarded zones
    generate_guarded_zones()
    """

    # 3. Create underlying Craftium environment
    self.env = MarlCraftiumEnv(
        env_dir_name="voxel-libre2-roles",
        num_agents=len(self.agent_roles),
        init_code=init_lua,
        obs_width=self.config.obs_width,
        obs_height=self.config.obs_height,
        max_timesteps=self.config.max_steps
    )

    # 4. Reset tracking variables
    self.team_milestones = {m: False for m in MILESTONE_REWARDS.keys()}
    self.total_deaths = 0
    self.cooperation_events = []

    # 5. Get initial observations
    obs_dict, info_dict = self.env.reset(seed=seed)

    # 6. Convert to PettingZoo format + add role info
    observations = {
        agent: obs for agent, obs in obs_dict.items()
    }
    infos = {
        agent: {**info, "role": self.agent_roles[agent]}
        for agent, info in info_dict.items()
    }

    return observations, infos
```

### 7.2 Action Gating (Capability Check)
```python
def _filter_actions(self, actions: Dict[str, int]) -> Dict[str, int]:
    """Block invalid role-specific actions."""
    filtered = {}

    for agent, action in actions.items():
        role = self.agent_roles[agent]
        role_config = ROLES[role]

        # Decode action to check type
        action_info = self._decode_craftium_action(action)

        # Check crafting restrictions
        if action_info["type"] == "craft":
            item = action_info["item"]
            if item not in role_config.crafting_whitelist:
                filtered[agent] = 0  # No-op
                self.infos[agent]["action_blocked"] = item
                continue

        # Check equipment restrictions
        if action_info["type"] == "equip":
            item = action_info["item"]
            if item not in role_config.equipment_whitelist:
                filtered[agent] = 0
                self.infos[agent]["action_blocked"] = item
                continue

        # Action is valid
        filtered[agent] = action

    return filtered

def _decode_craftium_action(self, action: int) -> Dict:
    """Map discrete action ID to semantic info.

    Craftium uses 17 discrete actions (DiscreteActionWrapper):
    0: no-op
    1-4: move (forward, back, left, right)
    5-8: look (up, down, left, right)
    9: jump
    10: sneak
    11: dig
    12: place
    13-17: use/interact with inventory slots
    """
    # Simplified - in reality, need to track agent state
    # (e.g., what item is selected, what they're looking at)

    if action == 11:  # Dig
        return {"type": "dig"}
    elif action == 12:  # Place
        return {"type": "place"}
    # ... would need state tracking for crafting/equipping
    # This is a placeholder - real implementation needs Lua integration

    return {"type": "movement"}
```

### 7.3 Guarded Resource Placement (Lua)
```lua
-- File: craftium-envs/voxel-libre2-roles/mods/world_gen/init.lua

function place_ore_cluster(ore_name, center_pos, radius)
    for x = -radius, radius do
        for y = -radius, radius do
            for z = -radius, radius do
                local pos = {
                    x = center_pos.x + x,
                    y = center_pos.y + y,
                    z = center_pos.z + z
                }

                -- Random distribution (not every block)
                if math.random() < 0.3 then
                    minetest.set_node(pos, {name = ore_name})
                end
            end
        end
    end
end

function spawn_monster(monster_type, pos)
    local entity = minetest.add_entity(pos, "mobs:" .. monster_type)
    if entity then
        entity:set_hp(20)  -- Set health
        entity:set_armor_groups({fleshy = 100})
    end
    return entity
end

function register_monster_spawner(center_pos, monster_type, interval_sec)
    -- Register ABM (Active Block Modifier) for continuous spawning
    minetest.register_abm({
        nodenames = {"default:stone"},
        neighbors = {},
        interval = interval_sec,
        chance = 1,

        action = function(pos, node)
            -- Check if near center_pos
            local dist = vector.distance(pos, center_pos)
            if dist < 20 then
                -- Count existing monsters
                local monsters = minetest.get_objects_inside_radius(center_pos, 20)
                local count = 0
                for _, obj in ipairs(monsters) do
                    if obj:get_luaentity().name:find(monster_type) then
                        count = count + 1
                    end
                end

                -- Spawn if below max density
                local max_monsters = (monster_type == "skeleton" and 8 or 15)
                if count < max_monsters then
                    local spawn_pos = {
                        x = center_pos.x + math.random(-15, 15),
                        y = center_pos.y,
                        z = center_pos.z + math.random(-15, 15)
                    }
                    spawn_monster(monster_type, spawn_pos)
                end
            end
        end
    })
end

-- Call during world initialization
minetest.register_on_generated(function(minp, maxp, seed)
    generate_guarded_zones()
end)
```

### 7.4 Milestone Unlock Gating
```python
def _compute_team_reward(self, infos: Dict) -> float:
    team_reward = 0.0

    for agent, info in infos.items():
        role = self.agent_roles[agent]

        # Check for newly unlocked milestones
        if "milestone_unlocked" in info:
            milestone = info["milestone_unlocked"]

            # Skip if already unlocked
            if self.team_milestones[milestone]:
                continue

            # Mark as unlocked (team-wide)
            self.team_milestones[milestone] = True

            # Get base reward
            base_reward = MILESTONE_REWARDS[milestone]

            # Apply role multiplier
            if self._is_correct_role(role, milestone):
                team_reward += base_reward * 1.0
            else:
                team_reward += base_reward * 0.1  # Penalty

            # Check dependencies (enforce progression order)
            if not self._check_milestone_dependencies(milestone):
                team_reward = 0.0  # Invalidate reward

    return team_reward

def _check_milestone_dependencies(self, milestone: str) -> bool:
    """Enforce progression order: T1 → T2 → T3 → T4."""
    dependencies = {
        "tools_t2": ["tools_t1"],
        "tools_t3": ["tools_t1", "tools_t2"],
        "tools_t4": ["tools_t1", "tools_t2", "tools_t3"],
        "defend_t3": ["tools_t2"],  # Can't clear iron zone without stone tools
        "defend_t4": ["tools_t3"],  # Can't clear diamond zone without iron tools
    }

    if milestone in dependencies:
        for dep in dependencies[milestone]:
            if not self.team_milestones[dep]:
                return False  # Dependency not met

    return True
```

### 7.5 Cooperation Detection
```python
def _detect_cooperation_events(self, infos: Dict) -> float:
    """Detect and reward emergent cooperation behaviors."""
    bonus = 0.0

    # Event 1: Guardian escorts Engineer to danger zone
    engineer_agent = [a for a, r in self.agent_roles.items() if r == "engineer"][0]
    guardian_agent = [a for a, r in self.agent_roles.items() if r == "guardian"][0]

    engineer_pos = infos[engineer_agent].get("position")
    guardian_pos = infos[guardian_agent].get("position")

    if engineer_pos and guardian_pos:
        distance = np.linalg.norm(np.array(engineer_pos) - np.array(guardian_pos))
        in_danger = self._is_danger_zone(engineer_pos)

        if distance < 10.0 and in_danger:
            bonus += self.config.rewards.cooperation_bonuses.guardian_escort

    # Event 2: Hunter shares food with low-health teammate
    hunter_agent = [a for a, r in self.agent_roles.items() if r == "hunter"][0]
    hunter_info = infos[hunter_agent]

    if "gave_food_to" in hunter_info:
        recipient = hunter_info["gave_food_to"]
        recipient_health = infos[recipient].get("health", 20)

        if recipient_health < 10:  # Low health
            bonus += self.config.rewards.cooperation_bonuses.food_sharing

    # Event 3: Engineer shares tools
    # (Requires tracking item transfers - simplified here)

    return bonus

def _is_danger_zone(self, position: np.ndarray) -> bool:
    """Check if position is in iron or diamond zone."""
    iron_center = np.array(self.config.world_gen.iron_zone_center)
    diamond_center = np.array(self.config.world_gen.diamond_zone_center)

    dist_to_iron = np.linalg.norm(position - iron_center)
    dist_to_diamond = np.linalg.norm(position - diamond_center)

    return dist_to_iron < 20 or dist_to_diamond < 25
```

---

## 8. Validation Tests

### Test Suite Design
**File**: `tests/test_role_cooperation.py`

```python
import pytest
from src.envs.openworld_roles import RoleBasedOpenWorldEnv

class TestRoleCooperation:

    @pytest.fixture
    def env(self):
        return RoleBasedOpenWorldEnv(
            num_agents=3,
            roles=["engineer", "hunter", "guardian"],
            max_steps=5000
        )

    # ========================================
    # Test 1: Single role cannot complete T4
    # ========================================
    def test_engineer_alone_fails_t4(self, env):
        """Engineer alone cannot reach diamond zone (dies to monsters)."""
        obs, info = env.reset(seed=42)

        # Simulate 1000 steps with only engineer acting
        # (Hunter and Guardian take no-ops)
        deaths = 0
        t4_unlocked = False

        for step in range(1000):
            actions = {
                "agent_0": env.action_space("agent_0").sample(),  # Engineer
                "agent_1": 0,  # Hunter no-op
                "agent_2": 0   # Guardian no-op
            }

            obs, rewards, terms, truncs, infos = env.step(actions)

            # Track deaths
            if infos["agent_0"].get("died", False):
                deaths += 1

            # Check if T4 unlocked
            if env.team_milestones.get("tools_t4", False):
                t4_unlocked = True
                break

        # Assertions
        assert deaths > 0, "Engineer should die at least once trying to reach diamond"
        assert not t4_unlocked, "T4 should not unlock with engineer alone"

    # ========================================
    # Test 2: Team cooperation enables T4
    # ========================================
    def test_team_cooperation_succeeds_t4(self, env):
        """Full team with coordination can unlock T4."""
        obs, info = env.reset(seed=42)

        # Use a scripted policy that mimics cooperation:
        # - Engineer mines in safe zone (T1, T2)
        # - Guardian clears iron zone
        # - Engineer mines iron (T3)
        # - Guardian clears diamond zone
        # - Engineer mines diamond (T4)

        # (Simplified - in practice, use trained agents)
        t4_unlocked = False

        for step in range(5000):
            # Scripted actions (placeholder)
            actions = self._get_scripted_cooperative_actions(env, obs, info, step)

            obs, rewards, terms, truncs, infos = env.step(actions)

            if env.team_milestones.get("tools_t4", False):
                t4_unlocked = True
                break

        assert t4_unlocked, "T4 should unlock with cooperative team"

    # ========================================
    # Test 3: Role swap breaks progression
    # ========================================
    def test_wrong_role_penalties(self, env):
        """Wrong role unlocking milestone gets 10% reward."""
        obs, info = env.reset(seed=42)

        # Simulate hunter unlocking tools_t3 (wrong role)
        # (In practice, hunter can't craft T3, but test reward logic)

        infos_fake = {
            "agent_1": {  # Hunter
                "milestone_unlocked": "tools_t3",
                "role": "hunter"
            }
        }

        reward = env._compute_team_reward(infos_fake)

        # Expected: 50.0 (base) * 0.1 (wrong role) = 5.0
        assert reward == pytest.approx(5.0), f"Expected 5.0, got {reward}"

    # ========================================
    # Test 4: Guardian escort bonus triggers
    # ========================================
    def test_guardian_escort_bonus(self, env):
        """Guardian near Engineer in danger zone triggers cooperation bonus."""
        obs, info = env.reset(seed=42)

        # Fake positions: both in iron zone, close together
        infos_fake = {
            "agent_0": {  # Engineer
                "position": np.array([120, 0, 120]),  # Iron zone center
                "role": "engineer"
            },
            "agent_1": {  # Hunter
                "position": np.array([0, 0, 0]),
                "role": "hunter"
            },
            "agent_2": {  # Guardian
                "position": np.array([125, 0, 122]),  # 5 blocks from engineer
                "role": "guardian"
            }
        }

        bonus = env._detect_cooperation_events(infos_fake)

        # Expected: guardian_escort bonus (5.0)
        assert bonus == pytest.approx(5.0), f"Expected 5.0, got {bonus}"

    # ========================================
    # Test 5: Milestone dependencies enforced
    # ========================================
    def test_milestone_dependencies(self, env):
        """Cannot unlock T4 without T1, T2, T3."""
        obs, info = env.reset(seed=42)

        # Fake unlocking T4 without prerequisites
        env.team_milestones = {
            "tools_t1": False,
            "tools_t2": False,
            "tools_t3": False,
            "tools_t4": False
        }

        infos_fake = {
            "agent_0": {
                "milestone_unlocked": "tools_t4",
                "role": "engineer"
            }
        }

        reward = env._compute_team_reward(infos_fake)

        # Expected: 0.0 (dependency check fails)
        assert reward == 0.0, f"Expected 0.0 (blocked), got {reward}"

        # Now unlock prerequisites and retry
        env.team_milestones["tools_t1"] = True
        env.team_milestones["tools_t2"] = True
        env.team_milestones["tools_t3"] = True

        reward = env._compute_team_reward(infos_fake)

        # Expected: 100.0 (full T4 reward)
        assert reward == 100.0, f"Expected 100.0, got {reward}"

    # ========================================
    # Helper: Scripted cooperative policy
    # ========================================
    def _get_scripted_cooperative_actions(self, env, obs, info, step):
        """Simple scripted policy for testing (not trainable)."""
        # Phase 1 (0-500): Engineer mines wood/stone, Guardian stays near spawn
        # Phase 2 (500-1000): Guardian moves to iron zone
        # Phase 3 (1000-2000): Engineer mines iron with Guardian escort
        # ... etc

        # Placeholder: random actions
        return {agent: env.action_space(agent).sample() for agent in env.agents}
```

### Test Execution
```bash
# Run validation tests
poetry run pytest tests/test_role_cooperation.py -v

# Expected output:
# test_engineer_alone_fails_t4 ✓
# test_team_cooperation_succeeds_t4 ✓
# test_wrong_role_penalties ✓
# test_guardian_escort_bonus ✓
# test_milestone_dependencies ✓
```

---

## 9. Summary: Implementation Checklist

- [ ] **Create role config system** (`src/envs/openworld_roles.py`)
  - [ ] Define `RoleConfig` dataclass
  - [ ] Define 3 roles (engineer, hunter, guardian)
  - [ ] Implement role assignment at reset

- [ ] **Implement action gating**
  - [ ] Action filtering in `step()` method
  - [ ] Crafting whitelist enforcement
  - [ ] Equipment restriction checks

- [ ] **Create guarded world generation**
  - [ ] New Craftium env: `voxel-libre2-roles/`
  - [ ] Lua script for ore placement
  - [ ] Monster spawning system (iron zone, diamond zone)

- [ ] **Build reward system**
  - [ ] Milestone reward computation
  - [ ] Role-based multipliers
  - [ ] Cooperation bonus detection
  - [ ] Dependency checking

- [ ] **Add Hydra config** (`configs/env/openworld_roles.yaml`)

- [ ] **Write validation tests** (`tests/test_role_cooperation.py`)

- [ ] **Integration with existing scaffold**
  - [ ] Update `src/envs/__init__.py` to export new env
  - [ ] Create training script for role-based env
  - [ ] Test with MAPPO/QMIX algorithms

---

## 10. Expected Emergent Behaviors

If implemented correctly, trained agents should exhibit:

1. **Division of labor**
   - Engineer focuses on mining/crafting
   - Hunter maintains team food supply
   - Guardian leads expeditions to danger zones

2. **Spatial coordination**
   - Guardian escorts Engineer to iron/diamond zones
   - Agents regroup at safe zone after resource runs
   - Guardian establishes "safe perimeter" before Engineer mines

3. **Resource sharing**
   - Hunter delivers food to teammates before long trips
   - Engineer crafts multiple pickaxes for team
   - Guardian protects shared resource stockpiles

4. **Failure modes (if lacking cooperation)**
   - Engineer dies repeatedly trying to solo mine
   - Team starves without Hunter contributions
   - Cannot progress past T2 without Guardian clearing zones

This design creates a **true cooperative Dec-POMDP** where single-agent play is provably suboptimal, forcing emergence of coordinated strategies.
