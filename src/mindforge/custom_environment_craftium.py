"""Environment adapter bridging Craftium OpenWorld multi-agent env to CausalForge's interface."""

import sys
import os
import numpy as np
import PIL.Image

# Import the PettingZoo env from src/craftium/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)  # src/
_craftium_dir = os.path.join(_src_dir, "craftium")
if _craftium_dir not in sys.path:
    sys.path.insert(0, _craftium_dir)

from openworld_multi_agents import OpenWorldMultiAgentEnv

# Load environment prompt
with open(os.path.join(_this_dir, "prompts", "environment_prompt.txt"), "r") as f:
    environment_prompt = f.read()

# Map from human-readable action names (used by the LLM) to Discrete(23) integers.
# The integer mapping matches _DISCRETE_ACTIONS in openworld_multi_agents.py:
#   0 = NOP
#   1-16 = forward, backward, left, right, jump, sneak, dig, place,
#           slot_1..slot_5, mouse_x+, mouse_x-, mouse_y+
ACTION_MAP = {
    "NoOp": 0,
    "MoveForward": 1,
    "MoveBackward": 2,
    "MoveLeft": 3,
    "MoveRight": 4,
    "Jump": 5,
    "Sneak": 6,
    "Dig": 7,
    "Place": 8,
    "Slot1": 9,
    "Slot2": 10,
    "Slot3": 11,
    "Slot4": 12,
    "Slot5": 13,
    "TurnRight": 14,
    "TurnLeft": 15,
    "LookDown": 16,
    "LookUp": 17,
    # --- added actions ---
    "Inventory": 0,    # mapped to NoOp — inventory info is now in the text prompt
    "Drop": 19,         # drop the currently held item
    "Slot6": 20,
    "Slot7": 21,
    "Slot8": 22,
}

VALID_ACTIONS = [k for k in ACTION_MAP if k != "Inventory"]


class CraftiumEnvironmentInterface:
    """Wraps OpenWorldMultiAgentEnv to match CausalForge's Environment_Interface contract.

    CausalForge expects:
      - step(action_str, agentId) -> (event, action_str)
      - environment_prompt (str)
      - pickedup_object(agentId) -> str | None
    """

    # Actions considered "idle" — repeating these wastes steps
    _IDLE_ACTIONS = frozenset({"NoOp", "Inventory"})

    # Camera pitch cap: agent may look at most this many LookDown steps below
    # horizontal, or this many LookUp steps above it.  Beyond these limits the
    # action is redirected to keep the crosshair in a useful range.
    # Each LookDown/LookUp step moves ~10-15° at _MOUSE_MOV=0.5.
    # Cap of 4 ≈ 40-60° down (enough to aim at ground), 2 up (enough to see trees).
    _PITCH_MAX_DOWN = 4   # positive pitch limit (looking down)
    _PITCH_MAX_UP   = 2   # negative pitch limit (looking up)
    _MAX_CONSECUTIVE_IDLE = 1

    # Actions that need to be held for multiple env ticks to take effect.
    # Minetest requires sustained key-press to break blocks / kill mobs.
    # With frameskip=3, each env.step() = 3 physics ticks, so multiply accordingly.
    # VoxeLibre bare-hand: wood ~15 physics ticks, stone ~30.
    # At frameskip=3: 3 env steps = 9 ticks (enough for wood), 6 steps = 18 ticks.
    _SUSTAINED_TICKS = {
        "Dig": 3,
    }

    # Position-stuck detection: if x/z haven't moved more than this threshold
    # after this many consecutive movement actions, the agent is physically stuck
    # (e.g. trapped in a self-dug pit or pressed against terrain).
    _STUCK_THRESHOLD_XZ = 1.0   # blocks — less than this = stuck (higher with frameskip=3)
    _STUCK_MAX_STEPS = 8        # consecutive movement steps before escape
    # Escape sequence to execute when stuck: (action, repeat_ticks)
    _ESCAPE_SEQUENCE = [
        ("Jump", 3),
        ("MoveBackward", 5),
        ("TurnRight", 4),
        ("Jump", 3),
        ("MoveForward", 8),
    ]

    def __init__(self, num_agents=3, obs_width=480, obs_height=480, max_steps=10000, seed=None, frameskip=3, pmul=20):
        self.num_agents = num_agents
        self.seed = seed
        self.env = OpenWorldMultiAgentEnv(
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            max_steps=max_steps,
            seed=seed,
            frameskip=frameskip,
            pmul=pmul,
        )
        self.environment_prompt = environment_prompt

        # Per-agent state
        self._observations = {}   # agent_name -> np.ndarray (H, W, 3)
        self._rewards = {}        # agent_name -> float (cumulative since last query)
        self._step_rewards = {}   # agent_name -> float (raw reward from last step)
        self._terminations = {}   # agent_name -> bool
        self._truncations = {}    # agent_name -> bool
        self._infos = {}          # agent_name -> dict
        self._last_actions = {}   # agent_name -> str
        self._consecutive_idle = {}  # agent_name -> int

        # Position-stuck tracking
        self._last_xz = {}           # agent_name -> (x, z) at last movement action
        self._stuck_move_steps = {}  # agent_name -> int, consecutive non-idle steps without xz change
        self._escape_queue = {}      # agent_name -> list of (action, ticks) remaining

        # Camera pitch tracker (net LookDown steps minus LookUp steps).
        # Positive = looking down, negative = looking up.
        # Capped at [-_PITCH_CAP, +_PITCH_CAP] to prevent staring at sky/ground.
        self._pitch = {}  # agent_name -> int

        # Server log tailer — surfaces [TOOLS]/[INVENTORY]/[TRACK STATUS] lines from Lua
        self._server_log_offset = 0   # byte offset into stderr.txt, tracks what we've already read
        self._server_log_path = None  # resolved lazily after first reset()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the environment and return initial observations."""
        observations, infos = self.env.reset()
        self._observations = observations
        self._infos = infos
        self._rewards = {a: 0.0 for a in self.env.possible_agents}
        self._terminations = {a: False for a in self.env.possible_agents}
        self._truncations = {a: False for a in self.env.possible_agents}
        self._last_actions = {a: "NoOp" for a in self.env.possible_agents}
        self._consecutive_idle = {a: 0 for a in self.env.possible_agents}
        self._last_xz = {}
        self._stuck_move_steps = {}
        self._escape_queue = {}
        self._pitch = {}
        self.reset_log_offset()
        return self._observations

    def step(self, action_str: str, agentId: int):
        """Execute one action for one agent, NoOp for the rest, then step the env.

        Args:
            action_str: Action name (e.g. "Dig", "MoveForward")
            agentId: Integer index of the acting agent

        Returns:
            (observations_dict, action_str) — matching CausalForge's contract

        Raises:
            ValueError: If action_str is not in ACTION_MAP
        """
        if action_str not in ACTION_MAP:
            import logging
            logging.warning(
                f"Invalid action: '{action_str}', clamping to NoOp. "
                f"Valid actions: {VALID_ACTIONS}"
            )
            action_str = "NoOp"

        import logging as _logging

        # Guard: break idle loops (NoOp/Inventory spam)
        agent_name = f"agent_{agentId}"
        if action_str in self._IDLE_ACTIONS:
            self._consecutive_idle[agent_name] = (
                self._consecutive_idle.get(agent_name, 0) + 1
            )
            if self._consecutive_idle[agent_name] >= self._MAX_CONSECUTIVE_IDLE:
                _logging.warning(
                    f"Agent {agentId} idle for {self._consecutive_idle[agent_name]} "
                    f"steps, forcing MoveForward"
                )
                action_str = "MoveForward"
                self._consecutive_idle[agent_name] = 0
        else:
            self._consecutive_idle[agent_name] = 0

        # Guard: position-stuck detection.
        # If an escape sequence is queued, consume from it first.
        if self._escape_queue.get(agent_name):
            esc_action, esc_ticks = self._escape_queue[agent_name][0]
            esc_ticks -= 1
            if esc_ticks <= 0:
                self._escape_queue[agent_name].pop(0)
            else:
                self._escape_queue[agent_name][0] = (esc_action, esc_ticks)
            action_str = esc_action
        else:
            # Track whether x/z position changed on movement actions
            _movement_actions = frozenset({
                "MoveForward", "MoveBackward", "MoveLeft", "MoveRight", "Jump"
            })
            if action_str in _movement_actions:
                try:
                    pos = self.env.env._positions[agentId]
                    cur_xz = (pos[0], pos[2]) if pos is not None else None
                except (AttributeError, IndexError, TypeError):
                    cur_xz = None

                if cur_xz is not None:
                    last_xz = self._last_xz.get(agent_name)
                    if last_xz is not None:
                        dx = abs(cur_xz[0] - last_xz[0])
                        dz = abs(cur_xz[1] - last_xz[1])
                        if dx < self._STUCK_THRESHOLD_XZ and dz < self._STUCK_THRESHOLD_XZ:
                            self._stuck_move_steps[agent_name] = (
                                self._stuck_move_steps.get(agent_name, 0) + 1
                            )
                        else:
                            self._stuck_move_steps[agent_name] = 0
                            self._last_xz[agent_name] = cur_xz
                    else:
                        self._last_xz[agent_name] = cur_xz
                        self._stuck_move_steps[agent_name] = 0

                    if self._stuck_move_steps.get(agent_name, 0) >= self._STUCK_MAX_STEPS:
                        _logging.warning(
                            f"Agent {agentId} physically stuck at xz={cur_xz} for "
                            f"{self._stuck_move_steps[agent_name]} steps — injecting escape sequence"
                        )
                        self._stuck_move_steps[agent_name] = 0
                        self._last_xz[agent_name] = None
                        self._escape_queue[agent_name] = [
                            list(step) for step in self._ESCAPE_SEQUENCE
                        ]
                        esc_action, esc_ticks = self._escape_queue[agent_name][0]
                        self._escape_queue[agent_name][0] = [esc_action, esc_ticks - 1]
                        if self._escape_queue[agent_name][0][1] <= 0:
                            self._escape_queue[agent_name].pop(0)
                        action_str = esc_action
            else:
                # Non-movement action — reset stuck counter (position change not expected)
                self._stuck_move_steps[agent_name] = 0

        # Reset pitch on respawn: if the agent died last step, the server respawns
        # them looking horizontally, so the stored pitch offset is stale.
        if self._terminations.get(agent_name, False):
            self._pitch[agent_name] = 0

        # Camera pitch cap: prevent runaway LookUp (staring at sky) / LookDown (ground only).
        pitch = self._pitch.get(agent_name, 0)
        if action_str == "LookDown":
            if pitch >= self._PITCH_MAX_DOWN:
                _logging.debug(f"Agent {agentId} pitch cap hit (down={pitch}), redirecting LookDown → NoOp")
                action_str = "NoOp"
            else:
                self._pitch[agent_name] = pitch + 1
        elif action_str == "LookUp":
            if pitch <= -self._PITCH_MAX_UP:
                _logging.debug(f"Agent {agentId} pitch cap hit (up={-pitch}), redirecting LookUp → NoOp")
                action_str = "NoOp"
                # pitch stays at its current capped value — no increment
            else:
                self._pitch[agent_name] = pitch - 1

        # Build action dict: acting agent gets the requested action, others NoOp
        actions = {}
        for i in range(self.num_agents):
            ag = f"agent_{i}"
            if ag not in self.env.agents:
                continue  # skip terminated agents
            if i == agentId:
                actions[ag] = ACTION_MAP[action_str]
            else:
                actions[ag] = ACTION_MAP["NoOp"]

        # Auto-equip: before Dig, switch to the best available tool.
        # Reads the inventory file and sends one slot-switch tick if needed.
        if action_str == "Dig":
            best_slot = self._find_best_tool(agentId)
            if best_slot is not None:
                equip_actions = {}
                slot_action_id = ACTION_MAP.get(f"Slot{best_slot}")
                for ag_name in actions:
                    i = int(ag_name.split("_")[1])
                    equip_actions[ag_name] = slot_action_id if i == agentId else ACTION_MAP["NoOp"]
                self.env.step(equip_actions)  # one tick to switch slot

        # Jump+forward pairing: a bare Jump rarely clears obstacles since the
        # agent doesn't move horizontally while airborne.  Send MoveForward on
        # the tick immediately after Jump so the agent actually vaults over terrain.
        if action_str == "Jump":
            fwd_actions = {ag: (ACTION_MAP["MoveForward"] if ag == f"agent_{agentId}" else ACTION_MAP["NoOp"])
                           for ag in actions}
            self.env.step(actions)      # jump tick
            actions = fwd_actions       # forward tick (becomes the "main" step below)

        # Sustained actions (Dig): repeat for multiple env ticks so blocks
        # actually break and mobs actually take damage.
        repeat = self._SUSTAINED_TICKS.get(action_str, 1)
        total_rewards = {ag: 0.0 for ag in actions}

        for tick in range(repeat):
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            for ag in rewards:
                if ag in total_rewards:
                    total_rewards[ag] += rewards[ag]
            # Stop early if acting agent died or episode ended
            acting_ag = f"agent_{agentId}"
            if terminations.get(acting_ag, False) or truncations.get(acting_ag, False):
                break

        self._observations = observations
        self._terminations = terminations
        self._truncations = truncations
        self._infos = infos

        # Surface any new Lua log lines (block breaks, kills, stage completions, etc.)
        self.tail_server_log()

        # Store raw per-step rewards (summed across sustained ticks) and accumulate
        self._step_rewards = dict(total_rewards)
        for agent_name, rew in total_rewards.items():
            self._rewards[agent_name] = self._rewards.get(agent_name, 0.0) + rew

        agent_name = f"agent_{agentId}"
        self._last_actions[agent_name] = action_str

        return self._observations, action_str

    # ------------------------------------------------------------------
    # Observation helpers (match CausalForge's usage patterns)
    # ------------------------------------------------------------------
    def get_agent_frame(self, agentId: int) -> np.ndarray:
        """Return the raw observation array (H, W, 3) for a specific agent."""
        agent_name = f"agent_{agentId}"
        return self._observations.get(agent_name)

    def get_pil_image(self, agentId: int) -> PIL.Image.Image:
        """Return a PIL Image for the given agent's current view."""
        frame = self.get_agent_frame(agentId)
        if frame is None:
            # Return a blank image if no observation available
            return PIL.Image.new("RGB", (self.env.obs_width, self.env.obs_height), (0, 0, 0))
        return PIL.Image.fromarray(frame)

    def get_reward_summary(self, agentId: int) -> str:
        """Return a text summary of the agent's reward and status.

        This is injected into the instruction prompt so the LLM sees its reward.
        Reading the summary resets the accumulated reward for that agent.
        """
        agent_name = f"agent_{agentId}"
        reward = self._rewards.get(agent_name, 0.0)
        terminated = self._terminations.get(agent_name, False)
        truncated = self._truncations.get(agent_name, False)

        if terminated:
            status = "Dead"
        elif truncated:
            status = "Episode ended"
        else:
            status = "Alive"

        # Reset accumulated reward after reading
        self._rewards[agent_name] = 0.0

        return f"Reward: {reward:.2f} / Status: {status}"

    def any_done(self) -> bool:
        """Check if any agent is terminated or truncated."""
        for agent_name in self.env.possible_agents:
            if self._terminations.get(agent_name, False):
                return True
            if self._truncations.get(agent_name, False):
                return True
        return False

    def all_done(self) -> bool:
        """Check if all agents are terminated or truncated."""
        for agent_name in self.env.possible_agents:
            if not self._terminations.get(agent_name, False) and not self._truncations.get(agent_name, False):
                return False
        return True

    def get_step_reward(self, agentId: int) -> float:
        """Return the raw reward from the last env.step() for this agent."""
        agent_name = f"agent_{agentId}"
        return self._step_rewards.get(agent_name, 0.0)

    def pickedup_object(self, agentId: int = 0):
        """Return a formatted inventory string read from a file written by the Lua mod.

        The server-side Lua mod writes per-player inventory files to the world
        directory every globalstep.  Format:
            wield_idx|item1_name count|item2_name count|...|item9_name count
        Empty slots are empty strings between pipes.
        """
        return self._read_inventory_file(agentId)

    @staticmethod
    def _tod_to_clock(tod: float) -> str:
        """Convert Minetest time-of-day float (0.0-1.0) to a readable clock string."""
        hours_f = tod * 24.0
        h = int(hours_f) % 24
        m = int((hours_f - int(hours_f)) * 60)
        period = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        phase = "day" if 6 <= h < 20 else "night"
        return f"{h12}:{m:02d} {period} ({phase})"

    def get_player_status_text(self, agentId: int) -> str:
        """Return a single string with health, hunger, and time for the given agent."""
        world_path = self._get_world_path()
        agent_name = f"agent{agentId}"

        def _read(path, fallback):
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except (FileNotFoundError, OSError):
                return fallback

        health = _read(os.path.join(world_path, f"health_{agent_name}.txt"), "?/20")
        hunger = _read(os.path.join(world_path, f"hunger_{agent_name}.txt"), "?/20")

        tod_raw = _read(os.path.join(world_path, "timeofday.txt"), None)
        if tod_raw is not None:
            try:
                time_str = self._tod_to_clock(float(tod_raw))
            except ValueError:
                time_str = "Unknown"
        else:
            time_str = "Unknown"

        return f"Health: {health} | Hunger: {hunger} | Time: {time_str}"

    def get_position_text(self, agentId: int) -> str:
        """Return a formatted position string for the given agent, or 'Unknown'."""
        try:
            pos = self.env.env._positions[agentId]
            if pos is not None:
                return f"x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}"
        except (AttributeError, IndexError, TypeError):
            pass
        return "Unknown"

    def warmup_noop(self):
        """Send NoOps to keep channels alive without incrementing step counters.

        Use this during media-loading warm-up instead of step().
        Returns list of observations (one per agent).
        """
        return self.env.warmup_noop()

    # ------------------------------------------------------------------
    # Inventory file reading
    # ------------------------------------------------------------------
    def _get_world_path(self):
        """Return the absolute path to the Minetest world directory."""
        if not hasattr(self, "_world_path"):
            srv_run_dir = self.env.env.mt_server.run_dir
            self._world_path = os.path.join(
                os.path.abspath(srv_run_dir), "worlds", "world"
            )
        return self._world_path

    @staticmethod
    def _pretty_item_name(raw_name: str) -> str:
        """Turn 'mcl_tools:pick_iron' into 'iron pickaxe', etc."""
        # Strip namespace prefix (e.g. "mcl_tools:", "mcl_torches:")
        if ":" in raw_name:
            raw_name = raw_name.split(":", 1)[1]
        # Swap order for tool names: pick_iron -> iron pick, sword_stone -> stone sword
        parts = raw_name.split("_")
        if len(parts) == 2 and parts[0] in ("pick", "sword", "axe", "shovel", "hoe"):
            tool_names = {"pick": "pickaxe"}
            tool = tool_names.get(parts[0], parts[0])
            return f"{parts[1]} {tool}"
        return raw_name.replace("_", " ")

    def _read_inventory_file(self, agentId: int):
        """Read and format the inventory file for the given agent."""
        world_path = self._get_world_path()
        inv_file = os.path.join(world_path, f"inv_agent{agentId}.txt")
        try:
            with open(inv_file, "r") as f:
                raw = f.read().strip()
        except (FileNotFoundError, OSError):
            return None

        if not raw:
            return None

        parts = raw.split("|")
        if len(parts) < 2:
            return None

        try:
            wield_idx = int(parts[0])
        except ValueError:
            return None

        slots = parts[1:]  # up to 9 slot entries
        lines = []
        for i, slot_str in enumerate(slots):
            slot_num = i + 1
            if not slot_str:
                lines.append(f"  [{slot_num}] empty")
            else:
                # "mcl_tools:pick_iron 1" -> name="mcl_tools:pick_iron", count="1"
                tokens = slot_str.rsplit(" ", 1)
                name = self._pretty_item_name(tokens[0])
                count = tokens[1] if len(tokens) > 1 else "1"
                marker = " <-- wielding" if slot_num == wield_idx else ""
                lines.append(f"  [{slot_num}] {name} x{count}{marker}")

        return "Hotbar:\n" + "\n".join(lines)

    # Tool tier ranking — higher = better. Covers pickaxes, swords, and axes.
    _TOOL_TIER = {
        "diamond": 5, "iron": 4, "stone": 3, "wood": 2, "gold": 1,
    }
    _TOOL_TYPES = {"pick", "sword", "axe", "shovel", "hoe"}

    def _find_best_tool(self, agentId: int):
        """Read inventory and return the slot number (1-8) of the best tool, or None."""
        world_path = self._get_world_path()
        inv_file = os.path.join(world_path, f"inv_agent{agentId}.txt")
        try:
            with open(inv_file, "r") as f:
                raw = f.read().strip()
        except (FileNotFoundError, OSError):
            return None

        if not raw:
            return None

        parts = raw.split("|")
        if len(parts) < 2:
            return None

        try:
            wield_idx = int(parts[0])
        except ValueError:
            return None

        slots = parts[1:]
        best_slot = None
        best_tier = -1

        for i, slot_str in enumerate(slots):
            if not slot_str:
                continue
            item_name = slot_str.rsplit(" ", 1)[0]  # "mcl_tools:pick_iron 1" -> "mcl_tools:pick_iron"
            if ":" in item_name:
                item_name = item_name.split(":", 1)[1]  # "pick_iron"
            item_parts = item_name.split("_")
            if len(item_parts) >= 2 and item_parts[0] in self._TOOL_TYPES:
                tier = self._TOOL_TIER.get(item_parts[1], 0)
                if tier > best_tier:
                    best_tier = tier
                    best_slot = i + 1  # 1-indexed

        # Only switch if we found a tool and it's not already wielded
        if best_slot is not None and best_slot != wield_idx:
            return best_slot
        return None

    # ------------------------------------------------------------------
    # Server log tailer
    # ------------------------------------------------------------------

    # Lines from the Lua server log that are worth surfacing in Python output.
    _LOG_TAGS = ("[TOOLS]", "[INVENTORY]", "[TRACK STATUS]", "[DIG]", "[HUNT]", "[DEFEND]")

    def _get_server_log_path(self):
        """Resolve and cache the path to the server's stderr.txt."""
        if self._server_log_path is None:
            try:
                srv_run_dir = self.env.env.mt_server.run_dir
                self._server_log_path = os.path.join(os.path.abspath(srv_run_dir), "stderr.txt")
            except AttributeError:
                pass
        return self._server_log_path

    def tail_server_log(self, tags=None, print_lines=True):
        """Read any new lines from the server's stderr.txt since the last call.

        Only lines that start with one of the watched tags are returned.
        Calls after reset() automatically re-anchor to the current EOF so old
        log lines from previous runs don't flood the output.

        Args:
            tags: tuple of tag strings to filter on (default: _LOG_TAGS)
            print_lines: if True, print matching lines to stdout

        Returns:
            list[str] of matching new lines (without trailing newline)
        """
        path = self._get_server_log_path()
        if path is None or not os.path.exists(path):
            return []

        tags = tags or self._LOG_TAGS
        matched = []
        try:
            with open(path, "r", errors="replace") as f:
                f.seek(self._server_log_offset)
                for line in f:
                    stripped = line.rstrip()
                    if any(stripped.startswith(t) for t in tags):
                        matched.append(stripped)
                        if print_lines:
                            print(f"  [SRV] {stripped}")
                self._server_log_offset = f.tell()
        except OSError:
            pass
        return matched

    def reset_log_offset(self):
        """Anchor the log tailer to the current EOF (call after env.reset()).

        This prevents lines from a previous episode from appearing in the
        current one.
        """
        path = self._get_server_log_path()
        if path is None:
            self._server_log_offset = 0
            return
        try:
            self._server_log_offset = os.path.getsize(path)
        except OSError:
            self._server_log_offset = 0

    def close(self):
        """Clean up the underlying environment."""
        self.env.close()
