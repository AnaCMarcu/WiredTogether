"""Environment adapter bridging the Craftium OpenWorld multi-agent env to the agent interface."""

import os
import re
import json
import numpy as np
import PIL.Image

from marl_craftium import OpenWorldMultiAgentEnv

_this_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment prompt
with open(os.path.join(_this_dir, "prompts", "environment_prompt.txt"), "r") as f:
    environment_prompt = f.read()

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
    "Drop": 19,         # drop the currently held item
    "Slot6": 20,
    "Slot7": 21,
    "Slot8": 22,
}

VALID_ACTIONS = list(ACTION_MAP)

_ACTION_ALIASES = {
    # combat / breaking -> Dig
    "attack": "Dig", "mine": "Dig", "hit": "Dig", "punch": "Dig",
    "break": "Dig", "chop": "Dig", "kill": "Dig", "strike": "Dig",
    "swing": "Dig", "fight": "Dig", "smash": "Dig",
    # forward / backward / strafe
    "forward": "MoveForward", "moveup": "MoveForward", "advance": "MoveForward",
    "walkforward": "MoveForward", "goforward": "MoveForward", "step": "MoveForward",
    "backward": "MoveBackward", "moveback": "MoveBackward", "back": "MoveBackward",
    "retreat": "MoveBackward",
    "strafeleft": "MoveLeft", "strafright": "MoveRight", "straferight": "MoveRight",
    # turning / camera. Bare "turn" defaults to TurnRight (direction unspecified).
    "turn": "TurnRight", "rotate": "TurnRight", "rotateright": "TurnRight",
    "turnclockwise": "TurnRight", "rotateleft": "TurnLeft",
    "turncounterclockwise": "TurnLeft",
    "turnup": "LookUp", "lookup": "LookUp", "lookupward": "LookUp", "tiltup": "LookUp",
    "raisecamera": "LookUp",
    "turndown": "LookDown", "lookdown": "LookDown", "lookdownward": "LookDown",
    "tiltdown": "LookDown", "lowercamera": "LookDown",
    # place / drop / jump / sneak
    "placeblock": "Place", "put": "Place", "build": "Place",
    "throw": "Drop", "discard": "Drop",
    "hop": "Jump", "leap": "Jump",
    "crouch": "Sneak",
    # explicit idling
    "wait": "NoOp", "stay": "NoOp", "idle": "NoOp", "stop": "NoOp",
    "donothing": "NoOp", "nothing": "NoOp", "none": "NoOp", "hold": "NoOp",
    "stand": "NoOp",
}


def _normalize_action_key(name) -> str:
    """Lowercase and strip spaces / underscores / hyphens for fuzzy matching."""
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


# normalized valid-name -> canonical valid name (e.g. "moveforward" -> "MoveForward")
_VALID_ACTION_NORMALIZED = {_normalize_action_key(a): a for a in ACTION_MAP}


def canonicalize_action(action_str):
    """Resolve a possibly-malformed action name to a valid ACTION_MAP key, or None.

    Tries in order: exact match, format-normalized match (case / spaces /
    underscores, e.g. 'move_forward' or 'MOVEFORWARD'), the synonym alias map
    (e.g. 'Attack' -> 'Dig', 'TurnUp' -> 'LookUp'), and finally the leading
    word of a multi-word string ('MoveForward to the door' -> 'MoveForward').
    Returns the canonical action name, or None if unrecoverable.
    """
    if action_str in ACTION_MAP:
        return action_str
    norm = _normalize_action_key(action_str)
    if norm in _VALID_ACTION_NORMALIZED:
        return _VALID_ACTION_NORMALIZED[norm]
    if norm in _ACTION_ALIASES:
        return _ACTION_ALIASES[norm]
    tokens = str(action_str).strip().split()
    if tokens:
        first = _normalize_action_key(tokens[0])
        if first in _VALID_ACTION_NORMALIZED:
            return _VALID_ACTION_NORMALIZED[first]
        if first in _ACTION_ALIASES:
            return _ACTION_ALIASES[first]
    return None

class CraftiumEnvironmentInterface:
    """Wraps OpenWorldMultiAgentEnv to match the agent stack's Environment interface contract.

    The agent stack expects:
      - step(action_str, agentId) -> (event, action_str)
      - environment_prompt (str)
      - pickedup_object(agentId) -> str | None
    """

    # Actions considered "idle" — repeating these wastes steps
    _IDLE_ACTIONS = frozenset({"NoOp"})

    _PITCH_MAX_DOWN = 2   # positive pitch limit (looking down)
    _PITCH_MAX_UP   = 1   # negative pitch limit (looking up); ~20-30° matches original ~30°
    _MAX_CONSECUTIVE_IDLE = 1

    # Actions that need to be held for multiple env ticks to take effect.
    _SUSTAINED_TICKS = {
        "Dig": 10,  # 10 env steps × frameskip=3 = 30 physics ticks — breaks stone in 1 action
    }

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

    def __init__(self, num_agents=3, obs_width=480, obs_height=480, max_steps=10000,
                 seed=None, frameskip=3, pmul=20, voxel_obs=False,
                 voxel_obs_rx=10, voxel_obs_ry=5, voxel_obs_rz=10):
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
            voxel_obs=voxel_obs,
            voxel_obs_rx=voxel_obs_rx,
            voxel_obs_ry=voxel_obs_ry,
            voxel_obs_rz=voxel_obs_rz,
        )
        self.voxel_obs_enabled = voxel_obs
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

        self._pitch = {}  # agent_name -> int

        self._futile_actions = {}  # agent_name -> int

        # Server log tailer — surfaces [TOOLS]/[INVENTORY]/[TRACK STATUS] lines from Lua
        self._server_log_offset = 0   # byte offset into stderr.txt, tracks what we've already read
        self._server_log_path = None  # resolved lazily after first reset()

        # Milestone event file polling (see §4.6, poll_milestone_events())
        self._milestone_file_offset = 0  # byte offset into milestone_events.jsonl
        self._anvil_coop_file_offset = 0
        self._death_file_offset = 0
        self._woulddie_charged: set = set()

        self._invalid_action_warning = {}  # agent_name -> str

        self._idle_force_counts = {}  # agent_name -> {cause -> int}

        self._action_recovered_counts = {}  # agent_name -> int

        self._step_cache: dict = {}

        self._anvil_hp_history = {}

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
        self._futile_actions = {}
        self._anvil_hp_history = {}
        self.reset_log_offset()
        return self._observations

    def _idle_guard(self, agent_name, agentId, action_str, invalid_original=None):
        """Break single-step idle loops, recording WHY the agent idled.

        A NoOp wastes a step, so the first one is converted to MoveForward
        (``_MAX_CONSECUTIVE_IDLE == 1``). Two distinct upstream failures feed
        NoOp into this guard:
          • ``invalid_action`` — the policy emitted an action name not in
            ACTION_MAP; it was clamped to NoOp and ``invalid_original`` holds
            the original (e.g. 'Attack', 'Turn', 'Mine').
          • ``explicit_noop``  — the policy deliberately chose NoOp.
        Both were previously logged identically as "idle for N steps", which
        hid the cause. We now name it and keep a per-agent run tally
        (:meth:`idle_force_summary`) so the band-aid is measurable.

        Shared verbatim by ``step()`` and ``_resolve_action_for_agent()`` so
        the two stepping paths cannot drift apart. Returns the (possibly
        rewritten) action name.
        """
        import logging as _logging

        if action_str not in self._IDLE_ACTIONS:
            self._consecutive_idle[agent_name] = 0
            return action_str

        self._consecutive_idle[agent_name] = (
            self._consecutive_idle.get(agent_name, 0) + 1
        )
        if self._consecutive_idle[agent_name] < self._MAX_CONSECUTIVE_IDLE:
            return action_str

        cause = "invalid_action" if invalid_original is not None else "explicit_noop"
        tally = self._idle_force_counts.setdefault(
            agent_name, {"invalid_action": 0, "explicit_noop": 0}
        )
        tally[cause] += 1
        detail = (
            f"invalid action '{invalid_original}' clamped to NoOp"
            if invalid_original is not None
            else "policy emitted NoOp"
        )
        _logging.warning(
            "Agent %d idle (%s: %s), forcing MoveForward "
            "[run tally agent_%d: invalid=%d explicit=%d]",
            agentId, cause, detail, agentId,
            tally["invalid_action"], tally["explicit_noop"],
        )
        self._consecutive_idle[agent_name] = 0
        return "MoveForward"

    def idle_force_summary(self):
        """Run-cumulative idle-force counts by agent and cause.

        Returns ``{agent_name: {"invalid_action": int, "explicit_noop": int}}``.
        Lets the trainer surface idle-forcing (and its dominant cause) in
        summary.txt instead of leaving it buried in the slurm .err firehose.
        """
        return {a: dict(c) for a, c in self._idle_force_counts.items()}

    def action_recovered_summary(self):
        """Run-cumulative count of synonym/format actions recovered per agent.

        Returns ``{agent_name: int}`` — how often canonicalize_action() rescued
        an invented action name (e.g. 'Attack' -> 'Dig') that would otherwise
        have been clamped to NoOp and force-moved.
        """
        return dict(self._action_recovered_counts)

    def _resolve_action_for_agent(self, agentId: int, action_str: str) -> str:
        """Per-agent action preprocessing shared by step_all() (and, eventually,
        step()).

        Runs the same guards step() has always run *before* it builds the
        underlying action dict: invalid-action clamping, the idle-loop guard,
        position-stuck / escape-queue handling, respawn pitch reset, and the
        camera pitch cap. It mutates the identical per-agent state dicts
        step() mutates (_escape_queue, _consecutive_idle, _last_xz,
        _stuck_move_steps, _pitch, _futile_actions, _invalid_action_warning)
        and returns the resolved PRIMITIVE action name.

        NOTE: this is intentionally a verbatim lift of step()'s preprocessing
        block. step() is left untouched until the --simultaneous path is
        parity-validated; once it is, step() should be migrated to call this
        helper so there is a single source of truth. The N=1 identity tier of
        scripts/parity_test_stepping.py is what licenses that later migration.
        """
        import logging as _logging

        agent_name = f"agent_{agentId}"

        invalid_original = None
        if action_str not in ACTION_MAP:
            canon = canonicalize_action(action_str)
            if canon is not None:
                # Synonym / format variant — recover intent instead of clamping.
                _logging.info(
                    "Recovered action '%s' -> '%s' for %s",
                    action_str, canon, agent_name,
                )
                self._action_recovered_counts[agent_name] = (
                    self._action_recovered_counts.get(agent_name, 0) + 1
                )
                action_str = canon
            else:
                invalid_original = action_str
                _logging.warning(
                    f"Invalid action: '{action_str}', clamping to NoOp. "
                    f"Valid actions: {VALID_ACTIONS}"
                )
                self._invalid_action_warning[agent_name] = (
                    f"WARNING: Your previous action '{action_str}' is NOT a "
                    f"valid action — it was silently clamped to NoOp. Valid "
                    f"actions are: {VALID_ACTIONS}. Do not emit invalid action "
                    f"names; use 'Dig' (not 'Mine' or 'Attack'), 'TurnRight' "
                    f"(not 'Turn'), etc."
                )
                action_str = "NoOp"

        action_str = self._idle_guard(
            agent_name, agentId, action_str, invalid_original
        )

        if self._escape_queue.get(agent_name):
            esc_action, esc_ticks = self._escape_queue[agent_name][0]
            esc_ticks -= 1
            if esc_ticks <= 0:
                self._escape_queue[agent_name].pop(0)
            else:
                self._escape_queue[agent_name][0] = (esc_action, esc_ticks)
            action_str = esc_action
        else:
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
                            f"Agent {agentId} stuck at xz={cur_xz} for "
                            f"{self._stuck_move_steps[agent_name]} steps — use Escape macro"
                        )
                        self._stuck_move_steps[agent_name] = 0
                        self._last_xz[agent_name] = None
            else:
                self._stuck_move_steps[agent_name] = 0

        if self._terminations.get(agent_name, False):
            self._pitch[agent_name] = 0

        pitch = self._pitch.get(agent_name, 0)
        if action_str == "LookDown":
            if pitch >= self._PITCH_MAX_DOWN:
                _logging.debug(f"Agent {agentId} pitch cap hit (down={pitch}), redirecting LookDown → NoOp")
                action_str = "NoOp"
                self._futile_actions[agent_name] = (
                    self._futile_actions.get(agent_name, 0) + 1
                )
            else:
                self._pitch[agent_name] = pitch + 1
        elif action_str == "LookUp":
            if pitch <= -self._PITCH_MAX_UP:
                _logging.debug(f"Agent {agentId} pitch cap hit (up={-pitch}), redirecting LookUp → NoOp")
                action_str = "NoOp"
                self._futile_actions[agent_name] = (
                    self._futile_actions.get(agent_name, 0) + 1
                )
            else:
                self._pitch[agent_name] = pitch - 1

        return action_str

    def step_all(self, actions: dict):
        """Simultaneous-move step: ALL agents act on the same underlying tick(s).

        Parameters
        ----------
        actions : dict[int, str]
            Maps agentId -> action name for every live agent. Terminated
            agents (absent from ``self.env.agents``) are skipped.

        Returns
        -------
        (observations, resolved) : (dict, dict[int, str])
            ``observations`` is the post-step obs dict; ``resolved`` maps each
            agentId to the primitive action that actually fired (after macro /
            clamp / escape resolution), for logging + RL bookkeeping.

        Contrast with step(): step() advances the underlying env once per
        agent (acting agent moves, the rest NoOp), so an N-agent outer step
        costs N underlying ticks and later agents observe the world already
        moved by earlier ones. step_all advances the env ONCE for all agents
        (with Jump/Dig tick-expansion), so every agent observes the same s_t
        and acts on it. This is a deliberate turn-based -> simultaneous-move
        change and is only reached when the caller opts in via --simultaneous.
        """
        self._step_cache.clear()
        live = list(self.env.agents)

        # 1. Resolve every agent's primitive via the shared preprocessing.
        resolved: dict[int, str] = {}
        for agentId, raw in actions.items():
            ag = f"agent_{agentId}"
            if ag not in live:
                continue  # terminated — no action this step
            resolved[agentId] = self._resolve_action_for_agent(agentId, raw)

        equip = {}
        for agentId, act in resolved.items():
            if act == "Dig":
                best_slot = self._find_best_tool(agentId)
                if best_slot is not None:
                    sid = ACTION_MAP.get(f"Slot{best_slot}")
                    if sid is not None:
                        equip[f"agent_{agentId}"] = sid
        if equip:
            equip_actions = {
                ag: equip.get(ag, ACTION_MAP["NoOp"]) for ag in live
            }
            self.env.step(equip_actions)

        schedules: dict[str, list] = {}
        for agentId, act in resolved.items():
            ag = f"agent_{agentId}"
            if act == "Jump":
                schedules[ag] = ["Jump", "MoveForward"]
            else:
                schedules[ag] = [act] * self._SUSTAINED_TICKS.get(act, 1)
        max_ticks = max((len(s) for s in schedules.values()), default=1)

        total_rewards = {ag: 0.0 for ag in live}
        observations = self._observations
        terminations = self._terminations
        truncations = self._truncations
        infos = self._infos
        for t in range(max_ticks):
            tick_actions = {}
            for ag in self.env.agents:  # re-read each tick; agents can drop on death
                seq = schedules.get(ag)
                act = seq[t] if (seq and t < len(seq)) else "NoOp"
                tick_actions[ag] = ACTION_MAP[act]
            observations, rewards, terminations, truncations, infos = self.env.step(tick_actions)
            for ag, rew in rewards.items():
                if ag in total_rewards:
                    total_rewards[ag] += rew
            # Stop once every agent has terminated/truncated.
            if all(
                terminations.get(ag, False) or truncations.get(ag, False)
                for ag in live
            ):
                break

        # 4. Commit state — mirrors step()'s tail exactly.
        self._observations = observations
        self._terminations = terminations
        self._truncations = truncations
        self._infos = infos
        self.tail_server_log()
        self._step_rewards = dict(total_rewards)
        for ag, rew in total_rewards.items():
            self._rewards[ag] = self._rewards.get(ag, 0.0) + rew
        for agentId, act in resolved.items():
            self._last_actions[f"agent_{agentId}"] = act

        return self._observations, resolved

    def step(self, action_str: str, agentId: int):
        """Execute one action for one agent, NoOp for the rest, then step the env.

        Args:
            action_str: Action name (e.g. "Dig", "MoveForward")
            agentId: Integer index of the acting agent

        Returns:
            (observations_dict, action_str) — matching the agent interface contract

        Raises:
            ValueError: If action_str is not in ACTION_MAP
        """
        import logging as _logging

        self._step_cache.clear()

        agent_name = f"agent_{agentId}"

        invalid_original = None
        if action_str not in ACTION_MAP:
            canon = canonicalize_action(action_str)
            if canon is not None:
                _logging.info(
                    "Recovered action '%s' -> '%s' for %s",
                    action_str, canon, agent_name,
                )
                self._action_recovered_counts[agent_name] = (
                    self._action_recovered_counts.get(agent_name, 0) + 1
                )
                action_str = canon
            else:
                invalid_original = action_str
                _logging.warning(
                    f"Invalid action: '{action_str}', clamping to NoOp. "
                    f"Valid actions: {VALID_ACTIONS}"
                )
                self._invalid_action_warning[agent_name] = (
                    f"WARNING: Your previous action '{action_str}' is NOT a "
                    f"valid action — it was silently clamped to NoOp. Valid "
                    f"actions are: {VALID_ACTIONS}. Do not emit invalid action "
                    f"names; use 'Dig' (not 'Mine' or 'Attack'), 'TurnRight' "
                    f"(not 'Turn'), etc."
                )
                action_str = "NoOp"

        action_str = self._idle_guard(
            agent_name, agentId, action_str, invalid_original
        )

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
                            f"Agent {agentId} stuck at xz={cur_xz} for "
                            f"{self._stuck_move_steps[agent_name]} steps — use Escape macro"
                        )
                        self._stuck_move_steps[agent_name] = 0
                        self._last_xz[agent_name] = None
            else:
                # Non-movement action — reset stuck counter (position change not expected)
                self._stuck_move_steps[agent_name] = 0

        if self._terminations.get(agent_name, False):
            self._pitch[agent_name] = 0

        pitch = self._pitch.get(agent_name, 0)
        if action_str == "LookDown":
            if pitch >= self._PITCH_MAX_DOWN:
                _logging.debug(f"Agent {agentId} pitch cap hit (down={pitch}), redirecting LookDown → NoOp")
                action_str = "NoOp"
                self._futile_actions[agent_name] = (
                    self._futile_actions.get(agent_name, 0) + 1
                )
            else:
                self._pitch[agent_name] = pitch + 1
        elif action_str == "LookUp":
            if pitch <= -self._PITCH_MAX_UP:
                _logging.debug(f"Agent {agentId} pitch cap hit (up={-pitch}), redirecting LookUp → NoOp")
                action_str = "NoOp"
                # pitch stays at its current capped value — no increment
                self._futile_actions[agent_name] = (
                    self._futile_actions.get(agent_name, 0) + 1
                )
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

        if action_str == "Dig":
            best_slot = self._find_best_tool(agentId)
            if best_slot is not None:
                slot_action_id = ACTION_MAP.get(f"Slot{best_slot}")
                if slot_action_id is not None:
                    equip_actions = {}
                    for ag_name in actions:
                        i = int(ag_name.split("_")[1])
                        equip_actions[ag_name] = slot_action_id if i == agentId else ACTION_MAP["NoOp"]
                    self.env.step(equip_actions)  # one tick to switch slot
                else:
                    _logging.warning(
                        "Auto-equip: Slot%d not in ACTION_MAP — "
                        "skipping equip and proceeding to Dig", best_slot,
                    )

        if action_str == "Jump":
            fwd_actions = {ag: (ACTION_MAP["MoveForward"] if ag == f"agent_{agentId}" else ACTION_MAP["NoOp"])
                           for ag in actions}
            self.env.step(actions)      # jump tick
            actions = fwd_actions       # forward tick (becomes the "main" step below)

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

    def episode_over(self) -> bool:
        """True once Lua signals a team wipe in Ch5 (deaths.lua writes
        episode_over.txt when every agent is permanently dead in the boss room).

        Explicit flag rather than relying on all_done(): in Ch4 agents are
        force-respawned (so they're never all 'terminated' together), and a
        permadeath in Ch5 may not map to a persistent PettingZoo termination.
        The flag is cleared at episode reset (init.lua reset handler).
        """
        return self._door_state_file_exists("episode_over.txt")

    def get_step_reward(self, agentId: int) -> float:
        """Return the raw reward from the last env.step() for this agent."""
        agent_name = f"agent_{agentId}"
        return self._step_rewards.get(agent_name, 0.0)

    def get_chamber_state(self, agentId: int) -> str:
        """Return a one-line LLM-readable summary of the current chamber's
        puzzle state for the given agent. Used to populate the
        ``{chamber_state}`` placeholder in the action-selection prompt so
        the policy can reason about hidden state (anvil HP, who's punching
        what, etc.) instead of inferring it from the visual frame.

        Returns an empty string when there's no chamber-specific state to
        report (e.g., agent is in Ch1 where everything is visible).
        """
        chamber = self.get_chamber(agentId)
        agent_name = f"agent_{agentId}"

        prefix = ""
        warning = self._invalid_action_warning.pop(agent_name, "")
        if warning:
            prefix = warning + " "

        if chamber == "ch1":
            return prefix + self._read_ch1_door_state()
        if chamber == "ch2":
            # Anvil HP + active punchers + Door 2 status.
            parts = [self._read_ch2_anvil_state(), self._read_ch2_door_state()]
            return prefix + " | ".join(p for p in parts if p)
        if chamber == "ch3":
            return prefix + self._read_ch3_state(agentId)
        if chamber == "ch4":
            return prefix + self._read_ch4_door_state()
        return prefix or ""

    def _read_ch1_door_state(self) -> str:
        """Return a one-line Door 1 status string for agents still in Ch1.
        Without this, an agent in Ch1 has no signal that Door 1 has
        unlocked beyond the visual delta (a red bedrock-textured block
        becomes air at the same coords) — the LLM consistently misses it.
        Lua writes {world_path}/door1_state.txt when open_door1() fires."""
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return ""
        if os.path.exists(os.path.join(world_path, "door1_state.txt")):
            return ("Door 1: OPEN — walk north (positive Z) to the opening "
                    "in the north wall and MoveForward through into Chamber 2.")
        return ("Door 1: LOCKED — complete any of M2/M3/M4/M5/M6/M7 "
                "(dig 3 blocks, pick up 3 items, dig 5 wood, kill an "
                "animal, kill 2 animals, or dig 3 stone) to unlock it "
                "for the whole team.")

    def _read_ch2_anvil_state(self) -> str:
        """Parse {world_path}/anvils.txt (written by player_state.lua) into
        a compact human-readable line, including a per-anvil HP delta
        over the last 3 reads.

        The delta is the key closed-loop signal for the LLM: after a
        punch, the agent reads chamber_state on the next prompt — if Δhp
        moved up, its coordination is working; if Δhp is 0, only ONE
        agent is punching and they need a teammate.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return ""
        path = os.path.join(world_path, "anvils.txt")
        try:
            with open(path, "r") as f:
                raw = f.read()
        except (FileNotFoundError, OSError):
            return ""

        from collections import deque

        parts = []
        for line in raw.strip().splitlines():
            fields = line.split("|")
            if len(fields) < 3:
                continue
            kind, hp_str, names = fields[0], fields[1], fields[2]
            try:
                hp = int(hp_str)
            except (TypeError, ValueError):
                hp = None

            delta_str = ""
            if hp is not None:
                hist = self._anvil_hp_history.setdefault(
                    kind, deque(maxlen=4)
                )
                old = hist[0] if hist else hp
                hist.append(hp)
                delta = hp - old
                if delta > 0:
                    delta_str = f" Δhp_last3={delta:+d} (coop working — keep punching)"
                elif delta < 0:
                    delta_str = f" Δhp_last3={delta:+d} (decay — at least 2 agents must punch within ~1s)"
                else:
                    delta_str = " Δhp_last3=0 (no progress — need ≥2 punchers on this anvil at the same time)"

            puncher_names = [n.strip() for n in names.split(",") if n.strip()]
            if puncher_names:
                who = " punchers: " + ", ".join(puncher_names)
            else:
                who = " (idle)"

            parts.append(f"{kind} anvil={hp_str} hp{delta_str}{who}")

        if not parts:
            return ""
        return "Anvils — " + "; ".join(parts)

    def _door_state_file_exists(self, filename: str) -> bool:
        """True iff `{world_path}/<filename>` exists. Used as a presence-
        based flag for door-open events (Lua writes the file on open,
        clear_state_files() deletes it at episode reset).
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return False
        return os.path.exists(os.path.join(world_path, filename))

    def _read_ch2_door_state(self) -> str:
        """Door 2 (Ch2→Ch3) status line. Door 2 opens 20 steps after BOTH
        anvils have been broken (see tick_door2 in doors.lua)."""
        if self._door_state_file_exists("door2_state.txt"):
            return ("Door 2: OPEN — walk north (positive Z) through Door 2 "
                    "to be teleported into your Chamber 3 isolation cell.")
        return ("Door 2: LOCKED — break BOTH purple anvils (M8 + M9) to "
                "open it. Each anvil needs at least 2 agents punching it "
                "within ~1 second to break (solo dig makes zero progress).")

    def _read_ch3_state(self, agentId: int) -> str:
        """Ch3 isolation-cell + communal status.

        Tells the agent both (a) whether their own cell door is OPEN (so
        they can walk out) and (b) whether Door 3 (communal → Ch4) is
        open. Switch rotational mapping: switch in cell i opens the door
        of cell (i+1) mod N; so agent_i's cell is unlocked by the agent
        whose cell index is (i-1) mod N.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return ""

        # Read the cell-doors file produced by open_cell_door().
        open_cells = set()
        cell_doors_path = os.path.join(world_path, "cell_doors_state.txt")
        try:
            with open(cell_doors_path, "r") as f:
                for line in f.read().strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    head = line.split(":", 1)[0].strip()
                    try:
                        open_cells.add(int(head))
                    except ValueError:
                        continue
        except (FileNotFoundError, OSError):
            pass

        my_cell_open = agentId in open_cells
        cell_line = (
            "Your cell door (cell %d): OPEN — walk north out of your cell "
            "into the communal room." % agentId
            if my_cell_open
            else "Your cell door (cell %d): LOCKED — a teammate must press "
                 "their switch to free you (rotational wiring: switch in "
                 "cell %d opens your door)." % (agentId, (agentId - 1) % self.num_agents)
        )

        door3_line = (
            "Door 3 (communal → Ch4): OPEN — walk north to enter Chamber 4."
            if self._door_state_file_exists("door3_state.txt")
            else "Door 3 (communal → Ch4): LOCKED — all %d agents must be "
                 "in the communal room together to open it."
                 % self.num_agents
        )
        return cell_line + " | " + door3_line

    def _read_ch4_door_state(self) -> str:
        """Door 4 (Ch4→Ch5) status line."""
        if self._door_state_file_exists("door4_state.txt"):
            return ("Door 4: OPEN — walk north into Chamber 5 to fight "
                    "the boss zombie.")
        return ("Door 4: LOCKED — kill all 3 zombies in Chamber 4 to open "
                "it (use Dig while facing each zombie; the diamond sword "
                "from Ch2 makes this much faster).")

    def consume_futile(self, agentId: int) -> int:
        """Pop and return the count of pitch-cap-redirected actions since
        the last call. The trainer multiplies this by a small penalty per
        outer step so the policy gets an action-validity signal that the
        team-reward path lacks.
        """
        agent_name = f"agent_{agentId}"
        n = self._futile_actions.get(agent_name, 0)
        self._futile_actions[agent_name] = 0
        return n

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
        """Return a single string with health and time for the given agent.

        Hunger is intentionally omitted: the env is permanently in exploration
        mode (survival was retired), so mcl_hunger drain is neutralised and
        hunger is force-pinned to 20/20 every step — a constant carries no
        information for the policy and its "hunger drains health" implication
        can never fire. See player_state.lua's hunger-neutralisation block.
        """
        world_path = self._get_world_path()
        agent_name = f"agent{agentId}"

        def _read(path, fallback):
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except (FileNotFoundError, OSError):
                return fallback

        # Health is per-agent → read every call.
        health = _read(os.path.join(world_path, f"health_{agent_name}.txt"), "?/20")

        if "timeofday" in self._step_cache:
            time_str = self._step_cache["timeofday"]
        else:
            tod_raw = _read(os.path.join(world_path, "timeofday.txt"), None)
            if tod_raw is not None:
                try:
                    time_str = self._tod_to_clock(float(tod_raw))
                except ValueError:
                    time_str = "Unknown"
            else:
                time_str = "Unknown"
            self._step_cache["timeofday"] = time_str

        return f"Health: {health} | Time: {time_str}"

    def get_position_text(self, agentId: int) -> str:
        """Return a formatted position + facing string for the agent.

        Reads pos AND yaw natively from the mt_channel observations
        (stashed on the patched env in T1.3). The yaw is the agent's
        camera horizontal angle in degrees from Lua (NUE convention:
        0 = north). The compass label gives the LLM a discrete direction
        token without it having to count TurnLeft/TurnRight actions.

        Returns "Unknown" when the env hasn't produced a position yet.
        """
        try:
            pos = self.env.env._positions[agentId]
            if pos is None:
                return "Unknown"
            text = f"x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}"
            try:
                yaw = self.env.env._yaws[agentId]
                # Normalize to [0, 360). NUE: 0=north, 90=east, 180=south, 270=west.
                yaw_norm = float(yaw) % 360.0
                _DIRECTIONS = [
                    ("north", 0), ("north-east", 45), ("east", 90), ("south-east", 135),
                    ("south", 180), ("south-west", 225), ("west", 270), ("north-west", 315),
                ]
                facing = min(_DIRECTIONS, key=lambda d: min(
                    abs(yaw_norm - d[1]), 360.0 - abs(yaw_norm - d[1])
                ))[0]
                text += f" facing {facing} (yaw={yaw_norm:.0f}°)"
            except (AttributeError, IndexError, TypeError):
                pass
            return text
        except (AttributeError, IndexError, TypeError):
            return "Unknown"

    def get_agent_velocity(self, agentId: int):
        """Return the agent's (vx, vy, vz) velocity tuple or None.

        Sourced from the mt_channel binary protocol (units: blocks/sec).
        Useful for a faster / more reliable stuck detector than the
        existing XZ-displacement heuristic — a velocity magnitude of
        near zero across consecutive steps is a positive signal the
        agent is wedged.
        """
        try:
            return self.env.env._velocities[agentId]
        except (AttributeError, IndexError, TypeError):
            return None

    def get_agent_pitch(self, agentId: int) -> float:
        """Return the agent's camera pitch in degrees (native, not the
        hand-rolled action-counter substitute). 0 = level; negative =
        looking up; positive = looking down. Returns 0.0 if not yet
        available."""
        try:
            return float(self.env.env._pitches[agentId])
        except (AttributeError, IndexError, TypeError):
            return 0.0

    def get_agent_yaw(self, agentId: int) -> float:
        """Return the agent's camera yaw in degrees (native). 0 = north.
        Returns 0.0 if not yet available."""
        try:
            return float(self.env.env._yaws[agentId])
        except (AttributeError, IndexError, TypeError):
            return 0.0

    def get_agent_dtime(self, agentId: int) -> float:
        """Return Minetest's wall-clock dtime for this agent's last server
        tick (seconds). Useful for profiling server-side stalls. Returns
        0.0 if not yet available."""
        try:
            return float(self.env.env._dtimes[agentId])
        except (AttributeError, IndexError, TypeError):
            return 0.0

    def get_voxel_obs(self, agentId: int):
        """Return the agent's voxel observation tensor or None.

        Shape: (2*voxel_obs_rx+1, 2*voxel_obs_ry+1, 2*voxel_obs_rz+1, 3).
        Last dim = (node_id, light, param2). Coordinate convention NUE
        (North=+x, Up=+y, East=+z). Available only when the env was
        constructed with ``voxel_obs=True``; otherwise returns None.
        """
        if not self.voxel_obs_enabled:
            return None
        try:
            return self.env.env._voxobs[agentId]
        except (AttributeError, IndexError, TypeError):
            return None

    def get_voxel_summary(self, agentId: int, top_k: int = 6) -> str:
        """Compact text summary of the voxel observation for the LLM prompt.

        Returns a one-line description of the most common non-air nodes
        around the agent, formatted as ``Nearby: stone×42, tree×8, dirt×6
        (plus 9 air-equivalent)``. Designed to be embedded in the
        per-step prompt as a hallucination-resistant grounding signal:
        the LLM cannot perceive a zombie that isn't in the voxel readout.

        Returns an empty string when voxel obs is disabled or the tensor
        hasn't been populated yet. The mapping from numeric node_id to
        human node-name (e.g. ``mcl_core:tree``) requires the env's node
        registry, which Craftium does NOT expose to Python — for now the
        summary reports raw node-id integers grouped by frequency. The
        LLM is told (via its system prompt's voxel section, added with
        this feature) that low IDs are usually engine air/walls and a
        handful of high IDs are the interactive blocks for the current
        chamber.
        """
        if not self.voxel_obs_enabled:
            return ""
        try:
            vox = self.env.env._voxobs[agentId]
        except (AttributeError, IndexError, TypeError):
            return ""
        if vox is None:
            return ""
        try:
            import numpy as _np
            node_ids = vox[..., 0].astype(_np.int64).ravel()
            uniques, counts = _np.unique(node_ids, return_counts=True)
            air_mask = (uniques == 0)
            air_count = int(counts[air_mask].sum()) if air_mask.any() else 0
            non_air = [
                (int(u), int(c))
                for u, c in zip(uniques, counts) if u != 0
            ]
            non_air.sort(key=lambda x: x[1], reverse=True)
            head = non_air[:top_k]
            parts = [f"node_id={u}×{c}" for u, c in head]
            tail = sum(c for _, c in non_air[top_k:])
            extras = []
            if tail > 0:
                extras.append(f"+{tail} other non-air")
            if air_count > 0:
                extras.append(f"{air_count} air")
            joined = ", ".join(parts) if parts else "no non-air voxels in range"
            if extras:
                joined += " (" + "; ".join(extras) + ")"
            return f"Nearby voxels: {joined}"
        except Exception:
            return ""

    def warmup_noop(self):
        """Send NoOps to keep channels alive without incrementing step counters.

        Use this during media-loading warm-up instead of step().
        Returns list of observations (one per agent).
        """
        return self.env.warmup_noop()

    def _get_world_path(self):
        """Return the absolute path to the Minetest world directory."""
        if not hasattr(self, "_world_path"):
            srv_run_dir = self.env.env.mt_server.run_dir
            self._world_path = os.path.join(
                os.path.abspath(srv_run_dir), "worlds", "world"
            )
        return self._world_path

    def signal_warmup_complete(self) -> bool:
        """Tell the Lua side that Python has finished warmup and is about to
        start real env steps. The Ch1-timeout fallback in ``doors.lua`` counts
        ticks from THIS signal, not from server start — otherwise the timeout
        fires during the warmup poll loop.

        Concrete failure mode this avoids: ``warmup_time=300`` s in Python
        runs ``warmup_noop()`` for 300 s. Lua's globalstep keeps ticking at
        20 Hz the whole time, so by the time Python issues step 0, the
        lua-tick counter has already exceeded ``CH1_TIMEOUT_TICKS`` (default
        1200 ≈ 60 s; thesis runs override to 400 ≈ 20 s) and the timeout
        fallback has already teleported agents to the Ch2 fallback spawn.
        Result: agents start at ``z=19`` (Ch2) instead of ``z=1`` (Ch1) and
        every Ch1 milestone is unreachable.

        Writes ``{world_path}/warmup_complete.txt`` atomically. ``doors.lua``
        polls for it and stamps ``warmup_complete_tick`` when it appears.
        Idempotent — safe to call multiple times.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return False
        tmp = os.path.join(world_path, "warmup_complete.tmp")
        dst = os.path.join(world_path, "warmup_complete.txt")
        try:
            with open(tmp, "w") as f:
                f.write("1")
            os.replace(tmp, dst)
            return True
        except OSError as e:
            import logging as _log
            _log.warning("[WARMUP] signal_warmup_complete failed: %s", e)
            return False

    def force_ch1_teleport(self) -> bool:
        """Signal the Lua side to force-fire the Ch1→Ch2 teleport now.

        Writes ``{world_path}/ch1_force_teleport.txt`` atomically. The doors.lua
        globalstep polls for this file; when present, it teleports every
        connected agent to its Ch2 fallback spawn and deletes the flag.

        We added this Python-driven trigger because the Lua-side
        ``step_counter`` based timeout was observed to never fire under our
        run config (agents stayed in Ch1 indefinitely). Counting env steps
        in Python is robust against any tick-rate mismatch.

        Returns True if the file was written, False if the world path was
        not yet resolved (e.g., called before reset()).
        """
        return self._write_force_teleport_flag(1)

    def force_ch2_teleport(self) -> bool:
        """Signal the Lua side to force-fire the Ch2→Ch3 teleport now.

        Same mechanism as ``force_ch1_teleport`` (write
        ``ch2_force_teleport.txt``, doors.lua polls + teleports each agent
        to its own Ch3 cell). Fired by the Python step loop at the 40%
        max_steps mark so agents who never broke the Ch2 anvils still get
        a chance to see Ch3+. Honest cooperation measurement is preserved
        in the milestone log — the timer doesn't credit any anvil/door
        milestone, only relocates the agents.
        """
        return self._write_force_teleport_flag(2)

    def force_ch3_teleport(self) -> bool:
        """Signal the Lua side to force-fire the Ch3→Ch4 teleport now.

        Writes ``ch3_force_teleport.txt``; doors.lua teleports every agent
        to a Ch4 spawn. Fired by the Python step loop at the 60% max_steps
        mark. Same milestone-honesty caveat as ``force_ch2_teleport``.
        """
        return self._write_force_teleport_flag(3)

    def force_ch4_teleport(self) -> bool:
        """Signal the Lua side to force-fire the Ch4→Ch5 teleport now.

        Writes ``ch4_force_teleport.txt``; doors.lua teleports every agent
        to a Ch5 spawn. Fired by the Python step loop at the 80% max_steps
        mark. Same milestone-honesty caveat as ``force_ch2_teleport``.
        """
        return self._write_force_teleport_flag(4)

    def _write_force_teleport_flag(self, chamber_from: int) -> bool:
        """Atomically write ``ch{N}_force_teleport.txt`` under world_path.

        Common helper for force_ch1/2/3/4_teleport. Returns False if the
        world path is not yet resolved (called before reset()) or if the
        atomic write fails.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return False
        fname_base = f"ch{chamber_from}_force_teleport"
        tmp = os.path.join(world_path, f"{fname_base}.tmp")
        dst = os.path.join(world_path, f"{fname_base}.txt")
        try:
            with open(tmp, "w") as f:
                f.write("1")
            os.replace(tmp, dst)
            return True
        except OSError as e:
            import logging as _log
            _log.warning(
                "[CH%d_TIMEOUT] _write_force_teleport_flag failed: %s",
                chamber_from, e,
            )
            return False

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

        _HOTBAR_LIMIT = 8
        slots = parts[1:1 + _HOTBAR_LIMIT]
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
                    best_slot = i + 1  # 1-indexed (always in [1, 8])

        # Only switch if we found a tool and it's not already wielded
        if best_slot is not None and best_slot != wield_idx:
            return best_slot
        return None


    # Lines from the Lua server log that are worth surfacing in Python output.
    _LOG_TAGS = (
        "[TOOLS]", "[INVENTORY]", "[TRACK STATUS]",
        "[DIG]", "[HUNT]", "[DEFEND]",
        "[PHASE]",       # phase transitions
        "[MILESTONE]",   # five_chambers milestone events
        "[ANVIL]", "[SWITCH]", "[DOOR]", "[MOB]", "[BOSS]",
        "[CH1_TIMEOUT]", "[CH1_TIMEOUT_DIAG]",
        "[CH2_TIMEOUT]", "[CH3_TIMEOUT]", "[CH4_TIMEOUT]",
    )

    _CHAMBER_BOUNDS = {
        "ch1": lambda p: 0  <= p[2] <= 15,
        "ch2": lambda p: 17 <= p[2] <= 25,
        "ch3": lambda p: 27 <= p[2] <= 45,
        "ch4": lambda p: 47 <= p[2] <= 57,
        "ch5": lambda p: 59 <= p[2] <= 67,
    }

    def get_agent_position(self, agentId: int):
        """Return current (x, y, z) tuple for agent, or None if unavailable."""
        try:
            return self.env.env._positions[agentId]
        except (AttributeError, IndexError, TypeError):
            return None

    def get_chamber(self, agentId: int):
        """Return the chamber name for agent's current position, or None."""
        pos = self.get_agent_position(agentId)
        if pos is None:
            return None
        for name, fn in self._CHAMBER_BOUNDS.items():
            if fn(pos):
                return name
        return None

    def poll_milestone_events(self) -> list:
        """Read any new milestone events from milestone_events.jsonl since the last call.

        Returns a list of dicts, each with keys: step, milestone, contributors, reward.
        The byte-offset is advanced so repeated calls never re-read the same lines.
        If the file is smaller than the tracked offset (episode reset / file cleared),
        the offset is automatically reset to 0.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return []

        path = os.path.join(world_path, "milestone_events.jsonl")
        if not os.path.exists(path):
            self._milestone_file_offset = 0
            return []

        # Detect file truncation / deletion + recreation (episode reset).
        try:
            if os.path.getsize(path) < self._milestone_file_offset:
                self._milestone_file_offset = 0
        except OSError:
            return []

        new_events = []
        try:
            with open(path, "r") as f:
                f.seek(self._milestone_file_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        new_events.append(ev)
                        for raw_name in ev.get("contributors", []):
                            _s = str(raw_name).removeprefix("agent_").removeprefix("agent")
                            try:
                                _aid = int(_s)
                            except ValueError:
                                continue
                            agent_name = f"agent_{_aid}"
                            self._rewards[agent_name] = (
                                self._rewards.get(agent_name, 0.0) + ev.get("reward", 0)
                            )
                    except json.JSONDecodeError:
                        pass
                self._milestone_file_offset = f.tell()
        except OSError:
            pass

        return new_events

    def reset_milestone_offset(self):
        """Anchor the milestone event reader to the current EOF (call after env.reset())."""
        try:
            world_path = self._get_world_path()
        except AttributeError:
            self._milestone_file_offset = 0
            return
        path = os.path.join(world_path, "milestone_events.jsonl")
        try:
            self._milestone_file_offset = os.path.getsize(path)
        except OSError:
            self._milestone_file_offset = 0

    def poll_anvil_coop_events(self) -> list:
        """Read any new anvil-coop diagnostic events since the last call.

        Returns a list of dicts each with keys: step, anvil, row,
        n_active, active. NO reward attached — these are for post-hoc
        analysis of "did the team try to coordinate at the anvils?".
        The Lua side (anvil.lua globalstep) edge-triggers one event per
        anvil per ~ACTIVE_WINDOW ticks when ≥2 agents are punching.

        Mirrors poll_milestone_events()'s file-offset bookkeeping so
        repeated calls never re-read the same lines, and a smaller
        file size (episode reset / file cleared) auto-rewinds the
        offset to 0.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return []
        path = os.path.join(world_path, "anvil_coop_events.jsonl")
        if not os.path.exists(path):
            self._anvil_coop_file_offset = 0
            return []
        try:
            if os.path.getsize(path) < self._anvil_coop_file_offset:
                self._anvil_coop_file_offset = 0
        except OSError:
            return []
        new_events = []
        try:
            with open(path, "r") as f:
                f.seek(self._anvil_coop_file_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        new_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                self._anvil_coop_file_offset = f.tell()
        except OSError:
            pass
        return new_events

    def reset_anvil_coop_offset(self):
        """Anchor the anvil-coop event reader to current EOF (post env.reset())."""
        try:
            world_path = self._get_world_path()
        except AttributeError:
            self._anvil_coop_file_offset = 0
            return
        path = os.path.join(world_path, "anvil_coop_events.jsonl")
        try:
            self._anvil_coop_file_offset = os.path.getsize(path)
        except OSError:
            self._anvil_coop_file_offset = 0

    def poll_death_events(self) -> list:
        """Read any new death / would-die events since the last call.

        Returns a list of dicts each with keys: step, kind ("death" |
        "woulddie"), agent, chamber, reward (a NEGATIVE int: −10 for a
        forgiving-chamber would-have-died, −50 for a real Ch5 permadeath).
        deaths.lua writes these to death_events.jsonl because the server-side
        craftium.reward() it also calls does NOT reach env.step()'s reward
        channel in the multi-agent five-chambers context — so this JSONL is the
        authoritative reward source. The caller (multi_agent_craftium.py) drains
        each event's reward into step_rewards_raw before Hebbian diffusion /
        record_reward, so the penalty propagates into the graph and into
        cumulative_returns / episode_return.

        WOULD-DIE RATE LIMIT (once per agent per EPISODE): the Lua hpchange
        callback fires once per *damage event*, so a continuously attacked agent
        emits dozens of would-die lines — stacking −10s until the penalty swamps
        every positive reward. We cap it hard: each agent is charged the −10
        would-die AT MOST ONCE PER EPISODE. Subsequent would-die lines for an
        already-charged agent are dropped (not returned, not mirrored into
        _rewards). The charged-agent set (self._woulddie_charged) is cleared at
        episode reset via reset_death_offset(). Real Ch5 deaths ("death") are
        one-shot/terminal and are never capped. The file offset still advances
        past every line, so dropped duplicates are not re-read; Lua keeps its
        own full [WOULDDIE] log + would_die_count for diagnostics.

        Mirrors poll_milestone_events()/poll_anvil_coop_events() file-offset
        bookkeeping so repeated calls never re-read the same lines, and a
        smaller file size (episode reset / file cleared) auto-rewinds to 0.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return []
        path = os.path.join(world_path, "death_events.jsonl")
        if not os.path.exists(path):
            self._death_file_offset = 0
            return []
        try:
            if os.path.getsize(path) < self._death_file_offset:
                self._death_file_offset = 0
        except OSError:
            return []
        new_events = []
        try:
            with open(path, "r") as f:
                f.seek(self._death_file_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _s = str(ev.get("agent", "")).removeprefix("agent_").removeprefix("agent")
                    try:
                        _aid = int(_s)
                    except ValueError:
                        _aid = None
                    if ev.get("kind") == "woulddie" and _aid is not None:
                        if _aid in self._woulddie_charged:
                            continue
                        self._woulddie_charged.add(_aid)
                    new_events.append(ev)
                    if _aid is not None:
                        agent_name = f"agent_{_aid}"
                        self._rewards[agent_name] = (
                            self._rewards.get(agent_name, 0.0) + ev.get("reward", 0)
                        )
                self._death_file_offset = f.tell()
        except OSError:
            pass
        return new_events

    def reset_death_offset(self):
        """Anchor the death-event reader to current EOF (post env.reset()) and
        clear the per-episode would-die charge set so each agent can be charged
        the −10 once again in the new episode."""
        self._woulddie_charged = set()
        try:
            world_path = self._get_world_path()
        except AttributeError:
            self._death_file_offset = 0
            return
        path = os.path.join(world_path, "death_events.jsonl")
        try:
            self._death_file_offset = os.path.getsize(path)
        except OSError:
            self._death_file_offset = 0

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
