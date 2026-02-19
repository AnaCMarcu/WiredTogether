"""PettingZoo ParallelEnv wrapper for Craftium's OpenWorld environment."""

import os
import socket as socket_mod
import time
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from craftium.multiagent_env import MarlCraftiumEnv


class _PatchedMarlCraftiumEnv(MarlCraftiumEnv):
    """MarlCraftiumEnv with two HPC fixes:

    1. **Binary name**: upstream ``MTServerOnly`` / ``MTClientOnly`` use
       ``./bin/minetest`` but newer Luanti builds renamed the binary to
       ``./bin/luanti``.  We detect and patch the launch commands.
    2. **Server polling**: upstream uses ``time.sleep(5)`` after starting the
       server, which is too short for VoxeLibre (~45-120 s).  We poll the
       server's stderr for ``"listening on"`` instead.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fix_binary_names()
        self._fix_headless_clients()

    # ------------------------------------------------------------------ #
    # Patch 1: binary name minetest -> luanti
    # ------------------------------------------------------------------ #
    def _fix_binary_names(self):
        """Replace ``./bin/minetest`` with ``./bin/luanti`` when needed."""
        srv_old = os.path.join(self.mt_server.run_dir, "bin", "minetest")
        srv_new = os.path.join(self.mt_server.run_dir, "bin", "luanti")
        if not os.path.exists(srv_old) and os.path.exists(srv_new):
            self.mt_server.launch_cmd = [
                s.replace("./bin/minetest", "./bin/luanti")
                for s in self.mt_server.launch_cmd
            ]
            print("* Patched server binary: minetest -> luanti")

        for i, client in enumerate(self.mt_clients):
            cli_old = os.path.join(client.run_dir, "bin", "minetest")
            cli_new = os.path.join(client.run_dir, "bin", "luanti")
            if not os.path.exists(cli_old) and os.path.exists(cli_new):
                client.launch_cmd = [
                    s.replace("./bin/minetest", "./bin/luanti")
                    for s in client.launch_cmd
                ]
                if i == 0:
                    print("* Patched client binaries: minetest -> luanti")

    # ------------------------------------------------------------------ #
    # Patch 1b: force all clients headless (upstream bug)
    # ------------------------------------------------------------------ #
    def _fix_headless_clients(self):
        """Ensure ALL clients use ``SDL_VIDEODRIVER=offscreen``.

        Upstream sets ``headless = (i == 0) and (render_mode != "human")``
        which leaves client 1+ non-headless — they crash on HPC nodes
        that have no display.
        """
        for client in self.mt_clients:
            if client.proc_env is None:
                client.proc_env = {"SDL_VIDEODRIVER": "offscreen"}
            elif "SDL_VIDEODRIVER" not in client.proc_env:
                client.proc_env["SDL_VIDEODRIVER"] = "offscreen"
        print(f"* Forced all {len(self.mt_clients)} clients to headless (offscreen SDL)")

    # ------------------------------------------------------------------ #
    # Patch 2: pre-listen on all channel sockets
    # ------------------------------------------------------------------ #
    def _pre_listen_channels(self):
        """Call ``listen(1)`` on every MtChannel socket BEFORE starting clients.

        ``mt_server.init_server()`` only does ``socket()`` + ``bind()`` — it
        does **not** call ``listen()``.  The ``listen()`` call normally happens
        inside ``server_listen()`` (triggered by ``open_conn()``), but by that
        time the client process may already have tried to ``connect()`` and
        received *Connection refused*.

        Fix: use Python's ``socket.fromfd()`` to put each socket into the
        LISTEN state early.  ``fromfd`` duplicates the fd; calling ``close()``
        on the Python wrapper only closes the duplicate — the original
        ``sockfd`` stays open for ``server_listen()`` to ``accept()`` on later.

        Calling ``listen()`` twice (here + inside ``server_listen()``) is
        harmless on Linux — the second call simply updates the backlog.
        """
        for i, ch in enumerate(self.mt_channs):
            sock = socket_mod.fromfd(ch.sockfd, socket_mod.AF_INET, socket_mod.SOCK_STREAM)
            sock.listen(1)
            sock.close()  # closes the dup'd fd; original ch.sockfd stays open
        print(f"* Pre-listened on {len(self.mt_channs)} channel sockets")

    # ------------------------------------------------------------------ #
    # Patch 3: server-ready polling + diagnostics
    # ------------------------------------------------------------------ #
    def reset(self, **kwargs):
        self.timesteps = 0
        observations = []

        if self.mt_server.proc is None:
            # --- diagnostics ---
            print(f"* Server launch cmd: {self.mt_server.launch_cmd}")
            print(f"* Server run dir:    {self.mt_server.run_dir}")

            self.mt_server.start_process()

            # Check server process didn't die immediately
            time.sleep(1)
            ret = self.mt_server.proc.poll()
            if ret is not None:
                stderr_path = os.path.join(self.mt_server.run_dir, "stderr.txt")
                stderr_content = ""
                try:
                    with open(stderr_path, "r", errors="ignore") as f:
                        stderr_content = f.read()
                except FileNotFoundError:
                    pass
                raise RuntimeError(
                    f"MT server process exited immediately with code {ret}.\n"
                    f"stderr:\n{stderr_content}"
                )

            # --- poll stderr for "listening on" ---
            print(
                "* Waiting for MT server to initialize (polling stderr). "
                "This is only required in the first call to reset."
            )
            deadline = time.time() + 120  # max 2 minutes
            server_ready = False
            stderr_path = os.path.join(self.mt_server.run_dir, "stderr.txt")
            while time.time() < deadline:
                time.sleep(2)
                ret = self.mt_server.proc.poll()
                if ret is not None:
                    try:
                        with open(stderr_path, "r", errors="ignore") as f:
                            stderr_content = f.read()
                    except FileNotFoundError:
                        stderr_content = "(no stderr.txt)"
                    raise RuntimeError(
                        f"MT server died during init (exit code {ret}).\n"
                        f"stderr:\n{stderr_content}"
                    )
                try:
                    with open(stderr_path, "r", errors="ignore") as f:
                        if "listening on" in f.read():
                            server_ready = True
                            break
                except (FileNotFoundError, OSError):
                    pass
            if not server_ready:
                # Dump last lines of stderr so user can debug
                try:
                    with open(stderr_path, "r", errors="ignore") as f:
                        tail = f.read()[-500:]
                except FileNotFoundError:
                    tail = "(file not found)"
                print(
                    f"* WARNING: server not confirmed ready after 120 s.\n"
                    f"  stderr tail:\n{tail}"
                )
            else:
                print("* MT server is ready!")

            # Small extra delay for server to stabilize
            time.sleep(3)

            # Put all channel sockets into LISTEN state before any client
            # starts, so clients never see "Connection refused".
            self._pre_listen_channels()

            for i in range(self.num_agents):
                print(f"* Starting client {i}: {self.mt_clients[i].launch_cmd}")
                self.mt_clients[i].start_process()

                # Call open_conn() immediately — it must be listening
                # BEFORE the client tries to connect to the Python TCP port.
                # Any sleep between start_process() and open_conn() risks the
                # client seeing "Connection refused".
                try:
                    self.mt_channs[i].open_conn()
                except ConnectionError:
                    ret = self.mt_clients[i].proc.poll()
                    cli_stderr = os.path.join(
                        self.mt_clients[i].run_dir, "stderr.txt"
                    )
                    content = ""
                    try:
                        with open(cli_stderr, "r", errors="ignore") as f:
                            content = f.read()
                    except FileNotFoundError:
                        pass
                    raise RuntimeError(
                        f"MT client {i} connection failed (exit code {ret}).\n"
                        f"stderr:\n{content}"
                    )

                for _ in range(self.init_frames):
                    _obs, *_ = self.mt_channs[i].receive()
                    self.mt_channs[i].send([0] * 21, 0, 0)

                observation, _voxobs, _pos, _vel, _pitch, _yaw, _dtime, reward, _term = (
                    self.mt_channs[i].receive()
                )
                if not self.gray_scale_keepdim and not self.rgb_observations:
                    observation = observation[:, :, 0]
                observations.append(observation)
                self.last_observations[i] = observation

        else:  # soft reset
            for i in range(self.num_agents):
                self.mt_channs[i].send_soft_reset()
                observation, _voxobs, _pos, _vel, _pitch, _yaw, _dtime, reward, _term = (
                    self.mt_channs[i].receive()
                )
                if not self.gray_scale_keepdim and not self.rgb_observations:
                    observation = observation[:, :, 0]
                observations.append(observation)
                self.last_observations[i] = observation

        infos = self._get_info()
        observations = np.vstack([np.expand_dims(obs, 0) for obs in observations])
        return observations, infos

_DISCRETE_ACTIONS = [
    "forward", "backward", "left", "right", "jump", "sneak",
    "dig", "place", "slot_1", "slot_2", "slot_3", "slot_4", "slot_5",
    "mouse x+", "mouse x-", "mouse y+",
]
_MOUSE_MOV = 0.5


def _discrete_to_dict(action: int) -> dict:
    """Convert a Discrete(17) integer to MarlCraftiumEnv dict format.

    Action 0 is NOP. Actions 1-16 map to _DISCRETE_ACTIONS.
    PettingZoo action_space.sample() return an integer, but Craftium expects {action, mouse} format
    """
    action = int(action)
    if action == 0:
        return {}  # NOP: no mouse movement

    name = _DISCRETE_ACTIONS[action - 1]
    mouse = [0.0, 0.0]

    if name == "mouse x+":
        mouse[0] = _MOUSE_MOV
        return {"mouse": mouse}
    elif name == "mouse x-":
        mouse[0] = -_MOUSE_MOV
        return {"mouse": mouse}
    elif name == "mouse y+":
        mouse[1] = _MOUSE_MOV
        return {"mouse": mouse}
    elif name == "mouse y-":
        mouse[1] = -_MOUSE_MOV
        return {"mouse": mouse}
    else:
        return {name: 1, "mouse": mouse}


class OpenWorldMultiAgentEnv(ParallelEnv):
    """Multi-agent OpenWorld environment with PettingZoo ParallelEnv API.

    This wrapper adapts Craftium's MarlCraftiumEnv to PettingZoo's ParallelEnv interface,
    allowing use with multi-agent RL libraries like RLlib, CleanRL, etc.

    Args:
        num_agents: Number of agents in the environment
        obs_width: Observation width in pixels (default: 320)
        obs_height: Observation height in pixels (default: 180)
        max_steps: Maximum steps per episode (default: 10000)
        task_focus: Optional task to focus agents on (e.g., "mining", "crafting")
        render_mode: Rendering mode ("rgb_array" or None)
    """

    metadata = {"render_modes": ["rgb_array"], "name": "openworld_multi_agent_v0"}

    def __init__(
        self,
        num_agents: int = 2,
        obs_width: int = 320,
        obs_height: int = 180,
        max_steps: int = 10000,
        task_focus: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Store config (use _num_agents to avoid conflict with PettingZoo property)
        self._num_agents = num_agents
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.max_steps = max_steps
        self.task_focus = task_focus
        self.render_mode = render_mode

        # Define agent names BEFORE creating env (needed for num_agents property)
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

        from craftium import root_path as craftium_root

        minetest_dir = os.environ.get(
            "CRAFTIUM_LUANTI_DIR",
            os.path.join(craftium_root, "luanti")
        )

        # Find voxel-libre2: prefer CRAFTIUM_ENV_DIR, then installed package,
        # then fall back to the local repo's craftium submodule.
        _this_file = os.path.abspath(__file__)
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))
        _pkg_env = os.path.join(craftium_root, "craftium-envs", "voxel-libre2")
        _local_env = os.path.join(_project_root, "craftium", "craftium-envs", "voxel-libre2")
        env_dir = (
            os.environ.get("CRAFTIUM_ENV_DIR")
            or (_pkg_env if os.path.isdir(_pkg_env) else _local_env)
        )

        self.env = _PatchedMarlCraftiumEnv(
            env_dir=env_dir,
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            max_timesteps=max_steps,
            minetest_dir=minetest_dir,
            mt_listen_timeout=300_000,  # 5 min per client; VoxeLibre loads slowly on HPC
        )

        # Define observation and action spaces
        self._observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )
        # Craftium uses 17 discrete actions (DiscreteActionWrapper)
        self._action_space = spaces.Discrete(17)

        self._step_count = 0

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """Return observation spaces for all agents."""
        return {agent: self._observation_space for agent in self.possible_agents}

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """Return action spaces for all agents."""
        return {agent: self._action_space for agent in self.possible_agents}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset the environment.

        Returns:
            observations: Dictionary of observations per agent
            infos: Dictionary of info dicts per agent
        """
        if seed is not None:
            np.random.seed(seed)

        obs_array, _ = self.env.reset()

        # Reset agents and step counter
        self.agents = self.possible_agents.copy()
        self._step_count = 0

        # Convert to PettingZoo format.
        # obs_array is a numpy array of shape (num_agents, obs_width, obs_height, 3).
        # info_dict is a plain dict (not per-agent), so we give each agent an empty dict.
        observations = {f"agent_{i}": obs for i, obs in enumerate(obs_array)}
        infos = {f"agent_{i}": {} for i in range(self._num_agents)}

        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Step the environment with actions from all agents.

        Args:
            actions: Dictionary mapping agent names to actions

        Returns:
            observations: Dictionary of observations per agent
            rewards: Dictionary of rewards per agent
            terminations: Dictionary of termination flags per agent
            truncations: Dictionary of truncation flags per agent
            infos: Dictionary of info dicts per agent
        """
        self._step_count += 1

        # Convert Discrete(17) integers to dict format expected by MarlCraftiumEnv
        action_list = [_discrete_to_dict(actions[agent]) for agent in self.agents]

        # Step the underlying environment
        obs_dict, reward_dict, done_dict, truncated_dict, _ = self.env.step(
            action_list
        )

        # Convert to PettingZoo format.
        # obs_dict, reward_dict, done_dict, truncated_dict are all numpy arrays.
        # info_dict is a plain merged dict, not per-agent.
        observations = {f"agent_{i}": obs for i, obs in enumerate(obs_dict)}
        rewards = {f"agent_{i}": float(rew) for i, rew in enumerate(reward_dict)}
        terminations = {f"agent_{i}": bool(done) for i, done in enumerate(done_dict)}
        truncations = {f"agent_{i}": bool(trunc) for i, trunc in enumerate(truncated_dict)}
        infos = {f"agent_{i}": {} for i in range(self._num_agents)}

        # Apply task-focused reward shaping if specified
        if self.task_focus:
            rewards = self._apply_task_focus(rewards, infos)

        # Remove agents that are done
        self.agents = [
            agent for agent in self.agents if not terminations.get(agent, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def _apply_task_focus(
        self, rewards: Dict[str, float], infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Apply task-focused reward shaping.

        This encourages agents to focus on specific tasks by modifying rewards
        based on the task_focus parameter.

        Args:
            rewards: Original rewards
            infos: Info dictionaries containing task progress

        Returns:
            Modified rewards dictionary
        """
        # Task-specific reward bonuses
        task_bonuses = {
            "mining": {"mined_blocks": 1.0},
            "crafting": {"items_crafted": 2.0},
            "building": {"blocks_placed": 1.0},
            "exploration": {"distance_traveled": 0.1},
        }

        if self.task_focus not in task_bonuses:
            return rewards

        bonuses = task_bonuses[self.task_focus]
        shaped_rewards = rewards.copy()

        for agent, info in infos.items():
            bonus = 0.0
            for metric, weight in bonuses.items():
                if metric in info:
                    bonus += info[metric] * weight
            shaped_rewards[agent] = rewards[agent] + bonus

        return shaped_rewards

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            return self.env.render()
        return None

    def close(self):
        """Close the environment and cleanup resources."""
        self.env.close()

    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for a specific agent."""
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for a specific agent."""
        return self._action_space
