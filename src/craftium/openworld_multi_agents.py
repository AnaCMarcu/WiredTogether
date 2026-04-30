"""PettingZoo ParallelEnv wrapper for Craftium's OpenWorld environment."""

import os
import socket as socket_mod
import time
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from craftium.multiagent_env import MarlCraftiumEnv, ACTION_ORDER


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
        self._fix_media_cache()
        # Per-agent position tracking for exploration reward
        self._prev_pos = [None] * self.num_agents
        self._positions = [None] * self.num_agents

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
    # Patch 1c: persistent media cache across resets
    # ------------------------------------------------------------------ #
    def _fix_media_cache(self):
        """Symlink a persistent cache dir into each client's run_dir.

        Craftium creates a fresh UUID-based run_dir for every client, and
        in RUN_IN_PLACE mode the media cache lives at ``{run_dir}/cache/``.
        This means every reset re-downloads all VoxeLibre media (5-60 min).

        Fix: create a single persistent cache directory and symlink each
        client's ``{run_dir}/cache`` to it.  The media files are keyed by
        SHA-1, so sharing across clients is safe.
        """
        import shutil

        # Prefer $SCRATCH (node-local SSD) over $HOME (NFS) for the media cache.
        # NFS latency during texture loading can cause the client to time out and
        # drop its Python TCP connection, producing the "Connection closed by peer" error.
        scratch = os.environ.get("SCRATCH", "")
        if scratch:
            persistent_cache = os.path.join(scratch, ".craftium_media_cache")
        else:
            persistent_cache = os.path.join(os.path.expanduser("~"), ".craftium_media_cache")
        os.makedirs(persistent_cache, exist_ok=True)

        for i, client in enumerate(self.mt_clients):
            client_cache = os.path.join(client.run_dir, "cache")
            # Remove any existing cache dir/symlink that the fresh run_dir may have
            if os.path.islink(client_cache):
                os.unlink(client_cache)
            elif os.path.isdir(client_cache):
                shutil.rmtree(client_cache)
            os.symlink(persistent_cache, client_cache)

        # Also symlink for the server (it caches world data)
        server_cache = os.path.join(self.mt_server.run_dir, "cache")
        if os.path.islink(server_cache):
            os.unlink(server_cache)
        elif os.path.isdir(server_cache):
            shutil.rmtree(server_cache)
        os.symlink(persistent_cache, server_cache)

        print(f"* Symlinked media cache -> {persistent_cache}")

    # ------------------------------------------------------------------ #
    # Patch 2a: warm-up NoOp — keep clients alive without incrementing timesteps
    # ------------------------------------------------------------------ #
    def warmup_noop(self):
        """Send a NoOp to every agent without incrementing the timestep counter.

        Used during the media-loading warm-up phase to keep TCP channels alive.
        Returns list of observations (one per agent) so caller can inspect them.
        """
        keys = [0] * 21
        observations = []
        for agent_id in range(self.num_agents):
            self.mt_channs[agent_id].send(keys, 0, 0)
            obs, *_ = self.mt_channs[agent_id].receive()
            observations.append(obs)
        return observations

    # ------------------------------------------------------------------ #
    # Patch 2: capture position data from step_agent
    # ------------------------------------------------------------------ #
    def step_agent(self, action):
        """Override to capture position data for exploration reward."""
        if self.current_agent_id == self.num_agents:
            self.current_agent_id = 0
        agent_id = self.current_agent_id
        self.current_agent_id += 1

        # Count one timestep per round, not per agent.
        # Increment only when the last agent in the round is processed.
        if agent_id == self.num_agents - 1:
            self.timesteps += 1

        keys = [0] * 21
        mouse_x, mouse_y = 0, 0
        for k, v in action.items():
            if k == "mouse":
                x, y = v[0], -v[1]
                mouse_x = int(x * (self.obs_width // 2))
                mouse_y = int(y * (self.obs_height // 2))
            else:
                keys[ACTION_ORDER.index(k)] = v

        self.mt_channs[agent_id].send(keys, mouse_x, mouse_y)

        observation, _voxobs, pos, _vel, _pitch, _yaw, _dtime, reward, termination = (
            self.mt_channs[agent_id].receive()
        )
        if not self.gray_scale_keepdim and not self.rgb_observations:
            observation = observation[:, :, 0]

        self.last_observations[agent_id] = observation

        # Store position for exploration reward
        self._prev_pos[agent_id] = self._positions[agent_id]
        self._positions[agent_id] = pos

        info = self._get_info()
        truncated = self.max_timesteps is not None and self.timesteps >= self.max_timesteps

        return observation, reward, termination, truncated, info

    # ------------------------------------------------------------------ #
    # Patch 3: pre-listen on all channel sockets
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
            server_ready_timeout = 1200  # 20 min — slow/contended HPC nodes need this headroom
            deadline = time.time() + server_ready_timeout
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
                try:
                    with open(stderr_path, "r", errors="ignore") as f:
                        tail = f.read()[-2000:]
                except FileNotFoundError:
                    tail = "(file not found)"
                raise RuntimeError(
                    f"MT server did not reach 'listening on' within "
                    f"{server_ready_timeout} s. Aborting before clients connect.\n"
                    f"stderr tail:\n{tail}"
                )
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
    "mouse x+", "mouse x-", "mouse y-", "mouse y+",  # y- = look down, y+ = look up (Minetest Y-axis is inverted)
    # --- added actions (indices 17-21) ---
    "inventory",                          # toggle inventory/crafting menu
    "drop",                               # drop held item
    "slot_6", "slot_7", "slot_8",         # extra hotbar slots
]
_MOUSE_MOV = 1.0  # doubled from 0.5: ~20-30° per step, halves steps needed for orientation


def _discrete_to_dict(action: int) -> dict:
    """Convert a Discrete(23) integer to MarlCraftiumEnv dict format.

    Action 0 is NOP. Actions 1-22 map to _DISCRETE_ACTIONS.
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
        obs_width: int = 480,
        obs_height: int = 480,
        max_steps: int = 10000,
        task_focus: Optional[str] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        frameskip: int = 3,
        pmul: int = 20,
    ):
        super().__init__()

        # Store config (use _num_agents to avoid conflict with PettingZoo property)
        self._num_agents = num_agents
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.max_steps = max_steps
        self.task_focus = task_focus
        self.render_mode = render_mode
        self.frameskip = frameskip
        self.pmul = pmul

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
            game_id="VoxeLibre",
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            # CraftiumEnvironmentInterface calls env.step() once per agent per
            # round, and each env.step() triggers one step_agent round (which
            # increments timesteps once after fix).  So timesteps = num_agents
            # * main_loop_steps.  Scale max_timesteps to match the user's
            # intended max_steps (main-loop rounds).
            max_timesteps=max_steps * num_agents,
            minetest_dir=minetest_dir,
            mt_listen_timeout=300_000,  # 5 min per client; VoxeLibre loads slowly on HPC
            seed=seed,  # fixed_map_seed for Minetest world generation
            frameskip=frameskip,
            pmul=pmul,
            # Scale HUD down to 0.5 (engine minimum) so hearts/food bar occupy
            # less of the 320×180 observation frame.
            mt_clients_conf={"hud_scaling": 0.5},
        )

        # Define observation and action spaces
        self._observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )
        # Craftium discrete actions: 0=NOP + 22 named actions
        self._action_space = spaces.Discrete(23)

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

        # Convert Discrete(23) integers to dict format expected by MarlCraftiumEnv
        # Always send actions for ALL agents (NoOp for terminated ones) —
        # MarlCraftiumEnv expects exactly num_agents actions every step.
        action_list = [
            _discrete_to_dict(actions.get(agent, 0))
            for agent in self.possible_agents
        ]

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

        # Dense reward shaping: exploration bonus based on distance moved
        for i in range(self._num_agents):
            prev = self.env._prev_pos[i]
            curr = self.env._positions[i]
            if prev is not None and curr is not None:
                # Euclidean distance in xz plane (ignore y to avoid jump-spam)
                dx = curr[0] - prev[0]
                dz = curr[2] - prev[2]
                dist = float(np.sqrt(dx * dx + dz * dz))
                # Scale: ~0.1 reward per node moved (small vs dig=1.0, stage=128+)
                rewards[f"agent_{i}"] += 0.1 * dist

        # Apply task-focused reward shaping if specified
        if self.task_focus:
            rewards = self._apply_task_focus(rewards, infos)

        # Remove agents that are done
        self.agents = [
            agent for agent in self.agents if not terminations.get(agent, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def warmup_noop(self):
        """Send NoOps to keep channels alive without incrementing step counters.

        Returns list of observations (one per agent).
        """
        return self.env.warmup_noop()

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
