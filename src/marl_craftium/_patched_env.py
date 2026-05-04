"""Patched ``MarlCraftiumEnv`` with HPC fixes.

Five concerns layered on top of upstream:

1. **Binary name** — newer Luanti builds renamed ``./bin/minetest`` → ``./bin/luanti``.
2. **Headless clients** — upstream only forces ``SDL_VIDEODRIVER=offscreen`` on
   client 0; on display-less HPC nodes clients 1+ crash.
3. **Persistent media cache** — without a stable cache symlink, every reset
   re-downloads VoxeLibre media (5-60 minutes). Symlink each client's
   ``cache/`` to ``$SCRATCH/.craftium_media_cache``.
4. **Pre-listen sockets** — upstream calls ``listen()`` only inside
   ``server_listen()`` which races with the client's ``connect()`` call.
5. **Server-ready polling** — upstream ``time.sleep(5)`` is far too short for
   VoxeLibre on HPC; we poll the server's stderr for ``"listening on"``.

All of these wrap upstream rather than fork it, so we stay forward-compatible
with the upstream package.
"""

from __future__ import annotations

import os
import shutil
import socket as socket_mod
import time

# Side-effect: ensure the in-tree craftium submodule is importable as the
# top-level `craftium` package before we touch any `from craftium...`.
from . import _bootstrap  # noqa: F401

import numpy as np

from craftium.multiagent_env import MarlCraftiumEnv, ACTION_ORDER


class _PatchedMarlCraftiumEnv(MarlCraftiumEnv):
    """``MarlCraftiumEnv`` with the HPC fixes listed in this module's docstring."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fix_binary_names()
        self._fix_headless_clients()
        self._fix_media_cache()
        # Per-agent position tracking for exploration reward.
        self._prev_pos = [None] * self.num_agents
        self._positions = [None] * self.num_agents

    # ─── Patches at construction time ─────────────────────────────────

    def _fix_binary_names(self) -> None:
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

    def _fix_headless_clients(self) -> None:
        """Force ``SDL_VIDEODRIVER=offscreen`` on every client.

        Upstream sets ``headless = (i == 0) and (render_mode != 'human')`` which
        leaves clients 1+ non-headless and they crash on display-less HPC nodes.
        """
        for client in self.mt_clients:
            if client.proc_env is None:
                client.proc_env = {"SDL_VIDEODRIVER": "offscreen"}
            elif "SDL_VIDEODRIVER" not in client.proc_env:
                client.proc_env["SDL_VIDEODRIVER"] = "offscreen"
        print(f"* Forced all {len(self.mt_clients)} clients to headless (offscreen SDL)")

    def _fix_media_cache(self) -> None:
        """Symlink a persistent cache dir into each client's run_dir.

        Craftium creates a fresh UUID-based run_dir for every client and the
        media cache lives at ``{run_dir}/cache/``. Without this fix, every
        reset re-downloads ~700 MB of VoxeLibre media (5-60 min). Sharing the
        cache across clients is safe — files are content-addressed by SHA-1.
        """
        cache_dir = self._persistent_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)

        for client in self.mt_clients:
            self._symlink_cache(client.run_dir, cache_dir)
        # The server caches world data too.
        self._symlink_cache(self.mt_server.run_dir, cache_dir)
        print(f"* Symlinked media cache -> {cache_dir}")

    @staticmethod
    def _persistent_cache_dir() -> str:
        """Prefer node-local SSD ($SCRATCH) over NFS $HOME — NFS latency
        causes texture-load timeouts that drop the Python TCP channel."""
        scratch = os.environ.get("SCRATCH", "")
        if scratch:
            return os.path.join(scratch, ".craftium_media_cache")
        return os.path.join(os.path.expanduser("~"), ".craftium_media_cache")

    @staticmethod
    def _symlink_cache(run_dir: str, cache_dir: str) -> None:
        target = os.path.join(run_dir, "cache")
        if os.path.islink(target):
            os.unlink(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
        os.symlink(cache_dir, target)

    # ─── Per-step overrides ───────────────────────────────────────────

    def warmup_noop(self):
        """NoOp every agent without incrementing the timestep counter.

        Used during the media-loading warm-up so TCP channels stay alive.
        Returns the per-agent observation list.
        """
        keys = [0] * 21
        observations = []
        for agent_id in range(self.num_agents):
            self.mt_channs[agent_id].send(keys, 0, 0)
            obs, *_ = self.mt_channs[agent_id].receive()
            observations.append(obs)
        return observations

    def step_agent(self, action):
        """Override upstream to capture per-agent positions for shaping."""
        if self.current_agent_id == self.num_agents:
            self.current_agent_id = 0
        agent_id = self.current_agent_id
        self.current_agent_id += 1

        # Count one timestep per round (when the last agent is processed),
        # not once per agent — otherwise max_timesteps fires num_agents× too soon.
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
        self._prev_pos[agent_id] = self._positions[agent_id]
        self._positions[agent_id] = pos

        info = self._get_info()
        truncated = self.max_timesteps is not None and self.timesteps >= self.max_timesteps
        return observation, reward, termination, truncated, info

    # ─── Reset: server polling + pre-listen + diagnostics ─────────────

    def reset(self, **kwargs):
        self.timesteps = 0
        observations = []

        if self.mt_server.proc is None:
            self._start_server_with_diagnostics()
            self._wait_until_server_ready()
            time.sleep(3)  # extra stabilisation
            self._pre_listen_channels()
            for i in range(self.num_agents):
                observations.append(self._start_client_and_collect_init_obs(i))
        else:
            for i in range(self.num_agents):
                observations.append(self._soft_reset_client(i))

        infos = self._get_info()
        observations = np.vstack([np.expand_dims(obs, 0) for obs in observations])
        return observations, infos

    # ─── Reset helpers ────────────────────────────────────────────────

    def _start_server_with_diagnostics(self) -> None:
        print(f"* Server launch cmd: {self.mt_server.launch_cmd}")
        print(f"* Server run dir:    {self.mt_server.run_dir}")
        self.mt_server.start_process()
        time.sleep(1)
        ret = self.mt_server.proc.poll()
        if ret is not None:
            raise RuntimeError(
                f"MT server process exited immediately with code {ret}.\n"
                f"stderr:\n{self._read_stderr(self.mt_server.run_dir)}"
            )

    def _wait_until_server_ready(self, timeout_s: float = 1200.0) -> None:
        """Poll stderr.txt for 'listening on'. 20-min ceiling on slow HPC nodes."""
        print(
            "* Waiting for MT server to initialize (polling stderr). "
            "This is only required in the first call to reset."
        )
        deadline = time.time() + timeout_s
        stderr_path = os.path.join(self.mt_server.run_dir, "stderr.txt")
        while time.time() < deadline:
            time.sleep(2)
            ret = self.mt_server.proc.poll()
            if ret is not None:
                raise RuntimeError(
                    f"MT server died during init (exit code {ret}).\n"
                    f"stderr:\n{self._read_stderr(self.mt_server.run_dir)}"
                )
            try:
                with open(stderr_path, "r", errors="ignore") as f:
                    if "listening on" in f.read():
                        print("* MT server is ready!")
                        return
            except (FileNotFoundError, OSError):
                pass
        # Did not reach "listening on" → raise with stderr tail.
        tail = self._read_stderr(self.mt_server.run_dir)[-2000:]
        raise RuntimeError(
            f"MT server did not reach 'listening on' within {timeout_s} s. "
            f"Aborting before clients connect.\nstderr tail:\n{tail}"
        )

    def _pre_listen_channels(self) -> None:
        """Call ``listen(1)`` on every MtChannel socket BEFORE any client starts.

        ``mt_server.init_server()`` does ``socket() + bind()`` only — ``listen()``
        normally fires inside ``server_listen()`` (via ``open_conn``), but by then
        the client has often already tried ``connect()`` and got *Connection
        refused*. ``socket.fromfd`` duplicates the fd so closing the wrapper
        leaves the original socket intact and in LISTEN state. Calling
        ``listen()`` twice (here + inside ``server_listen``) is harmless on Linux.
        """
        for ch in self.mt_channs:
            sock = socket_mod.fromfd(ch.sockfd, socket_mod.AF_INET, socket_mod.SOCK_STREAM)
            sock.listen(1)
            sock.close()
        print(f"* Pre-listened on {len(self.mt_channs)} channel sockets")

    def _start_client_and_collect_init_obs(self, i: int):
        """Launch client `i`, open the TCP channel, run init_frames, return first obs."""
        print(f"* Starting client {i}: {self.mt_clients[i].launch_cmd}")
        self.mt_clients[i].start_process()
        # open_conn() must run RIGHT AFTER start_process() — any sleep risks
        # the client's connect() arriving before accept() fires.
        try:
            self.mt_channs[i].open_conn()
        except ConnectionError:
            ret = self.mt_clients[i].proc.poll()
            raise RuntimeError(
                f"MT client {i} connection failed (exit code {ret}).\n"
                f"stderr:\n{self._read_stderr(self.mt_clients[i].run_dir)}"
            )

        for _ in range(self.init_frames):
            _obs, *_ = self.mt_channs[i].receive()
            self.mt_channs[i].send([0] * 21, 0, 0)

        observation, _voxobs, _pos, _vel, _pitch, _yaw, _dtime, _reward, _term = (
            self.mt_channs[i].receive()
        )
        if not self.gray_scale_keepdim and not self.rgb_observations:
            observation = observation[:, :, 0]
        self.last_observations[i] = observation
        return observation

    def _soft_reset_client(self, i: int):
        """Send a soft-reset signal and read the first post-reset observation."""
        self.mt_channs[i].send_soft_reset()
        observation, _voxobs, _pos, _vel, _pitch, _yaw, _dtime, _reward, _term = (
            self.mt_channs[i].receive()
        )
        if not self.gray_scale_keepdim and not self.rgb_observations:
            observation = observation[:, :, 0]
        self.last_observations[i] = observation
        return observation

    @staticmethod
    def _read_stderr(run_dir: str) -> str:
        """Best-effort read of run_dir/stderr.txt; returns '' if missing."""
        path = os.path.join(run_dir, "stderr.txt")
        try:
            with open(path, "r", errors="ignore") as f:
                return f.read()
        except FileNotFoundError:
            return ""
