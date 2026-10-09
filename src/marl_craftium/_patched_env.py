"""Patched ``MarlCraftiumEnv`` with HPC fixes.

Six concerns layered on top of upstream:

1. **Binary name** — newer Luanti builds renamed ``./bin/minetest`` → ``./bin/luanti``.
2. **Headless clients** — upstream only forces ``SDL_VIDEODRIVER=offscreen`` on
   client 0; on display-less HPC nodes clients 1+ crash.
3. **Subprocess env inheritance** — upstream sets ``proc_env = {"SDL_VIDEODRIVER":
   "offscreen"}`` and passes it as ``env=`` to Popen, which *replaces* the parent
   env. Anything the launcher set in ``os.environ`` (e.g. ``CH1_TIMEOUT_TICKS``
   read by Lua) is dropped; we merge ``os.environ`` in.
4. **Persistent media cache** — without a stable cache symlink, every reset
   re-downloads VoxeLibre media (5-60 minutes). Symlink each client's
   ``cache/`` to ``$SCRATCH/.craftium_media_cache``.
5. **Pre-listen sockets** — upstream calls ``listen()`` only inside
   ``server_listen()`` which races with the client's ``connect()`` call.
6. **Server-ready polling** — upstream ``time.sleep(5)`` is far too short for
   VoxeLibre on HPC; we poll the server's stderr for ``"listening on"``.
7. **Client recovery at episode reset** — a client occasionally exits at the
   episode boundary ("Connection closed by peer: is MT down?" in the soft
   reset; about 1 reset in 10 with 7–10 agents on Snellius), which used to
   end the run. The state and log tail of every MT process are printed to
   the run log, the client is restarted and soft-reset again, and the run
   continues. ``WT_MT_RECOVER=0`` restores the old fail-fast behaviour.
8. **Shared game tree** — upstream copies ``games/`` (6k files) into every
   server and client run dir; it is read-only at runtime, so each run dir
   gets a symlink instead (see ``_game_tree``). ``WT_COPY_GAMES=1`` restores
   the copy.

All of these wrap upstream rather than fork it, so we stay forward-compatible
with the upstream package.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket as socket_mod
import time

from . import _bootstrap  # noqa: F401

import numpy as np

from craftium.multiagent_env import MarlCraftiumEnv, ACTION_ORDER

from . import _game_tree

try:
    from craftium import minetest as _craftium_minetest
except ImportError:  # pragma: no cover - test stand-ins have no minetest module
    _craftium_minetest = None
if _craftium_minetest is not None and _game_tree.install(_craftium_minetest):
    print("* craftium: games/ is symlinked into each run dir, not copied "
          "(WT_COPY_GAMES=1 to copy)", flush=True)


class _PatchedMarlCraftiumEnv(MarlCraftiumEnv):
    """``MarlCraftiumEnv`` with the HPC fixes listed in this module's docstring."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fix_binary_names()
        self._fix_headless_clients()
        self._inherit_parent_env()
        self._fix_media_cache()
        # Per-agent position tracking for exploration reward.
        self._prev_pos = [None] * self.num_agents
        self._positions = [None] * self.num_agents
        self._velocities = [None] * self.num_agents
        self._pitches    = [0.0] * self.num_agents
        self._yaws       = [0.0] * self.num_agents
        self._dtimes     = [0.0] * self.num_agents
        self._voxobs = [None] * self.num_agents

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

    def _inherit_parent_env(self) -> None:
        """Merge ``os.environ`` into each subprocess's ``proc_env``.

        Upstream sets ``proc_env = {"SDL_VIDEODRIVER": "offscreen"}`` and passes
        that as ``env=`` to Popen, which **replaces** the parent env. That drops
        anything the launcher set in ``os.environ`` — e.g. ``CH1_TIMEOUT_TICKS``
        (read by Lua's ``config.lua``), so Lua silently falls back to its
        hardcoded default (1200 ticks ≈ 133 env steps) and agents get
        teleported out of Ch1 almost immediately.

        Merging with proc_env overrides on top preserves both: the parent env
        propagates, and the headless SDL override still wins.
        """
        for sub in (self.mt_server, *self.mt_clients):
            sub.proc_env = {**os.environ, **(sub.proc_env or {})}

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
        observation, voxobs, pos, vel, pitch, yaw, dtime, reward, termination = (
            self.mt_channs[agent_id].receive()
        )
        if not self.gray_scale_keepdim and not self.rgb_observations:
            observation = observation[:, :, 0]

        self.last_observations[agent_id] = observation
        self._prev_pos[agent_id] = self._positions[agent_id]
        self._positions[agent_id] = pos
        self._velocities[agent_id] = vel
        self._pitches[agent_id]    = pitch
        self._yaws[agent_id]       = yaw
        self._dtimes[agent_id]     = dtime
        self._voxobs[agent_id] = voxobs

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
                observations.append(self._soft_reset_with_recovery(i))

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

    # ─── Client recovery at episode reset ─────────────────────────────

    #: Seconds to wait before each restart attempt. The server keeps a dead
    #: client's session until it times out and refuses the same player name
    #: meanwhile ("Another client is connected with this name"), so the
    #: waits grow.
    _MT_RECOVER_WAITS_S = (20, 45, 90)
    #: Accept timeout (ms) while a restarted client connects. The default
    #: listen_timeout is minutes, and a refused client never connects.
    _MT_RECOVER_LISTEN_MS = 180_000

    def _soft_reset_with_recovery(self, i: int):
        """``_soft_reset_client``, restarting client ``i`` if it has died.

        Recovery happens only at an episode boundary, where every agent is
        placed back at its spawn anyway, so no trajectory is cut short. A
        rejoining player spawns at its Ch1 tile (wire on_joinplayer), and the
        second soft reset gives it the same clean state as everyone else.
        """
        try:
            return self._soft_reset_client(i)
        except ConnectionError as exc:
            if os.environ.get("WT_MT_RECOVER", "1") == "0":
                raise
            self._report_mt_state(i, exc)
            server = self.mt_server.proc
            if server is None or server.poll() is not None:
                raise  # the server itself is gone: nothing to reconnect to

        last_exc = None
        for attempt, wait_s in enumerate(self._MT_RECOVER_WAITS_S, start=1):
            print(f"[MT-RECOVER] client {i}: restart attempt {attempt} in {wait_s} s",
                  flush=True)
            time.sleep(wait_s)
            try:
                self._restart_client(i)
                observation = self._soft_reset_client(i)
            except (ConnectionError, RuntimeError) as exc:
                last_exc = exc
                print(f"[MT-RECOVER] client {i}: attempt {attempt} failed: "
                      f"{str(exc)[:500]}", flush=True)
                continue
            self.mt_recoveries = getattr(self, "mt_recoveries", 0) + 1
            print(f"[MT-RECOVER] client {i} restarted and reset "
                  f"(recoveries in this run: {self.mt_recoveries})", flush=True)
            return observation
        raise ConnectionError(
            f"MT client {i} could not be restarted after "
            f"{len(self._MT_RECOVER_WAITS_S)} attempts"
        ) from last_exc

    def _restart_client(self, i: int) -> None:
        """Kill what is left of client ``i`` and start it again on its channel.

        The channel's listening socket outlives the client, so the new process
        connects to the same port.
        """
        client, chan = self.mt_clients[i], self.mt_channs[i]
        if client.proc is not None and client.proc.poll() is None:
            try:  # the whole process group, as upstream close() does
                os.killpg(os.getpgid(client.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            except AttributeError:  # no process groups (not Linux)
                client.proc.kill()
            client.proc.wait()
        chan.close_conn()
        client.close_pipes()
        saved_timeout = chan.listen_timeout
        chan.listen_timeout = self._MT_RECOVER_LISTEN_MS
        try:
            self._start_client_and_collect_init_obs(i)
        finally:
            chan.listen_timeout = saved_timeout

    def _report_mt_state(self, i: int, exc: BaseException) -> None:
        """Print every MT process's state, and the log tails of the ones that
        matter, to the run log. craftium deletes the run dirs on close, so
        this is the only copy that survives the job."""
        print(f"[MT-RECOVER] client {i} lost at episode reset: {exc}", flush=True)
        procs = [("server", self.mt_server)] + [
            (f"client {k}", c) for k, c in enumerate(self.mt_clients)
        ]
        for name, mt in procs:
            code = None if mt.proc is None else mt.proc.poll()
            state = "running" if code is None else f"exited with code {code}"
            print(f"[MT-RECOVER]   {name}: {state}  ({mt.run_dir})", flush=True)
        for name, mt in procs:
            exited = mt.proc is not None and mt.proc.poll() is not None
            if name in ("server", f"client {i}") or exited:
                for fname in ("stderr.txt", "debug.txt"):
                    tail = self._tail(os.path.join(mt.run_dir, fname))
                    if tail:
                        print(f"[MT-RECOVER] --- {name} {fname}, last lines ---",
                              flush=True)
                        print(tail, flush=True)

    @staticmethod
    def _tail(path: str, n_bytes: int = 3000) -> str:
        """Last ``n_bytes`` of a text file, or '' if it is missing."""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - n_bytes))
                return f.read().decode("utf-8", errors="ignore").strip()
        except OSError:
            return ""

    @staticmethod
    def _read_stderr(run_dir: str) -> str:
        """Best-effort read of run_dir/stderr.txt; returns '' if missing."""
        path = os.path.join(run_dir, "stderr.txt")
        try:
            with open(path, "r", errors="ignore") as f:
                return f.read()
        except FileNotFoundError:
            return ""
