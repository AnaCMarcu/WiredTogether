"""Start, health-check, restart and stop one AgentWorld (Kaetram) game server.

One world per job. Ports come from ``SLURM_JOB_ID`` (disjoint from the vLLM
range 30000–45999 used by hpc/slurm/experiments/_common.sh) so jobs sharing a
node never collide. With ``SKIP_DATABASE=true`` all world state is in memory,
so a restart is the episode reset: characters, the chat buffer, mobs, ground
items and harvested resources all start fresh.

The server command is whatever ``hpc/agentworld/run_world.sh`` (inside the
Apptainer image) or a local ``yarn`` invocation provides; Kaetram reads
``PORT``, ``API_PORT``, ``SKIP_DATABASE`` and ``MAX_PLAYERS`` from the
environment (P0: confirm the process env overrides its ``.env``).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx


def job_ports(job_id: Optional[int] = None) -> tuple[int, int]:
    """(game websocket port, HTTP API port) for this job."""
    if job_id is None:
        job_id = int(os.environ.get("SLURM_JOB_ID", os.getpid()))
    game = 20000 + 2 * (int(job_id) % 4000)
    return game, game + 1


@dataclass
class ServerConfig:
    command: Sequence[str]
    cwd: Optional[Path] = None
    game_port: int = 7030
    api_port: int = 7031
    host: str = "127.0.0.1"
    max_players: int = 200
    log_path: Optional[Path] = None
    ready_timeout_s: float = 300.0     # VoxeLibre-scale startups were 45–120 s on DAIC
    extra_env: Optional[Dict[str, str]] = None


class KaetramServer:
    def __init__(self, config: ServerConfig):
        self.cfg = config
        self.proc: Optional[subprocess.Popen] = None
        self._log = None

    @property
    def api_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.api_port}"

    def env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update({
            "PORT": str(self.cfg.game_port),
            "API_PORT": str(self.cfg.api_port),
            "API_ENABLED": "true",
            "SKIP_DATABASE": "true",
            "MAX_PLAYERS": str(self.cfg.max_players),
            "HOST": self.cfg.host,
            "HUSKY": "0",
        })
        env.update(self.cfg.extra_env or {})
        return env

    def start(self) -> float:
        """Launch and wait until the API answers; returns startup seconds."""
        if self.proc is not None and self.proc.poll() is None:
            return 0.0
        if self.cfg.log_path:
            self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.cfg.log_path, "a", encoding="utf-8")
        t0 = time.time()
        self.proc = subprocess.Popen(
            list(self.cfg.command), cwd=str(self.cfg.cwd) if self.cfg.cwd else None,
            env=self.env(), stdout=self._log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"))
        self.wait_ready()
        return time.time() - t0

    def healthy(self) -> bool:
        try:
            r = httpx.get(f"{self.api_url}/ai/world-status", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def wait_ready(self) -> None:
        deadline = time.time() + self.cfg.ready_timeout_s
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"game server exited with code {self.proc.returncode}"
                                   f" (log: {self.cfg.log_path})")
            if self.healthy():
                return
            time.sleep(2.0)
        raise TimeoutError(f"game server not ready after {self.cfg.ready_timeout_s:.0f}s")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(self.proc.pid, signal.SIGTERM)
                else:
                    self.proc.terminate()
                self.proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                self.proc.kill()
        self.proc = None
        if self._log:
            self._log.close()
            self._log = None

    def restart(self) -> float:
        self.stop()
        return self.start()


class SupervisedServer:
    """A server run by a host-side supervisor loop (hpc/agentworld/aw_common.sh).

    On the cluster the harness runs inside its own container, which cannot
    start the game server's container. The job script supervises the server
    on the host instead; :meth:`restart` asks for a restart by creating
    ``restart_file`` and waits until the API goes down and comes back up.
    """

    def __init__(self, api_url: str, restart_file: Path, ready_timeout_s: float = 300.0):
        self.api_url = api_url.rstrip("/")
        self.restart_file = Path(restart_file)
        self.ready_timeout_s = ready_timeout_s

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.api_url}/ai/world-status", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def start(self) -> float:
        t0 = time.time()
        self._wait(up=True)
        return time.time() - t0

    def _wait(self, up: bool, timeout: Optional[float] = None) -> None:
        deadline = time.time() + (timeout or self.ready_timeout_s)
        while time.time() < deadline:
            if self.healthy() == up:
                return
            time.sleep(0.5)
        raise TimeoutError(f"game server did not go {'up' if up else 'down'} in time")

    def restart(self) -> float:
        """Request a restart; the supervisor deletes the file once the old server
        is dead (the acknowledgement), then we wait for the new one to answer.
        Watching the file rather than the API avoids missing a fast restart."""
        t0 = time.time()
        self.restart_file.parent.mkdir(parents=True, exist_ok=True)
        self.restart_file.touch()
        deadline = time.time() + 60.0
        while self.restart_file.exists():
            if time.time() > deadline:
                raise TimeoutError("supervisor did not acknowledge the restart "
                                   f"(is it running? {self.restart_file})")
            time.sleep(0.2)
        self._wait(up=True)
        return time.time() - t0

    def stop(self) -> None:
        """The supervisor owns the process; the job's EXIT trap stops it."""


def world_status(api_url: str, timeout: float = 10.0) -> Dict:
    """``GET /ai/world-status`` (no auth): every player's position, one call."""
    r = httpx.get(f"{api_url}/ai/world-status", timeout=timeout)
    r.raise_for_status()
    return r.json()


def local_command(agentworld_root: Path) -> List[str]:
    """Development launch without a container (Node 20 + Yarn 4 on PATH)."""
    return ["yarn", "--cwd", str(agentworld_root), "workspace", "@kaetram/server", "start"]
