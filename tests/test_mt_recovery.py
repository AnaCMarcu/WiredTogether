"""Client recovery at episode reset (_PatchedMarlCraftiumEnv).

A Luanti client occasionally exits at an episode boundary, and the soft reset
then fails with "Connection closed by peer: is MT down?". These tests drive
the reset with fake processes and channels (conftest stubs craftium) and pin:

  * a healthy reset never restarts anything (legacy path unchanged);
  * a lost client is restarted, soft-reset again, and the run continues,
    with the channel's accept timeout restored afterwards;
  * the MT process states are reported to the run log;
  * no recovery when the server itself is gone, or with WT_MT_RECOVER=0;
  * a client that cannot be brought back still ends the run, after the
    configured number of attempts.
"""

import numpy as np
import pytest

from marl_craftium import _patched_env as pe


class FakeProc:
    def __init__(self, code=None):
        self.code = code
        self.pid = 12345

    def poll(self):
        return self.code

    def wait(self):
        return self.code

    def kill(self):
        self.code = -9


class FakeMT:
    """A server or client: a process and a run dir."""

    def __init__(self, run_dir, code=None):
        self.run_dir = str(run_dir)
        self.proc = FakeProc(code)
        self.starts = 0
        self.launch_cmd = ["luanti"]

    def start_process(self):
        self.starts += 1
        self.proc = FakeProc(None)

    def close_pipes(self):
        pass


class FakeChannel:
    """Answers receive() with an observation unless told to fail."""

    def __init__(self, fail_receives=0):
        self.fail_receives = fail_receives
        self.listen_timeout = 300_000
        self.soft_resets = 0
        self.opened = 0

    def send_soft_reset(self):
        self.soft_resets += 1

    def send(self, keys, mouse_x, mouse_y):
        pass

    def receive(self):
        if self.fail_receives > 0:
            self.fail_receives -= 1
            raise ConnectionError("Failed to receive from MT. Connection closed by peer: is MT down?")
        obs = np.zeros((2, 2, 3), dtype=np.uint8)
        return obs, None, (0, 0, 0), (0, 0, 0), 0.0, 0.0, 0.0, 0.0, False

    def close_conn(self):
        pass

    def open_conn(self):
        self.opened += 1


def make_env(tmp_path, channels, server_code=None):
    env = object.__new__(pe._PatchedMarlCraftiumEnv)
    env.num_agents = len(channels)
    env.timesteps = 5
    env.init_frames = 1
    env.gray_scale_keepdim = False
    env.rgb_observations = True
    env.last_observations = [None] * env.num_agents
    env.mt_channs = channels
    env.mt_server = FakeMT(tmp_path / "server", code=server_code)
    env.mt_clients = [FakeMT(tmp_path / f"client{i}", code=None) for i in range(env.num_agents)]
    env._get_info = lambda: {}
    return env


@pytest.fixture(autouse=True)
def no_waits(monkeypatch):
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.delenv("WT_MT_RECOVER", raising=False)


def test_healthy_reset_restarts_nothing(tmp_path):
    env = make_env(tmp_path, [FakeChannel(), FakeChannel(), FakeChannel()])
    observations, _ = env.reset()
    assert observations.shape[0] == 3
    assert [c.starts for c in env.mt_clients] == [0, 0, 0]
    assert [ch.soft_resets for ch in env.mt_channs] == [1, 1, 1]
    assert getattr(env, "mt_recoveries", 0) == 0


def test_lost_client_is_restarted_and_run_continues(tmp_path, capsys):
    (tmp_path / "client1").mkdir()
    (tmp_path / "client1" / "stderr.txt").write_text("ERROR: something broke\n")
    lost = FakeChannel(fail_receives=1)
    env = make_env(tmp_path, [FakeChannel(), lost, FakeChannel()])
    env.mt_clients[1].proc = FakeProc(code=-11)       # the client segfaulted

    observations, _ = env.reset()

    assert observations.shape[0] == 3
    assert env.mt_clients[1].starts == 1              # only the lost client
    assert [env.mt_clients[i].starts for i in (0, 2)] == [0, 0]
    assert lost.soft_resets == 2                      # failed reset + the retry
    assert lost.listen_timeout == 300_000             # restored after restart
    assert env.mt_recoveries == 1
    out = capsys.readouterr().out
    assert "client 1: exited with code -11" in out
    assert "ERROR: something broke" in out
    assert "client 1 restarted and reset" in out


def test_no_recovery_when_the_server_is_gone(tmp_path):
    env = make_env(tmp_path, [FakeChannel(fail_receives=1)], server_code=1)
    with pytest.raises(ConnectionError):
        env.reset()
    assert env.mt_clients[0].starts == 0


def test_recovery_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("WT_MT_RECOVER", "0")
    env = make_env(tmp_path, [FakeChannel(fail_receives=1)])
    with pytest.raises(ConnectionError):
        env.reset()
    assert env.mt_clients[0].starts == 0


def test_client_that_cannot_come_back_ends_the_run(tmp_path):
    attempts = len(pe._PatchedMarlCraftiumEnv._MT_RECOVER_WAITS_S)
    env = make_env(tmp_path, [FakeChannel(fail_receives=1 + attempts)])
    with pytest.raises(ConnectionError, match="could not be restarted"):
        env.reset()
    assert env.mt_clients[0].starts == attempts


# ── Warm-up right after a reset (2026-10-10, N=15 on Snellius) ─────────────
# Every failed N=15 run died with "is MT down?" after the episode ended but
# with no [MT-RECOVER] line: the client survived its own soft reset and died
# a moment later, so the first warm-up NoOp was the call that noticed.


def test_client_lost_during_warmup_is_restarted(tmp_path, capsys):
    lost = FakeChannel(fail_receives=1)
    env = make_env(tmp_path, [FakeChannel(), lost, FakeChannel()])
    env.mt_clients[1].proc = FakeProc(code=-11)

    observations = env.warmup_noop()

    assert len(observations) == 3
    assert env.mt_clients[1].starts == 1              # only the lost client
    assert [env.mt_clients[i].starts for i in (0, 2)] == [0, 0]
    assert lost.soft_resets == 1                      # the clean reset after the restart
    assert env.mt_recoveries == 1
    out = capsys.readouterr().out
    assert "client 1 lost during warm-up" in out
    assert "client 1 restarted and reset" in out


def test_healthy_warmup_restarts_nothing(tmp_path):
    env = make_env(tmp_path, [FakeChannel(), FakeChannel()])
    assert len(env.warmup_noop()) == 2
    assert [c.starts for c in env.mt_clients] == [0, 0]
    assert getattr(env, "mt_recoveries", 0) == 0


def test_warmup_recovery_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("WT_MT_RECOVER", "0")
    env = make_env(tmp_path, [FakeChannel(fail_receives=1)])
    with pytest.raises(ConnectionError):
        env.warmup_noop()
    assert env.mt_clients[0].starts == 0


def test_no_warmup_recovery_when_the_server_is_gone(tmp_path):
    env = make_env(tmp_path, [FakeChannel(fail_receives=1)], server_code=1)
    with pytest.raises(ConnectionError, match="is MT down"):
        env.warmup_noop()
    assert env.mt_clients[0].starts == 0
