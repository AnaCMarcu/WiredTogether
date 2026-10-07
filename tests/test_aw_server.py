"""Game-server helpers: port plan, environment, supervised restart handshake."""

import threading
import time

import httpx
import pytest

from agentworld import server as srv


def test_job_ports_are_disjoint_from_vllm_and_luanti():
    for job in (0, 1, 3999, 4000, 123456789):
        game, api = srv.job_ports(job)
        assert api == game + 1
        assert 20000 <= game < 28000          # vLLM: 30000–45999, Luanti: 49152+
    assert srv.job_ports(7) != srv.job_ports(8)


def test_server_env_keeps_world_in_memory():
    s = srv.KaetramServer(srv.ServerConfig(command=["true"], game_port=21000, api_port=21001,
                                           max_players=300))
    env = s.env()
    assert env["PORT"] == "21000" and env["API_PORT"] == "21001"
    assert env["SKIP_DATABASE"] == "true" and env["MAX_PLAYERS"] == "300"
    assert s.api_url.endswith(":21001")


def test_supervised_restart_waits_for_down_then_up(tmp_path, monkeypatch):
    state = {"up": True}
    restart = tmp_path / "restart_world"

    def fake_get(url, timeout=0):
        if not state["up"]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"players": []})

    monkeypatch.setattr(srv.httpx, "get", fake_get)

    def supervisor():  # what aw_common.sh does on the host
        while not restart.exists():
            time.sleep(0.05)
        state["up"] = False
        restart.unlink()
        time.sleep(0.3)
        state["up"] = True

    threading.Thread(target=supervisor, daemon=True).start()
    s = srv.SupervisedServer("http://127.0.0.1:21001", restart, ready_timeout_s=10)
    assert s.start() < 1.0
    took = s.restart()
    assert 0.2 < took < 5.0 and state["up"] and not restart.exists()


def test_supervised_restart_times_out_if_server_never_returns(tmp_path, monkeypatch):
    def fake_get(url, timeout=0):
        raise httpx.ConnectError("down for good")

    monkeypatch.setattr(srv.httpx, "get", fake_get)
    restart = tmp_path / "r"

    def supervisor():
        while not restart.exists():
            time.sleep(0.02)
        restart.unlink()      # acknowledges, but the server never comes back

    threading.Thread(target=supervisor, daemon=True).start()
    s = srv.SupervisedServer("http://x", restart, ready_timeout_s=0.3)
    with pytest.raises(TimeoutError):
        s.restart()
