"""RemoteModelClient connection settings (LLM_HTTP_KEEPALIVE, LLM_REQUEST_TIMEOUT).

An RL smoke on vLLM lost 105 of 137 minutes to requests stuck on keep-alive
connections the server had already closed, each waiting the 900 s timeout
before the SDK retried. LLM_HTTP_KEEPALIVE=0 opens a fresh connection per
request and LLM_REQUEST_TIMEOUT bounds the wait; unset, nothing changes.
A local HTTP/1.1 server records which client port each request came from.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("openai")

import social_stubs  # noqa: F401,E402  (autogen stand-ins, src on sys.path)
from mindforge.agent_modules import remote_model_client as rmc  # noqa: E402

_REPLY = json.dumps({
    "id": "x", "object": "chat.completion", "created": 0, "model": "wt",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}).encode()


@pytest.fixture
def server():
    ports = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"          # keep-alive capable

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            ports.append(self.client_address[1])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_REPLY)))
            self.end_headers()
            self.wfile.write(_REPLY)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.url = "http://127.0.0.1:%d/v1" % httpd.server_address[1]
    httpd.ports = ports
    yield httpd
    httpd.shutdown()


def _client(monkeypatch, url, **env):
    monkeypatch.setattr(rmc, "_load_tokenizer_only", lambda path: None)
    for key in ("LLM_HTTP_KEEPALIVE", "LLM_REQUEST_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return rmc.RemoteModelClient(model_path="unused", base_url=url)


def _two_requests(client):
    async def run():
        for _ in range(2):
            await client._client.chat.completions.create(
                model="wt", messages=[{"role": "user", "content": "hi"}])
    asyncio.run(run())


def test_keepalive_off_opens_a_connection_per_request(server, monkeypatch):
    _two_requests(_client(monkeypatch, server.url, LLM_HTTP_KEEPALIVE="0"))
    assert len(server.ports) == 2 and len(set(server.ports)) == 2


def test_default_still_reuses_the_connection(server, monkeypatch):
    _two_requests(_client(monkeypatch, server.url))
    assert len(server.ports) == 2 and len(set(server.ports)) == 1


def test_request_timeout_comes_from_the_environment(monkeypatch):
    client = _client(monkeypatch, "http://127.0.0.1:9/v1", LLM_REQUEST_TIMEOUT="120")
    assert client._client.timeout == 120.0


def test_empty_settings_keep_the_defaults(monkeypatch):
    client = _client(monkeypatch, "http://127.0.0.1:9/v1",
                     LLM_REQUEST_TIMEOUT="", LLM_HTTP_KEEPALIVE="")
    assert client._client.timeout == 900.0
