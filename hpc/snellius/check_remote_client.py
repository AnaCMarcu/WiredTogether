#!/usr/bin/env python3
"""Offline check of RemoteModelClient against a fake vLLM server. No GPU.

Runs on a login node inside the experiment image:

    apptainer exec --bind $WT_WORKSPACE $WT_IMAGE \
        env PYTHONPATH=$WT_WORKSPACE/WiredTogether/src \
        python hpc/snellius/check_remote_client.py $MODEL_LLM

It builds an agent-style request (system prompt + frame + text, with a
response schema), sends it through util.create_model_client with
LLM_BACKEND=vllm, and checks what the fake server received:
  * the prompt carries LocalModelClient's injected JSON instruction
  * the frame arrives as a lossless PNG data URL
  * sampling = temperature 0.7, top_p 0.9, max_tokens 1024, thinking off
  * no response_format (so vLLM does not constrain the output)
  * the answer is post-processed like the in-process path (<think> stripped)
  * a chat template that rejects the system role triggers the merge fallback
  * count_text_tokens (comm-budget ledger) uses the real tokenizer
Prints PASS/FAIL per check and exits non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_LLM", "")
if not MODEL or not os.path.isfile(os.path.join(MODEL, "config.json")):
    sys.exit(f"usage: check_remote_client.py <model dir>  (got {MODEL!r})")

RECEIVED: list = []
REJECT_SYSTEM_ONCE = {"armed": False}


class FakeVLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": req})
        has_system = any(m["role"] == "system" for m in req["messages"])
        if REJECT_SYSTEM_ONCE["armed"] and has_system:
            REJECT_SYSTEM_ONCE["armed"] = False
            self._send(400, {"error": {"message": "System role not supported by chat template",
                                       "type": "BadRequestError", "code": 400}})
            return
        self._send(200, {
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "wt",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant",
                "content": '<think>hmm</think>{"thoughts": "t", "action": "MoveForward"}'}}],
            "usage": {"prompt_tokens": 321, "completion_tokens": 12, "total_tokens": 333},
        })


server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
threading.Thread(target=server.serve_forever, daemon=True).start()
port = server.server_address[1]

os.environ.update({
    "LLM_BACKEND": "vllm",
    "LLM_MODEL_PATH": MODEL,
    "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
    "LLM_SERVER_KEY": "k123",
    "LLM_ENABLE_THINKING": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import PIL.Image  # noqa: E402
from autogen_core import Image as AgImage  # noqa: E402
from autogen_core.models import SystemMessage, UserMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from mindforge.agent_modules import local_model_client as lmc  # noqa: E402
from mindforge.agent_modules.util import create_model_client  # noqa: E402


class Resp(BaseModel):
    thoughts: str
    action: str


results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))


async def main() -> None:
    client = create_model_client(response_format=Resp)
    check("factory returns RemoteModelClient", type(client).__name__ == "RemoteModelClient",
          type(client).__name__)
    check("tokenizer loaded; vision detected", lmc._shared_tokenizer is not None,
          f"vision={lmc._shared_is_vision}")
    print(f"  info  model vision={lmc._shared_is_vision}")

    frame = PIL.Image.new("RGB", (480, 480), (10, 200, 30))
    msgs = [SystemMessage(content="You are agent_0 in a team of 3."),
            UserMessage(content=["Here is your view.", AgImage(frame)], source="user")]

    out = await client.create(msgs)
    body = RECEIVED[-1]["body"]
    check("request hit /v1/chat/completions", RECEIVED[-1]["path"] == "/v1/chat/completions",
          RECEIVED[-1]["path"])
    check("per-job API key sent", RECEIVED[-1]["auth"] == "Bearer k123", str(RECEIVED[-1]["auth"]))
    check("served model name 'wt'", body.get("model") == "wt", str(body.get("model")))
    check("temperature 0.7", body.get("temperature") == 0.7, str(body.get("temperature")))
    check("top_p 0.9", body.get("top_p") == 0.9, str(body.get("top_p")))
    check("max_tokens 1024", body.get("max_tokens") == 1024, str(body.get("max_tokens")))
    check("thinking off via chat_template_kwargs",
          body.get("chat_template_kwargs") == {"enable_thinking": False},
          str(body.get("chat_template_kwargs")))
    check("no response_format (no constrained decoding)", "response_format" not in body,
          str(body.get("response_format")))

    sys_msg = body["messages"][0]
    check("system prompt first", sys_msg["role"] == "system", sys_msg["role"])
    sys_text = sys_msg["content"] if isinstance(sys_msg["content"], str) else json.dumps(sys_msg["content"])
    check("injected JSON instruction present", "You MUST respond with valid JSON" in sys_text)
    check("schema fields listed", '"action"' in sys_text and "REQUIRED" in sys_text)

    user = body["messages"][1]["content"]
    if lmc._shared_is_vision:
        imgs = [p for p in user if p.get("type") == "image_url"]
        check("frame sent as PNG data URL", len(imgs) == 1
              and imgs[0]["image_url"]["url"].startswith("data:image/png;base64,"))
    else:
        check("text-only model: frame dropped with note",
              isinstance(user, str) and "cannot process images" in user, str(user)[:120])

    check("<think> stripped, JSON kept", out.content == '{"thoughts": "t", "action": "MoveForward"}',
          out.content)
    check("usage from server", out.usage.prompt_tokens == 321 and out.usage.completion_tokens == 12)

    REJECT_SYSTEM_ONCE["armed"] = True
    n_before = len(RECEIVED)
    out2 = await client.create(msgs)
    retry = RECEIVED[-1]["body"]["messages"]
    check("system-role rejection -> retried once, merged into user turn",
          len(RECEIVED) == n_before + 2 and retry[0]["role"] == "user"
          and not any(m["role"] == "system" for m in retry), f"{len(RECEIVED) - n_before} requests")
    check("fallback answer returned", out2.content.startswith("{"))

    n = lmc.count_text_tokens("hello team, the door is open")
    check("count_text_tokens uses the real tokenizer", isinstance(n, int) and 3 <= n <= 20, str(n))

    # Concurrency: gathered calls must be in flight together (what vLLM batches).
    n_before = len(RECEIVED)
    await asyncio.gather(*(client.create(msgs) for _ in range(3)))
    check("3 gathered calls all served", len(RECEIVED) == n_before + 3)
    await client.close()


asyncio.run(main())
server.shutdown()
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
