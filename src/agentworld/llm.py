"""Per-agent, per-module accounting around the shared model clients.

All agents share one inner client per response schema (vLLM:
``RemoteModelClient`` via ``LLM_BACKEND=vllm``; API anchor:
``OpenAIChatCompletionClient`` via ``LLM_BASE_URL``/``LLM_MODEL`` — both from
``mindforge.agent_modules.util.create_model_client``). :class:`AccountingClient`
wraps one inner client for one (agent, module) pair and adds only:

* a token ledger (calls, prompt and completion tokens) for the
  tokens-per-agent-vs-N plots;
* a concurrency cap shared by all wrappers (vLLM wants hundreds in flight, an
  API endpoint a handful);
* async retry with backoff, so ``llm_call``'s blocking ``time.sleep`` retry
  path is almost never reached while 100 agents are being gathered.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class TokenLedger:
    calls: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    prompt: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    completion: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    errors: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def record(self, agent: str, module: str, usage: Any) -> None:
        key = (agent, module)
        self.calls[key] += 1
        if usage is not None:
            self.prompt[key] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion[key] += int(getattr(usage, "completion_tokens", 0) or 0)

    def per_agent(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for (agent, _), v in self.calls.items():
            out[agent]["calls"] += v
        for (agent, _), v in self.prompt.items():
            out[agent]["prompt_tokens"] += v
        for (agent, _), v in self.completion.items():
            out[agent]["completion_tokens"] += v
        return {a: dict(v) for a, v in out.items()}

    def per_module(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for (_, module), v in self.calls.items():
            out[module]["calls"] += v
        for (_, module), v in self.prompt.items():
            out[module]["prompt_tokens"] += v
        for (_, module), v in self.completion.items():
            out[module]["completion_tokens"] += v
        return {m: dict(v) for m, v in out.items()}

    def snapshot(self) -> Dict[str, Any]:
        return {"per_agent": self.per_agent(), "per_module": self.per_module(),
                "errors": {f"{a}/{m}": n for (a, m), n in self.errors.items()}}


class LoopSemaphore:
    """A concurrency cap that works across successive ``asyncio.run`` calls
    (a plain Semaphore stays bound to the first loop that used it)."""

    def __init__(self, limit: int):
        self.limit = limit
        self._by_loop: Dict[int, asyncio.Semaphore] = {}

    def _sem(self) -> asyncio.Semaphore:
        key = id(asyncio.get_running_loop())
        if key not in self._by_loop:
            self._by_loop = {key: asyncio.Semaphore(self.limit)}
        return self._by_loop[key]

    async def __aenter__(self):
        await self._sem().acquire()
        return self

    async def __aexit__(self, *exc):
        self._sem().release()
        return False


class AccountingClient:
    def __init__(self, inner: Any, ledger: TokenLedger, agent: str, module: str,
                 semaphore: Any = None, retries: int = 3,
                 backoff_s: float = 1.0):
        self._inner = inner
        self._ledger = ledger
        self._agent = agent
        self._module = module
        self._sem = semaphore
        self._retries = retries
        self._backoff = backoff_s

    async def create(self, messages, **kwargs):
        attempt = 0
        while True:
            try:
                if self._sem is not None:
                    async with self._sem:
                        result = await self._inner.create(messages, **kwargs)
                else:
                    result = await self._inner.create(messages, **kwargs)
                self._ledger.record(self._agent, self._module, getattr(result, "usage", None))
                return result
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception:
                self._ledger.errors[(self._agent, self._module)] += 1
                attempt += 1
                if attempt > self._retries:
                    raise
                await asyncio.sleep(self._backoff * attempt)

    def __getattr__(self, name):
        return getattr(self._inner, name)
