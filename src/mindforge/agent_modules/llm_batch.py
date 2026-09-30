"""Cross-call micro-batching for the shared in-process LLM (``--llm-batch``).

WHY
  Profiling finished runs (analysis/profile_run.py) shows 90-96 % of the
  wall-clock is LLM calls and ~95 % of THAT is decode, at batch size 1: the
  one shared model generates for one agent at a time while the GPU sits
  mostly idle. Every agent's calls within a step are independent of each
  other (they all read the shared pre-step state s_t), so they can share one
  ``generate()``.

WHAT IT DOES
  With the switch on, callers stop invoking the model themselves. They hand a
  request to a :class:`MicroBatcher` and await a future. One drain task owns
  the model: it waits until every coroutine that can currently run has queued
  its request (a few bare event-loop yields, until the queue stops growing),
  then runs everything pending as one batch per compatible group and resolves
  the futures. Nothing else touches the model, so interleaved coroutines can
  no longer reach it concurrently - the failure that forced the simultaneous
  pre-pass to stay sequential.

  The batcher only batches what is pending AT THE SAME TIME, so the call
  sites that used to be sequential-by-construction switch to
  ``asyncio.gather`` when (and only when) the switch is on; see
  :func:`gather_or_sequential`.

WHAT STAYS IDENTICAL
  With ``--llm-batch`` unset (every existing suite) the switch is off: no
  batcher is built, ``LocalModelClient.create`` takes its original path and
  :func:`gather_or_sequential` awaits one coroutine after another in the
  original order. Prompts, sampling parameters and token caps are the same in
  both modes; batching changes WHICH random draws a call gets, not the
  distribution it is drawn from.

  The master switch is the ``WT_LLM_BATCH`` environment variable, set by
  multi_agent_craftium.py when the flag is given (mirrors ``WT_COMM_BUDGET``),
  so modules that never see ``args`` agree with the training loop.

Pure stdlib; safe to import anywhere (tests included).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable, Hashable, List, Optional, Sequence

#: Master switch (set by the training loop; read by the client + call sites).
ENV_SWITCH = "WT_LLM_BATCH"
#: Largest batch one ``generate()`` may carry; bigger queues are split.
ENV_MAX_BATCH = "WT_LLM_BATCH_MAX"
DEFAULT_MAX_BATCH = 16

#: The queue counts as settled after this many consecutive event-loop yields
#: with no new request. One yield lets every READY coroutine run up to its
#: next suspension, so 2 covers a caller that fans out (an inner gather)
#: right after it was resumed.
_SETTLE_STABLE = 2
#: Hard cap on yields per settle, so a pathological caller cannot starve the
#: queue forever.
_SETTLE_MAX_YIELDS = 200


# ── Master switch ───────────────────────────────────────────────────────

def llm_batch_enabled() -> bool:
    """The master switch as seen via the environment."""
    return os.environ.get(ENV_SWITCH) == "1"


def max_batch_from_env() -> int:
    try:
        return max(1, int(os.environ.get(ENV_MAX_BATCH, DEFAULT_MAX_BATCH)))
    except ValueError:
        return DEFAULT_MAX_BATCH


def set_env_switch(enabled: bool, max_batch: int = DEFAULT_MAX_BATCH) -> None:
    """Set (or clear) the master switch for this process."""
    if enabled:
        os.environ[ENV_SWITCH] = "1"
        os.environ[ENV_MAX_BATCH] = str(max(1, int(max_batch)))
    else:
        os.environ.pop(ENV_SWITCH, None)
        os.environ.pop(ENV_MAX_BATCH, None)


# ── Call-site helper ────────────────────────────────────────────────────

async def gather_or_sequential(factories: Sequence[Callable[[], Awaitable[Any]]]) -> List[Any]:
    """Run zero-argument coroutine factories; results in input order.

    Switch off: one after another, in order - exactly the legacy control
    flow. Switch on: concurrently, so their LLM calls are pending together
    and the batcher can merge them.
    """
    if llm_batch_enabled():
        return list(await asyncio.gather(*(f() for f in factories)))
    return [await f() for f in factories]


# ── The batcher ─────────────────────────────────────────────────────────

class MicroBatcher:
    """Merge concurrently pending requests into batched calls.

    ``run_group(requests) -> results`` is a SYNCHRONOUS callable that serves
    one compatible group and returns one result per request, in order. It
    may raise: every request of that group then sees the exception (so the
    caller's own retry logic applies per request), other groups are
    unaffected.

    ``group_key(request)`` says which requests may share a call (same
    sampling parameters, same modality, ...). Groups run in order of first
    arrival; a group larger than ``max_batch`` is split in arrival order.
    """

    def __init__(
        self,
        run_group: Callable[[list], list],
        group_key: Callable[[Any], Hashable] = lambda _request: 0,
        max_batch: Optional[int] = None,
    ) -> None:
        self._run_group = run_group
        self._group_key = group_key
        self._max_batch = max_batch
        self._pending: list = []            # [(request, future)]
        self._task: Optional[asyncio.Task] = None
        #: sizes of every call made, in order - for tests and the run summary
        self.call_sizes: List[int] = []

    async def submit(self, request: Any) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append((request, future))
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        while self._pending:
            await self._settle()
            batch, self._pending = self._pending, []
            self._flush(batch)

    async def _settle(self) -> None:
        last, stable = -1, 0
        for _ in range(_SETTLE_MAX_YIELDS):
            await asyncio.sleep(0)
            n = len(self._pending)
            stable = stable + 1 if n == last else 0
            last = n
            if stable >= _SETTLE_STABLE:
                return

    def _flush(self, batch: list) -> None:
        groups: dict = {}
        for item in batch:
            groups.setdefault(self._group_key(item[0]), []).append(item)
        cap = self._max_batch or max_batch_from_env()
        for items in groups.values():
            for start in range(0, len(items), cap):
                chunk = items[start:start + cap]
                self.call_sizes.append(len(chunk))
                try:
                    results = self._run_group([request for request, _ in chunk])
                    if len(results) != len(chunk):
                        raise RuntimeError(
                            f"run_group returned {len(results)} results for "
                            f"{len(chunk)} requests"
                        )
                except Exception as exc:  # KeyboardInterrupt/SystemExit propagate
                    for _, future in chunk:
                        if not future.done():
                            future.set_exception(exc)
                    continue
                for (_, future), result in zip(chunk, results):
                    if not future.done():
                        future.set_result(result)

    def summary(self) -> str:
        n = sum(self.call_sizes)
        if not self.call_sizes:
            return "no batched calls"
        return (f"{n} requests in {len(self.call_sizes)} generate() calls "
                f"(mean batch {n / len(self.call_sizes):.2f}, max {max(self.call_sizes)})")
