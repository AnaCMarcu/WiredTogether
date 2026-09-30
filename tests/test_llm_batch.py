"""--llm-batch: cross-call micro-batching of the shared in-process LLM.

Three layers under test:

- ``agent_modules/llm_batch.py`` — the pure-asyncio MicroBatcher and the
  ``gather_or_sequential`` call-site helper (no torch, no model);
- ``local_model_client.py`` — the batched generate path (left padding,
  per-row EOS trimming, per-row usage, OOM splitting) against a fake
  tokenizer + model, and ``create()`` routing on/off the switch;
- ``belief_system.py`` — the per-partner updates issued together under the
  switch and strictly in order without it.

The contract that matters most: with the switch OFF nothing changes — no
batcher is built and the legacy control flow runs call by call, in order.
"""

import asyncio

import pytest
import torch

import social_stubs  # noqa: F401  (autogen/chromadb stand-ins, src on sys.path)

from mindforge.agent_modules import llm_batch
from mindforge.agent_modules import local_model_client as lmc
from mindforge.agent_modules.llm_batch import MicroBatcher, gather_or_sequential


@pytest.fixture(autouse=True)
def _clean_switch(monkeypatch):
    """Every test starts with the switch off and no process-wide batcher."""
    monkeypatch.delenv(llm_batch.ENV_SWITCH, raising=False)
    monkeypatch.delenv(llm_batch.ENV_MAX_BATCH, raising=False)
    monkeypatch.setattr(lmc, "_batcher", None)


def _echo_batcher(**kwargs):
    calls = []

    def run_group(requests):
        calls.append(list(requests))
        return [f"out:{r}" for r in requests]

    return MicroBatcher(run_group, **kwargs), calls


# ── master switch ───────────────────────────────────────────────────────────

def test_switch_defaults_off_and_round_trips():
    assert not llm_batch.llm_batch_enabled()
    llm_batch.set_env_switch(True, 8)
    assert llm_batch.llm_batch_enabled()
    assert llm_batch.max_batch_from_env() == 8
    llm_batch.set_env_switch(False)
    assert not llm_batch.llm_batch_enabled()
    assert llm_batch.max_batch_from_env() == llm_batch.DEFAULT_MAX_BATCH


# ── MicroBatcher ────────────────────────────────────────────────────────────

def test_concurrent_submits_share_one_call_and_route_results():
    batcher, calls = _echo_batcher()

    async def main():
        return await asyncio.gather(*(batcher.submit(i) for i in range(5)))

    assert asyncio.run(main()) == [f"out:{i}" for i in range(5)]
    assert calls == [[0, 1, 2, 3, 4]]


def test_nested_fan_out_lands_in_the_same_batch():
    """Each 'agent' issues three calls at once (the belief gather) — all 3N batch."""
    batcher, calls = _echo_batcher()

    async def agent(a):
        return await asyncio.gather(*(batcher.submit((a, k)) for k in range(3)))

    async def main():
        return await asyncio.gather(*(agent(a) for a in range(4)))

    out = asyncio.run(main())
    assert out[2] == ["out:(2, 0)", "out:(2, 1)", "out:(2, 2)"]
    assert [len(c) for c in calls] == [12]


def test_agents_out_of_phase_batch_round_by_round():
    """A does three calls in a row, B and C one each: rounds of 3, 1, 1."""
    batcher, calls = _echo_batcher()

    async def agent(name, depth):
        return [await batcher.submit(f"{name}{d}") for d in range(depth)]

    async def main():
        return await asyncio.gather(agent("A", 3), agent("B", 1), agent("C", 1))

    out = asyncio.run(main())
    assert out[0] == ["out:A0", "out:A1", "out:A2"]
    assert calls == [["A0", "B0", "C0"], ["A1"], ["A2"]]


def test_sequential_submits_are_not_delayed_into_batches():
    batcher, calls = _echo_batcher()

    async def main():
        return [await batcher.submit(i) for i in range(3)]

    assert asyncio.run(main()) == ["out:0", "out:1", "out:2"]
    assert calls == [[0], [1], [2]]


def test_incompatible_requests_run_in_separate_groups_in_arrival_order():
    batcher, calls = _echo_batcher(group_key=lambda r: r[0])

    async def main():
        reqs = [("img", 1), ("txt", 2), ("img", 3), ("txt", 4)]
        return await asyncio.gather(*(batcher.submit(r) for r in reqs))

    out = asyncio.run(main())
    assert out == ["out:('img', 1)", "out:('txt', 2)", "out:('img', 3)", "out:('txt', 4)"]
    assert calls == [[("img", 1), ("img", 3)], [("txt", 2), ("txt", 4)]]


def test_max_batch_splits_in_arrival_order():
    batcher, calls = _echo_batcher(max_batch=2)

    async def main():
        return await asyncio.gather(*(batcher.submit(i) for i in range(5)))

    asyncio.run(main())
    assert calls == [[0, 1], [2, 3], [4]]
    assert batcher.call_sizes == [2, 2, 1]
    assert "5 requests in 3 generate() calls" in batcher.summary()


def test_max_batch_comes_from_the_env_switch():
    llm_batch.set_env_switch(True, 3)
    batcher, calls = _echo_batcher()

    async def main():
        await asyncio.gather(*(batcher.submit(i) for i in range(4)))

    asyncio.run(main())
    assert [len(c) for c in calls] == [3, 1]


def test_group_failure_reaches_every_caller_of_that_group_only():
    def run_group(requests):
        if requests[0][0] == "bad":
            raise ValueError("boom")
        return [r[1] for r in requests]

    batcher = MicroBatcher(run_group, group_key=lambda r: r[0])

    async def main():
        return await asyncio.gather(
            batcher.submit(("bad", 1)), batcher.submit(("ok", 2)),
            batcher.submit(("bad", 3)), return_exceptions=True,
        )

    first, second, third = asyncio.run(main())
    assert isinstance(first, ValueError) and isinstance(third, ValueError)
    assert second == 2


def test_batcher_survives_a_failure_and_a_new_event_loop():
    state = {"fail": True}

    def run_group(requests):
        if state["fail"]:
            raise RuntimeError("first call fails")
        return requests

    batcher = MicroBatcher(run_group)
    with pytest.raises(RuntimeError):
        asyncio.run(batcher.submit("a"))
    state["fail"] = False
    assert asyncio.run(batcher.submit("b")) == "b"     # fresh loop, same batcher


def test_wrong_result_count_is_an_error_not_a_misroute():
    batcher = MicroBatcher(lambda requests: requests[:-1])

    async def main():
        return await asyncio.gather(batcher.submit(1), batcher.submit(2),
                                    return_exceptions=True)

    assert all(isinstance(r, RuntimeError) for r in asyncio.run(main()))


# ── gather_or_sequential ────────────────────────────────────────────────────

def _traced_jobs(trace):
    async def job(name):
        trace.append(f"start {name}")
        await asyncio.sleep(0)
        trace.append(f"end {name}")
        return name

    return [lambda n=n: job(n) for n in ("a", "b", "c")]


def test_switch_off_runs_strictly_one_after_another():
    trace = []
    assert asyncio.run(gather_or_sequential(_traced_jobs(trace))) == ["a", "b", "c"]
    assert trace == ["start a", "end a", "start b", "end b", "start c", "end c"]


def test_switch_on_runs_concurrently_and_keeps_result_order():
    llm_batch.set_env_switch(True)
    trace = []
    assert asyncio.run(gather_or_sequential(_traced_jobs(trace))) == ["a", "b", "c"]
    assert trace[:3] == ["start a", "start b", "start c"]


# ── local_model_client: batched generate against fakes ──────────────────────

PAD, EOS, END_OF_TURN = 0, 1, 106


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """One token per whitespace word (id = 1000 + word length); pads per ``padding_side``."""
    pad_token_id = PAD
    eos_token_id = EOS
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return " ".join(m["content"] for m in msgs)

    def __call__(self, text=None, padding=False, return_tensors=None, **kwargs):
        prompts = [text] if isinstance(text, str) else list(text)
        rows = [[1000 + len(w) for w in p.split()] for p in prompts]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            fill = [PAD] * (width - len(r))
            ones, zeros = [1] * len(r), [0] * (width - len(r))
            if self.padding_side == "left":
                ids.append(fill + r); mask.append(zeros + ones)
            else:
                ids.append(r + fill); mask.append(ones + zeros)
        return _FakeBatch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in tokens if int(t) not in (PAD, EOS, END_OF_TURN))


class _FakeModel:
    """Row i 'generates' i+1 tokens (value 7), then an end token, then padding."""
    device = "cpu"

    class generation_config:  # noqa: N801 - mirrors the transformers attribute
        eos_token_id = [EOS, END_OF_TURN]

    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask, **kwargs})
        n = input_ids.shape[0]
        new = torch.full((n, n + 1), PAD)
        for i in range(n):
            new[i, : i + 1] = 7
            new[i, i + 1] = END_OF_TURN if i % 2 else EOS
        return torch.cat([input_ids, new], dim=1)


@pytest.fixture
def fake_model(monkeypatch):
    model, tok = _FakeModel(), _FakeTokenizer()
    monkeypatch.setattr(lmc, "_shared_model", model)
    monkeypatch.setattr(lmc, "_shared_tokenizer", tok)
    monkeypatch.setattr(lmc, "_shared_is_vision", False)
    return model, tok


def _req(text, temperature=0.7):
    return lmc._GenRequest([{"role": "user", "content": text}], [], 64, False, temperature, 0.9)


def test_generate_batch_pads_left_and_reports_each_rows_own_usage(fake_model):
    model, tok = fake_model
    out = lmc._generate_batch([_req("a b c d e"), _req("a b"), _req("a b c")])

    assert tok.padding_side == "left"
    call = model.calls[0]
    assert call["input_ids"].shape == (3, 5)
    assert call["input_ids"][1].tolist()[:3] == [PAD, PAD, PAD]      # short row padded on the left
    assert call["attention_mask"].sum(dim=1).tolist() == [5, 2, 3]
    assert call["max_new_tokens"] == 64 and call["do_sample"] is True
    assert call["temperature"] == 0.7 and call["top_p"] == 0.9

    # (text, prompt_tokens, completion_tokens, batch_size, secs): the row's
    # own prompt length, and its output cut after ITS end token (inclusive).
    assert [(t, p, c, b) for t, p, c, b, _ in out] == [
        ("7", 5, 2, 3), ("7 7", 2, 3, 3), ("7 7 7", 3, 4, 3),
    ]


def test_trim_at_eos_keeps_an_unterminated_row_whole():
    row = torch.tensor([7, 7, 7])
    assert lmc._trim_at_eos(row, {EOS}).tolist() == [7, 7, 7]
    assert lmc._trim_at_eos(torch.tensor([7, EOS, PAD, PAD]), {EOS}).tolist() == [7, EOS]


def test_group_key_separates_sampling_settings(fake_model):
    assert lmc._group_key(_req("x")) == lmc._group_key(_req("y"))
    assert lmc._group_key(_req("x")) != lmc._group_key(_req("x", temperature=0.0))


def test_oom_halves_the_batch_and_keeps_the_order(monkeypatch):
    sizes = []

    def fake_generate_batch(reqs):
        sizes.append(len(reqs))
        if len(reqs) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [r.chat_messages for r in reqs]

    monkeypatch.setattr(lmc, "_generate_batch", fake_generate_batch)
    reqs = [lmc._GenRequest(i, [], 8, False, 0.7, 0.9) for i in range(5)]
    assert lmc._run_group(reqs) == [0, 1, 2, 3, 4]
    assert sizes == [5, 2, 3, 1, 2]


def test_non_oom_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(lmc, "_generate_batch",
                        lambda reqs: (_ for _ in ()).throw(ValueError("bad prompt")))
    with pytest.raises(ValueError):
        lmc._run_group([lmc._GenRequest(0, [], 8, False, 0.7, 0.9)] * 3)


# ── LocalModelClient.create routing ─────────────────────────────────────────

def _user(text):
    from autogen_core.models import UserMessage
    return UserMessage(content=text, source="user")


def test_create_switch_off_takes_the_legacy_path(fake_model, monkeypatch):
    seen = []

    def legacy_generate(self, chat_messages, images, max_new, enable_thinking):
        seen.append(chat_messages[0]["content"])
        return '{"ok": 1}', 11, 4

    monkeypatch.setattr(lmc.LocalModelClient, "_generate", legacy_generate)
    client = lmc.LocalModelClient(model_path="")

    async def main():
        return await asyncio.gather(*(client.create([_user(f"p{i}")]) for i in range(3)))

    results = asyncio.run(main())
    assert seen == ["p0", "p1", "p2"]                 # one call each, in order
    assert lmc._batcher is None                       # no batcher was ever built
    assert fake_model[0].calls == []
    assert [r.usage.prompt_tokens for r in results] == [11, 11, 11]


def test_create_switch_on_batches_concurrent_calls(fake_model, monkeypatch):
    llm_batch.set_env_switch(True)
    monkeypatch.setattr(lmc.LocalModelClient, "_generate",
                        lambda *a, **k: pytest.fail("legacy path used under --llm-batch"))
    model, _ = fake_model
    client = lmc.LocalModelClient(model_path="")

    async def main():
        prompts = ["a b c d", "a", "a b"]
        return await asyncio.gather(*(client.create([_user(p)]) for p in prompts))

    results = asyncio.run(main())
    assert len(model.calls) == 1 and model.calls[0]["input_ids"].shape[0] == 3
    assert [r.usage.prompt_tokens for r in results] == [4, 1, 2]
    assert [r.usage.completion_tokens for r in results] == [2, 3, 4]
    assert [r.content for r in results] == ["7", "7 7", "7 7 7"]
    assert client.total_usage().prompt_tokens == 7
    assert lmc._batcher.call_sizes == [3]


# ── belief system: per-partner updates ──────────────────────────────────────

def _belief_system(monkeypatch, trace):
    from mindforge.agent_modules import belief_system as bs

    async def fake_llm_call(client, **kwargs):
        trace.append(f"start {kwargs['convo']}")
        await asyncio.sleep(0)
        trace.append(f"end {kwargs['convo']}")
        return {"beliefs": f"belief about {kwargs['convo']} (was {kwargs['previous_partner_belief']!r})"}

    monkeypatch.setattr(bs, "llm_call", fake_llm_call)
    system = bs.BeliefSystem(number_of_agents=4, belief_model_client=object())
    system.partner_beliefs = {0: "old0", 1: "old1", 2: "old2"}
    return system


def test_partner_beliefs_switch_off_update_one_by_one_in_order(monkeypatch):
    trace = []
    system = _belief_system(monkeypatch, trace)
    out = asyncio.run(system.update_partner_beliefs(["c0", "c1"], None))
    assert trace == ["start c0", "end c0", "start c1", "end c1"]
    assert out[0] == "belief about c0 (was 'old0')"
    assert out[1] == "belief about c1 (was 'old1')"
    assert out[2] == "old2"                           # no conversation -> untouched


def test_partner_beliefs_switch_on_are_issued_together(monkeypatch):
    llm_batch.set_env_switch(True)
    trace = []
    system = _belief_system(monkeypatch, trace)
    out = asyncio.run(system.update_partner_beliefs(["c0", "c1", "c2"], None))
    assert trace[:3] == ["start c0", "start c1", "start c2"]
    assert [out[i] for i in range(3)] == [
        "belief about c0 (was 'old0')", "belief about c1 (was 'old1')",
        "belief about c2 (was 'old2')",
    ]


def test_partner_beliefs_without_conversations_make_no_call(monkeypatch):
    trace = []
    system = _belief_system(monkeypatch, trace)
    assert asyncio.run(system.update_partner_beliefs(None, None)) == {0: "old0", 1: "old1", 2: "old2"}
    assert trace == []
