"""The batched action scorer may ask the model for the last K_max+1 logits only.

``RLLayer._score_actions`` (batched path: one ``[prompt + candidate]`` row per
candidate, right-padded) reads logits only at the positions that predict
candidate tokens. It passes ``logits_to_keep=K_max+1`` so a model that
supports it skips the rest of the C x L x V logits (Qwen3.5, which cannot use
the packed scorer, has a 248k vocab). These tests require the same log-probs,
pooled state and gradients as computing every position, and the fallback for
models that reject the argument.

A tiny random Llama stands in: the wrappers below give it the
``logits_to_keep`` semantics of current transformers (keep the LAST k
positions), so the tests run on old transformers releases too.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from rl_layer.config import RLConfig
from rl_layer.rl_layer import RLLayer

PROMPT = [5, 17, 33, 2, 9, 41, 12, 7, 60, 3, 28, 44, 19, 8, 51, 6, 30, 22, 11, 47, 14, 39, 25]
# Lengths 1-3 so rows are padded differently; shared first tokens as in
# "MoveForward" / "MoveBackward".
CANDIDATES = {"A": [70], "B": [71, 72], "C": [71, 73], "D": [74, 75, 76],
              "E": [74, 75, 77], "F": [78]}
K_MAX = max(len(v) for v in CANDIDATES.values())


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids
        self.attention_mask = torch.ones_like(ids)

    def to(self, device):
        return self


class _Tokenizer:
    """Prompt text is space-separated token ids."""
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
        return _Enc(torch.tensor([[int(t) for t in text.split()]]))


class _Base(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.seen = []

    def set_adapter(self, name):
        pass

    @property
    def config(self):
        return self.inner.config

    def get_input_embeddings(self):
        return self.inner.get_input_embeddings()


class _KeepsLast(_Base):
    """Supports logits_to_keep as transformers does: the last k positions."""

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache,
                logits_to_keep=0):
        self.seen.append(logits_to_keep)
        out = self.inner(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=output_hidden_states, use_cache=use_cache)
        if logits_to_keep:
            out.logits = out.logits[:, -logits_to_keep:, :]
        return out


class _NoKeep(_Base):
    """A model whose forward has no logits_to_keep: every position comes back."""

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
        self.seen.append(None)
        return self.inner(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=output_hidden_states, use_cache=use_cache)


def _model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=256, pad_token_id=0)
    return LlamaForCausalLM(cfg).float().eval()


def _score(wrapped, with_grad):
    layer = object.__new__(RLLayer)
    layer.config = RLConfig(shared_prefix_scoring=False, rl_prompt_max_tokens=4096)
    layer.model = wrapped
    layer.tokenizer = _Tokenizer()
    layer._device = torch.device("cpu")
    layer._adapter_name = "agent"
    layer._candidate_actions = tuple(CANDIDATES)
    layer._candidate_token_ids = [torch.tensor(v) for v in CANDIDATES.values()]
    wrapped.inner.zero_grad(set_to_none=True)
    logp, pooled = layer._score_actions(" ".join(map(str, PROMPT)), tuple(CANDIDATES),
                                        with_grad=with_grad)
    grads = None
    if with_grad:
        weights = torch.linspace(-1.0, 1.0, len(CANDIDATES))
        ((logp * weights).sum() + 0.01 * pooled.sum()).backward()
        grads = {n: p.grad.detach().clone() for n, p in wrapped.inner.named_parameters()
                 if p.grad is not None}
    return logp.detach(), pooled.detach(), grads


def _assert_same(a, b):
    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a[1], b[1], rtol=1e-5, atol=1e-5)
    if a[2] is not None:
        assert a[2] and set(a[2]) == set(b[2])
        for name, g in b[2].items():
            scale = g.abs().max().item()
            err = (a[2][name] - g).abs().max().item()
            assert err <= 1e-5 * scale + 1e-9, f"{name}: max |diff| {err:.2e}, scale {scale:.2e}"


@pytest.mark.parametrize("with_grad", [True, False])
def test_kept_logits_match_all_positions(with_grad):
    model = _model()
    keep, full = _KeepsLast(model), _NoKeep(model)
    _assert_same(_score(keep, with_grad), _score(full, with_grad))
    assert keep.seen == [K_MAX + 1]            # asked for the window only
    assert full.seen == [None]                 # rejected at the call (TypeError), then the plain call


@pytest.mark.skipif("logits_to_keep" not in inspect.signature(LlamaForCausalLM.forward).parameters,
                    reason="this transformers release has no logits_to_keep")
def test_native_logits_to_keep_matches_all_positions():
    model = _model()

    class _Native(_Base):
        def forward(self, **kwargs):
            self.seen.append(kwargs.get("logits_to_keep"))
            return self.inner(**kwargs)

    native = _Native(model)
    _assert_same(_score(native, True), _score(_NoKeep(model), True))
    assert native.seen == [K_MAX + 1]
