"""Packed (shared-prefix) action scoring must equal the batched path.

``RLLayer._score_actions`` scores each candidate action as a continuation of
the prompt. The default path runs one ``[prompt + candidate]`` row per
candidate; ``shared_prefix_scoring`` packs the prompt once plus every
candidate's tokens into one sequence with a custom attention mask
(rl_layer/shared_prefix.py). These tests run the production method both ways
on a tiny random Gemma 4 model and require the same candidate log-probs,
pooled hidden state and parameter gradients. The model has sliding-window
layers with a window SHORTER than the prompt and KV-shared layers, so the
window and the sharing are actually exercised.

The Gemma 4 tests need a transformers release that ships Gemma 4 (the
cluster image does); with an older one they are skipped. The mask tests are
pure torch and always run.
"""

from __future__ import annotations

import pytest
import torch

from rl_layer.config import RLConfig
from rl_layer.rl_layer import RLLayer
from rl_layer.shared_prefix import packed_layout, packed_masks, packing_support

try:
    from transformers import (Gemma4Config, Gemma4ForCausalLM,
                              Gemma4ForConditionalGeneration, Gemma4TextConfig)
except Exception:  # transformers predates Gemma 4
    Gemma4ForCausalLM = None

needs_gemma4 = pytest.mark.skipif(Gemma4ForCausalLM is None,
                                  reason="transformers has no Gemma 4")

PROMPT = [5, 17, 33, 2, 9, 41, 12, 7, 60, 3, 28, 44, 19, 8, 51, 6, 30, 22, 11, 47, 14, 39, 25]
# Lengths 1-3, two candidates sharing a first token (as "MoveForward" /
# "MoveBackward" can), one sharing a two-token prefix.
CANDIDATES = {"A": [70], "B": [71, 72], "C": [71, 73], "D": [74, 75, 76],
              "E": [74, 75, 77], "F": [78]}


# ── mask construction (pure torch) ──────────────────────────────────────────
def test_layout_gives_candidates_their_own_row_positions():
    owner, positions, starts = packed_layout(4, [1, 2, 3])
    assert owner.tolist() == [-1, -1, -1, -1, 0, 1, 1, 2, 2, 2]
    assert positions.tolist() == [0, 1, 2, 3, 4, 4, 5, 4, 5, 6]
    assert starts == [4, 5, 7]


def test_masks_isolate_candidates_and_window_by_position():
    owner, positions, _ = packed_layout(4, [1, 2, 3])
    m = packed_masks(owner, positions, sliding_window=3)
    full, sliding = m["full_attention"], m["sliding_attention"]
    # Prompt rows: causal over the prompt, never a candidate token.
    for q in range(4):
        assert full[q].tolist() == [k <= q for k in range(4)] + [False] * 6
    # Candidate 2's last token (index 9): the whole prompt + its own 3 tokens.
    assert full[9].tolist() == [True] * 4 + [False] * 3 + [True] * 3
    # Candidate 1's second token never sees candidate 0 or candidate 2.
    assert full[6].tolist() == [True] * 4 + [False, True, True] + [False] * 3
    # Sliding window 3 measured on positions: index 9 has position 6, so it
    # sees positions 4-6 (its own tokens) and no prompt token (positions 0-3).
    assert sliding[9].tolist() == [False] * 7 + [True] * 3
    # Index 7 (position 4) sees prompt positions 2-3 and itself.
    assert sliding[7].tolist() == [False, False, True, True, False, False, False, True, False, False]


def test_models_without_attention_masks_are_not_packed():
    class Cfg:
        layer_types = ["linear_attention", "full_attention"]
        _attn_implementation = "sdpa"

    class M:
        config = Cfg()

    ok, why = packing_support(M())
    assert not ok and "linear_attention" in why


# ── production _score_actions, batched vs packed ────────────────────────────
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


class _PeftLike(torch.nn.Module):
    """What RLLayer needs from its PeftModel: set_adapter, config, forward."""

    def __init__(self, model):
        super().__init__()
        self.inner = model

    def set_adapter(self, name):
        pass

    @property
    def config(self):
        return self.inner.config

    def get_input_embeddings(self):
        return self.inner.get_input_embeddings()

    def forward(self, **kwargs):
        return self.inner(**kwargs)


def _layer(model, shared: bool) -> RLLayer:
    layer = object.__new__(RLLayer)
    layer.config = RLConfig(shared_prefix_scoring=shared, rl_prompt_max_tokens=4096)
    layer.model = _PeftLike(model)
    layer.tokenizer = _Tokenizer()
    layer._device = torch.device("cpu")
    layer._adapter_name = "agent"
    layer._candidate_actions = tuple(CANDIDATES)
    layer._candidate_token_ids = [torch.tensor(v) for v in CANDIDATES.values()]
    RLLayer._packing_checked = None
    return layer


def _text_config(attn: str):
    cfg = Gemma4TextConfig(
        vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        max_position_embeddings=256, sliding_window=7,
        layer_types=["sliding_attention", "sliding_attention", "full_attention",
                     "sliding_attention", "sliding_attention", "full_attention"],
        num_kv_shared_layers=2, vocab_size_per_layer_input=97,
        hidden_size_per_layer_input=8, final_logit_softcapping=30.0, pad_token_id=0,
    )
    cfg._attn_implementation = attn
    return cfg


def _score(model, shared: bool, with_grad: bool):
    layer = _layer(model, shared)
    model.zero_grad(set_to_none=True)
    prompt = " ".join(map(str, PROMPT))
    logp, pooled = layer._score_actions(prompt, tuple(CANDIDATES), with_grad=with_grad)
    grads = None
    if with_grad:
        weights = torch.linspace(-1.0, 1.0, len(CANDIDATES))
        ((logp * weights).sum() + 0.01 * pooled.sum()).backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
    return logp.detach(), pooled.detach(), grads


def _assert_same(model):
    model.eval()                                   # no dropout: both paths deterministic
    ref_logp, ref_pooled, ref_grads = _score(model, shared=False, with_grad=True)
    new_logp, new_pooled, new_grads = _score(model, shared=True, with_grad=True)
    assert RLLayer._packing_checked is True       # the packed path actually ran
    torch.testing.assert_close(new_logp, ref_logp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_pooled, ref_pooled, rtol=1e-5, atol=1e-5)
    assert ref_grads and set(new_grads) == set(ref_grads)
    # Relative to each parameter's own gradient scale: elementwise rtol fails
    # on entries near zero, where float32 summation order alone moves ~1e-7.
    for name, g in ref_grads.items():
        scale = g.abs().max().item()
        err = (new_grads[name] - g).abs().max().item()
        assert err <= 1e-5 * scale + 1e-9, f"{name}: max |diff| {err:.2e}, grad scale {scale:.2e}"
    # Sampling-time call (no grad) agrees too.
    nograd_logp, _, _ = _score(model, shared=True, with_grad=False)
    torch.testing.assert_close(nograd_logp, ref_logp, rtol=1e-5, atol=1e-5)


@needs_gemma4
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_matches_batched_gemma4_text(attn):
    torch.manual_seed(0)
    _assert_same(Gemma4ForCausalLM(_text_config(attn)).float())


@needs_gemma4
def test_packed_matches_batched_gemma4_multimodal_wrapper():
    # The RL base for gemma-4-E4B-it is loaded through the multimodal class.
    torch.manual_seed(0)
    cfg = Gemma4Config(text_config=_text_config("sdpa").to_dict())
    cfg._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    _assert_same(Gemma4ForConditionalGeneration(cfg).float())
