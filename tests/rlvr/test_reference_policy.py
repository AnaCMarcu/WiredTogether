"""Tests for ``rlvr.reference_policy``.

The torch / PEFT / transformers integration paths (``GRPOLanguageModel``,
``ReferencePolicy.compute_kl``, ``LLMPolicy.act`` against a real model)
require a GPU + base model and are HPC-only. These local tests cover only
the parts that can run with stubs:

    * ``default_prompt`` formatting
    * ``LLMPolicy`` action parsing + fallback against a stub language model

Anything that requires actual gradient flow / adapter switching is
covered by HPC integration runs, not pytest.
"""

from __future__ import annotations

import pytest

from rlvr.reference_policy import LLMPolicy, default_prompt
from rlvr.rollout_sampler import RolloutTensors


# ──── prompt template ──────────────────────────────────────────────────


def test_default_prompt_contains_chamber_and_hp():
    s = default_prompt(None, {"chamber": "ch3", "hp": 20})
    assert "ch3" in s
    assert "20" in s
    assert "JSON" in s
    assert "action" in s


def test_default_prompt_handles_missing_fields():
    s = default_prompt(None, {})
    assert "unknown" in s


# ──── LLMPolicy.act ─────────────────────────────────────────────────────


class _StubModel:
    """Mimics ``GRPOLanguageModel.generate`` enough for ``LLMPolicy`` tests."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None

    def generate(self, prompt: str):
        self.last_prompt = prompt
        # tokens / logprobs slots — None is fine for LLMPolicy's contract
        return self.response_text, None, None


def test_llm_policy_parses_valid_response():
    model = _StubModel('{"action": "dig", "communication_target": null, "thoughts": "x"}')
    policy = LLMPolicy(model, n_agents=3)
    action, tensors = policy.act(None, {"chamber": "ch3"})
    assert action["action"] == "dig"
    assert isinstance(tensors, RolloutTensors)
    assert tensors.prompt_text == model.last_prompt


def test_llm_policy_falls_back_on_parse_failure():
    model = _StubModel("this is not json at all")
    policy = LLMPolicy(model, n_agents=3)
    action, _ = policy.act(None, {})
    # Fallback is NOP — env step won't crash.
    assert action["action"] == "nop"


def test_llm_policy_falls_back_on_invalid_action_name():
    model = _StubModel('{"action": "fly", "communication_target": null, "thoughts": ""}')
    policy = LLMPolicy(model, n_agents=3)
    action, _ = policy.act(None, {})
    assert action["action"] == "nop"


def test_llm_policy_custom_prompt_template():
    seen = {}

    def template(obs, info):
        seen["info"] = info
        return f"custom_{info.get('chamber','?')}"

    model = _StubModel('{"action": "forward", "communication_target": null, "thoughts": ""}')
    policy = LLMPolicy(model, n_agents=3, prompt_template=template)
    _action, tensors = policy.act(None, {"chamber": "ch5"})
    assert seen["info"] == {"chamber": "ch5"}
    assert tensors.prompt_text == "custom_ch5"


def test_llm_policy_custom_fallback_action():
    model = _StubModel("garbage")
    policy = LLMPolicy(
        model, n_agents=3,
        fallback_action={"action": "jump", "communication_target": 0, "thoughts": "stuck"},
    )
    action, _ = policy.act(None, {})
    assert action["action"] == "jump"
    # Custom fallback is copied — caller can't mutate the policy's prototype
    action["action"] = "mutated"
    again, _ = policy.act(None, {})
    assert again["action"] == "jump"


# ──── GRPOLanguageModel — only importable here, can't load a model locally
# without torch. The actual unit tests for it live in an HPC notebook.


@pytest.mark.skipif(True, reason="Requires torch + a HF model — HPC-only")
def test_grpo_language_model_loads():  # pragma: no cover
    pass
