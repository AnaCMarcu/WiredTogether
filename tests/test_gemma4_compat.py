"""Compatibility shims that let the stack load Gemma 4 instead of Qwen3.5.

Gemma 4 differs from the Qwen checkpoints the code was written against in three
ways that each have a shim under test here:

- it is multimodal at every size, so the vision path engages by default and the
  ``LLM_VISION_MODE`` override is the only way back to text-only parity;
- it loads through a newer auto-class (``AutoModelForMultimodalLM``) and, on a
  transformers new enough to know it, ``from_pretrained(dtype=...)`` rather
  than the deprecated ``torch_dtype=``;
- multimodal configs nest the text width under ``text_config``, which is the
  width the RL value head must match.

``autogen_core`` is not installed on the dev machine (and conftest deliberately
does not stub it globally), so a minimal stand-in is installed here — module
local, so no other test's imports change.
"""

import importlib
import sys
import types

import pytest


# ── autogen_core stand-in (only when the real package is absent) ────────────
def _install_autogen_stub() -> None:
    if importlib.util.find_spec("autogen_core") is not None:  # pragma: no cover
        return

    core = types.ModuleType("autogen_core")

    class Image:
        @staticmethod
        def to_base64(img):
            return img

    core.CancellationToken = type("CancellationToken", (), {})
    core.Image = Image

    models = types.ModuleType("autogen_core.models")

    class _Msg:
        def __init__(self, content, source="test"):
            self.content = content
            self.source = source

    class RequestUsage:
        def __init__(self, prompt_tokens=0, completion_tokens=0):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class CreateResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    models.ChatCompletionClient = type("ChatCompletionClient", (), {})
    models.CreateResult = CreateResult
    models.LLMMessage = _Msg
    models.ModelInfo = dict
    models.RequestUsage = RequestUsage
    models.SystemMessage = type("SystemMessage", (_Msg,), {})
    models.UserMessage = type("UserMessage", (_Msg,), {})

    core.models = models
    sys.modules["autogen_core"] = core
    sys.modules["autogen_core.models"] = models


_install_autogen_stub()

from mindforge.agent_modules import local_model_client as lmc  # noqa: E402
from rl_layer.rl_layer import _text_hidden_size  # noqa: E402


# ── LLM_VISION_MODE override ────────────────────────────────────────────────

class _MultimodalConfig:
    """Stand-in for a Gemma 4 config: multimodal, with a nested text config."""
    model_type = "gemma4"

    class text_config:  # noqa: N801 - mirrors the transformers attribute name
        hidden_size = 2048

    class vision_config:  # noqa: N801
        hidden_size = 1152


class _TextOnlyConfig:
    model_type = "qwen3"
    hidden_size = 3584


def test_vision_mode_auto_detects_multimodal_config(monkeypatch, tmp_path):
    """Default (auto): vision_config on the config is enough to pick the VL path."""
    monkeypatch.delenv("LLM_VISION_MODE", raising=False)
    assert lmc._detect_is_vision(str(tmp_path), _MultimodalConfig) is True
    assert lmc._detect_is_vision(str(tmp_path), _TextOnlyConfig) is False


def test_vision_mode_text_forces_text_only(monkeypatch, tmp_path):
    """LLM_VISION_MODE=text is the parity switch: frames dropped, as under Qwen."""
    monkeypatch.setenv("LLM_VISION_MODE", "text")
    assert lmc._detect_is_vision(str(tmp_path), _MultimodalConfig) is False


def test_vision_mode_vision_forces_vl_path(monkeypatch, tmp_path):
    """LLM_VISION_MODE=vision skips the sniff entirely, even for a text-only config."""
    monkeypatch.setenv("LLM_VISION_MODE", "vision")
    assert lmc._detect_is_vision(str(tmp_path), _TextOnlyConfig) is True


@pytest.mark.parametrize("filename", [
    "preprocessor_config.json",   # Qwen-VL generation
    "processor_config.json",      # what Gemma 4 actually ships
])
def test_processor_file_alone_selects_vision(monkeypatch, tmp_path, filename):
    """A staged checkpoint is sniffed by file too, not just by config attributes."""
    monkeypatch.delenv("LLM_VISION_MODE", raising=False)
    (tmp_path / filename).write_text("{}", encoding="utf-8")
    assert lmc._detect_is_vision(str(tmp_path), _TextOnlyConfig) is True


# ── auto-class preference order ─────────────────────────────────────────────

def test_vision_model_class_prefers_multimodal(monkeypatch):
    """Gemma 4's documented class wins when the installed transformers has it."""
    import transformers

    sentinel = type("AutoModelForMultimodalLM", (), {})
    monkeypatch.setattr(transformers, "AutoModelForMultimodalLM", sentinel, raising=False)
    assert lmc._vision_model_class() is sentinel


def test_vision_model_class_falls_back_to_image_text_to_text(monkeypatch):
    """Without the newest class, the Gemma-3/Qwen-VL-era class is used."""
    import transformers

    monkeypatch.delattr(transformers, "AutoModelForMultimodalLM", raising=False)
    sentinel = type("AutoModelForImageTextToText", (), {})
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", sentinel, raising=False)
    assert lmc._vision_model_class() is sentinel


# ── dtype kwarg rename ──────────────────────────────────────────────────────

class _Loader:
    """from_pretrained that records the kwargs it was handed.

    Deliberately permissive (``**kwargs``) — that is exactly how the real
    ``from_pretrained`` behaves, and why the wrong dtype kwarg is swallowed
    silently instead of raising.
    """

    def __init__(self, loaded_dtype=None):
        self.seen = None
        self.loaded_dtype = loaded_dtype

    def from_pretrained(self, path, **kwargs):
        self.seen = kwargs
        return types.SimpleNamespace(
            dtype=self.loaded_dtype,
            to=lambda dt: types.SimpleNamespace(dtype=dt, cast=True),
        )


@pytest.mark.parametrize("version,expected", [
    ("4.40.2", "torch_dtype"),
    ("4.55.4", "torch_dtype"),
    ("4.56.0", "dtype"),
    ("5.1.0", "dtype"),
])
def test_dtype_kwarg_name_follows_transformers_version(monkeypatch, version, expected):
    """4.56 renamed torch_dtype → dtype; the wrong name is silently ignored."""
    import transformers

    monkeypatch.setattr(transformers, "__version__", version)
    assert lmc._dtype_kwarg_name() == expected


def test_from_pretrained_passes_the_dtype_under_the_expected_name(monkeypatch):
    import torch
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.62.0")
    loader = _Loader(loaded_dtype=torch.bfloat16)
    lmc._from_pretrained(loader, "p", torch.bfloat16, device_map="auto")
    assert loader.seen == {"dtype": torch.bfloat16, "device_map": "auto"}


def test_from_pretrained_casts_when_the_kwarg_was_ignored(monkeypatch):
    """A model that came back fp32 anyway gets cast, not run in the wrong dtype."""
    import torch
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.62.0")
    loader = _Loader(loaded_dtype=torch.float32)
    model = lmc._from_pretrained(loader, "p", torch.bfloat16)
    assert model.dtype is torch.bfloat16
    assert model.cast is True


# ── system-role merge ───────────────────────────────────────────────────────

def test_merge_system_into_first_user_string_content():
    merged = lmc._merge_system_into_first_user([
        {"role": "system", "content": "You are agent 0."},
        {"role": "user", "content": "What now?"},
        {"role": "assistant", "content": "Mine."},
    ])
    assert merged == [
        {"role": "user", "content": "You are agent 0.\n\nWhat now?"},
        {"role": "assistant", "content": "Mine."},
    ]


def test_merge_system_into_first_user_keeps_frame_after_system_text():
    """VL content stays a parts list, with the system text as the leading part."""
    frame = {"type": "image", "image": object()}
    merged = lmc._merge_system_into_first_user([
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": [{"type": "text", "text": "Look."}, frame]},
    ])
    assert merged == [{"role": "user", "content": [
        {"type": "text", "text": "Rules."},
        {"type": "text", "text": "Look."},
        frame,
    ]}]


def test_merge_system_into_first_user_without_system_is_identity():
    msgs = [{"role": "user", "content": "hi"}]
    assert lmc._merge_system_into_first_user(msgs) == msgs


# ── apply_chat_template fallbacks ───────────────────────────────────────────

class _Tokenizer:
    """Chat template with configurable strictness, recording what it rendered."""

    def __init__(self, *, accepts_thinking=True, accepts_system=True):
        self.accepts_thinking = accepts_thinking
        self.accepts_system = accepts_system
        self.rendered = None

    def apply_chat_template(self, msgs, *, tokenize, add_generation_prompt, **kwargs):
        if "enable_thinking" in kwargs and not self.accepts_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        if not self.accepts_system and any(m["role"] == "system" for m in msgs):
            raise ValueError("System role not supported")
        self.rendered = msgs
        return "PROMPT"


def _render(tok, msgs):
    return lmc._apply_chat_template(tok, msgs, tokenize=False, enable_thinking=False)


def test_apply_chat_template_drops_enable_thinking_when_unsupported():
    tok = _Tokenizer(accepts_thinking=False)
    assert _render(tok, [{"role": "user", "content": "hi"}]) == "PROMPT"


def test_apply_chat_template_retries_with_merged_system():
    """A template that rejects role=system must not take the run down."""
    tok = _Tokenizer(accepts_system=False)
    msgs = [
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": "Go."},
    ]
    assert _render(tok, msgs) == "PROMPT"
    assert tok.rendered == [{"role": "user", "content": "Rules.\n\nGo."}]


def test_apply_chat_template_reraises_when_no_system_to_merge():
    """Without a system turn the merge cannot help — surface the real error."""
    tok = _Tokenizer(accepts_system=False)
    tok.apply_chat_template = lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        _render(tok, [{"role": "user", "content": "hi"}])


# ── RL value-head width ─────────────────────────────────────────────────────

def test_text_hidden_size_prefers_nested_text_config():
    """Multimodal config: the value head must match the TEXT tower's width."""
    assert _text_hidden_size(_MultimodalConfig) == 2048


def test_text_hidden_size_falls_back_to_flat_attribute():
    assert _text_hidden_size(_TextOnlyConfig) == 3584


# ── LoRA target resolution for multimodal towers (RL arms) ──────────────────
#
# The gemma4_smoke run (2026-08-19, exp03_mappo) showed the real failure
# shape: Gemma 4's VISION and AUDIO towers implement q/v_proj as
# Gemma4ClippableLinear — a wrapper holding an inner nn.Linear — while the
# text tower's projections are plain nn.Linear. Suffix-matching "q_proj"
# therefore reached into towers PEFT cannot wrap (56 rejections: 32 vision,
# 24 audio, 0 text) AND towers the text-only RL forward would never train.
# resolve_lora_targets confines targeting to the text tower and returns
# explicit module paths.

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rl_layer.peft_compat import resolve_lora_targets  # noqa: E402
from rl_layer.rl_layer import LORA_TARGET_MODULES  # noqa: E402

peft = pytest.importorskip("peft")


class _ClippableLinear(nn.Module):
    """Mirrors Gemma4ClippableLinear as the smoke-run repr showed it:
    a wrapper around an inner nn.Linear, with no weight of its own."""

    def __init__(self, h):
        super().__init__()
        self.linear = nn.Linear(h, h, bias=False)
        self.clip = 4.0

    def forward(self, x):
        return self.linear(x).clamp(-self.clip, self.clip)


def _attn(h, proj_cls):
    block = nn.Module()
    block.q_proj = proj_cls(h)
    block.k_proj = proj_cls(h)
    block.v_proj = proj_cls(h)
    return block


class _TinyGemma4(nn.Module):
    """Multimodal layout from the smoke log: text tower under
    model.language_model with plain Linears, vision/audio towers with
    ClippableLinear wrappers."""

    def __init__(self, h=8, layers=2):
        super().__init__()
        plain = lambda h: nn.Linear(h, h)  # noqa: E731
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [_attn(h, plain) for _ in range(layers)])
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.layers = nn.ModuleList(
            [_attn(h, _ClippableLinear) for _ in range(layers)])
        self.model.audio_tower = nn.Module()
        self.model.audio_tower.layers = nn.ModuleList(
            [_attn(h, _ClippableLinear) for _ in range(layers)])

    def forward(self, x):
        # Text-only path, like RLLayer feeding input_ids: the vision and
        # audio towers are never entered.
        for block in self.model.language_model.layers:
            x = block.q_proj(x) + block.v_proj(x)
        return x


class _TinyQwen(nn.Module):
    """Text-only layout: no towers, projections are plain nn.Linear."""

    def __init__(self, h=8, layers=2):
        super().__init__()
        plain = lambda h: nn.Linear(h, h)  # noqa: E731
        self.layers = nn.ModuleList([_attn(h, plain) for _ in range(layers)])

    def forward(self, x):
        for block in self.layers:
            x = block.q_proj(x) + block.v_proj(x)
        return x


def _lora_cfg(targets):
    return peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=list(targets), bias="none",
    )


def test_suffix_targeting_reproduces_the_smoke_run_crash():
    """The pre-fix behavior: bare q/v_proj suffixes reach the vision tower's
    ClippableLinear and PEFT rejects it — exp03_mappo's exact failure."""
    with pytest.raises(ValueError, match="not supported"):
        peft.get_peft_model(_TinyGemma4(), _lora_cfg(LORA_TARGET_MODULES))


def test_resolution_confines_targets_to_the_text_tower():
    targets = resolve_lora_targets(_TinyGemma4(), LORA_TARGET_MODULES)
    assert len(targets) == 4  # q+v over 2 text layers
    assert all(t.startswith("model.language_model.") for t in targets)
    assert not any("vision" in t or "audio" in t for t in targets)
    assert not any(t.endswith("k_proj") for t in targets)


def test_resolution_on_text_only_matches_old_suffix_behavior():
    """Qwen path unchanged: every q/v_proj, none skipped, exact paths."""
    targets = resolve_lora_targets(_TinyQwen(), LORA_TARGET_MODULES)
    assert sorted(targets) == [
        "layers.0.q_proj", "layers.0.v_proj",
        "layers.1.q_proj", "layers.1.v_proj",
    ]


def test_wrapped_text_projection_resolves_to_inner_linear():
    """If the TEXT tower ever ships ClippableLinear too, LoRA lands on the
    inner nn.Linear — inside the clip, per the checkpoint's own arithmetic."""
    model = _TinyGemma4()
    h = 8
    model.model.language_model.layers[0].q_proj = _ClippableLinear(h)
    targets = resolve_lora_targets(model, LORA_TARGET_MODULES)
    assert "model.language_model.layers.0.q_proj.linear" in targets
    assert len(targets) == 4


def test_ambiguous_wrapper_is_skipped_not_guessed():
    model = _TinyGemma4()
    amb = nn.Module()
    amb.a = nn.Linear(8, 8)
    amb.b = nn.Linear(8, 8)
    model.model.language_model.layers[0].q_proj = amb
    targets = resolve_lora_targets(model, LORA_TARGET_MODULES)
    assert len(targets) == 3  # the ambiguous q_proj dropped, rest intact


def test_nothing_resolved_raises_instead_of_training_nothing():
    with pytest.raises(RuntimeError, match="no LoRA target"):
        resolve_lora_targets(_TinyQwen(), ("nonexistent_proj",))


def test_peft_bootstrap_and_per_agent_adapters_with_resolved_targets():
    """The full RLLayer flow on the multimodal fake: bootstrap wrap,
    add_adapter per role, adapter-tag optimizer filtering, grad flow —
    and the vision/audio towers left untouched."""
    torch.manual_seed(0)
    model = _TinyGemma4()
    targets = resolve_lora_targets(model, LORA_TARGET_MODULES)

    peft_model = peft.get_peft_model(
        model, _lora_cfg(targets), adapter_name="__bootstrap__")
    wrapped = peft_model.base_model.model
    text0 = wrapped.model.language_model.layers[0]
    assert type(text0.q_proj).__name__ == "Linear"      # peft.tuners.lora
    assert hasattr(text0.q_proj, "lora_A")
    assert type(text0.k_proj) is nn.Linear              # untouched
    vis0 = wrapped.model.vision_tower.layers[0]
    assert type(vis0.q_proj) is _ClippableLinear        # untouched

    # _ensure_adapter path: one adapter per role on the shared base.
    peft_model.add_adapter("gatherer", _lora_cfg(targets))
    peft_model.set_adapter("gatherer")

    out = peft_model(torch.randn(2, 8))
    out.sum().backward()

    # RLLayer's optimizer selects params whose name contains ".<adapter>.".
    tagged = [
        (n, p) for n, p in peft_model.named_parameters()
        if ".gatherer." in n and p.requires_grad
    ]
    assert len(tagged) == 8  # lora_A + lora_B for q+v over 2 text layers
    assert all(p.grad is not None for _, p in tagged)
    # No adapter parameters anywhere in the vision/audio towers.
    assert not any(
        ("vision_tower" in n or "audio_tower" in n)
        for n, _ in peft_model.named_parameters() if "lora" in n
    )
