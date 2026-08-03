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


def test_preprocessor_file_alone_selects_vision(monkeypatch, tmp_path):
    """A staged checkpoint is sniffed by file too, not just by config attributes."""
    monkeypatch.delenv("LLM_VISION_MODE", raising=False)
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
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
