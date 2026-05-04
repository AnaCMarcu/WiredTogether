"""Local transformers-based ChatCompletionClient for autogen_core.

Loads a HuggingFace model once and serves it in-process — no HTTP server needed.
Implements the autogen_core ChatCompletionClient protocol.

Supports both text-only models (Qwen3.5 etc.) and vision-language models
(Qwen2.5-VL etc.). Vision models are auto-detected and images are processed
through the model's processor pipeline.
"""

import base64
import io
import logging
import os
import re
import threading
from typing import Any, List, Optional, Sequence, Tuple

import PIL.Image
import torch
from autogen_core import CancellationToken, Image as AutogenImage
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)


# ─── Shared singleton state (one model per process) ────────────────────

_shared_model = None
_shared_tokenizer = None       # AutoTokenizer (text-only) or AutoProcessor (VL)
_shared_model_name = ""
_shared_is_vision = False
_warned_no_vision = False      # warn once, not every call
_load_lock = threading.Lock()


# ─── Stateless helpers ─────────────────────────────────────────────────

def _autogen_image_to_pil(img) -> PIL.Image.Image:
    b64 = AutogenImage.to_base64(img)
    return PIL.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _inner_tokenizer(tok):
    """Return the underlying tokenizer (handles both AutoProcessor and plain)."""
    return getattr(tok, "tokenizer", tok)


def _detect_is_vision(model_path: str, config) -> bool:
    """Vision model = vision keyword in model_type, OR preprocessor file, OR vision_config."""
    model_type = getattr(config, "model_type", "unknown").lower()
    has_keyword = any(kw in model_type for kw in ("vl", "vision", "visual", "multimodal"))
    has_preprocessor = os.path.isfile(
        os.path.join(model_path, "preprocessor_config.json")
    )
    has_vision_config = hasattr(config, "vision_config")
    return has_keyword or has_preprocessor or has_vision_config


def _ensure_pad_token(processor):
    """VL processors often miss pad_token_id; copy from eos."""
    tok = _inner_tokenizer(processor)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token = tok.eos_token
        logger.info("Set pad_token_id=%s (copied from eos_token_id)", tok.pad_token_id)


def _vision_model_class():
    """Prefer AutoModelForImageTextToText; fall back to AutoModelForCausalLM."""
    try:
        from transformers import AutoModelForImageTextToText
        logger.info("Using AutoModelForImageTextToText for VL model")
        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM
        logger.info("AutoModelForImageTextToText not available, using AutoModelForCausalLM")
        return AutoModelForCausalLM


def _apply_chat_template(tokenizer, chat_messages, *, tokenize, enable_thinking):
    """apply_chat_template with graceful fallback for tokenizers that don't accept enable_thinking."""
    base_kwargs = dict(tokenize=tokenize, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            chat_messages, **base_kwargs, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(chat_messages, **base_kwargs)


def _strip_thinking_and_extract_json(text: str) -> str:
    """Remove <think>...</think> blocks; if text has a JSON object, keep from its opening brace."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text.startswith('{') and '{' in text:
        text = text[text.rfind('{'):]
    return text


# ─── Model loading ─────────────────────────────────────────────────────

def _load_shared_model(model_path: str, dtype: str = "bfloat16"):
    """Load model + tokenizer/processor once into GPU memory."""
    global _shared_model, _shared_tokenizer, _shared_model_name, _shared_is_vision

    if _shared_model is not None:
        return

    with _load_lock:
        if _shared_model is not None:
            return  # double-check inside lock

        logger.info("Loading local model: %s", model_path)
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _shared_is_vision = _detect_is_vision(model_path, config)

        if _shared_is_vision:
            _shared_tokenizer, _shared_model = _load_vision_model(model_path, torch_dtype)
            logger.info("Loaded VISION model: %s (type=%s)", model_path,
                        getattr(config, "model_type", "unknown"))
        else:
            _shared_tokenizer, _shared_model = _load_text_only_model(model_path, torch_dtype)
            _warn_text_only(getattr(config, "model_type", "unknown"))

        _shared_model.eval()
        _shared_model_name = model_path.rstrip("/").split("/")[-1]
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info("Model loaded: %s, vision=%s, GPU memory: %.2f GB",
                    _shared_model_name, _shared_is_vision, mem_gb)


def _load_vision_model(model_path, torch_dtype):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _ensure_pad_token(processor)
    model = _vision_model_class().from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return processor, model


def _load_text_only_model(model_path, torch_dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def _warn_text_only(model_type: str):
    logger.warning(
        "=" * 70 + "\n"
        f"  LOCAL MODEL IS TEXT-ONLY ({model_type}).\n"
        f"  All image inputs will be DROPPED. The agent will NOT see game frames.\n"
        f"  For vision, use a VL model (Qwen3.5, Qwen2.5-VL, etc.).\n"
        + "=" * 70
    )


# ─── Message conversion ────────────────────────────────────────────────

def _convert_messages(
    messages: Sequence[LLMMessage],
) -> Tuple[List[dict], List[PIL.Image.Image]]:
    """autogen messages → HF chat dicts. Returns (chat_messages, collected_images)."""
    result = []
    images: List[PIL.Image.Image] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            result.append(_convert_user_message(msg, images))
        else:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append({"role": "assistant", "content": content})

    return result, images


def _convert_user_message(msg: UserMessage, images: List[PIL.Image.Image]) -> dict:
    content = msg.content
    if isinstance(content, str):
        return {"role": "user", "content": content}
    if isinstance(content, list):
        if _shared_is_vision:
            return {"role": "user", "content": _build_vl_content(content, images)}
        return {"role": "user", "content": _build_text_only_content(content)}
    return {"role": "user", "content": str(content)}


def _build_vl_content(parts, images: List[PIL.Image.Image]) -> List[dict]:
    """VL multimodal message: collect images into images_list and reference them inline."""
    out = []
    for part in parts:
        if isinstance(part, str):
            out.append({"type": "text", "text": part})
        else:
            pil_img = _autogen_image_to_pil(part)
            images.append(pil_img)
            out.append({"type": "image", "image": pil_img})
    return out


def _build_text_only_content(parts) -> str:
    """Drop images; once-only warning + a note appended to text so the LLM doesn't hallucinate."""
    global _warned_no_vision
    text_parts, image_count = [], 0
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        else:
            image_count += 1
    text = " ".join(text_parts)
    if image_count > 0:
        if not _warned_no_vision:
            logger.warning(
                "LocalModelClient: %d image(s) dropped — text-only model cannot process images.",
                image_count,
            )
            _warned_no_vision = True
        text += (
            "\n\n[NOTE: A game screenshot was attached but this model "
            "cannot process images. Base your response on the text "
            "context above (task, beliefs, skills, communications). "
            "Do NOT hallucinate visual details you cannot see.]"
        )
    return text


# ─── JSON schema injection ─────────────────────────────────────────────

def _inject_json_instruction(chat_messages: list, response_format) -> list:
    """Append a 'reply with JSON' instruction PLUS the actual schema fields to
    the system message. We don't have constrained decoding for the local
    model, so this is the only way the LLM knows which fields are REQUIRED
    (e.g. `communication_target`) — without the schema it silently drops
    fields it considers optional, breaking downstream targeted-comm logic."""
    if response_format is None:
        return chat_messages
    schema_name = getattr(response_format, "__name__", "JSON")

    # Try Pydantic v2 then v1; fall back to a name-only hint.
    schema_block = ""
    try:
        if hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
        elif hasattr(response_format, "schema"):
            schema = response_format.schema()
        else:
            schema = None
        if schema:
            props = schema.get("properties", {}) or {}
            required = schema.get("required", []) or []
            field_lines = []
            for fname, fdef in props.items():
                ftype = fdef.get("type", "string")
                req = "REQUIRED" if fname in required else "optional"
                field_lines.append(f'  "{fname}": <{ftype}>   ({req})')
            schema_block = (
                "\n\nSchema fields:\n" + "\n".join(field_lines) +
                "\nALL fields marked REQUIRED must appear in your JSON output."
            )
    except Exception:
        pass

    instruction = (
        f"\n\nYou MUST respond with valid JSON matching the {schema_name} schema. "
        f"Output ONLY the JSON object, no markdown, no commentary, no surrounding text."
        f"{schema_block}"
    )
    if chat_messages and chat_messages[0]["role"] == "system":
        first = chat_messages[0]
        if isinstance(first["content"], str):
            first["content"] += instruction
        else:
            first["content"].append({"type": "text", "text": instruction})
    else:
        chat_messages.insert(0, {"role": "system", "content": instruction.strip()})
    return chat_messages


# ─── Main client ───────────────────────────────────────────────────────

class LocalModelClient(ChatCompletionClient):
    """ChatCompletionClient backed by a local HuggingFace transformers model."""

    def __init__(
        self,
        model_path: str | None = None,
        dtype: str = "bfloat16",
        response_format: Any = None,
        model_info: dict | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        **kwargs,
    ):
        self._model_path = model_path or os.environ.get("LLM_MODEL_PATH", "")
        self._dtype = dtype
        self._response_format = response_format
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        if self._model_path:
            _load_shared_model(self._model_path, self._dtype)

    # ─── public API ────────────────────────────────────────────────

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs,
    ) -> CreateResult:
        if _shared_model is None:
            raise RuntimeError(
                "Local model not loaded. Set LLM_MODEL_PATH env var or pass model_path."
            )

        chat_messages, images = _convert_messages(messages)
        chat_messages = _inject_json_instruction(chat_messages, self._response_format)

        enable_thinking = os.environ.get("LLM_ENABLE_THINKING", "0") == "1"
        max_new = self._max_tokens * 4 if enable_thinking else self._max_tokens

        text, input_len, completion_tokens = self._generate(
            chat_messages, images, max_new, enable_thinking
        )
        text = _strip_thinking_and_extract_json(text)
        logger.info("[LocalModel PARSED output]: %s", text[:300])

        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + input_len,
            completion_tokens=self._total_usage.completion_tokens + completion_tokens,
        )
        return CreateResult(
            content=text,
            finish_reason="stop",
            usage=RequestUsage(
                prompt_tokens=input_len,
                completion_tokens=completion_tokens,
            ),
            cached=False,
        )

    # ─── generation ────────────────────────────────────────────────

    def _generate(self, chat_messages, images, max_new, enable_thinking):
        """Tokenize → generate → decode. Returns (text, input_len, completion_tokens)."""
        if _shared_is_vision and images:
            inputs, input_len = self._tokenize_vision(chat_messages, images, enable_thinking)
        else:
            inputs, input_len = self._tokenize_text(chat_messages, enable_thinking)

        tok = _inner_tokenizer(_shared_tokenizer)
        with torch.no_grad():
            outputs = _shared_model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=max(self._temperature, 0.01),
                top_p=self._top_p,
                do_sample=self._temperature > 0.01,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new_tokens = outputs[0][input_len:]
        text = _shared_tokenizer.decode(new_tokens, skip_special_tokens=True)
        logger.info("[LocalModel RAW output] (%d tokens): %s", len(new_tokens), text[:500])
        return text, input_len, len(new_tokens)

    @staticmethod
    def _tokenize_vision(chat_messages, images, enable_thinking):
        """VL path: processor handles text + images in one call."""
        text_prompt = _apply_chat_template(
            _shared_tokenizer, chat_messages,
            tokenize=False, enable_thinking=enable_thinking,
        )
        inputs = _shared_tokenizer(
            text=[text_prompt],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(_shared_model.device)
        return inputs, inputs["input_ids"].shape[1]

    @staticmethod
    def _tokenize_text(chat_messages, enable_thinking):
        """Text-only path. Also handles a VL model invoked without images."""
        text_prompt = _apply_chat_template(
            _shared_tokenizer, chat_messages,
            tokenize=False, enable_thinking=enable_thinking,
        )
        if _shared_is_vision:
            tokenized = _shared_tokenizer(
                text=[text_prompt], return_tensors="pt", padding=True
            )
            input_ids = tokenized.input_ids.to(_shared_model.device)
        else:
            input_ids = _shared_tokenizer(
                text_prompt, return_tensors="pt"
            ).input_ids.to(_shared_model.device)
        return {"input_ids": input_ids}, input_ids.shape[1]

    # ─── boilerplate ───────────────────────────────────────────────

    async def create_stream(self, messages, *, cancellation_token=None, **kwargs):
        raise NotImplementedError("Streaming not supported for local model client")

    def actual_usage(self) -> RequestUsage:
        return self._total_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    @property
    def capabilities(self) -> ModelInfo:
        return ModelInfo(
            vision=_shared_is_vision,
            function_calling=False,
            json_output=True,
            family="unknown",
            structured_output=False,
        )

    @property
    def model_info(self) -> ModelInfo:
        return self.capabilities

    def count_tokens(self, messages, **kwargs) -> int:
        return 0

    def remaining_tokens(self, messages, **kwargs) -> int:
        return self._max_tokens

    async def close(self) -> None:
        pass
