"""Local transformers-based ChatCompletionClient for autogen_core.

Loads a HuggingFace model once and serves it in-process — no HTTP server needed.
Implements the autogen_core ChatCompletionClient protocol.

Supports both text-only models (Qwen3.5 etc.) and vision-language models
(Qwen2.5-VL etc.). Vision models are auto-detected and images are processed
through the model's processor pipeline.
"""

import base64
import io
import json
import logging
import os
import re
import threading
from typing import Any, List, Optional, Sequence

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

# Singleton: model + tokenizer/processor loaded once, shared across all client instances
_shared_model = None
_shared_tokenizer = None  # AutoTokenizer (text-only) or AutoProcessor (VL)
_shared_model_name = ""
_shared_is_vision = False
_warned_no_vision = False  # warn once, not every call
_load_lock = threading.Lock()


def _autogen_image_to_pil(img) -> PIL.Image.Image:
    """Convert an autogen_core Image to a PIL Image."""
    b64 = AutogenImage.to_base64(img)
    return PIL.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _load_shared_model(model_path: str, dtype: str = "bfloat16"):
    """Load model and tokenizer/processor once into GPU memory."""
    global _shared_model, _shared_tokenizer, _shared_model_name, _shared_is_vision

    if _shared_model is not None:
        return

    with _load_lock:
        # Double-check after acquiring lock
        if _shared_model is not None:
            return

        logger.info(f"Loading local model: {model_path}")
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"

        # ── Detect vision model ──
        # Check model_type in config AND whether preprocessor_config.json exists
        # (unified VL models like Qwen3.5 may not have "vl" in their model_type
        #  but still ship a preprocessor for image handling)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "unknown").lower()
        has_vision_keyword = any(
            kw in model_type for kw in ("vl", "vision", "visual", "multimodal")
        )
        has_preprocessor = os.path.isfile(
            os.path.join(model_path, "preprocessor_config.json")
        )
        has_vision_config = hasattr(config, "vision_config")
        _shared_is_vision = has_vision_keyword or has_preprocessor or has_vision_config

        if _shared_is_vision:
            # VL model: use AutoProcessor (handles text + images).
            # For model class, try AutoModelForImageTextToText first (correct
            # for models with pipeline_tag=image-text-to-text like Qwen3.5),
            # then fall back to AutoModelForCausalLM.
            from transformers import AutoProcessor
            _shared_tokenizer = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            # Ensure pad_token_id is set — many VL processors (Qwen3-VL etc.)
            # ship without one, which breaks generate() and padding.
            _tok = getattr(_shared_tokenizer, "tokenizer", _shared_tokenizer)
            if getattr(_tok, "pad_token_id", None) is None:
                _tok.pad_token_id = _tok.eos_token_id
                _tok.pad_token = _tok.eos_token
                logger.info(f"Set pad_token_id={_tok.pad_token_id} (copied from eos_token_id)")
            try:
                from transformers import AutoModelForImageTextToText
                _VLModelClass = AutoModelForImageTextToText
                logger.info("Using AutoModelForImageTextToText for VL model")
            except ImportError:
                from transformers import AutoModelForCausalLM
                _VLModelClass = AutoModelForCausalLM
                logger.info("AutoModelForImageTextToText not available, using AutoModelForCausalLM")
            _shared_model = _VLModelClass.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info(f"Loaded VISION model: {model_path} (type={model_type}, class={_VLModelClass.__name__})")
        else:
            # Text-only model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            _shared_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            _shared_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.warning(
                "=" * 70 + "\n"
                f"  LOCAL MODEL IS TEXT-ONLY ({model_type}).\n"
                f"  All image inputs will be DROPPED. The agent will NOT see game frames.\n"
                f"  For vision, use a VL model (Qwen3.5, Qwen2.5-VL, etc.).\n"
                "=" * 70
            )

        _shared_model.eval()
        _shared_model_name = model_path.rstrip("/").split("/")[-1]

        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Model loaded: {_shared_model_name}, vision={_shared_is_vision}, GPU memory: {mem_gb:.2f} GB")


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
        # Allow model_path from env var
        self._model_path = model_path or os.environ.get("LLM_MODEL_PATH", "")
        self._dtype = dtype
        self._response_format = response_format
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        if self._model_path:
            _load_shared_model(self._model_path, self._dtype)

    def _messages_to_dicts(
        self, messages: Sequence[LLMMessage]
    ) -> tuple[list[dict], list[PIL.Image.Image]]:
        """Convert autogen messages to HF chat format.

        Returns (chat_messages, images_list).
        For VL models, images are kept inline in the message content list and
        also collected into images_list for the processor.
        For text-only models, images are dropped with a warning note.
        """
        global _warned_no_vision
        result = []
        images: list[PIL.Image.Image] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    result.append({"role": "user", "content": msg.content})
                elif isinstance(msg.content, list):
                    if _shared_is_vision:
                        # VL model: build multimodal content list
                        content_parts = []
                        for part in msg.content:
                            if isinstance(part, str):
                                content_parts.append({"type": "text", "text": part})
                            else:
                                # autogen Image object → PIL
                                pil_img = _autogen_image_to_pil(part)
                                images.append(pil_img)
                                content_parts.append({"type": "image", "image": pil_img})
                        result.append({"role": "user", "content": content_parts})
                    else:
                        # Text-only: drop images
                        text_parts = []
                        image_count = 0
                        for part in msg.content:
                            if isinstance(part, str):
                                text_parts.append(part)
                            else:
                                image_count += 1
                        text = " ".join(text_parts)
                        if image_count > 0:
                            if not _warned_no_vision:
                                logger.warning(
                                    "LocalModelClient: %d image(s) dropped — "
                                    "text-only model cannot process images.",
                                    image_count,
                                )
                                _warned_no_vision = True
                            text += (
                                "\n\n[NOTE: A game screenshot was attached but this model "
                                "cannot process images. Base your response on the text "
                                "context above (task, beliefs, skills, communications). "
                                "Do NOT hallucinate visual details you cannot see.]"
                            )
                        result.append({"role": "user", "content": text})
                else:
                    result.append({"role": "user", "content": str(msg.content)})
            else:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                result.append({"role": "assistant", "content": content})

        return result, images

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs,
    ) -> CreateResult:
        global _shared_model, _shared_tokenizer

        if _shared_model is None:
            raise RuntimeError(
                "Local model not loaded. Set LLM_MODEL_PATH env var or pass model_path."
            )

        chat_messages, images = self._messages_to_dicts(messages)

        # If structured output requested, add JSON instruction to system prompt
        if self._response_format is not None:
            schema_name = getattr(self._response_format, "__name__", "JSON")
            json_instruction = (
                f"\n\nYou MUST respond with valid JSON matching the {schema_name} schema. "
                f"Output ONLY the JSON object, no markdown or extra text."
            )
            if chat_messages and chat_messages[0]["role"] == "system":
                # System content may be a string or a list (VL format)
                if isinstance(chat_messages[0]["content"], str):
                    chat_messages[0]["content"] += json_instruction
                else:
                    chat_messages[0]["content"].append(
                        {"type": "text", "text": json_instruction}
                    )
            else:
                chat_messages.insert(0, {
                    "role": "system",
                    "content": json_instruction.strip(),
                })

        enable_thinking = os.environ.get("LLM_ENABLE_THINKING", "0") == "1"
        max_new = self._max_tokens * 4 if enable_thinking else self._max_tokens

        if _shared_is_vision and images:
            # ── VL model path: use processor to handle text + images ──
            template_kwargs = dict(
                tokenize=False, add_generation_prompt=True,
            )
            try:
                template_kwargs["enable_thinking"] = enable_thinking
                text_prompt = _shared_tokenizer.apply_chat_template(
                    chat_messages, **template_kwargs
                )
            except TypeError:
                del template_kwargs["enable_thinking"]
                text_prompt = _shared_tokenizer.apply_chat_template(
                    chat_messages, **template_kwargs
                )
            inputs = _shared_tokenizer(
                text=[text_prompt],
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(_shared_model.device)
            input_len = inputs["input_ids"].shape[1]

            _tok = getattr(_shared_tokenizer, "tokenizer", _shared_tokenizer)
            with torch.no_grad():
                outputs = _shared_model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=max(self._temperature, 0.01),
                    top_p=self._top_p,
                    do_sample=self._temperature > 0.01,
                    pad_token_id=_tok.pad_token_id or _tok.eos_token_id,
                )

            new_tokens = outputs[0][input_len:]
            text = _shared_tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            # ── Text-only path (or VL model with no images) ──
            # When using AutoProcessor (VL model) for text-only calls,
            # apply_chat_template may return a string instead of tensors.
            # Handle both cases: get text first, then tokenize if needed.
            template_kwargs = dict(
                add_generation_prompt=True,
            )
            try:
                template_kwargs["enable_thinking"] = enable_thinking
                text_or_tokens = _shared_tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, **template_kwargs
                )
            except TypeError:
                del template_kwargs["enable_thinking"]
                text_or_tokens = _shared_tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, **template_kwargs
                )
            # Now tokenize the text prompt
            if _shared_is_vision:
                # VL processor: use its __call__ with text only (no images)
                tokenized = _shared_tokenizer(
                    text=[text_or_tokens], return_tensors="pt", padding=True
                )
                input_ids = tokenized.input_ids.to(_shared_model.device)
            else:
                # Text-only tokenizer: tokenize directly
                input_ids = _shared_tokenizer(
                    text_or_tokens, return_tensors="pt"
                ).input_ids.to(_shared_model.device)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                outputs = _shared_model.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    temperature=max(self._temperature, 0.01),
                    top_p=self._top_p,
                    do_sample=self._temperature > 0.01,
                    pad_token_id=_tok.pad_token_id or _tok.eos_token_id,
                )

            new_tokens = outputs[0][input_len:]
            text = _shared_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Debug: show raw model output (remove after confirming it works)
        logger.info(f"[LocalModel RAW output] ({len(new_tokens)} tokens): {text[:500]}")

        # Strip thinking tags (Qwen3.5 etc.)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if not text.startswith('{') and '{' in text:
            text = text[text.rfind('{'):]

        logger.info(f"[LocalModel PARSED output]: {text[:300]}")

        completion_tokens = len(new_tokens)
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