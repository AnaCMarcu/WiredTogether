"""Local transformers-based ChatCompletionClient for autogen_core.

Loads a HuggingFace model once and serves it in-process — no HTTP server needed.
Implements the autogen_core ChatCompletionClient protocol.

IMPORTANT: Text-only models (Qwen3.5-9B etc.) cannot process images.
When an image is attached to a message, it is DROPPED and replaced with a
warning note. For vision, use a VL model via the HTTP API backend instead.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch
from autogen_core import CancellationToken
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

# Singleton: model + tokenizer loaded once, shared across all client instances
_shared_model = None
_shared_tokenizer = None
_shared_model_name = ""
_warned_no_vision = False  # warn once, not every call


def _load_shared_model(model_path: str, dtype: str = "bfloat16"):
    """Load model and tokenizer once into GPU memory."""
    global _shared_model, _shared_tokenizer, _shared_model_name

    if _shared_model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading local model: {model_path}")
    torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"

    _shared_tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    _shared_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _shared_model.eval()
    _shared_model_name = model_path.rstrip("/").split("/")[-1]

    # ── Check if model supports vision ──
    model_type = getattr(_shared_model.config, "model_type", "unknown")
    is_vision = any(
        kw in model_type.lower()
        for kw in ("vl", "vision", "visual", "multimodal")
    )
    if not is_vision:
        logger.warning(
            "=" * 70 + "\n"
            f"  LOCAL MODEL '{_shared_model_name}' IS TEXT-ONLY.\n"
            f"  All image inputs will be DROPPED. The agent will NOT see game frames.\n"
            f"  For vision, use a VL model (Qwen2.5-VL, etc.) via HTTP API.\n"
            "=" * 70
        )

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"Model loaded: {_shared_model_name}, GPU memory: {mem_gb:.2f} GB")


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

    def _messages_to_dicts(self, messages: Sequence[LLMMessage]) -> list[dict]:
        """Convert autogen messages to HF chat format.

        For text-only models, images are replaced with a warning note so the
        LLM knows it was supposed to see a frame but can't.
        """
        global _warned_no_vision
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                # UserMessage.content can be a list (text + images) or string
                if isinstance(msg.content, str):
                    text = msg.content
                elif isinstance(msg.content, list):
                    text_parts = []
                    image_count = 0
                    for part in msg.content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        else:
                            # This is an image object — text-only model can't use it
                            image_count += 1
                    text = " ".join(text_parts)
                    if image_count > 0:
                        if not _warned_no_vision:
                            logger.warning(
                                "LocalModelClient: %d image(s) dropped from message — "
                                "text-only model cannot process images. "
                                "Agent perception beliefs will be unreliable.",
                                image_count,
                            )
                            _warned_no_vision = True
                        # ── FIX: tell the LLM it can't see the image ──
                        text += (
                            "\n\n[NOTE: A game screenshot was attached but this model "
                            "cannot process images. Base your response on the text "
                            "context above (task, beliefs, skills, communications). "
                            "Do NOT hallucinate visual details you cannot see.]"
                        )
                else:
                    text = str(msg.content)
                result.append({"role": "user", "content": text})
            else:
                # AssistantMessage or other
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                result.append({"role": "assistant", "content": content})
        return result

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

        chat_messages = self._messages_to_dicts(messages)

        # If structured output requested, add JSON instruction to system prompt
        if self._response_format is not None:
            schema_name = getattr(self._response_format, "__name__", "JSON")
            # Check if there's already a system message
            if chat_messages and chat_messages[0]["role"] == "system":
                chat_messages[0]["content"] += (
                    f"\n\nYou MUST respond with valid JSON matching the {schema_name} schema. "
                    f"Output ONLY the JSON object, no markdown or extra text."
                )
            else:
                chat_messages.insert(0, {
                    "role": "system",
                    "content": (
                        f"You MUST respond with valid JSON matching the {schema_name} schema. "
                        f"Output ONLY the JSON object, no markdown or extra text."
                    ),
                })

        enable_thinking = os.environ.get("LLM_ENABLE_THINKING", "0") == "1"
        template_kwargs = dict(
            add_generation_prompt=True, return_tensors="pt",
        )
        # Qwen3.5 supports enable_thinking toggle
        try:
            template_kwargs["enable_thinking"] = enable_thinking
            tokenized = _shared_tokenizer.apply_chat_template(
                chat_messages, **template_kwargs
            )
        except TypeError:
            # Older tokenizers don't support enable_thinking
            del template_kwargs["enable_thinking"]
            tokenized = _shared_tokenizer.apply_chat_template(
                chat_messages, **template_kwargs
            )
        # apply_chat_template may return a BatchEncoding or a plain tensor
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(_shared_model.device)
        else:
            input_ids = tokenized.to(_shared_model.device)

        with torch.no_grad():
            outputs = _shared_model.generate(
                input_ids,
                max_new_tokens=self._max_tokens * 4 if enable_thinking else self._max_tokens,
                temperature=max(self._temperature, 0.01),
                top_p=self._top_p,
                do_sample=self._temperature > 0.01,
                pad_token_id=(
                    _shared_tokenizer.pad_token_id
                    or _shared_tokenizer.eos_token_id
                ),
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        text = _shared_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Qwen3.5 emits <think>...</think> reasoning before the answer — strip it
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Handle untagged thinking blocks (e.g. "Thinking Process: ...")
        if not text.startswith('{') and '{' in text:
            text = text[text.rfind('{'):]

        prompt_tokens = input_ids.shape[1]
        completion_tokens = len(new_tokens)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + completion_tokens,
        )

        return CreateResult(
            content=text,
            finish_reason="stop",
            usage=RequestUsage(
                prompt_tokens=prompt_tokens,
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
            vision=False,
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