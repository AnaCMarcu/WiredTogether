"""ChatCompletionClient that sends LocalModelClient's exact requests to a vLLM server.

Selected with ``LLM_BACKEND=vllm`` (see ``util.create_model_client``); the
launcher starts the server inside the job (``LLM_SERVER=vllm`` in
hpc/slurm/experiments/_common.sh) and sets ``LLM_BASE_URL``.

WHY NOT THE EXISTING HTTP PATH
  ``OpenAIChatCompletionClient`` (the API path in util.py) sends the pydantic
  response_format as a JSON schema, which vLLM turns into constrained
  decoding, and it skips the schema instruction LocalModelClient injects into
  the prompt. It also leaves Gemma's thinking mode at the template default and
  writes no ``[LocalModel usage]`` line. A run through it would not be
  comparable with the in-process runs.

WHAT STAYS IDENTICAL TO LocalModelClient
  * the prompt: messages are converted by the same ``_convert_messages``
    (frames inline, or dropped with the same note on a text-only model) and
    get the same ``_inject_json_instruction``; the chat template is the
    model's own, applied by the server, with the same ``enable_thinking``
    switch (``LLM_ENABLE_THINKING``) and the same system-role fallback;
  * sampling: temperature 0.7, top_p 0.9, the same token cap (x4 when
    thinking is on); any other default (top_k, ...) comes from the model's
    generation_config.json in both backends;
  * post-processing: ``_strip_thinking_and_extract_json``;
  * logging: the ``[LocalModel usage]``, RAW and PARSED lines that
    analysis/compute_flops.py and the log parsers read;
  * the comm-budget ledger: the model's tokenizer is loaded (CPU only) into
    local_model_client, so ``count_text_tokens`` stays exact.

WHAT DIFFERS
  Different kernels and batching mean different random draws from the same
  sampling distribution, the same trade-off --llm-batch already makes. Calls
  are real coroutines, so callers that gather them (--llm-batch) reach the
  server together and are batched there; sequential callers still gain the
  faster decode and the server's prefix cache.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Optional, Sequence

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
)

from mindforge.agent_modules import local_model_client as lmc

logger = logging.getLogger(__name__)

#: The name the launcher gives the model (vllm serve --served-model-name).
DEFAULT_SERVED_NAME = "wt"

_tokenizer_loaded = False


def _load_tokenizer_only(model_path: str) -> None:
    """Load the model's tokenizer/processor on the CPU into local_model_client.

    Sets the two globals the shared helpers read: ``_shared_is_vision`` (how
    ``_convert_messages`` treats frames) and ``_shared_tokenizer`` (what
    ``count_text_tokens`` counts with). No weights, no GPU.
    """
    global _tokenizer_loaded
    if _tokenizer_loaded:
        return
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    is_vision = lmc._detect_is_vision(model_path, config)
    if is_vision:
        tok = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        lmc._warn_text_only(getattr(config, "model_type", "unknown"))
    lmc._shared_is_vision = is_vision
    lmc._shared_tokenizer = tok
    lmc._shared_model_name = model_path.rstrip("/").split("/")[-1]
    _tokenizer_loaded = True
    logger.info("RemoteModelClient: tokenizer for %s loaded (vision=%s)", model_path, is_vision)


def _to_openai_messages(chat_messages: list) -> list:
    """HF chat dicts (PIL images inline) -> OpenAI chat messages (data-URL images).

    Frames are sent as PNG, which is lossless, so the server decodes exactly
    the pixels the in-process processor would have received.
    """
    out = []
    for m in chat_messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts = []
        for p in content:
            if p.get("type") == "image":
                buf = io.BytesIO()
                p["image"].save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{b64}"}})
            else:
                parts.append({"type": "text", "text": p.get("text", "")})
        out.append({"role": m["role"], "content": parts})
    return out


def _is_system_role_rejection(exc: Exception) -> bool:
    """A 400 from the server's chat template refusing the system role."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status == 400 and "system" in text and ("role" in text or "template" in text)


class RemoteModelClient(ChatCompletionClient):
    """LocalModelClient's requests, served by an OpenAI-compatible vLLM server."""

    def __init__(
        self,
        model_path: str | None = None,
        base_url: str | None = None,
        served_name: str | None = None,
        response_format: Any = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        timeout: float = 900.0,
        extract_json: bool = True,
        **kwargs,
    ):
        from openai import AsyncOpenAI

        self._model_path = model_path or os.environ.get("LLM_MODEL_PATH", "")
        self._base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        if not self._model_path or not self._base_url:
            raise RuntimeError(
                "RemoteModelClient needs LLM_MODEL_PATH (for the tokenizer) and "
                "LLM_BASE_URL (the vLLM server)."
            )
        self._served_name = served_name or os.environ.get("LLM_SERVED_NAME", DEFAULT_SERVED_NAME)
        self._response_format = response_format
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        # False = raw-text client (no brace slicing), as in LocalModelClient.
        self._extract_json = extract_json
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        # The server is local to the job; its key (if any) only keeps other
        # users on a shared node out.
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=os.environ.get("LLM_SERVER_KEY", "none"),
            timeout=timeout,
            max_retries=3,
        )
        _load_tokenizer_only(self._model_path)

    # ─── public API ────────────────────────────────────────────────

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs,
    ) -> CreateResult:
        chat_messages, _images = lmc._convert_messages(messages)
        chat_messages = lmc._inject_json_instruction(chat_messages, self._response_format)

        enable_thinking = os.environ.get("LLM_ENABLE_THINKING", "0") == "1"
        max_new = self._max_tokens * 4 if enable_thinking else self._max_tokens

        try:
            resp = await self._request(chat_messages, max_new, enable_thinking)
        except Exception as exc:
            # Same fallback as LocalModelClient._apply_chat_template.
            if not (_is_system_role_rejection(exc)
                    and any(m["role"] == "system" for m in chat_messages)):
                raise
            logger.warning("Chat template rejected the system role (%s) — merging it "
                           "into the first user turn", exc)
            resp = await self._request(lmc._merge_system_into_first_user(chat_messages),
                                       max_new, enable_thinking)

        raw = resp.choices[0].message.content or ""
        input_len = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0
        logger.info("[LocalModel RAW output] (%d tokens): %s", completion_tokens, raw[:500])
        # Same leading fields as the in-process line (compute_flops.py parses them).
        logger.info("[LocalModel usage] prompt_tokens=%d completion_tokens=%d backend=vllm",
                    input_len, completion_tokens)
        text = lmc._strip_thinking_and_extract_json(raw, self._extract_json)
        logger.info("[LocalModel PARSED output]: %s", text[:300])

        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + input_len,
            completion_tokens=self._total_usage.completion_tokens + completion_tokens,
        )
        return CreateResult(
            content=text,
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=input_len, completion_tokens=completion_tokens),
            cached=False,
        )

    async def _request(self, chat_messages: list, max_new: int, enable_thinking: bool):
        return await self._client.chat.completions.create(
            model=self._served_name,
            messages=_to_openai_messages(chat_messages),
            max_tokens=max_new,
            temperature=self._temperature,
            top_p=self._top_p,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )

    # ─── boilerplate (mirrors LocalModelClient) ────────────────────

    async def create_stream(self, messages, *, cancellation_token=None, **kwargs):
        raise NotImplementedError("Streaming not supported for the remote model client")

    def actual_usage(self) -> RequestUsage:
        return self._total_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    @property
    def capabilities(self) -> ModelInfo:
        return ModelInfo(
            vision=lmc._shared_is_vision,
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
        await self._client.close()
