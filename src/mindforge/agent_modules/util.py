"""Shared utility helpers: env config, pydantic schemas, prompt formatting,
JSON parsing, image conversion, and the unified model-client factory."""

import base64
import io
import json
import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import PIL
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from matplotlib import pyplot as plt
from pydantic import BaseModel


# ─── Environment config ────────────────────────────────────────────────
# Two ways to wire up an LLM:
#   A) Local model:    LLM_MODEL_PATH=/path/to/model
#   B) HTTP endpoint:  LLM_BASE_URL + LLM_MODEL + LLM_API_KEY (default: OpenRouter)

local_model_path = os.environ.get("LLM_MODEL_PATH", "")
base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

# Sentence-transformer for ChromaDB embeddings. On HPC point at a local path:
#   export ST_MODEL_NAME=/scratch/acmarcu/models/all-MiniLM-L6-v2
ST_MODEL_NAME = os.environ.get("ST_MODEL_NAME", "all-MiniLM-L6-v2")


# ─── Pydantic response schemas ─────────────────────────────────────────

class AgentResponse(BaseModel):
    thoughts: str
    action: str
    communication: str
    # REQUIRED (not Optional): the JSON schema enforcer must always inject a
    # value, otherwise the LLM treats the field as skippable and we lose
    # targeted-comm enforcement. Routing in multi_agent_craftium.py rescues
    # malformed targets ("all", self-target, missing agent_N) by re-routing
    # via Hebbian-strongest or random teammate, but only if the field exists.
    communication_target: str


class TargetedCommunicationResponse(BaseModel):
    """All comm is targeted: communication_target is a required string (not Optional)
    so the schema enforcer guarantees the model always picks a recipient."""
    communication: str
    communication_target: str


class CurruliculumResponse(BaseModel):
    reasoning: str
    task: str


class CurriculumQuestionResponse(BaseModel):
    reasoning: str
    questions: List[str]


class CurriculumAnswerResponse(BaseModel):
    answer: str


class CriticResponse(BaseModel):
    reasoning: str
    success: bool
    critique: str


class SkillResponse(BaseModel):
    name: str
    description: str


class EpisodeResponse(BaseModel):
    summary: str


class BeliefResponse(BaseModel):
    beliefs: str


# ─── Prompt formatting ─────────────────────────────────────────────────

def safe_format(template: str, **kwargs) -> str:
    """Format a template, defaulting any missing placeholders to 'N/A'."""
    placeholders = set(re.findall(r'(?<!\{)\{(\w+)\}(?!\})', template))
    for key in placeholders:
        if key not in kwargs:
            kwargs[key] = "N/A"
            logging.warning("Prompt placeholder '{%s}' not provided, using 'N/A'", key)
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError) as e:
        logging.error("Template formatting failed even after defaults: %s", e)
        return template


# ─── JSON parsing ──────────────────────────────────────────────────────

def _fix_common_json_errors(text: str) -> str:
    """Repair common JSON formatting mistakes from LLMs.

    - Missing commas between key-value pairs
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes (only when text has zero double quotes)
    """
    # <value>\n<key>: → <value>,\n<key>:
    text = re.sub(
        r'("(?:[^"\\]|\\.)*"|true|false|null|\d+\.?\d*|\]|\})'
        r'(\s*\n\s*)'
        r'("(?:[^"\\]|\\.)*"\s*:)',
        r'\1,\2\3',
        text,
    )
    text = re.sub(r',\s*([\}\]])', r'\1', text)  # trailing commas
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def _strip_markdown_fences(response: str) -> str:
    """Remove ```json / ``` wrappers and {{...}} double-brace escaping."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    if response.startswith("{{") and response.endswith("}}"):
        response = response[1:-1]
    return response


def _try_parse(candidate: str) -> Optional[dict]:
    """Try parsing as-is, then with common-error fixes. Return None on failure."""
    for attempt in (candidate, _fix_common_json_errors(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def load_json(response: str) -> dict:
    """Parse a model response into a dict.

    Accepts pure JSON, ```json fenced blocks, thinking-text-followed-by-JSON,
    and slightly malformed JSON (missing commas, trailing commas, single quotes).
    Returns {} when no candidate parses cleanly.
    """
    response = _strip_markdown_fences(response)

    # 1. Direct parse (fast path).
    parsed = _try_parse(response)
    if parsed is not None:
        return parsed

    # 2. Walk all {...} substrings; prefer the LAST (thinking text often precedes
    #    the real JSON, so the final brace span is more likely to be the answer).
    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
    for match in reversed(matches):
        parsed = _try_parse(match)
        if parsed is not None:
            return parsed

    # 3. Fallback: outermost { ... } span across the whole text.
    first = response.find('{')
    last = response.rfind('}')
    if first != -1 and last > first:
        parsed = _try_parse(response[first:last + 1])
        if parsed is not None:
            return parsed

    logging.error("Failed to decode JSON response: %s", response[:300])
    return {}


# ─── Image utilities ───────────────────────────────────────────────────

def visualize_frames(
    rgb_frames: List[np.ndarray],
    title: str = "",
    figsize: Tuple[int, int] = (8, 2),
) -> plt.Figure:
    """Plot one frame per agent in a horizontal strip."""
    fig, axs = plt.subplots(
        1, len(rgb_frames), figsize=figsize, facecolor="white", dpi=300
    )
    for i, frame in enumerate(rgb_frames):
        ax = axs[i] if len(rgb_frames) > 1 else axs
        ax.imshow(frame)
        ax.set_title(f"AgentId: {i}")
        ax.axis("off")
    if title:
        fig.suptitle(title)
    return fig


def autogenImg_to_Pil(autogen_image):
    """Convert an autogen_core Image to a PIL Image."""
    b64 = Image.to_base64(autogen_image)
    return PIL.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ─── Model client factory ──────────────────────────────────────────────

def _resolve_api_key(key_path: str) -> str:
    """Resolve the LLM API key from env, falling back to the key file, then a placeholder."""
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        return api_key
    try:
        with open(key_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "no-key-needed"


def create_model_client(response_format, key_path="api.key"):
    """Build a ChatCompletionClient.

    - Local in-process model if LLM_MODEL_PATH is set.
    - Otherwise an OpenAI-compatible HTTP client (vLLM, SGLang, OpenRouter…).
    """
    if local_model_path:
        from agent_modules.local_model_client import LocalModelClient
        return LocalModelClient(
            model_path=local_model_path,
            response_format=response_format,
        )
    return OpenAIChatCompletionClient(
        model=model,
        base_url=base_url,
        api_key=_resolve_api_key(key_path),
        response_format=response_format,
        model_info={
            "vision": True,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )
