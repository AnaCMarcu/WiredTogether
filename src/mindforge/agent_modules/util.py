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

local_model_path = os.environ.get("LLM_MODEL_PATH", "")
base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

ST_MODEL_NAME = os.environ.get("ST_MODEL_NAME", "all-MiniLM-L6-v2")


# ─── Agent-ID normalization ────────────────────────────────────────────

_AGENT_ID_RE = re.compile(r"^\s*agent_?(\d+)\s*$", re.IGNORECASE)


def normalize_agent_target(s):
    """Canonicalize an LLM-emitted agent name to the form ``agent_<N>``.

    The LLM sometimes drops the underscore (``agent0``), capitalizes
    (``Agent_1``), or pads with spaces. Without normalization the same
    teammate ends up indexed under two distinct keys downstream — observed
    in ``exp1_llm/seed_42`` where milestone_log contributors contained
    BOTH ``agent_0`` and ``agent0`` for the same agent, double-counting
    per-agent contribution metrics.

    Returns the canonical ``agent_<N>`` string when ``s`` parses cleanly,
    otherwise returns the input unchanged (so the existing routing-layer
    fallback in multi_agent_craftium.py can still rescue unparseable
    targets like ``all`` / ``none`` / ``self``).
    """
    if not isinstance(s, str):
        return s
    m = _AGENT_ID_RE.match(s)
    if m is None:
        return s
    return f"agent_{int(m.group(1))}"


# ─── Pydantic response schemas ─────────────────────────────────────────

class AgentResponse(BaseModel):
    thoughts: str
    action: str
    communication: str
    communication_target: str


class TargetedCommunicationResponse(BaseModel):
    """All comm is targeted: communication_target is a required string (not Optional)
    so the schema enforcer guarantees the model always picks a recipient.

    ``thoughts`` is required too: in RL mode the action is chosen by the
    constrained-generation scorer (no freeform LLM output), so this is the
    only LLM-produced reasoning text we get per step. Without it the run
    log just shows a hard-coded placeholder ('RL policy (step N): selected
    X') and we have no insight into why the LLM picked any particular
    teammate or message phrasing.
    """
    thoughts: str
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


class SocialThought(BaseModel):
    """Output of the SocialModule's deliberation step.

    Lives between the Hebbian graph and the action LLM: takes (bond weights,
    bond deltas, incoming messages, recent self-state) and produces an
    explicit social directive that the action prompt consumes. The
    ``referenced_bonds`` + ``bond_change_explanation`` fields are the
    interpretability artifact for the Hebbian-as-social-intelligence claim —
    they show whether the LLM actually conditioned on the graph.
    """
    bond_change_explanation: dict[str, str]
    # Free-text rationale tying bonds + incoming messages to the decision.
    reasoning: str
    referenced_bonds: dict[str, float]
    # Asker side: who to send a help request to, and the suggested text.
    ask_target: Optional[str] = None
    ask_message: Optional[str] = None
    # Responder side: subset of incoming senders to help this step.
    respond_to: List[str] = []
    confidence: float = 0.5


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


def coerce_belief_text(value, previous: str = "") -> str:
    """Normalise a parsed ``beliefs`` field into a single string.

    Accepts a string ("A. B."), a list (["A.", "B."]), or anything empty.
    Falls back to ``previous`` when there is nothing usable, so a blank
    update never clobbers an existing belief.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        joined = " ".join(str(x).strip() for x in value if str(x).strip())
        if joined:
            return joined
    return previous


def load_belief(response: str) -> dict:
    """Parse a belief-module response into ``{"beliefs": <str>}``.

    Belief modules only ever read a single free-text ``beliefs`` field, so
    this is far more forgiving than :func:`load_json` and **never triggers a
    retry**. It accepts the canonical ``{"beliefs": "..."}`` and
    ``{"beliefs": [...]}`` shapes, and salvages the common malformation where
    the model emits a bag of bare strings — ``{"beliefs": "A", "B", "C"}`` —
    by harvesting every string literal and joining them. Returns
    ``{"beliefs": ""}`` when nothing is salvageable (the caller keeps its
    previous belief), which is always truthy and so will not provoke the
    "Empty JSON response" retry path in ``llm_call``.
    """
    parsed = load_json(response)
    if isinstance(parsed, dict):
        text = coerce_belief_text(parsed.get("beliefs"))
        if text:
            return {"beliefs": text}

    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', _strip_markdown_fences(response))
    if strings and strings[0].strip() == "beliefs":
        strings = strings[1:]
    joined = " ".join(s.strip() for s in strings if s.strip())
    return {"beliefs": joined}


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
