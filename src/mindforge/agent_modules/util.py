import json
import logging
from typing import List, Literal, Tuple

import PIL
import base64
import io
from autogen_core import Image

from matplotlib import pyplot as plt
import numpy as np
import logging

from pydantic import BaseModel

import os
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Configure via environment variables:
#
# Option A — Local model (no server needed):
#   export LLM_MODEL_PATH=/scratch/acmarcu/models/Qwen3.5-2B
#
# Option B — HTTP server (SGLang, vLLM, llm_server.py, OpenRouter):
#   export LLM_BASE_URL=http://localhost:8000/v1
#   export LLM_MODEL=Qwen3.5-2B
#   export LLM_API_KEY=no-key-needed
#
local_model_path = os.environ.get("LLM_MODEL_PATH", "")
base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

# LLM response format
class AgentResponse(BaseModel):
    task: str
    thoughts: str
    action: str
    communication: str


class CandidateResponse(BaseModel):
    candidate_actions: List[str]


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

class Observation(BaseModel):
    observation: str

class Belief(BaseModel):
    belief: str

class Action(BaseModel):
    action: str

class Feedback(BaseModel):
    content: str
# parse model response from json to dict
# Accepted formats:
# 1. {...}
# 2. ```json
#    {...}
#    ```
def load_json(response: str) -> AgentResponse:
    import re
    # Strip markdown code fences
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object — prefer the last one (thinking text may contain braces)
    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    logging.error(f"Failed to decode JSON response: {response[:200]}")
    return {}


# print vision frame for each agent
def visualize_frames(
    rgb_frames: List[np.ndarray], title: str = "", figsize: Tuple[int, int] = (8, 2)
) -> plt.Figure:
    """Plots the rgb_frames for each agent."""
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


def create_model_client(
    resonse_format, key_path="api.key"
):
    # Option A: local transformers model (no HTTP server)
    if local_model_path:
        from agent_modules.local_model_client import LocalModelClient
        return LocalModelClient(
            model_path=local_model_path,
            response_format=resonse_format,
        )

    # Option B: OpenAI-compatible HTTP endpoint
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        try:
            with open(key_path) as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            api_key = "no-key-needed"
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        response_format=resonse_format,
        model_info={
            "vision": True,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )
    return model_client


def autogenImg_to_Pil(autogen_image):
    # Convert autogen_core Image to Pil Image
    base_img = Image.to_base64(autogen_image)
    image_data = base64.b64decode(base_img)
    image = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")
    return image
