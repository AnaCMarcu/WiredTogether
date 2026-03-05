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

from autogen_ext.models.openai import OpenAIChatCompletionClient
""
base_url = "https://openrouter.ai/api/v1" 
# base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model = "google/gemini-2.5-flash" # "google/gemma-3-27b-it:free" # "nvidia/nemotron-nano-12b-v2-vl:free" # "google/gemma-3-27b-it:free" # "qwen/qwen-2.5-vl-7b-instruct:free" 
# model = "gemini-2.5-flash-lite" # "google/gemini-2.0-flash-001",

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
    if response.startswith("```json") and response.endswith("```"):
        response = response[8:-3].strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON response: ```json`...```", response)
            return {}
    else:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON response: else", response)
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
) -> OpenAIChatCompletionClient:
    with open(key_path) as f:
        api_key = f.read().strip()
    model_client = OpenAIChatCompletionClient(
        # configure which model and API to use
        # model="qwen/qwen2.5-vl-32b-instruct:free",
        model= model,
        base_url=base_url,
        api_key=api_key,
        response_format=resonse_format,  # constrain response format
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
