"""Minimal OpenAI-compatible LLM server using transformers.

No JIT compilation, no nvcc, no flashinfer — just PyTorch + transformers.
Designed for HPC nodes where only the GPU driver (no CUDA toolkit) is available.

Usage:
    python llm_server.py --model-path /path/to/Qwen3.5-2B --port 8000
"""

import argparse
import json
import logging
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Global state — set during startup
model = None
tokenizer = None
model_name = ""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    stop: list[str] | None = None


def generate(input_ids, temperature, top_p, max_tokens, stop_strings=None):
    """Generate text from input_ids."""
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Only decode newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Apply stop strings
    if stop_strings:
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]

    return text, len(new_tokens)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "owned_by": "local",
        }]
    })


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    t0 = time.time()

    # Apply chat template
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    text, n_tokens = generate(
        input_ids, req.temperature, req.top_p, req.max_tokens, req.stop
    )

    elapsed = time.time() - t0
    logger.info(f"Chat completion: {n_tokens} tokens in {elapsed:.2f}s "
                f"({n_tokens / elapsed:.1f} tok/s)")

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": n_tokens,
            "total_tokens": input_ids.shape[1] + n_tokens,
        }
    })


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    t0 = time.time()

    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to(model.device)

    text, n_tokens = generate(
        input_ids, req.temperature, req.top_p, req.max_tokens, req.stop
    )

    elapsed = time.time() - t0
    logger.info(f"Completion: {n_tokens} tokens in {elapsed:.2f}s "
                f"({n_tokens / elapsed:.1f} tok/s)")

    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": n_tokens,
            "total_tokens": len(input_ids[0]) + n_tokens,
        }
    })


def main():
    global model, tokenizer, model_name

    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible LLM server")
    parser.add_argument("--model-path", required=True, help="Path to HF model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "auto"])
    args = parser.parse_args()

    model_name = args.model_path.rstrip("/").split("/")[-1]

    logger.info(f"Loading model: {args.model_path}")
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"Model loaded: {model_name}, GPU memory: {mem_gb:.2f} GB")
    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
