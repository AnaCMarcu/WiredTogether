#!/usr/bin/env python3
"""bench_inference.py - GPU micro-benchmark for the shared in-process LLM.

analysis/profile_run.py shows where a finished run's wall-clock went (90-96 %
LLM calls, of which ~95 % is DECODE at 16-26 output tokens/s, batch size 1).
This script measures, on one GPU in ~10 minutes, how much each inference lever
would buy BEFORE anything in the training loop is touched:

  batch       the same call at batch size 1 / 2 / 3 / 5 / 7 / 9 - what
              batching the N agents' action-selection calls would give
  greedy      sampling off (logits-processor overhead)
  static      cache_implementation="static" (pre-allocated KV cache)
  compile     --compile: static cache + torch.compile of the forward
  short       the belief-update shape (~470 prompt tokens, no frame, ~55 out)
              at batch 1 and 8
  prefill     max_new_tokens=1, to separate prefill from decode
  op profile  --torch-profile: torch.profiler (CPU + CUDA) over one short
              generate. Prints the top operators and the GPU-busy fraction
              (CUDA kernel time / wall time) and writes a chrome://tracing
              file. A LOW busy fraction means decode is bound by Python/CPU
              overhead per token (static cache + compile, more CPU cores and
              batching all help); a HIGH one means the GPU itself is the
              limit (only batching or a faster GPU helps).

The two call shapes are the measured means of the real runs: action selection
= ~4500 prompt tokens incl. one 480x480 frame, ~150 output tokens; belief
update = ~470 prompt tokens, ~55 output tokens. Output length is pinned with
min_new_tokens so every variant decodes exactly the same number of tokens.

The model is loaded through mindforge's own loader, so dtype, model class and
device map are exactly what a run uses. Self-contained otherwise: copy this
one file anywhere and run it inside the experiment container with
PYTHONPATH=<repo>/src.

    python bench_inference.py --model $WT_WORKSPACE/models/gemma-4-E4B-it
    python bench_inference.py --model ... --batches 1 3 7 --compile --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

FILLER = (
    "You are an agent in a cooperative voxel world with five chambers. Observe the frame, "
    "recall your beliefs about teammates, and choose exactly one primitive action. "
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_text(tok, n_tokens: int, salt: int) -> str:
    """Filler text of ~n_tokens; ``salt`` makes each batch row a different length."""
    per = len(tok(FILLER, add_special_tokens=False).input_ids)
    return f"[agent_{salt}] " + FILLER * max(1, (n_tokens + 7 * salt) // per)


def make_inputs(L, n_prompt: int, batch: int, with_frame: bool, rng):
    proc = L._shared_tokenizer
    tok = L._inner_tokenizer(proc)
    tok.padding_side = "left"           # decoder-only batching needs left padding
    prompts, images = [], []
    for b in range(batch):
        text = build_text(tok, n_prompt, b)
        if with_frame and L._shared_is_vision:
            img = Image.fromarray(rng.integers(0, 255, (480, 480, 3), dtype=np.uint8))
            images.append([img])
            content = [{"type": "image", "image": img}, {"type": "text", "text": text}]
        else:
            content = text if not L._shared_is_vision else [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        prompts.append(L._apply_chat_template(proc, msgs, tokenize=False, enable_thinking=False))
    if L._shared_is_vision:
        kw = dict(text=prompts, padding=True, return_tensors="pt")
        if images:
            kw["images"] = images
        inputs = proc(**kw)
    else:
        inputs = proc(prompts, padding=True, return_tensors="pt")
    return inputs.to(L._shared_model.device)


def timed_generate(L, inputs, n_new: int, reps: int, **gen_kw) -> dict:
    tok = L._inner_tokenizer(L._shared_tokenizer)
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new,
              pad_token_id=tok.pad_token_id or tok.eos_token_id,
              do_sample=True, temperature=0.7, top_p=0.9)   # LocalModelClient defaults
    kw.update(gen_kw)
    if not kw.get("do_sample"):
        kw.pop("temperature", None)
        kw.pop("top_p", None)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(reps + 1):                # first rep = warm-up, discarded
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = L._shared_model.generate(**inputs, **kw)
        _sync()
        if i:
            times.append(time.perf_counter() - t0)
    n_in = inputs["input_ids"].shape[1]
    assert out.shape[1] - n_in == n_new, (out.shape, n_in, n_new)
    return {"secs": statistics.median(times), "prompt_tokens": n_in,
            "batch": inputs["input_ids"].shape[0],
            "vram_gb": torch.cuda.max_memory_allocated() / 1e9}


def op_profile(L, inputs, n_new: int, trace_path: str) -> None:
    """torch.profiler over one generate: is a decode step GPU-bound or CPU-bound?"""
    from torch.profiler import ProfilerActivity, profile

    tok = L._inner_tokenizer(L._shared_tokenizer)
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new, do_sample=True, temperature=0.7,
              top_p=0.9, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    with torch.no_grad():
        L._shared_model.generate(**inputs, **kw)          # warm-up outside the profile
    _sync()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            L._shared_model.generate(**inputs, **kw)
        _sync()
    wall = time.perf_counter() - t0
    avgs = prof.key_averages()

    def _dev(e):   # torch >= 2.4 renamed cuda -> device
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)

    # Count kernel rows only: an aten:: op row repeats the device time of the
    # kernels it launched, so summing every row double-counts.
    from torch.autograd import DeviceType
    kernels = [e for e in avgs if getattr(e, "device_type", None) == DeviceType.CUDA]
    gpu_s = sum(_dev(e) for e in (kernels or avgs)) / 1e6
    cpu_s = sum(e.self_cpu_time_total for e in avgs) / 1e6
    print(f"profiled wall {wall:.2f} s (profiler overhead included) for {n_new} new tokens: "
          f"CUDA kernels {gpu_s:.2f} s = {100 * gpu_s / wall:.0f}% GPU-busy, "
          f"CPU-side ops {cpu_s:.2f} s")
    for key in ("self_cuda_time_total", "self_device_time_total"):
        try:
            print(avgs.table(sort_by=key, row_limit=12, max_name_column_width=48))
            break
        except Exception:
            continue
    print(avgs.table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=48))
    prof.export_chrome_trace(trace_path)
    print(f"chrome trace -> {trace_path}  (open in chrome://tracing or ui.perfetto.dev)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL_PATH", ""))
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 5, 7, 9])
    ap.add_argument("--prompt-tokens", type=int, default=4200,
                    help="text tokens of the action-selection shape (the frame adds ~280)")
    ap.add_argument("--new-tokens", type=int, default=150)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compile", action="store_true",
                    help="also try static cache + torch.compile (slow first call)")
    ap.add_argument("--torch-profile", action="store_true",
                    help="operator-level torch.profiler pass over one short generate")
    ap.add_argument("--trace", default="bench_trace.json")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.model:
        sys.exit("pass --model or set LLM_MODEL_PATH")

    from mindforge.agent_modules import local_model_client as L
    import transformers

    L._load_shared_model(args.model)
    model = L._shared_model
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"host={os.uname().nodename}  gpu={gpu}  cpus={cpus}  torch={torch.__version__}  "
          f"transformers={transformers.__version__}  dtype={model.dtype}  "
          f"attn={getattr(model.config, '_attn_implementation', '?')}  vision={L._shared_is_vision}")

    rng = np.random.default_rng(0)
    results = []

    def run(name, n_prompt, batch, with_frame, n_new, **gen_kw):
        try:
            inputs = make_inputs(L, n_prompt, batch, with_frame, rng)
            r = timed_generate(L, inputs, n_new, args.reps, **gen_kw)
        except Exception as exc:                      # one variant must not kill the rest
            print(f"{name:28s} bs={batch}  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
            torch.cuda.empty_cache()
            return None
        r.update(name=name, new_tokens=n_new,
                 tok_s_stream=n_new / r["secs"], tok_s_total=batch * n_new / r["secs"])
        results.append(r)
        print(f"{name:28s} bs={batch}  in={r['prompt_tokens']:5d}  {r['secs']:6.2f} s  "
              f"{r['tok_s_stream']:6.1f} tok/s per stream  {r['tok_s_total']:7.1f} tok/s total  "
              f"VRAM {r['vram_gb']:.1f} GB")
        return r

    print("\n-- action-selection shape (frame + long prompt) --")
    run("prefill only", args.prompt_tokens, 1, True, 1)
    base = run("baseline (as in a run)", args.prompt_tokens, 1, True, args.new_tokens)
    run("greedy", args.prompt_tokens, 1, True, args.new_tokens, do_sample=False)
    run("static cache", args.prompt_tokens, 1, True, args.new_tokens, cache_implementation="static")
    for b in args.batches:
        if b > 1:
            run("batched", args.prompt_tokens, b, True, args.new_tokens)

    print("\n-- belief-update shape (short, no frame) --")
    run("short baseline", 470, 1, False, 55)
    run("short batched", 470, 8, False, 55)
    run("short batched", 470, 16, False, 55)

    if args.torch_profile:
        print("\n-- torch.profiler: one generate, action-selection shape, 30 new tokens --")
        try:
            op_profile(L, make_inputs(L, args.prompt_tokens, 1, True, rng), 30, args.trace)
        except Exception as exc:
            print(f"torch.profiler FAILED: {type(exc).__name__}: {str(exc)[:200]}")

    if args.compile:
        print("\n-- torch.compile (first call compiles; the median excludes it) --")
        try:
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
            run("compiled + static cache", args.prompt_tokens, 1, True, args.new_tokens,
                cache_implementation="static")
            run("compiled + static, short", 470, 1, False, 55, cache_implementation="static")
        except Exception as exc:
            print(f"compile FAILED: {type(exc).__name__}: {str(exc)[:200]}")

    if base:
        print(f"\nspeed-up vs baseline total throughput ({base['tok_s_total']:.1f} tok/s):")
        for r in results:
            if r["new_tokens"] == args.new_tokens and r is not base:
                print(f"  {r['name']:28s} bs={r['batch']}  x{r['tok_s_total'] / base['tok_s_total']:.2f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"gpu": gpu, "host": os.uname().nodename, "results": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
