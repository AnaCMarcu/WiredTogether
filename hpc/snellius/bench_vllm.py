#!/usr/bin/env python3
"""bench_vllm.py - in-process HF generate() vs a vLLM server, same GPU, same calls.

Answers one question before any training-loop code changes: how much faster is
an agent round if the LLM calls go to a vLLM server instead of the shared
in-process transformers model?

Two call shapes, the measured means of real runs:
  action   ~4500 prompt tokens + one 480x480 frame, 150 output tokens
  belief   ~470 prompt tokens, no frame, 55 output tokens
Output length is pinned (HF: min_new_tokens; vLLM: min_tokens + ignore_eos), so
both backends decode exactly the same number of tokens. Sampling matches
LocalModelClient: temperature 0.7, top_p 0.9, thinking off.

A "round" is C agents each making one call:
  hf     today's path: batch 1, one call after another -> C x single-call time
  vllm   C concurrent HTTP requests -> the server batches them itself
vLLM is measured twice: "unique" prompts (no reuse between calls) and "shared"
prompts (the first 80 % identical across calls, like the common system prompt
and rules in a real run), which is where prefix caching helps.

    python bench_vllm.py hf   --model DIR --json hf.json          # WiredTogether image
    python bench_vllm.py vllm --url http://127.0.0.1:PORT --json vllm.json   # any python
    python bench_vllm.py compare hf.json vllm.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SHAPES = {
    "action": {"prompt_words": 4000, "frame": True, "new_tokens": 150},
    "belief": {"prompt_words": 420, "frame": False, "new_tokens": 55},
}
TEMPERATURE, TOP_P = 0.7, 0.9
WORDS = ("agent chamber door switch zombie wood stone anvil iron craft build team "
         "help ask partner follow plan milestone reward bond trust message north "
         "south east west lever torch pick sword gate bridge river cave health").split()


# ── identical inputs for both backends ────────────────────────────────────

def prompt_text(shape: str, uid: int, shared: bool) -> str:
    """Deterministic filler. shared=True: the first 80 % is the same for every uid."""
    n = SHAPES[shape]["prompt_words"]
    n_common = int(n * 0.8) if shared else 0
    common = " ".join(random.Random(f"{shape}-common").choice(WORDS) for _ in range(n_common))
    rng = random.Random(f"{shape}-{uid}")
    unique = " ".join(rng.choice(WORDS) for _ in range(n - n_common))
    head = common + " " if common else ""
    return f"{head}Request {uid}. {unique}\nReply with one short sentence."


def frame_png(uid: int) -> bytes:
    from PIL import Image
    rng = random.Random(uid)
    img = Image.new("RGB", (480, 480))
    px = img.load()
    base = [rng.randrange(256) for _ in range(3)]
    for y in range(480):
        for x in range(480):
            px[x, y] = ((base[0] + x) % 256, (base[1] + y) % 256, (base[2] + x * y // 480) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── HF in-process (today's path) ──────────────────────────────────────────

def bench_hf(args) -> dict:
    import torch
    from PIL import Image
    from mindforge.agent_modules import local_model_client as lmc

    lmc._load_shared_model(args.model)
    tok = lmc._inner_tokenizer(lmc._shared_tokenizer)
    print(f"[hf] loaded {args.model}  vision={lmc._shared_is_vision}", flush=True)

    def one_call(shape: str, uid: int) -> dict:
        spec = SHAPES[shape]
        text = prompt_text(shape, uid, shared=False)
        if spec["frame"] and lmc._shared_is_vision:
            img = Image.open(io.BytesIO(frame_png(uid))).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": text}]}]
            inputs, n_in = lmc.LocalModelClient._tokenize_vision(msgs, [img], False)
        else:
            msgs = [{"role": "user", "content": text}]
            inputs, n_in = lmc.LocalModelClient._tokenize_text(msgs, False)
        n = spec["new_tokens"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = lmc._shared_model.generate(
                **inputs, max_new_tokens=n, min_new_tokens=n,
                temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        torch.cuda.synchronize()
        return {"secs": time.perf_counter() - t0, "prompt_tokens": n_in,
                "completion_tokens": out.shape[1] - n_in}

    results = {"backend": "hf", "model": args.model, "gpu": torch.cuda.get_device_name(0),
               "shapes": {}}
    for shape in SHAPES:
        one_call(shape, 10_000)                                  # warm-up
        calls = [one_call(shape, uid) for uid in range(args.reps)]
        secs = statistics.mean(c["secs"] for c in calls)
        results["shapes"][shape] = {
            "single_call_secs": secs,
            "prompt_tokens": statistics.mean(c["prompt_tokens"] for c in calls),
            "completion_tokens": statistics.mean(c["completion_tokens"] for c in calls),
            # batch 1, sequential: a round of C calls takes C single calls
            "rounds": {str(c): {"round_secs": secs * c} for c in args.concurrency},
        }
        print(f"[hf] {shape}: {secs:.2f} s/call  "
              f"({results['shapes'][shape]['prompt_tokens']:.0f} in, "
              f"{results['shapes'][shape]['completion_tokens']:.0f} out)", flush=True)
    return results


# ── vLLM server ───────────────────────────────────────────────────────────

def _post(url: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bench_vllm(args) -> dict:
    base = args.url.rstrip("/")
    models = json.loads(urllib.request.urlopen(f"{base}/v1/models", timeout=30).read())
    served = models["data"][0]["id"]
    print(f"[vllm] serving {served}", flush=True)
    frames = {}

    def one_call(shape: str, uid: int, shared: bool) -> dict:
        spec = SHAPES[shape]
        content = []
        if spec["frame"]:
            if uid not in frames:
                frames[uid] = base64.b64encode(frame_png(uid)).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{frames[uid]}"}})
        content.append({"type": "text", "text": prompt_text(shape, uid, shared)})
        n = spec["new_tokens"]
        body = {"model": served, "messages": [{"role": "user", "content": content}],
                "max_tokens": n, "min_tokens": n, "ignore_eos": True,
                "temperature": TEMPERATURE, "top_p": TOP_P,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        resp = _post(f"{base}/v1/chat/completions", body)
        return {"secs": time.perf_counter() - t0,
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "completion_tokens": resp["usage"]["completion_tokens"]}

    results = {"backend": "vllm", "served_model": served, "shapes": {}}
    uid = 0
    for shape in SHAPES:
        for mode in ("unique", "shared"):
            shared = mode == "shared"
            one_call(shape, 90_000 + (1 if shared else 0), shared)   # warm-up
            key = f"{shape}/{mode}"
            results["shapes"][key] = {"rounds": {}}
            for c in args.concurrency:
                rounds, calls = [], []
                for _ in range(args.reps):
                    uids = list(range(uid, uid + c))
                    uid += c
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=c) as pool:
                        got = list(pool.map(lambda u: one_call(shape, u, shared), uids))
                    rounds.append(time.perf_counter() - t0)
                    calls.extend(got)
                results["shapes"][key]["rounds"][str(c)] = {
                    "round_secs": statistics.mean(rounds),
                    "call_secs": statistics.mean(x["secs"] for x in calls),
                }
                results["shapes"][key]["prompt_tokens"] = statistics.mean(
                    x["prompt_tokens"] for x in calls)
                results["shapes"][key]["completion_tokens"] = statistics.mean(
                    x["completion_tokens"] for x in calls)
                print(f"[vllm] {key} C={c}: {statistics.mean(rounds):.2f} s/round", flush=True)
    return results


# ── comparison table ──────────────────────────────────────────────────────

def compare(hf_path: str, vllm_path: str) -> None:
    hf = json.load(open(hf_path))
    vl = json.load(open(vllm_path))
    print(f"\nGPU: {hf.get('gpu', '?')}   model: {hf.get('model', '?')}")
    print("seconds per round of C agents each making one call (lower is better)\n")
    print(f"{'shape':<8} {'C':>3} {'hf (today)':>11} {'vllm unique':>12} {'vllm shared':>12} "
          f"{'speedup':>8}")
    for shape in SHAPES:
        h = hf["shapes"][shape]
        for c, r in h["rounds"].items():
            u = vl["shapes"][f"{shape}/unique"]["rounds"][c]["round_secs"]
            s = vl["shapes"][f"{shape}/shared"]["rounds"][c]["round_secs"]
            print(f"{shape:<8} {c:>3} {r['round_secs']:>10.2f}s {u:>11.2f}s {s:>11.2f}s "
                  f"{r['round_secs'] / min(u, s):>7.1f}x")
    print("\nprompt tokens  hf/vllm:  " + "  ".join(
        f"{s}={hf['shapes'][s]['prompt_tokens']:.0f}/{vl['shapes'][s + '/unique']['prompt_tokens']:.0f}"
        for s in SHAPES))
    print("(the counts differ slightly if the two backends tokenize the frame differently)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("hf", "vllm"):
        s = sub.add_parser(name)
        if name == "hf":
            s.add_argument("--model", required=True)
        else:
            s.add_argument("--url", required=True)
        s.add_argument("--concurrency", type=int, nargs="+", default=[1, 3, 9])
        s.add_argument("--reps", type=int, default=3)
        s.add_argument("--json", required=True)
    c = sub.add_parser("compare")
    c.add_argument("hf_json")
    c.add_argument("vllm_json")
    args = p.parse_args()

    if args.cmd == "compare":
        compare(args.hf_json, args.vllm_json)
        return 0
    results = bench_hf(args) if args.cmd == "hf" else bench_vllm(args)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
