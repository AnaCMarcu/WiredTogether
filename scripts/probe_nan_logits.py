"""Isolate the NaN-logits failure seen on the sharded Gemma 4 checkpoints.

The 2026-08-20 pareto smoke found that gemma-4-26B-A4B-it and gemma-4-31B-it
fail EVERY generate() call with

    _assert_async_cuda_kernel: Assertion 'probability tensor contains either
    inf, nan or element < 0' failed.

which is torch.multinomial rejecting NaN/Inf logits. e2b/12b (single GPU) are
clean, so the failure correlates perfectly with device_map="auto" sharding
across 2 GPUs -- but 26b/31b are also the only never-before-run checkpoints, so
sharding and something checkpoint-specific are both live hypotheses.

This probe runs ONE vision generate per config, straight against the loader
helpers the real agent uses, with no Craftium/Minetest around it -- minutes
instead of the ~20+ min a full smoke costs.

Configs (comma-separated, --configs):
    sdpa_auto     device_map=auto, attn=sdpa      <- reproduces the failure
    eager_auto    device_map=auto, attn=eager     <- Gemma sliding-window +
                                                     sdpa can yield an
                                                     all-masked row -> NaN
                                                     softmax; eager is the
                                                     standard workaround
    sdpa_single   device_map={"":0}, attn=sdpa    <- no sharding (needs a card
                                                     that holds the whole
                                                     model, e.g. 80GB A100)
    eager_single  device_map={"":0}, attn=eager
    fp32_auto     device_map=auto, dtype=float32  <- overflow hypothesis; only
                                                     viable for small models

Per config it reports, in order:
    device map       where each block landed; any 'cpu'/'disk' entry means
                     accelerate offloaded and the run would be both slow AND a
                     candidate NaN source
    logits finite    THE discriminator. A forward pass is checked directly, so
                     this separates "the forward pass produces NaN" from
                     "sampling params are bad" -- the assert alone cannot.
    greedy / sample  whether generate() actually returns text

Usage (inside the container, see hpc/daic/probe_nan.sbatch):
    python scripts/probe_nan_logits.py --model $WORKSPACE/models/gemma-4-31B-it
    python scripts/probe_nan_logits.py --model ... --configs sdpa_auto,eager_auto
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import torch
import PIL.Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    # Preferred: exercise the SAME helpers the agent uses, so a fix proven here
    # transfers to the real run path unchanged.
    from mindforge.agent_modules.local_model_client import (  # noqa: E402
        _from_pretrained, _vision_model_class,
    )
    _HELPERS = "project"
except Exception as _exc:  # autogen &c. missing outside the container
    print("note: project helpers unavailable ({}); using local equivalents"
          .format(type(_exc).__name__))
    _HELPERS = "fallback"

    def _vision_model_class():
        import transformers
        for _name in ("AutoModelForMultimodalLM",
                      "AutoModelForImageTextToText",
                      "AutoModelForCausalLM"):
            cls = getattr(transformers, _name, None)
            if cls is not None:
                print("  using {} for VL model".format(_name))
                return cls
        raise ImportError("transformers exposes no usable VL auto-class")

    def _from_pretrained(model_cls, model_path, torch_dtype, **kwargs):
        # Mirrors the dtype-kwarg rename absorbed by the project helper.
        for _kw in ("dtype", "torch_dtype"):
            try:
                return model_cls.from_pretrained(
                    model_path, **{_kw: torch_dtype}, **kwargs)
            except TypeError:
                continue
        raise TypeError("neither dtype= nor torch_dtype= accepted")

CONFIGS = {
    "sdpa_auto":    dict(device_map="auto",  attn="sdpa",  dtype=torch.bfloat16),
    "eager_auto":   dict(device_map="auto",  attn="eager", dtype=torch.bfloat16),
    "sdpa_single":  dict(device_map={"": 0}, attn="sdpa",  dtype=torch.bfloat16),
    "eager_single": dict(device_map={"": 0}, attn="eager", dtype=torch.bfloat16),
    "fp32_auto":    dict(device_map="auto",  attn="sdpa",  dtype=torch.float32),
    # Vision components pinned to cuda:0, only the text decoder sharded.
    # Rationale: probe 12809912 showed sharded 12b fail with "Image features
    # and image tokens do not match, tokens: 266, features: 0" -- the vision
    # tower's output is lost crossing the device boundary, while the SAME model
    # on one GPU describes the frame correctly. Keeping the whole vision path
    # on one device removes that boundary; the decoder still splits, which is
    # what actually buys the memory.
    #
    # TWO pin sets, because probe 12821339 showed the first attempt create a
    # WEIGHT-TYING conflict: Gemma ties embed_tokens to lm_head, so pinning
    # embed_tokens to cuda:0 while accelerate placed lm_head (end of the
    # decoder) on cuda:1 left the single shared tensor on 1 and every lookup
    # died with "index is on cuda:0, ... other tensors on cuda:1".
    #   _min  : pin ONLY the vision/audio towers + projector; leave
    #           embed_tokens/lm_head wherever accelerate puts the tied pair.
    #   full  : pin the towers AND embed_tokens AND lm_head together on 0,
    #           so the tied tensor and both its users agree.
    # Both are plain-Python failures if wrong (not CUDA asserts), so they can
    # share one job; run _min first -- fewer forced placements is the better
    # fix if it suffices.
    "sdpa_vispin_min": dict(device_map="__vision_pinned__", attn="sdpa",
                            dtype=torch.bfloat16, pin_keys="min"),
    "sdpa_vispin":     dict(device_map="__vision_pinned__", attn="sdpa",
                            dtype=torch.bfloat16, pin_keys="full"),
}

# Substrings identifying modules forced to cuda:0. Module names seen on
# gemma4_unified 12b: model.embed_vision, model.embed_audio,
# model.language_model.embed_tokens, lm_head.
_PIN_KEYS = {
    "min":  ("vision", "multi_modal", "multimodal", "projector", "audio",
             "vision_tower"),
    "full": ("vision", "multi_modal", "multimodal", "projector", "audio",
             "vision_tower", "embed_tokens", "lm_head"),
}


def vision_pinned_device_map(model_cls, model_path, dtype, pin_keys,
                             gpu0_budget_gib=None):
    """device_map that keeps the whole vision path on cuda:0, shards the rest.

    Device 0 deliberately gets a LOWER weight budget than the other cards, for
    two reasons:
      - the vision modules are forced onto it AFTER placement, so real
        headroom must exist for that forced move plus the KV cache;
      - a model that fits entirely inside device 0's budget is not sharded at
        all, which makes a validation run VACUOUS: probe 12810410 ran 12b
        "vispin" and silently reproduced sdpa_single ("device map: single
        device cuda:0", "0 modules forced"), proving nothing about the
        cross-device boundary. For such a validation, pass --gpu0-budget-gib
        (e.g. 14 for 12b) to force the decoder to actually split.
    """
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        empty = model_cls.from_config(cfg, trust_remote_code=True)

    n_dev = torch.cuda.device_count()
    if gpu0_budget_gib:
        budget0 = int(gpu0_budget_gib * 2 ** 30)
    else:
        budget0 = int(torch.cuda.get_device_properties(0).total_memory * 0.70)
    max_memory = {0: budget0}
    for i in range(1, n_dev):
        max_memory[i] = int(
            torch.cuda.get_device_properties(i).total_memory * 0.85)

    dmap = infer_auto_device_map(
        empty, max_memory=max_memory, dtype=dtype,
        no_split_module_classes=list(getattr(empty, "_no_split_modules", None) or []),
    )
    moved = [k for k in dmap if any(t in k.lower() for t in _PIN_KEYS[pin_keys])]
    for k in moved:
        dmap[k] = 0
    print("  vision pinned   : {} modules forced to cuda:0 ({})".format(
        len(moved), moved[:4]))
    devs_used = sorted({str(v) for v in dmap.values()})
    if len(devs_used) < 2:
        print("  !! NOT SHARDED — the whole model fit device {} and no "
              "cross-device boundary exists. As a sharding test this run is "
              "VACUOUS; re-run with --gpu0-budget-gib to force a real split."
              .format(devs_used))
    return dmap


def build_inputs(processor, model, sys_chars, user_chars, size):
    """A prompt shaped like the real agent call: long system + user + one frame."""
    # Filler that tokenizes like prose, sized to the real call
    # (sys_chars=8917 user_chars~2617 in the failing runs).
    sys_text = ("You are an agent in a five-chamber cooperative environment. " * 200)[:sys_chars]
    user_text = ("Describe the scene and choose an action. " * 200)[:user_chars]
    image = PIL.Image.new("RGB", size, (110, 140, 90))
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": sys_text + "\n\n" + user_text},
        ]},
    ]
    # ONE call does template + image processing, so the count of image
    # placeholder tokens the template emits always matches the count of
    # features the processor produces. Building text and pixels in two separate
    # calls let them disagree — probe 12790126 died with
    # "Image features and image tokens do not match, tokens: 266, features: 280"
    # which was this probe's bug, not the agent's (the real runs pass 12b fine).
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    except (TypeError, ValueError, KeyError):
        # Older processors reject tokenize/return_dict here.
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True,
                           return_tensors="pt")
    return inputs.to(model.device)


def probe(name, model_path, args):
    cfg = CONFIGS[name]
    print("\n" + "=" * 72)
    print("== CONFIG {}: device_map={} attn={} dtype={}".format(
        name, cfg["device_map"], cfg["attn"], cfg["dtype"]))
    print("=" * 72, flush=True)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_cls = _vision_model_class()

    device_map = cfg["device_map"]
    if device_map == "__vision_pinned__":
        device_map = vision_pinned_device_map(
            model_cls, model_path, cfg["dtype"], cfg["pin_keys"],
            gpu0_budget_gib=args.gpu0_budget_gib)

    model = _from_pretrained(
        model_cls, model_path, cfg["dtype"],
        device_map=device_map,
        attn_implementation=cfg["attn"],
        trust_remote_code=True,
    )
    model.eval()

    dev_map = getattr(model, "hf_device_map", None)
    if dev_map:
        devs = sorted({str(v) for v in dev_map.values()})
        print("  device map      : {} modules over {}".format(len(dev_map), devs))
        offloaded = [k for k, v in dev_map.items() if str(v) in ("cpu", "disk")]
        if offloaded:
            print("  !! OFFLOADED    : {} modules to cpu/disk (e.g. {}) — "
                  "a NaN candidate and a speed killer".format(
                      len(offloaded), offloaded[:3]))
    else:
        print("  device map      : single device {}".format(model.device))

    inputs = build_inputs(processor, model, args.sys_chars, args.user_chars,
                          (args.width, args.height))
    print("  prompt tokens   : {}".format(inputs["input_ids"].shape[1]))

    # THE discriminator: inspect logits directly, before any sampling.
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[:, -1, :].float()
    finite = bool(torch.isfinite(logits).all().item())
    n_nan = int(torch.isnan(logits).sum().item())
    n_inf = int(torch.isinf(logits).sum().item())
    print("  logits finite   : {}   (nan={} inf={} of {})".format(
        finite, n_nan, n_inf, logits.numel()))
    if finite:
        print("  logits range    : min={:.3f} max={:.3f}".format(
            logits.min().item(), logits.max().item()))

    tok = getattr(processor, "tokenizer", processor)
    pad = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)

    # Prefill being finite is NOT enough: probe 12798909 showed 31b produce
    # healthy prefill logits (min=-10.9 max=8.8) and then NaN during DECODE —
    # greedy silently returned '' (argmax of NaN picks token 0) and sampling
    # tripped the multinomial assert. Step greedily with per-step logits so we
    # can see WHICH decode step first goes non-finite; that separates a
    # prefill/mask problem from a KV-cache/incremental-decode one.
    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                 pad_token_id=pad,
                                 return_dict_in_generate=True,
                                 output_logits=True, output_scores=True)
        per_step = getattr(gen, "logits", None) or getattr(gen, "scores", None)
        bad = None
        if per_step:
            for i, step in enumerate(per_step):
                if not torch.isfinite(step).all():
                    bad = i
                    break
            print("  decode steps    : {} generated, first non-finite = {}".format(
                len(per_step), "none" if bad is None else "step {}".format(bad)))
        text = tok.decode(gen.sequences[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        print("  greedy          : OK -> {}".format(repr(text[:90])))
        if not text.strip():
            print("    !! empty output — with non-finite logits argmax silently "
                  "returns token 0, so this is a FAILURE that did not raise")
    except Exception as exc:
        first = str(exc).splitlines()[0][:120] if str(exc) else ""
        print("  greedy          : FAILED -> {}: {}".format(
            type(exc).__name__, first))

    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=24, do_sample=True,
                                 temperature=0.7, top_p=0.9, pad_token_id=pad)
        text = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        print("  sample          : OK -> {}".format(repr(text[:90])))
    except Exception as exc:
        first = str(exc).splitlines()[0][:120] if str(exc) else ""
        print("  sample          : FAILED -> {}: {}".format(
            type(exc).__name__, first))

    # A device-side assert poisons the CUDA context for the WHOLE process, so
    # even empty_cache() raises and any later config would report a misleading
    # failure. Detect that and stop rather than emit junk for the rest.
    del model, processor, inputs
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        print("  !! CUDA context is poisoned ({}) — remaining configs cannot "
              "run in this process. Submit them as separate jobs, one "
              "--configs each.".format(type(exc).__name__))
        raise SystemExit(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="path to the staged checkpoint")
    ap.add_argument("--configs", default="sdpa_auto,eager_auto",
                    help="comma-separated; available: " + ",".join(CONFIGS))
    ap.add_argument("--sys-chars", type=int, default=8917)
    ap.add_argument("--user-chars", type=int, default=2617)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--gpu0-budget-gib", type=float, default=None,
                    help="cap device 0's weight budget (GiB) in sdpa_vispin — "
                         "force a real split when validating on a model that "
                         "would otherwise fit one card (e.g. 14 for 12b)")
    args = ap.parse_args()

    print("model : {}".format(args.model))
    print("torch : {}  CUDA devices: {}".format(
        torch.__version__, torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print("  cuda:{} {} {:.1f} GiB".format(i, p.name, p.total_memory / 2 ** 30))

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        sys.exit("unknown config(s): {}; available: {}".format(bad, list(CONFIGS)))

    for name in names:
        try:
            probe(name, args.model, args)
        except Exception:
            # One config OOMing or erroring must not hide the others' results.
            # (SystemExit from a poisoned CUDA context is NOT caught here — it
            # is a BaseException, so it correctly ends the process.)
            print("  CONFIG {} ABORTED:".format(name))
            traceback.print_exc()
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                print("  !! CUDA context unusable — stopping.")
                raise SystemExit(3)


if __name__ == "__main__":
    main()
