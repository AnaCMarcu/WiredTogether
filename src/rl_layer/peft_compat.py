"""Resolve LoRA target modules for multimodal checkpoints (Gemma 4).

PEFT matches ``target_modules`` by name suffix, so the bare names
``q_proj`` / ``v_proj`` select attention projections in EVERY tower of a
multimodal checkpoint. On Gemma 4 that is a problem twice over:

1. **It crashes.** The vision and audio towers implement their projections
   as ``Gemma4ClippableLinear`` — a *wrapper* holding an inner ``nn.Linear``
   (``Gemma4ClippableLinear((linear): Linear(in_features=768, ...))``), not
   an ``nn.Linear`` itself. PEFT's LoRA dispatch requires
   ``isinstance(target, nn.Linear)`` and raises::

       ValueError: Target module Gemma4ClippableLinear(...) is not supported.

   which killed every MAPPO/IPPO arm at model load (the 2026-08-04/06
   batches, and again in the 2026-08-19 gemma4_smoke run). The text tower's
   projections are plain ``nn.Linear`` — in the smoke run all 56 rejected
   modules were vision (32) or audio (24), none text.

2. **The adapters would be dead weight even if it did not crash.** The RL
   policy scores actions through ``RLLayer._score_actions``, which feeds the
   model ``input_ids`` only; the vision/audio towers are never entered, so
   LoRA parameters placed on them receive no gradient. They would sit in
   each agent's optimizer forever at their initial values, inflating the
   trainable-parameter count and making it incomparable with the Qwen runs
   (which have no such towers at all).

So targeting is restricted to the text tower, and the resolved module names
are passed to PEFT *explicitly* rather than as suffixes — the injection then
cannot wander into a tower we did not choose, on this or any future
checkpoint layout. Text-only checkpoints (Qwen) have no separate towers, so
resolution returns exactly the same set the old suffix matching produced.

A wrapper-descent step is kept for the case where a text-tower projection is
itself a ``ClippableLinear``-style wrapper: LoRA is then attached to the
inner ``nn.Linear``, which also puts the low-rank delta *inside* the clip
rather than after it — the placement the checkpoint's own arithmetic implies.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch.nn as nn

logger = logging.getLogger(__name__)

# Submodule names that root the text tower of a multimodal wrapper. The
# outermost match wins, and targeting is confined to its subtree.
_TEXT_SUBTREE_NAMES = ("language_model", "text_model")

# Fallback only, for a multimodal layout that names its text tower something
# we do not know: these subtrees are excluded instead. Positive selection via
# _TEXT_SUBTREE_NAMES is preferred because it excludes unknown future towers
# automatically, whereas this list only excludes what it already lists.
_NON_TEXT_SUBTREE_NAMES = (
    "vision_tower", "audio_tower", "vision_model", "audio_model",
    "vision_encoder", "audio_encoder", "multi_modal_projector",
)


def _text_subtree_prefix(model: nn.Module) -> Optional[str]:
    """Dotted path of the outermost text tower, or None if there isn't one.

    Shallowest match wins: a checkpoint that nests ``language_model`` inside
    another ``language_model`` should be confined by the outer one.
    """
    candidates = [
        name for name, _ in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in _TEXT_SUBTREE_NAMES
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda n: (n.count("."), len(n)))


def _resolve_linear(name: str, module: nn.Module) -> Optional[str]:
    """Return the name of the ``nn.Linear`` LoRA should wrap for this target.

    ``name`` itself when the module already is one; otherwise the path of its
    single ``nn.Linear`` descendant (the ``Gemma4ClippableLinear.linear``
    shape). Ambiguous or non-linear targets return None and are skipped —
    PEFT would only reject them later with a less informative error.
    """
    if isinstance(module, nn.Linear):
        return name
    inner = [
        sub for sub, m in module.named_modules()
        if sub and isinstance(m, nn.Linear)
    ]
    if len(inner) == 1:
        return f"{name}.{inner[0]}"
    logger.warning(
        "peft_compat: %s (%s) is a LoRA target but holds %d nn.Linear "
        "submodules; skipping it (expected exactly 1)",
        name, type(module).__name__, len(inner),
    )
    return None


def resolve_lora_targets(model: nn.Module,
                         target_names: Sequence[str]) -> List[str]:
    """Full module names PEFT should attach LoRA adapters to.

    ``target_names`` are the logical projection names (``q_proj``,
    ``v_proj``). Matching is on the final path component, confined to the
    text tower, with wrapper modules resolved to their inner ``nn.Linear``.

    Must be called on the bare base model BEFORE ``get_peft_model``. The
    returned list is reused verbatim for every per-agent ``add_adapter``, so
    all adapters cover an identical parameter set — which the per-agent
    optimizer filtering in ``RLLayer.__init__`` assumes.

    Raises
    ------
    RuntimeError
        If nothing resolved. Training would otherwise proceed with zero
        adaptable parameters and silently learn nothing for the whole run —
        the failure mode this check exists to prevent.
    """
    prefix = _text_subtree_prefix(model)
    targets: List[str] = []
    skipped_towers = 0

    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in target_names:
            continue
        if prefix is not None:
            if not name.startswith(prefix + "."):
                skipped_towers += 1
                continue
        elif any(part in _NON_TEXT_SUBTREE_NAMES for part in name.split(".")):
            skipped_towers += 1
            continue
        resolved = _resolve_linear(name, module)
        if resolved is not None:
            targets.append(resolved)

    if not targets:
        raise RuntimeError(
            f"peft_compat: no LoRA target resolved for {tuple(target_names)} "
            f"in {type(model).__name__} (text tower: {prefix or '<none>'}). "
            "Adapters would train nothing — check the projection names for "
            "this checkpoint."
        )

    logger.info(
        "peft_compat: %d LoRA targets in text tower '%s'%s",
        len(targets), prefix or "<root>",
        f"; {skipped_towers} vision/audio projections excluded"
        if skipped_towers else "",
    )
    return targets
