"""PEFT/LoRA compatibility for checkpoints whose projections are not nn.Linear.

Gemma 4 (any size, incl. the 4B/E4B checkpoints) implements its attention
projections as ``Gemma4ClippableLinear`` — functionally a linear layer with
output clipping, but a direct ``nn.Module`` subclass rather than an
``nn.Linear``. PEFT's LoRA dispatch matches target modules with
``isinstance(target_base_layer, torch.nn.Linear)``, so adapter injection on a
Gemma 4 base dies at load with::

    ValueError: Target module Gemma4ClippableLinear(...) is not supported.

which is what killed every MAPPO/IPPO arm on the 2026-08-04/06 batches
(Qwen3.5 projections are plain ``nn.Linear``, so RL only ever ran on Qwen).

Fix: reparent each *targeted* instance to a dynamically created subclass of
``(OriginalClass, nn.Linear)``. The MRO keeps the original ``forward`` first,
so numerics are bit-identical and the clipping stays exactly where the
checkpoint put it; ``isinstance(module, nn.Linear)`` becomes True, which is
the one predicate every PEFT injection path (``get_peft_model``,
``add_adapter``, ``load_adapter``) dispatches on. LoRA then wraps the module
as its ``base_layer`` and adds the low-rank delta to the clipped output,
exactly as it does for Qwen.

Two alternatives were rejected:

- ``LoraConfig._register_custom_module`` (PEFT >= 0.11 dynamic dispatch) is
  runtime-only state that is not serialised into ``adapter_config.json`` —
  the ``load_adapter``-from-checkpoint path in ``RLLayer._ensure_adapter``
  would rebuild the config from disk and crash all over again.
- Re-implementing the clipped forward on a fresh ``nn.Linear`` risks silently
  diverging from whatever transformers does (clip value semantics, dtype of
  the clamp), and would break again on the next upstream tweak.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# One compat class per original class, so repeated loads (and repeated
# RLLayer constructions in tests) reuse the same type object.
_COMPAT_CLASSES: Dict[type, type] = {}


def _compat_class(cls: type) -> type:
    compat = _COMPAT_CLASSES.get(cls)
    if compat is None:
        try:
            compat = type(cls.__name__, (cls, nn.Linear), {})
        except TypeError as exc:  # inconsistent MRO / incompatible layout
            raise RuntimeError(
                f"peft_compat: cannot graft nn.Linear onto {cls.__module__}."
                f"{cls.__qualname__}; PEFT will not be able to LoRA-wrap it"
            ) from exc
        _COMPAT_CLASSES[cls] = compat
    return compat


def make_lora_targets_linear(model: nn.Module,
                             target_modules: Sequence[str]) -> int:
    """Make LoRA-targeted linear-like modules pass PEFT's nn.Linear dispatch.

    Walks ``model`` and, for every submodule whose name matches
    ``target_modules`` (same suffix semantics PEFT uses: exact name or
    ``.<target>`` suffix) and that is linear-like (2-D ``weight``) but NOT an
    ``nn.Linear``, swaps ``__class__`` to a subclass that also inherits
    ``nn.Linear``. Parameters, buffers, attributes and ``forward`` are
    untouched. ``in_features``/``out_features`` are derived from the weight
    shape when absent, because PEFT's ``LoraLayer.__init__`` reads them off
    the base layer.

    Must run on the bare base model BEFORE ``get_peft_model``; afterwards all
    PEFT paths — bootstrap wrap, per-agent ``add_adapter``, and
    ``load_adapter`` from a saved checkpoint — see plain Linears.

    Returns the number of modules reparented (0 for Qwen-style checkpoints
    whose projections already are ``nn.Linear`` — the call is then a no-op).
    """
    patched = 0
    for name, module in model.named_modules():
        if not any(name == t or name.endswith("." + t) for t in target_modules):
            continue
        if isinstance(module, nn.Linear):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            logger.warning(
                "peft_compat: %s (%s) matches a LoRA target but has no 2-D "
                "weight; leaving it for PEFT to reject explicitly",
                name, type(module).__name__,
            )
            continue
        # nn.Linear stores weight as (out_features, in_features); any
        # checkpoint-compatible linear variant must follow the same layout.
        if not hasattr(module, "in_features"):
            module.in_features = weight.shape[1]
        if not hasattr(module, "out_features"):
            module.out_features = weight.shape[0]
        module.__class__ = _compat_class(type(module))
        patched += 1
    if patched:
        logger.info(
            "peft_compat: reparented %d LoRA target modules (e.g. Gemma 4's "
            "ClippableLinear) so PEFT can inject adapters", patched,
        )
    return patched
