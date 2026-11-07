"""Score every candidate action in ONE forward that reads the prompt once.

``RLLayer._score_actions`` (the default) builds one row per candidate,
``[prompt + cand_c]``, and runs them as a batch: the prompt is encoded C
times (C = 14 actions with Slot* masked) at sampling and at every PPO pass.
Here the prompt and all candidates' tokens are packed into one sequence,

    [prompt | cand_0 | cand_1 | ... | cand_{C-1}]

and a custom attention mask lets each candidate token attend to the whole
prompt and to the earlier tokens of its OWN candidate, never to another
candidate. Candidate tokens carry the positions they would have had in their
own row (prompt_len + j), so rotary embeddings and the sliding window see
exactly the per-row computation. The log-probs, the pooled hidden state and
their gradients are therefore the same as the batched path's (up to
floating-point order); the cost is one prompt forward instead of C.

The mask goes in as the per-layer-type dict that hybrid models accept
(``{"full_attention": ..., "sliding_attention": ...}``, Gemma 3/4) or as a
single 4D mask for full-attention-only models. Models whose layers take no
attention mask (linear attention, e.g. Qwen3.5's gated DeltaNet) cannot be
packed: ``packing_support`` says so and the caller keeps the batched path.
"""

from __future__ import annotations

from typing import Sequence

import torch

PACKABLE_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})
PACKABLE_ATTENTION = frozenset({"sdpa", "eager"})


def _text_config(model):
    cfg = getattr(model, "config", None)
    return getattr(cfg, "text_config", None) or cfg


def packing_support(model) -> "tuple[bool, str]":
    """(True, "") when the packed forward is exact for this model, else (False, why)."""
    cfg = _text_config(model)
    if cfg is None:
        return False, "model has no config"
    layer_types = set(getattr(cfg, "layer_types", None) or ["full_attention"])
    other = layer_types - PACKABLE_LAYER_TYPES
    if other:
        return False, "layer types %s take no attention mask" % sorted(other)
    impl = getattr(cfg, "_attn_implementation", None) or "eager"
    if impl not in PACKABLE_ATTENTION:
        return False, "attention implementation %r" % impl
    if "sliding_attention" in layer_types and getattr(cfg, "sliding_window", None) is None:
        return False, "sliding layers without a sliding_window"
    if getattr(cfg, "use_bidirectional_attention", None) == "all":
        return False, "bidirectional attention"
    return True, ""


def packed_layout(prompt_len: int, cand_lens: Sequence[int], device=None):
    """Token bookkeeping of the packed sequence.

    Returns (owner, positions, starts): owner[t] is -1 for prompt tokens and c
    for candidate c's tokens; positions[t] is the position the token has in
    its own unpacked row; starts[c] is the packed index of candidate c's
    first token.
    """
    total = prompt_len + sum(cand_lens)
    owner = torch.full((total,), -1, dtype=torch.long, device=device)
    positions = torch.arange(total, device=device)
    starts = []
    s = prompt_len
    for c, k in enumerate(cand_lens):
        owner[s:s + k] = c
        positions[s:s + k] = torch.arange(prompt_len, prompt_len + k, device=device)
        starts.append(s)
        s += k
    return owner, positions, starts


def packed_masks(owner: torch.Tensor, positions: torch.Tensor,
                 sliding_window: "int | None") -> "dict[str, torch.Tensor]":
    """Boolean (T, T) masks, True = query row may attend to key column.

    full:    the prompt causally (by position), plus the earlier-or-same tokens
             of the query's own candidate.
    sliding: full AND within the window, ``q_pos - k_pos < sliding_window``
             (transformers' sliding_window_overlay, applied to positions).
    """
    idx = torch.arange(owner.numel(), device=owner.device)
    q_pos, k_pos = positions[:, None], positions[None, :]
    see_prompt = (owner[None, :] < 0) & (k_pos <= q_pos)
    see_own = ((owner[None, :] == owner[:, None]) & (owner[:, None] >= 0)
               & (idx[None, :] <= idx[:, None]))
    masks = {"full_attention": see_prompt | see_own}
    if sliding_window is not None:
        masks["sliding_attention"] = masks["full_attention"] & (q_pos - k_pos < int(sliding_window))
    return masks


def _format_mask(mask: torch.Tensor, impl: str, dtype: torch.dtype) -> torch.Tensor:
    mask = mask[None, None]                       # (1, 1, T, T)
    if impl == "sdpa":
        return mask                               # SDPA: bool, True = attend
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~mask, torch.finfo(dtype).min)   # eager: additive


def score_candidates_packed(model, prompt_ids: torch.Tensor,
                            cand_ids_list: Sequence[torch.Tensor],
                            with_grad: bool) -> "tuple[torch.Tensor, torch.Tensor]":
    """(cand_log_probs (C,) float32, pooled (1, H) float32), as the batched path.

    ``prompt_ids`` are the real (unpadded) prompt token ids; candidate c's
    log-prob is the sum of its tokens' log-probs given the prompt.
    """
    cfg = _text_config(model)
    device = prompt_ids.device
    prompt_ids = prompt_ids.reshape(-1)
    tails = [ids.to(device).reshape(-1) for ids in cand_ids_list]
    lens = [int(t.numel()) for t in tails]
    L = int(prompt_ids.numel())

    owner, positions, starts = packed_layout(L, lens, device=device)
    layer_types = set(getattr(cfg, "layer_types", None) or ["full_attention"])
    sliding = "sliding_attention" in layer_types
    masks = packed_masks(owner, positions, cfg.sliding_window if sliding else None)
    impl = getattr(cfg, "_attn_implementation", None) or "eager"
    dtype = model.get_input_embeddings().weight.dtype
    if sliding:
        attention_mask = {k: _format_mask(m, impl, dtype) for k, m in masks.items()}
    else:
        attention_mask = _format_mask(masks["full_attention"], impl, dtype)

    # Logits are needed only where a candidate token is predicted: the last
    # prompt position (every candidate's first token) and each candidate
    # token but its own last. Computing the rest over a 262k vocab is waste.
    keep = [L - 1]
    row_of = {}
    for c, k in enumerate(lens):
        for j in range(1, k):
            row_of[(c, j)] = len(keep)
            keep.append(starts[c] + j - 1)
    keep_t = torch.tensor(keep, dtype=torch.long, device=device)

    input_ids = torch.cat([prompt_ids] + tails).unsqueeze(0)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=positions.unsqueeze(0),
                      output_hidden_states=True, use_cache=False)
        try:
            out = model(**kwargs, logits_to_keep=keep_t)
        except TypeError:                          # no logits_to_keep support
            out = model(**kwargs)
        logits = out.logits[0]
        if logits.shape[0] != len(keep):
            logits = logits[keep_t]

        lse = torch.logsumexp(logits.float(), dim=-1)
        rows, targets, who = [], [], []
        for c, ids in enumerate(tails):
            for j in range(lens[c]):
                rows.append(0 if j == 0 else row_of[(c, j)])
                targets.append(int(ids[j]))
                who.append(c)
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=device)
        who_t = torch.tensor(who, dtype=torch.long, device=device)
        token_logp = logits[rows_t, targets_t].float() - lse[rows_t]
        cand_log_probs = torch.zeros(len(tails), dtype=torch.float32, device=device)
        cand_log_probs = cand_log_probs.index_add(0, who_t, token_logp)

        pooled = out.hidden_states[-1][0:1, L - 1, :].float()
    return cand_log_probs, pooled
