"""GRPOLanguageModel, ReferencePolicy, LLMPolicy — torch/PEFT-dependent.

Three concerns, one file because they share the underlying PEFT model:

* ``GRPOLanguageModel`` — wraps a HuggingFace causal-LM with two LoRA
  adapters loaded simultaneously (``grpo_policy`` + ``grpo_reference``),
  using the same ``adapter_name=`` mechanism the legacy stack uses at
  ``src/rl_layer/rl_layer.py:77-119``. Exposes ``generate()`` for rollouts
  and ``logprobs()`` for both the surrogate ratio (policy adapter active)
  and the KL term (reference adapter active).

* ``ReferencePolicy`` — owns the frozen reference adapter and exposes
  ``compute_kl(prompt, response_tokens)``. The reference adapter is
  snapshotted from the policy adapter at construction time so they share
  the same starting point.

* ``LLMPolicy`` — concrete implementation of ``rollout_sampler.Policy``.
  Calls ``model.generate()``, parses the JSON, returns
  ``(action_dict, RolloutTensors)``.

Everything in this module requires torch + transformers + peft at runtime.
The minimal-dev env has none of those, so most of the code can only be
verified on HPC. The interfaces are designed to be mockable for unit tests
(see ``tests/rlvr/test_reference_policy.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

from rlvr.action_parser import parse_action_json
from rlvr.rollout_sampler import RolloutTensors

if TYPE_CHECKING:
    import torch
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ──── prompt template ──────────────────────────────────────────────────


class PromptTemplate(Protocol):
    """Build the LLM prompt string from an env step's observation + info.

    Stage 2 ships a minimal default; the entry point can pass a richer
    template (chamber narration, comm history, belief state) via the YAML
    config.
    """

    def __call__(self, observation: object, info: dict) -> str: ...


def default_prompt(observation: object, info: dict) -> str:
    """A tiny default. Real runs override this in the entry point."""
    chamber = info.get("chamber", "unknown")
    hp = info.get("hp", "?")
    return (
        f"You are an agent in Five Chambers. Chamber: {chamber}. HP: {hp}.\n"
        f'Output JSON: {{"action": <name>, "communication_target": <int|null>, '
        f'"thoughts": <str>}}.\n'
    )


# ──── model wrapper ────────────────────────────────────────────────────


@dataclass
class GRPOModelConfig:
    base_model_name: str = "MUST_BE_SET_IN_YAML"
    """HuggingFace ID or local path. The sentinel default lets the config
    dataclass be constructed with no args (for test-only paths that never
    load the model); a real run will fail clearly when the loader hits the
    sentinel."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    policy_adapter: str = "grpo_policy"
    reference_adapter: str = "grpo_reference"

    device: str = "cuda"
    """Set to ``"cpu"`` for small-scale smoke tests."""

    max_new_tokens: int = 256
    """Maximum tokens generated per env step's action."""

    temperature: float = 1.0
    """Sampling temperature for ``generate``. The trainer relies on the
    sampling being stochastic — a temperature of 0 would make GRPO's
    advantage signal degenerate."""


class GRPOLanguageModel:
    """A HuggingFace causal-LM with two PEFT LoRA adapters loaded
    simultaneously. ``set_active_adapter()`` switches between them.

    Construction order matters:
        1. Load base model + tokenizer
        2. Apply PEFT with ``policy_adapter`` as the first adapter
        3. Apply PEFT again to register ``reference_adapter`` as a
           SECOND adapter
        4. Copy the policy adapter's initial weights into the reference
           adapter so they start identical
        5. Freeze the reference adapter's parameters
    """

    def __init__(self, config: GRPOModelConfig):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        base = base.to(config.device)

        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base, lora_cfg, adapter_name=config.policy_adapter)
        peft_model.add_adapter(config.reference_adapter, lora_cfg)
        self.model: PeftModel = peft_model

        # Copy policy → reference at init so they share a starting point.
        self._snapshot(config.policy_adapter, config.reference_adapter)

        # Freeze reference adapter — only the policy is trainable.
        self._freeze_adapter(config.reference_adapter)

        self.set_active_adapter(config.policy_adapter)

    # ──── adapter management ────────────────────────────────────────

    def set_active_adapter(self, name: str) -> None:
        """Activate ``name`` for forward/backward passes."""
        self.model.set_adapter(name)

    def _snapshot(self, src: str, dst: str) -> None:
        """Copy LoRA weights from ``src`` adapter to ``dst`` adapter."""
        import torch

        with torch.no_grad():
            for module_name, module in self.model.named_modules():
                if not hasattr(module, "lora_A"):
                    continue
                if src in module.lora_A and dst in module.lora_A:
                    module.lora_A[dst].weight.copy_(module.lora_A[src].weight)
                if src in module.lora_B and dst in module.lora_B:
                    module.lora_B[dst].weight.copy_(module.lora_B[src].weight)

    def _freeze_adapter(self, name: str) -> None:
        """Set ``requires_grad=False`` for every parameter in adapter ``name``."""
        for param_name, param in self.model.named_parameters():
            if name in param_name:
                param.requires_grad_(False)

    # ──── generation / scoring ──────────────────────────────────────

    def generate(self, prompt: str) -> tuple[str, "torch.Tensor", "torch.Tensor"]:
        """Generate a completion with the currently-active adapter.

        Returns ``(text, response_token_ids, per_token_logprobs)``. The
        token ids and logprobs cover only the response, not the prompt.
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response_ids = out.sequences[0, prompt_len:]
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        logprobs = _logprobs_from_scores(out.scores, response_ids)
        return text, response_ids, logprobs

    def logprobs(self, prompt: str, response_tokens: "torch.Tensor") -> "torch.Tensor":
        """Per-token logprobs of ``response_tokens`` given ``prompt``, under
        the currently-active adapter.

        Differentiable (no ``torch.no_grad`` context). The trainer relies on
        this for the surrogate-loss backward pass.
        """
        import torch
        import torch.nn.functional as F

        device = self.config.device
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        response_tokens = response_tokens.to(device)
        full = torch.cat([prompt_ids[0], response_tokens], dim=0).unsqueeze(0)
        out = self.model(full)
        logits = out.logits[0]   # (T, vocab)
        # Predicting token t from logits[t-1]. The response starts at
        # prompt_len; the predictor for response[i] is logits[prompt_len-1 + i].
        prompt_len = prompt_ids.shape[1]
        response_logits = logits[prompt_len - 1: -1]   # (R, vocab)
        log_probs = F.log_softmax(response_logits, dim=-1)
        per_token = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        return per_token


# ──── reference policy ─────────────────────────────────────────────────


class ReferencePolicy:
    """Convenience wrapper: activate the reference adapter, compute logprobs
    or KL, restore the previous adapter.

    Stage 2 of the trainer uses ``compute_kl`` per minibatch. The class
    exists to enforce safe adapter switching — never leave the reference
    adapter active accidentally, that would silently disable training.
    """

    def __init__(self, model: GRPOLanguageModel):
        self.model = model

    def compute_kl(
        self,
        prompt: str,
        response_tokens: "torch.Tensor",
        policy_logprobs: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Approximate per-token KL ``logp_policy - logp_reference``.

        If ``policy_logprobs`` is supplied (recomputed by the trainer
        anyway for the surrogate ratio), we skip recomputing under the
        policy adapter — one forward pass instead of two.

        Returns a 1-D tensor of length ``len(response_tokens)`` so the
        trainer can either ``.mean()`` or ``.sum()`` over tokens.
        """
        cfg = self.model.config
        previous = cfg.policy_adapter   # assume the trainer leaves it active

        if policy_logprobs is None:
            self.model.set_active_adapter(cfg.policy_adapter)
            policy_logprobs = self.model.logprobs(prompt, response_tokens)

        try:
            self.model.set_active_adapter(cfg.reference_adapter)
            reference_logprobs = self.model.logprobs(prompt, response_tokens)
        finally:
            # Restore policy adapter — this is the dangerous case where
            # an exception in logprobs would otherwise leave the model
            # in reference-mode for the rest of training.
            self.model.set_active_adapter(previous)

        return policy_logprobs - reference_logprobs


# ──── LLM policy (rollout side) ────────────────────────────────────────


class LLMPolicy:
    """Concrete ``Policy`` for ``RolloutSampler``. Wraps a
    ``GRPOLanguageModel`` (which it does NOT own — the trainer owns it)
    plus a prompt template.

    The active adapter for ``act()`` is whatever is currently set on the
    model, which during training is the policy adapter. The trainer is
    responsible for never leaving the reference adapter active when
    ``act()`` is called.
    """

    def __init__(
        self,
        model: GRPOLanguageModel,
        n_agents: int,
        prompt_template: PromptTemplate | Callable[[object, dict], str] = default_prompt,
        fallback_action: dict | None = None,
    ):
        self.model = model
        self.n_agents = n_agents
        self.prompt_template = prompt_template
        self.fallback_action = fallback_action or {
            "action": "nop",
            "communication_target": None,
            "thoughts": "",
        }

    def act(self, observation: object, info: dict) -> tuple[dict, RolloutTensors]:
        prompt = self.prompt_template(observation, info)
        text, tokens, logprobs = self.model.generate(prompt)
        parsed, _score = parse_action_json(text, self.n_agents)
        # Format reward at score=0.0 means parse failed — fall back to NOP
        # so the env step doesn't crash. The verifier still penalises the
        # failed format via per-step format reward.
        action = parsed if parsed is not None else dict(self.fallback_action)
        return action, RolloutTensors(
            prompt_text=prompt,
            response_tokens=tokens,
            response_logprobs=logprobs,
        )


# ──── helpers ──────────────────────────────────────────────────────────


def _logprobs_from_scores(scores: tuple, response_ids: "torch.Tensor") -> "torch.Tensor":
    """Convert HF ``generate`` per-step logit tuples to per-token logprobs.

    ``scores[t]`` is the logits used to pick ``response_ids[t]``. We
    extract its log-softmax at that token index. Detached from the graph
    because generation runs under ``no_grad``.
    """
    import torch
    import torch.nn.functional as F

    out = []
    for t, logits in enumerate(scores):
        lp = F.log_softmax(logits[0], dim=-1)
        out.append(lp[response_ids[t]].detach())
    return torch.stack(out) if out else torch.empty(0)
