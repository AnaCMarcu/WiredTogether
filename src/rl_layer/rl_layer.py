"""RLLayer — modular RL wrapper for MindForge agents.

Sits between MindForge's belief/context pipeline and the environment.
When ``config.enabled`` is True it replaces ActionSelection's LLM call
with a LoRA-adapted local model + classification head, collects
trajectories, and periodically runs PPO updates (per-agent IPPO baseline,
or true MAPPO when a ``CentralizedCritic`` is provided).

When ``config.enabled`` is False every public method is a fast no-op,
so the rest of the system is unaffected.

Most of the heavy lifting lives in sibling modules:
- ``rl_layer.heads``        — RunningMeanStd, ActionHead, ValueHead
- ``rl_layer.ppo_update``   — body of update() (GAE, social replay, PPO loop)
- ``rl_layer.token_opt``    — agent-decided token-level fine-tuning
- ``rl_layer.persistence``  — save/load checkpoint helpers
- ``rl_layer.ippo``         — per-mini-batch policy + value losses
- ``rl_layer.centralized_critic`` — shared MAPPO critic
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

import torch.nn.functional as F

from rl_layer.config import RLConfig
from rl_layer.heads import RunningMeanStd, ValueHead
from rl_layer.trajectory_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class RLLayer:
    """Modular RL layer — drop-in alongside MindForge agents.

    Parameters
    ----------
    config : RLConfig
        Full configuration (see config.py).
    role : str
        This agent's role name (``"gatherer"``, ``"hunter"``, ``"defender"``).
        Determines which LoRA adapter is loaded/saved.
    agent_id : int
        Numeric agent identifier.

    Notes
    -----
    The base LLM is **shared across all RLLayer instances** within a
    process: the first construction loads ``AutoModelForCausalLM`` and
    wraps it as a multi-adapter PeftModel; subsequent constructions reuse
    that single base and register their own LoRA adapter on it via
    ``model.add_adapter(...)``. Each agent activates its own adapter
    before every forward pass (``_score_actions`` / ``_encode_prompt``)
    via ``model.set_adapter(...)``, and the per-agent optimizer is built
    from parameters whose name contains its adapter token, so
    ``agent_i.optimizer.step()`` only mutates adapter_i's LoRA weights +
    its own value head.

    This brings 3-agent Qwen-9B from 3 × 18 GB (OOM on V100) to
    1 × 18 GB + 3 × ~1 MB LoRA + 3 small value heads — well within the
    32 GB ceiling.
    """

    # ── Process-wide shared base model + tokenizer (class state) ──
    # First constructor populates these; subsequent constructors reuse them.
    # Keyed loosely by model_path so a config switch within the same
    # process triggers a clean reload (defensive — we don't actually do
    # this in production but it prevents stale state in tests).
    _shared_model = None
    _shared_tokenizer = None
    _shared_model_path: Optional[str] = None

    def __init__(self, config: RLConfig, role: str, agent_id: int,
                 centralized_critic: "Optional[object]" = None):
        """When ``centralized_critic`` is provided, the per-agent value head is
        bypassed at update time (MAPPO mode) and GAE uses the critic's V_global
        baseline. The per-agent value head is still constructed for
        compatibility with the IPPO ``critic_mode='independent'`` config.
        """
        self.config = config
        self.role = role
        self.agent_id = agent_id
        self.step_count = 0
        self._update_count = 0
        self.centralized_critic = centralized_critic
        # Effective mode: centralised iff a critic was passed AND config says so.
        self._use_centralized = (
            centralized_critic is not None
            and getattr(config, "critic_mode", "centralized") == "centralized"
        )

        if not config.enabled:
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = getattr(torch, config.dtype)

        # ── Acquire shared base model + tokenizer (load on first call) ──
        self.tokenizer = self._get_or_load_shared_base(config)
        self.model = RLLayer._shared_model

        # ── Register or look up this agent's LoRA adapter on the shared base ──
        adapter_name = role if config.lora_per_role else "shared"
        self._adapter_name = adapter_name
        self._ensure_adapter(adapter_name, config)
        # Activate this adapter on the shared model so any immediate forward
        # uses the right LoRA. Every public forward in this class re-sets
        # before computing (cheap; idempotent if already active), so cross-
        # agent calls in the main loop don't corrupt each other.
        self.model.set_adapter(adapter_name)

        # Gradient checkpointing: recompute activations during backward instead of
        # storing them all live.  Reduces peak VRAM by ~40-50% at ~33% compute cost.
        # Apply only once per shared model — idempotent enable is safe.
        if config.gradient_checkpointing and not getattr(
            RLLayer._shared_model, "_grad_ckpt_enabled", False
        ):
            RLLayer._shared_model.gradient_checkpointing_enable()
            RLLayer._shared_model._grad_ckpt_enabled = True
            logger.info("RLLayer: gradient checkpointing enabled on shared base")

        # ── Value head — float32 ──
        # pooled hidden states are upcast to float32 (in _encode_prompt and
        # ippo.action_level_ppo_step) to prevent NaN from fp16 overflow, so
        # the head must also be float32. (ActionHead was removed: the policy
        # is now pi(a|s) = softmax over LLM sequence-log-probs of valid
        # action strings, computed in _score_actions — see GLAM-style
        # constrained generation. No classifier head needed.)
        hidden_size = self.model.config.hidden_size
        self.value_head = ValueHead(hidden_size, config.value_hidden).to(device=self._device, dtype=torch.float32)

        # ── Optimizer: ONLY this agent's LoRA adapter + value head ──
        # The shared model holds adapters for every agent; we must select
        # only this agent's params so agent_i.optimizer.step() doesn't
        # walk onto agent_j's gradients (zero anyway, but Adam moments
        # would still get a "saw this param" entry).
        # PEFT names LoRA params like
        #   base_model.model.[...].lora_A.<adapter_name>.weight
        # so a substring match on ".<adapter_name>." is exact.
        adapter_tag = "." + adapter_name + "."
        adapter_params = [
            p for n, p in self.model.named_parameters()
            if adapter_tag in n and p.requires_grad
        ]
        trainable = adapter_params + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(trainable, lr=config.lr)
        logger.info(
            "RLLayer agent %d: optimizer over %d LoRA params (adapter='%s') + value head",
            agent_id, len(adapter_params), adapter_name,
        )

        # ── Rollout buffer ──
        self.buffer = RolloutBuffer(max_size=config.buffer_size)

        # ── Action-index mapping (canonical, indexes into config.actions) ──
        self._action_to_idx = {a: i for i, a in enumerate(config.actions)}
        self._idx_to_action = {i: a for i, a in enumerate(config.actions)}

        # ── Candidate set: actions minus Slot* (the only masking) ──
        # Membership in this set IS the masking — no separate _action_mask
        # tensor anymore. Stored as an ordered tuple so the candidate-index
        # used by Categorical is stable across select_action and the PPO
        # update path.
        self._candidate_actions: tuple = self._build_candidate_actions()
        # Reverse map: index into config.actions → index within candidate set
        # (or -1 if masked out). Used by ippo to look up the action_idx that
        # was stored in the buffer (a config.actions index) inside the new
        # softmax distribution that's defined over the candidate set.
        self._full_idx_to_cand_idx = {
            self._action_to_idx[a]: c
            for c, a in enumerate(self._candidate_actions)
        }
        logger.info(
            "RLLayer: candidate action set = %d / %d total (mask_slot_actions=%s)",
            len(self._candidate_actions), len(config.actions),
            getattr(config, "mask_slot_actions", False),
        )

        # ── Pre-tokenize candidate action strings (one tensor of ids per
        #    candidate). Done once at init so the SAME token sequence is
        #    used for scoring at sampling time AND at update time — this
        #    is the consistency that makes ratio = exp(new_lp − old_lp)
        #    equal 1.0 on the first PPO epoch (sanity check in ppo_update).
        #
        # We tokenize each candidate as " <name>" (leading space) for two
        # reasons:
        #  1. BPE tokenizers (Qwen, LLaMA, GPT2) emit DIFFERENT tokens for
        #     "Dig" at sentence start vs " Dig" mid-sentence. The natural
        #     continuation of a prompt is the mid-sentence variant, so
        #     using the leading-space form makes log-probs comparable
        #     across single-token and multi-token candidates.
        #  2. Without the leading space, short common words like "Sneak"
        #     / "Jump" / "Drop" / "TurnRight" get artificially high
        #     log-probs (they match the LLM's natural sentence-start
        #     prior), while multi-token / less common words ("MoveForward",
        #     "MoveBackward") get penalised — observed in the first
        #     test run as the agent only picking short words.
        # add_special_tokens=False keeps us from adding BOS/EOS tokens
        # that would also drift the encoding.
        self._candidate_token_ids: tuple = tuple(
            torch.tensor(
                self.tokenizer(" " + a, add_special_tokens=False)["input_ids"],
                dtype=torch.long, device=self._device,
            )
            for a in self._candidate_actions
        )

        # ── Reward normaliser ──
        self._reward_rms = RunningMeanStd() if config.normalize_rewards else None

        # ── Recent outcomes (for token-opt trigger) ──
        self._recent_successes: List[bool] = []
        self._recent_actions: List[str] = []
        self._recent_rewards: List[float] = []
        self._current_task: str = "Explore"
        self._last_token_opt_step: int = 0  # cooldown tracking

    # ──────────────────────────────────────────────
    # Public API (called by CustomAgent)
    # ──────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def select_action(
        self,
        prompt_text: str,
        thoughts_prefix: Optional[str] = None,
    ) -> Optional[Dict]:
        """Run the LoRA-adapted model and return an action dict.

        Returns ``None`` if RL is disabled or mode is "token" (caller falls
        back to vanilla LLM).  In "token" mode the RL layer is still active
        for token-level optimisation but does not override action selection.

        The actor is LLM constrained-generation: each candidate action
        string is scored by summing the model's token log-probabilities of
        emitting that string as a continuation of the prompt. We sample a
        Categorical over those scores. No classifier head is involved.

        When ``thoughts_prefix`` is given (a one-sentence rationale produced
        by an EARLIER LLM call), it is appended to the scoring prompt so
        the policy is ``p(action | prompt, thoughts)`` — i.e. thoughts
        DRIVE the action, not the other way around. The augmented prompt
        is what we store in the buffer; the PPO update re-scores with the
        same augmented prompt so the ratio stays consistent.

        Returns
        -------
        dict with keys ``action``, ``thoughts``, ``communication``
        """
        if not self.config.enabled:
            return None
        if self.config.mode == "token":
            return None  # token-opt only — let LLM choose actions

        if thoughts_prefix:
            # Append the LLM's pre-action reasoning to the prompt so the
            # constrained scorer conditions on it. The "Action:" suffix is
            # a soft prefix that biases the model toward emitting an
            # action token next — this is the same shape as p2 prompts
            # use, just without the JSON wrapper (the actual action
            # selection is the argmax over the candidate set).
            scoring_prompt = (
                f"{prompt_text}\n\nThoughts: {thoughts_prefix}\nAction:"
            )
        else:
            scoring_prompt = prompt_text

        self.model.eval()
        with torch.no_grad():
            cand_logp, pooled = self._score_actions(
                scoring_prompt, self._candidate_actions, with_grad=False,
            )
            # Skip the per-agent value head when the centralised critic is in
            # charge — the main loop will populate old_value_global + joint_state
            # via set_pending_value_global() after all agents have acted.
            if self._use_centralized:
                value_scalar = 0.0
            else:
                value = self.value_head(pooled).squeeze(-1)  # (1,)
                value_scalar = value.item()

            # Categorical over the candidate set. The softmax in
            # Categorical renormalises the summed-log-prob scores into a
            # proper distribution.
            dist = torch.distributions.Categorical(logits=cand_logp)
            cand_idx = dist.sample()              # scalar tensor in [0, C)
            log_prob = dist.log_prob(cand_idx)    # scalar tensor

        cand_idx_int = int(cand_idx.item())
        action_name = self._candidate_actions[cand_idx_int]
        # The buffer stores the index into config.actions (the canonical
        # space), NOT the candidate-set index. ippo's action_level_ppo_step
        # uses _full_idx_to_cand_idx to round-trip back to the candidate
        # softmax. This keeps existing Transition fields untouched.
        full_action_idx = self._action_to_idx[action_name]

        # Store the AUGMENTED prompt (with thoughts) so the PPO update
        # re-scores against the same context. If we stored the bare
        # prompt_text here, the update's _score_actions would see no
        # thoughts and the ratio would drift away from 1.0 on epoch 1.
        self.buffer.store_action(
            prompt_text=scoring_prompt,
            action_idx=full_action_idx,
            log_prob=log_prob.item(),
            value=value_scalar,
        )
        self.step_count += 1

        logger.info("RLLayer step=%d action=%s prompt:\n%s", self.step_count, action_name, scoring_prompt)

        # Communication + thoughts are produced by an EARLIER LLM call
        # (action_selection.generate_thoughts_and_comm) before this method
        # runs; the caller threads `thoughts_prefix` into our scoring
        # prompt so the policy is conditioned on them. The empty strings
        # here are structural placeholders the caller overwrites with the
        # real values before the main loop reads them.
        return {
            "action": action_name,
            "thoughts": thoughts_prefix or "",
            "communication": "",
        }

    def get_pending_value(self) -> Optional[float]:
        """Return V(s_t) from the pending transition (stored during select_action).

        Used to compute a one-step advantage estimate δ_t = r_t - V(s_t) before
        store_reward() is called, so Hebbian updates can use per-agent advantages
        at the same step rather than one step behind. In centralised-critic mode
        this returns V_global if it has been attached, otherwise the per-agent
        value (which is 0 when the value head was skipped).

        Returns None if RL is disabled, mode is 'token', or no action has been
        selected yet this step.
        """
        if not self.config.enabled or self.config.mode == "token":
            return None
        pending = self.buffer._pending
        if pending is None:
            return None
        if pending.old_value_global is not None:
            return pending.old_value_global
        return pending.old_value

    def set_pending_value_global(self, value_global: float,
                                 joint_state=None) -> None:
        """Attach the centralised critic's V_global (and the joint state it was
        computed from) to the currently-pending transition. No-op when RL is
        disabled, in token mode, or when no critic was provided."""
        if not self.config.enabled or self.config.mode == "token":
            return
        if not self._use_centralized:
            return
        self.buffer.set_pending_value_global(value_global, joint_state)

    def store_reward(self, reward: float, done: bool = False,
                     reward_task: float = 0.0, reward_comm: float = 0.0) -> None:
        """Feed the environment reward back into the buffer.

        Applies up to two transforms before storage:
        1. Death penalty: subtracts ``config.death_penalty`` on termination so
           the value head learns that dying is worse than running out of steps.
        2. Reward normalisation: scales by running 1/std so the 0.1–2048 range
           the environment produces maps to roughly unit variance, stabilising
           value function learning.

        The reward normalisation is **disabled** when a centralised critic is
        active (MAPPO mode). The centralised critic is trained against the
        raw team-mean reward (multi_agent_craftium.py Phase 3a), so V_global
        learns on the raw scale. If we normalised the per-agent reward here,
        GAE on this buffer would mix scales:
            δ_t = r_normalised + γ · V_global_raw − V_global_raw
        and the V_global terms would drown out the reward signal. With
        normalisation off in this mode, both streams are raw and GAE is
        consistent. Per-rollout advantage normalisation in compute_gae() still
        makes the policy gradient see unit-variance advantages.

        In ``critic_mode='independent'`` (legacy IPPO), normalisation stays on
        because the per-agent value head IS trained against ``tr.returns``, and
        keeping returns near unit variance avoids the value-clip / value-coef
        coupling problems that motivated normalisation in the first place.
        """
        if not self.config.enabled:
            return
        if done and self.config.death_penalty != 0.0:
            reward += self.config.death_penalty
        if self._reward_rms is not None and not self._use_centralized:
            self._reward_rms.update(reward)
            reward = self._reward_rms.normalize(reward)
        self.buffer.store_reward(reward, done,
                                 reward_task=reward_task, reward_comm=reward_comm)

    def record_success(self, success: bool) -> None:
        """Track critic success/failure for token-opt self-trigger."""
        if not self.config.enabled:
            return
        self._recent_successes.append(success)
        if len(self._recent_successes) > self.config.token_opt_window:
            self._recent_successes.pop(0)

    def record_context(self, action: str, reward: float, task: str) -> None:
        """Track recent actions/rewards/task for the learning belief prompt."""
        if not self.config.enabled:
            return
        self._recent_actions.append(action)
        self._recent_rewards.append(reward)
        self._current_task = task
        if len(self._recent_actions) > self.config.token_opt_window:
            self._recent_actions.pop(0)
        if len(self._recent_rewards) > self.config.token_opt_window:
            self._recent_rewards.pop(0)

    def should_update(self) -> bool:
        """True when enough steps have been collected for a MAPPO update."""
        return (
            self.config.enabled
            and len(self.buffer) >= self.config.update_interval
        )

    def update(self, neighbour_buffers: Optional[Dict[int, "RolloutBuffer"]] = None,
               hebbian_graph=None) -> Dict:
        """Run a full PPO update. See ``rl_layer.ppo_update.run_ppo_update``."""
        from rl_layer.ppo_update import run_ppo_update
        return run_ppo_update(self, neighbour_buffers, hebbian_graph)

    async def maybe_token_optimize(self, cancellation_token=None,
                                    hebbian_graph=None) -> Optional[Dict]:
        """Agent-decided token-level fine-tuning.
        See ``rl_layer.token_opt.maybe_token_optimize``.
        """
        from rl_layer.token_opt import maybe_token_optimize
        return await maybe_token_optimize(self, cancellation_token, hebbian_graph)

    # ── Persistence ──

    def save(self, path: Optional[str] = None) -> None:
        """Save LoRA + heads + optimizer + RMS state. See ``rl_layer.persistence.save_rl_layer``."""
        from rl_layer.persistence import save_rl_layer
        save_rl_layer(self, path)

    def load(self, path: Optional[str] = None) -> None:
        """Restore heads + optimizer + RMS state. See ``rl_layer.persistence.load_rl_layer``."""
        from rl_layer.persistence import load_rl_layer
        load_rl_layer(self, path)

    # ── Internal ──

    @classmethod
    def _get_or_load_shared_base(cls, config: RLConfig):
        """Return the process-wide tokenizer, loading the base LLM lazily.

        The base ``AutoModelForCausalLM`` and tokenizer are cached on the
        class so every RLLayer in this process points at the SAME base
        weights (3 × 18 GB → 1 × 18 GB for Qwen-9B on a V100). The first
        construction does the heavy load + wraps the base as a PeftModel
        with a placeholder adapter; subsequent constructions call
        ``_ensure_adapter`` to register their own LoRA on the shared base.
        """
        if (
            cls._shared_model is not None
            and cls._shared_tokenizer is not None
            and cls._shared_model_path == config.model_path
        ):
            return cls._shared_tokenizer

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        logger.info("RLLayer: loading shared tokenizer from %s", config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        logger.info("RLLayer: loading shared base model from %s", config.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=getattr(torch, config.dtype),
            trust_remote_code=True,
        ).to(device)

        # Wrap in a PeftModel with a SENTINEL adapter named "__bootstrap__".
        # Real per-agent adapters are added later via _ensure_adapter; the
        # bootstrap adapter exists so get_peft_model has something to
        # initialise around (PeftModel requires at least one adapter at
        # construction time). It's never activated for inference/update.
        boot_lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base, boot_lora, adapter_name="__bootstrap__")
        cls._shared_model = peft_model
        cls._shared_tokenizer = tokenizer
        cls._shared_model_path = config.model_path
        return tokenizer

    def _ensure_adapter(self, adapter_name: str, config: RLConfig) -> None:
        """Register this agent's LoRA on the shared base if not already present.

        If a saved adapter exists at ``<lora_save_dir>/<adapter_name>/``,
        load it via ``model.load_adapter``. Otherwise add a fresh LoRA
        with ``model.add_adapter``. After this call the adapter is
        registered on the shared model and can be activated via
        ``model.set_adapter(adapter_name)``.

        Idempotent: re-calling with an already-registered name is a no-op
        (handles the test/restart paths where the same class state
        survives across RLLayer constructions).
        """
        from peft import LoraConfig

        model = RLLayer._shared_model
        # PEFT keeps a dict of adapter configs; presence == registered.
        if adapter_name in getattr(model, "peft_config", {}):
            return

        adapter_path = Path(config.lora_save_dir) / adapter_name
        if (adapter_path / "adapter_config.json").exists():
            logger.info(
                "RLLayer agent %d: loading existing LoRA adapter '%s' from %s",
                self.agent_id, adapter_name, adapter_path,
            )
            model.load_adapter(str(adapter_path), adapter_name=adapter_name)
        else:
            logger.info(
                "RLLayer agent %d: initialising new LoRA adapter '%s'",
                self.agent_id, adapter_name,
            )
            lora_cfg = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter(adapter_name, lora_cfg)

    def _build_candidate_actions(self) -> tuple:
        """Return the ordered tuple of action strings the policy can emit.

        Excludes Slot* actions when config.mask_slot_actions is True. This
        replaces the old _action_mask tensor — membership in this tuple IS
        the masking. The ordering matches the iteration order of
        config.actions, so candidate_set[c] gives a deterministic mapping
        into the canonical action space via _action_to_idx.
        """
        mask_slot = getattr(self.config, "mask_slot_actions", False)
        return tuple(
            a for a in self.config.actions
            if not (mask_slot and a.startswith("Slot"))
        )

    def _score_actions(
        self,
        prompt_text: str,
        candidate_strings,
        with_grad: bool,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Sum the LLM's token log-probs of each candidate as a prompt continuation.

        Returns (cand_log_probs (C,) float32, pooled_hidden (1, H) float32).

        Implementation: ONE batched forward over [prompt + cand_i] rows
        (right-padded), then read per-token log-probs at the candidate
        positions. The KV-cached approach (one prompt forward + per-
        candidate tail forwards using past_key_values) was abandoned
        because gradient checkpointing silently demotes use_cache=True
        to False, returning past_kv=None and making the tail forwards
        score the candidate WITHOUT prompt context — broken scores for
        multi-token candidates plus extra graph state in the per-
        candidate loop that OOM'd the GPU.

        The full-forward version is correct under any combination of
        eval / train mode and grad_ckpt on/off. Per-candidate cost is
        O(L+K) per row; batched as (C, L+K_max) it's a single forward
        pass through the model + LoRA. With C ≈ 22 candidates and
        K_cand ≤ 3, the batched activation footprint is ~C × L × H,
        which fits when grad_ckpt is enabled (the typical setup).

        Args
        ----
        prompt_text : str
            The full LLM prompt.
        candidate_strings : sequence of str
            Action strings to score. Each must already be in
            self._candidate_actions so its pre-tokenized IDs are cached.
        with_grad : bool
            True at update time (gradient flows through LoRA → log-prob
            extraction → ratio). False at sampling time.
        """
        # Resolve pre-tokenized candidate IDs from the cache. Re-tokenizing
        # at every call would risk drift between sampling and update.
        cand_ids_list = []
        for s in candidate_strings:
            cand_idx_in_full = self._candidate_actions.index(s)
            cand_ids_list.append(self._candidate_token_ids[cand_idx_in_full])

        # Activate THIS agent's LoRA adapter. The shared base model carries
        # an adapter per agent; without this set_adapter call a forward
        # could end up using whichever adapter the LAST agent in the main
        # loop activated, producing inconsistent logits across calls.
        self.model.set_adapter(self._adapter_name)

        # Tokenize the prompt ONCE.
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.rl_prompt_max_tokens,
        ).to(self._device)
        prompt_ids = enc.input_ids[0]                          # (L_pad,) — may include left-pad
        prompt_attn = enc.attention_mask[0]                    # (L_pad,)
        # Strip any left-padding so concatenation with candidates lands at
        # contiguous real positions [0, L). With padding_side="left", real
        # tokens are right-aligned in the encoded tensor; we keep only
        # those.
        real_mask = prompt_attn.bool()
        prompt_ids_real = prompt_ids[real_mask]                # (L,)
        prompt_len = int(prompt_ids_real.numel())

        # Build right-padded batch [prompt + cand_i + pad] per row.
        C = len(cand_ids_list)
        seq_lens = [prompt_len + int(ids.numel()) for ids in cand_ids_list]
        max_len = max(seq_lens) if seq_lens else prompt_len
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        batch_ids = torch.full(
            (C, max_len), pad_id, dtype=torch.long, device=self._device,
        )
        batch_attn = torch.zeros(
            (C, max_len), dtype=torch.long, device=self._device,
        )
        for ci, ids in enumerate(cand_ids_list):
            row = torch.cat([prompt_ids_real, ids.to(self._device)], dim=0)
            L_row = int(row.numel())
            batch_ids[ci, :L_row] = row
            batch_attn[ci, :L_row] = 1

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            out = self.model(
                input_ids=batch_ids,
                attention_mask=batch_attn,
                output_hidden_states=True,
                use_cache=False,
            )

            # ── Slice + gather + logsumexp (avoid materializing softmax) ──
            # We only need log-probs at the positions that predict
            # candidate tokens. Slicing the window cuts the V-dimensional
            # tensor down to (C, K_max, V); using gather+logsumexp instead
            # of log_softmax+index means we never RETAIN a full-vocab
            # softmax output in the autograd graph for backward. Memory
            # footprint for the log-prob computation goes from
            # (C, max_len, V) ≈ 6.87 GB to (C, K_max) ≈ <1 KB live across
            # backward; the V-sized intermediates inside logsumexp are
            # temporary and freed before backward.
            #
            # log p(tok | context) = logit[tok] − logsumexp(logits, dim=-1)
            K_max = max((int(ids.numel()) for ids in cand_ids_list), default=1)
            window_start = prompt_len - 1
            window_end = window_start + K_max
            window_logits = out.logits[:, window_start:window_end, :]  # (C, K_max, V) — model dtype

            # Build the (C, K_max) target tensor and a mask for padded
            # candidate positions (when K_i < K_max).
            target_ids = torch.zeros((C, K_max), dtype=torch.long, device=self._device)
            mask = torch.zeros((C, K_max), dtype=torch.bool, device=self._device)
            for ci, ids in enumerate(cand_ids_list):
                K = int(ids.numel())
                if K > 0:
                    target_ids[ci, :K] = ids.to(self._device)
                    mask[ci, :K] = True

            # Gather logit at each (candidate, position) → (C, K_max)
            gathered = window_logits.gather(
                dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            # logsumexp over the vocab dim (fp32 for stability) → (C, K_max)
            lse = torch.logsumexp(window_logits.float(), dim=-1)
            # Per-token log-prob; masked-fill pad positions to 0 so they
            # contribute nothing to the sum below.
            log_p = (gathered.float() - lse).masked_fill(~mask, 0.0)
            cand_log_probs = log_p.sum(dim=1).to(torch.float32)  # (C,)

            # Pool last-prompt-position hidden state for the value head.
            # All rows share the same prompt at positions [0, L), so row 0
            # is canonical.
            last_hidden = out.hidden_states[-1]                # (C, max_len, H)
            pooled = last_hidden[0:1, prompt_len - 1, :].float()  # (1, H)

            # OPTIONAL length-normalisation hook (default OFF — sequence
            # length is constant within candidate set for short action
            # strings, so this rarely matters; flip via config when the
            # action vocab gets longer/shorter mixes).
            if getattr(self.config, "length_normalize_action_logp", False):
                lengths = torch.tensor(
                    [max(1, int(ids.numel())) for ids in cand_ids_list],
                    dtype=torch.float32, device=self._device,
                )
                cand_log_probs = cand_log_probs / lengths

        return cand_log_probs, pooled

    def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
        """Tokenize prompt and return pooled last-token hidden state (1, H).

        Kept for the GAE bootstrap path (`ppo_update._bootstrap_last_value`)
        which only needs the value head, not the candidate scores.
        """
        self.model.set_adapter(self._adapter_name)
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.rl_prompt_max_tokens,
        ).to(self._device)
        outputs = self.model(**enc, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (1, L, H)
        seq_len = enc.attention_mask.sum(dim=1) - 1  # (1,)
        return last_hidden[:, seq_len.item(), :].float()  # (1, H) — upcast to fp32 to prevent NaN from fp16 overflow
