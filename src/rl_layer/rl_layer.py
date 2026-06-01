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
    """

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

        # ── Load tokenizer + base model ──
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, PeftModel

        logger.info("RLLayer: loading tokenizer from %s", config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding is standard for decoder-only (causal) models in batched
        # inference.  It keeps real tokens right-aligned so causal attention
        # naturally ignores the left-side padding without needing a separate mask
        # check.  Combined with explicit pad_token_id on the model config this
        # suppresses the "attention_mask cannot be inferred" warning that fires
        # when pad_token_id == eos_token_id.
        self.tokenizer.padding_side = "left"

        logger.info("RLLayer: loading base model from %s", config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device)

        # ── LoRA adapter ──
        adapter_name = role if config.lora_per_role else "shared"
        adapter_path = Path(config.lora_save_dir) / adapter_name

        if (adapter_path / "adapter_config.json").exists():
            logger.info("RLLayer: loading existing LoRA adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), adapter_name=adapter_name,
            )
        else:
            logger.info("RLLayer: initialising new LoRA adapter '%s'", adapter_name)
            lora_cfg = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj"],  # standard for most LLMs
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg, adapter_name=adapter_name)

        self.model.print_trainable_parameters()
        self._adapter_name = adapter_name

        # Gradient checkpointing: recompute activations during backward instead of
        # storing them all live.  Reduces peak VRAM by ~40-50% at ~33% compute cost.
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("RLLayer: gradient checkpointing enabled")

        # ── Value head — float32 ──
        # pooled hidden states are upcast to float32 (in _encode_prompt and
        # ippo.action_level_ppo_step) to prevent NaN from fp16 overflow, so
        # the head must also be float32. (ActionHead was removed: the policy
        # is now pi(a|s) = softmax over LLM sequence-log-probs of valid
        # action strings, computed in _score_actions — see GLAM-style
        # constrained generation. No classifier head needed.)
        hidden_size = self.model.config.hidden_size
        self.value_head = ValueHead(hidden_size, config.value_hidden).to(device=self._device, dtype=torch.float32)

        # ── Optimizer (LoRA adapter + value head only) ──
        # ActionHead's parameters are gone; the LoRA adapter alone parametrises
        # the actor (via the LLM's own logits at the candidate-action tokens).
        # Net VRAM: <= the head-based path because (a) no head params + optim
        # state and (b) candidate scoring uses no_grad sampling + tiny 1-2
        # token incremental forwards off a reused prompt-KV cache.
        trainable = (
            list(filter(lambda p: p.requires_grad, self.model.parameters()))
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(trainable, lr=config.lr)

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
        # We tokenize without special tokens. Whether the BPE tokenizer
        # prefixes a leading-space marker depends on the model; using the
        # same `add_special_tokens=False` call at score time keeps the
        # encoding deterministic.
        self._candidate_token_ids: tuple = tuple(
            torch.tensor(
                self.tokenizer(a, add_special_tokens=False)["input_ids"],
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

    def select_action(self, prompt_text: str) -> Optional[Dict]:
        """Run the LoRA-adapted model and return an action dict.

        Returns ``None`` if RL is disabled or mode is "token" (caller falls
        back to vanilla LLM).  In "token" mode the RL layer is still active
        for token-level optimisation but does not override action selection.

        The actor is now LLM constrained-generation: each candidate action
        string is scored by summing the model's token log-probabilities of
        emitting that string as a continuation of the prompt. We sample a
        Categorical over those scores. No classifier head is involved.

        Returns
        -------
        dict with keys ``action``, ``thoughts``, ``communication``
        """
        if not self.config.enabled:
            return None
        if self.config.mode == "token":
            return None  # token-opt only — let LLM choose actions

        self.model.eval()
        with torch.no_grad():
            cand_logp, pooled = self._score_actions(
                prompt_text, self._candidate_actions, with_grad=False,
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

        # Store in buffer (reward comes later via store_reward)
        self.buffer.store_action(
            prompt_text=prompt_text,
            action_idx=full_action_idx,
            log_prob=log_prob.item(),
            value=value_scalar,
        )
        self.step_count += 1

        logger.info("RLLayer step=%d action=%s prompt:\n%s", self.step_count, action_name, prompt_text)

        # Communication is left empty here as a placeholder. The RL action
        # head doesn't pick words — it only picks discrete actions. The
        # natural-language message is produced by a SEPARATE LLM call in
        # custom_agent.on_messages → action_selection.generate_communication
        # (uses prompts/rl_communication_prompt.txt and is conditioned on
        # the chosen action + task + frame), which then overwrites
        # content["communication"] and content["communication_target"]
        # before the main loop reads them. So this empty string is never
        # what the env sees in production — it's just a structural
        # placeholder for the schema.
        return {
            "action": action_name,
            "thoughts": f"RL policy (step {self.step_count}): selected {action_name}",
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

        The pooled hidden state is the prompt's last-token representation,
        for the value head to read (IPPO mode). In MAPPO mode the caller
        ignores the value head, so the returned hidden state costs nothing
        extra (it's already computed by the prompt forward).

        Implementation:
          1. Tokenize the prompt ONCE; forward through the model with
             use_cache=True so we can reuse past_key_values across all
             candidate scoring forwards.
          2. The prompt forward's last-position logits already give us the
             conditional distribution over the FIRST candidate token. No
             extra forward needed for single-token candidates.
          3. For multi-token candidates, do ONE incremental forward feeding
             cand_ids[:-1] as input with past_key_values=prompt_kv. The
             resulting tail logits at position i predict cand_ids[i+1].

        VRAM accounting (vs the old action-head path):
          - No new optimizer parameters (head removed).
          - Sampling (with_grad=False) runs under torch.no_grad — same as
            the old _encode_prompt + action_head forward.
          - Update path (with_grad=True) replaces ONE classifier forward
            with C tiny (1-2 token) incremental forwards through the
            base model + LoRA. Since the prompt's KV cache is reused, the
            per-candidate cost is O(K_cand) tokens of attention, not
            O(L_prompt + K_cand). At C ≈ 22 candidates and K_cand ≤ 3,
            this is comparable to or cheaper than the old head's
            embedding-table-sized matmul.
          - No new model copies. No extra activation graph beyond the
            prompt forward + per-candidate tail forwards (gradient
            checkpointing still applies).

        Args
        ----
        prompt_text : str
            The full LLM prompt (system + user, exactly what _encode_prompt
            would receive).
        candidate_strings : sequence of str
            Action strings to score. Each must already be in
            self._candidate_actions so its pre-tokenized IDs are cached.
        with_grad : bool
            True at update time (gradient flows through LoRA → tail
            forwards → log_softmax → ratio). False at sampling time.
        """
        # Resolve pre-tokenized candidate IDs from the cache. Re-tokenizing
        # at every call would risk drift between sampling and update.
        cand_ids_list = []
        for s in candidate_strings:
            cand_idx_in_full = self._candidate_actions.index(s)
            cand_ids_list.append(self._candidate_token_ids[cand_idx_in_full])

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            # 1) Prompt forward — emit logits, last hidden state, and KV cache.
            enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.rl_prompt_max_tokens,
            ).to(self._device)
            out = self.model(
                **enc,
                output_hidden_states=True,
                use_cache=True,
            )
            last_hidden_all = out.hidden_states[-1]  # (1, L, H)
            seq_last = int(enc.attention_mask.sum(dim=1).item() - 1)
            pooled = last_hidden_all[:, seq_last, :].float()  # (1, H)

            # Last-position logits predict the FIRST candidate token.
            # log_softmax once over the vocab; we index it for each cand.
            last_logits = out.logits[:, -1, :].float()  # (1, V)
            last_logp = F.log_softmax(last_logits, dim=-1)  # (1, V)
            past_kv = out.past_key_values

            # 2) Per-candidate scoring via cached KV + tiny tail forwards.
            cand_log_probs = []
            for ids in cand_ids_list:
                if ids.numel() == 0:
                    # Degenerate (empty action name) — score as 0 log-prob.
                    cand_log_probs.append(
                        torch.zeros((), device=self._device, dtype=torch.float32)
                    )
                    continue
                first_id = int(ids[0].item())
                total = last_logp[0, first_id]
                if ids.numel() > 1:
                    # Forward (ids[0], ..., ids[-2]) with prompt KV cache.
                    tail_input = ids[:-1].unsqueeze(0)  # (1, K-1)
                    tail_out = self.model(
                        input_ids=tail_input,
                        past_key_values=past_kv,
                        use_cache=False,
                    )
                    tail_logits = tail_out.logits.float()  # (1, K-1, V)
                    tail_logp = F.log_softmax(tail_logits, dim=-1)
                    # tail_logp[0, i, :] predicts the token after
                    # tail_input[0, i] = ids[i], which is ids[i+1].
                    for i in range(ids.numel() - 1):
                        tok_id = int(ids[i + 1].item())
                        total = total + tail_logp[0, i, tok_id]
                cand_log_probs.append(total)

            cand_log_probs = torch.stack(cand_log_probs).to(torch.float32)  # (C,)

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
