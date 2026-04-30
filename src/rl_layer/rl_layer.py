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
from typing import Dict, List, Optional

import torch

from rl_layer.config import RLConfig
from rl_layer.heads import ActionHead, RunningMeanStd, ValueHead
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

        # ── Heads — always float32 ──
        # pooled hidden states are upcast to float32 (in _encode_prompt and mappo.py)
        # to prevent NaN from fp16 overflow, so heads must also be float32.
        hidden_size = self.model.config.hidden_size
        n_actions = len(config.actions)
        self.action_head = ActionHead(hidden_size, n_actions).to(device=self._device, dtype=torch.float32)
        self.value_head = ValueHead(hidden_size, config.value_hidden).to(device=self._device, dtype=torch.float32)

        # ── Optimizer (LoRA params + heads) ──
        trainable = (
            list(filter(lambda p: p.requires_grad, self.model.parameters()))
            + list(self.action_head.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(trainable, lr=config.lr)

        # ── Rollout buffer ──
        self.buffer = RolloutBuffer(max_size=config.buffer_size)

        # ── Action-index mapping ──
        self._action_to_idx = {a: i for i, a in enumerate(config.actions)}
        self._idx_to_action = {i: a for i, a in enumerate(config.actions)}

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
            pooled = self._encode_prompt(prompt_text)  # (1, H)
            action_logits = self.action_head(pooled)    # (1, A)
            # Skip the per-agent value head when the centralised critic is in
            # charge — the main loop will populate old_value_global + joint_state
            # via set_pending_value_global() after all agents have acted.
            if self._use_centralized:
                value_scalar = 0.0
            else:
                value = self.value_head(pooled).squeeze(-1)  # (1,)
                value_scalar = value.item()

            dist = torch.distributions.Categorical(logits=action_logits)
            action_idx = dist.sample()  # (1,)
            log_prob = dist.log_prob(action_idx)

        action_name = self._idx_to_action[action_idx.item()]

        # Store in buffer (reward comes later via store_reward)
        self.buffer.store_action(
            prompt_text=prompt_text,
            action_idx=action_idx.item(),
            log_prob=log_prob.item(),
            value=value_scalar,
        )
        self.step_count += 1

        # Extract task line from prompt for communication (first line: "Task: ...")
        task_line = prompt_text.split("\n", 1)[0] if prompt_text else ""
        communication = f"[RL step {self.step_count}] {task_line} → {action_name}"

        logger.info("RLLayer step=%d action=%s prompt:\n%s", self.step_count, action_name, prompt_text)

        return {
            "action": action_name,
            "thoughts": f"RL policy (step {self.step_count}): selected {action_name}",
            "communication": communication,
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

        Applies two transforms before storage:
        1. Death penalty: subtracts ``config.death_penalty`` on termination so
           the value head learns that dying is worse than running out of steps.
        2. Reward normalisation: scales by running 1/std so the 0.1–2048 range
           the environment produces maps to roughly unit variance, stabilising
           value function learning.
        """
        if not self.config.enabled:
            return
        if done and self.config.death_penalty != 0.0:
            reward += self.config.death_penalty
        if self._reward_rms is not None:
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

    def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
        """Tokenize prompt and return pooled last-token hidden state (1, H)."""
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
