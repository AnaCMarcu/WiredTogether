"""RLLayer — modular RL wrapper for MindForge agents.

Sits between MindForge's belief/context pipeline and the environment.
When ``config.enabled`` is True it replaces ActionSelection's LLM call
with a LoRA-adapted local model + classification head, collects
trajectories, and periodically runs MAPPO updates.

When ``config.enabled`` is False every public method is a fast no-op,
so the rest of the system is unaffected.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rl_layer.config import RLConfig
from rl_layer.trajectory_buffer import RolloutBuffer
from rl_layer.mappo import action_level_ppo_step, token_level_ppo_step

logger = logging.getLogger(__name__)

# Load learning belief prompt (for agent-decided token-opt)
_prompt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
_learning_belief_path = os.path.join(_prompt_dir, "learning_belief.txt")
try:
    with open(_learning_belief_path, "r") as _f:
        LEARNING_BELIEF_PROMPT = _f.read()
except FileNotFoundError:
    LEARNING_BELIEF_PROMPT = None


# ── Lightweight heads ──

class ActionHead(nn.Module):
    """Maps LLM hidden state → logits over discrete Craftium actions."""
    def __init__(self, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Linear(hidden_size, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueHead(nn.Module):
    """Maps LLM hidden state → scalar state-value estimate."""
    def __init__(self, hidden_size: int, value_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, value_hidden),
            nn.Tanh(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    def __init__(self, config: RLConfig, role: str, agent_id: int):
        self.config = config
        self.role = role
        self.agent_id = agent_id
        self.step_count = 0
        self._update_count = 0

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

        logger.info("RLLayer: loading base model from %s", config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=self._dtype,
            trust_remote_code=True,
        ).to(self._device)

        # ── LoRA adapter ──
        adapter_name = role if config.lora_per_role else "shared"
        adapter_path = Path(config.lora_save_dir) / adapter_name

        if adapter_path.exists():
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
            value = self.value_head(pooled).squeeze(-1)  # (1,)

            dist = torch.distributions.Categorical(logits=action_logits)
            action_idx = dist.sample()  # (1,)
            log_prob = dist.log_prob(action_idx)

        action_name = self._idx_to_action[action_idx.item()]

        # Store in buffer (reward comes later via store_reward)
        self.buffer.store_action(
            prompt_text=prompt_text,
            action_idx=action_idx.item(),
            log_prob=log_prob.item(),
            value=value.item(),
        )
        self.step_count += 1

        return {
            "action": action_name,
            "thoughts": f"RL policy (step {self.step_count})",
            "communication": "",
        }

    def store_reward(self, reward: float, done: bool = False) -> None:
        """Feed the environment reward back into the buffer."""
        if not self.config.enabled:
            return
        self.buffer.store_reward(reward, done)

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

    def update(self) -> Dict:
        """Run a full MAPPO update over the collected rollout."""
        if not self.config.enabled or not self.buffer.ready:
            return {}

        self.model.train()

        # Compute last value for GAE bootstrap
        last_value = 0.0
        if len(self.buffer) > 0:
            last_tr = self.buffer.get_all()[-1]
            if not last_tr.done:
                with torch.no_grad():
                    pooled = self._encode_prompt(last_tr.prompt_text)
                    last_value = self.value_head(pooled).squeeze(-1).item()

        self.buffer.compute_gae(
            self.config.gamma, self.config.gae_lambda, last_value
        )

        all_info = {}
        # GradScaler requires FP32 model weights; since the model is loaded
        # directly in FP16/BF16, gradients are already in that dtype and the
        # scaler would raise "Attempting to unscale FP16 gradients." Disable it.
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for epoch in range(self.config.ppo_epochs):
            for batch in self.buffer.sample_batches(self.config.mini_batch_size):
                with torch.amp.autocast(self._device.type, dtype=self._dtype):
                    loss, info = action_level_ppo_step(
                        model=self.model,
                        action_head=self.action_head,
                        value_head=self.value_head,
                        tokenizer=self.tokenizer,
                        batch=batch,
                        clip_eps=self.config.clip_eps,
                        value_clip_eps=self.config.value_clip_eps,
                        entropy_coef=self.config.entropy_coef,
                        value_coef=self.config.value_coef,
                        device=self._device,
                    )
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                scaler.step(self.optimizer)
                scaler.update()
                all_info = info  # keep last batch info

        self._update_count += 1
        self.buffer.clear()

        logger.info(
            "RLLayer agent %d update #%d: %s",
            self.agent_id, self._update_count, all_info,
        )
        return all_info

    async def maybe_token_optimize(self, cancellation_token=None) -> Optional[Dict]:
        """Agent-decided token-level fine-tuning.

        The agent is shown its recent performance stats and decides for
        itself whether deeper training is needed.  The LLM call uses the
        learning_belief.txt prompt — the agent reasons about its own
        competence gap and outputs ``needs_training: true/false``.

        Hard guards prevent degenerate behaviour:
        - Cooldown: at least ``token_opt_window`` steps since last training
        - Minimum data: need ``token_opt_min_samples`` transitions
        - Enough history: need a full window of success/failure data
        """
        if not self.config.enabled or not self.config.auto_token_opt:
            return None
        if LEARNING_BELIEF_PROMPT is None:
            return None

        window = self._recent_successes
        if len(window) < self.config.token_opt_window:
            return None  # not enough data to assess

        # Cooldown — don't retrain too frequently
        if (self.step_count - self._last_token_opt_step) < self.config.token_opt_window:
            return None

        # Need enough transitions to actually learn from
        transitions = self.buffer.get_all()
        if len(transitions) < self.config.token_opt_min_samples:
            return None

        # ── Build stats for the agent ──
        successes = sum(window)
        total = len(window)
        success_rate = successes / total

        # Reward trend: compare first half vs second half
        if len(self._recent_rewards) >= 4:
            mid = len(self._recent_rewards) // 2
            first_half = sum(self._recent_rewards[:mid]) / mid
            second_half = sum(self._recent_rewards[mid:]) / (len(self._recent_rewards) - mid)
            if second_half > first_half * 1.1:
                reward_trend = "improving"
            elif second_half < first_half * 0.9:
                reward_trend = "declining"
            else:
                reward_trend = "stable"
        else:
            reward_trend = "not enough data"

        # Failure description
        recent_outcomes = ", ".join("pass" if s else "FAIL" for s in window)
        fail_streak = 0
        for s in reversed(window):
            if not s:
                fail_streak += 1
            else:
                break
        if fail_streak >= 3:
            failure_description = f"{fail_streak} consecutive failures on '{self._current_task}'"
        else:
            failure_description = "no clear repeated failure pattern"

        # ── Ask the agent ──
        prompt = LEARNING_BELIEF_PROMPT.format(
            window_size=total,
            recent_outcomes=recent_outcomes,
            success_rate=success_rate,
            successes=successes,
            total=total,
            reward_trend=reward_trend,
            current_task=self._current_task,
            failure_description=failure_description,
            min_samples=self.config.token_opt_min_samples,
            recent_actions=", ".join(self._recent_actions[-total:]) if self._recent_actions else "none",
        )

        # LLM call through the same infrastructure as the rest of MindForge
        from agent_modules.llm_call import llm_call
        from agent_modules.util import create_model_client
        from pydantic import BaseModel

        class LearningBeliefResponse(BaseModel):
            needs_training: bool
            reason: str
            skill_focus: str

        try:
            response = await llm_call(
                model_client=create_model_client(response_format=LearningBeliefResponse),
                user_prompt=prompt,
                cancellation_token=cancellation_token,
                log_prefix=f"RLLayer agent {self.agent_id} learning-belief",
            )
        except Exception as e:
            logger.warning("RLLayer agent %d: learning-belief LLM call failed: %s",
                           self.agent_id, e)
            return None

        needs_training = response.get("needs_training", False)
        reason = response.get("reason", "")
        skill_focus = response.get("skill_focus", "")

        logger.info(
            "RLLayer agent %d: learning-belief → needs_training=%s, reason='%s', skill='%s'",
            self.agent_id, needs_training, reason, skill_focus,
        )

        if not needs_training:
            return {"decision": "skip", "reason": reason}

        # ── Agent decided to train — run token-level PPO ──
        # Optionally filter transitions by skill_focus
        if skill_focus:
            relevant = self.buffer.filter_by_keyword(skill_focus)
            if len(relevant) >= self.config.token_opt_min_samples:
                transitions = relevant

        logger.info(
            "RLLayer agent %d: token-opt triggered by agent (reason='%s'), %d transitions",
            self.agent_id, reason, len(transitions),
        )

        self.model.train()
        all_info = {"decision": "train", "reason": reason, "skill_focus": skill_focus}
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for epoch in range(self.config.token_opt_epochs):
            for start in range(0, len(transitions), self.config.mini_batch_size):
                batch = transitions[start:start + self.config.mini_batch_size]
                with torch.amp.autocast(self._device.type, dtype=self._dtype):
                    loss, info = token_level_ppo_step(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        batch=batch,
                        clip_eps=self.config.clip_eps,
                        device=self._device,
                    )
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                scaler.step(self.optimizer)
                scaler.update()
                all_info.update(info)

        self._last_token_opt_step = self.step_count
        logger.info("RLLayer agent %d: token-opt done: %s", self.agent_id, all_info)
        return all_info

    # ── Persistence ──

    def save(self, path: Optional[str] = None) -> None:
        """Save LoRA adapter + heads to disk."""
        if not self.config.enabled:
            return
        save_dir = Path(path or self.config.lora_save_dir) / self._adapter_name
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_dir))
        torch.save(self.action_head.state_dict(), save_dir / "action_head.pt")
        torch.save(self.value_head.state_dict(), save_dir / "value_head.pt")
        logger.info("RLLayer agent %d: saved to %s", self.agent_id, save_dir)

    def load(self, path: Optional[str] = None) -> None:
        """Load heads from disk (LoRA loaded at init)."""
        if not self.config.enabled:
            return
        load_dir = Path(path or self.config.lora_save_dir) / self._adapter_name
        ah_path = load_dir / "action_head.pt"
        vh_path = load_dir / "value_head.pt"
        if ah_path.exists():
            self.action_head.load_state_dict(torch.load(ah_path, map_location=self._device))
        if vh_path.exists():
            self.value_head.load_state_dict(torch.load(vh_path, map_location=self._device))

    # ── Internal ──

    def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
        """Tokenize prompt and return pooled last-token hidden state (1, H)."""
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self._device)
        outputs = self.model(**enc, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (1, L, H)
        seq_len = enc.attention_mask.sum(dim=1) - 1  # (1,)
        return last_hidden[:, seq_len.item(), :].float()  # (1, H) — upcast to fp32 to prevent NaN from fp16 overflow
