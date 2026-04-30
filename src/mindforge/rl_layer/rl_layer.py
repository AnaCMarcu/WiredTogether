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


# ── Running reward normaliser ──

class RunningMeanStd:
    """Online running mean/variance using Welford's algorithm.

    Used to normalise rewards before storing in the rollout buffer so the
    value head sees a roughly unit-variance signal regardless of whether
    the episode returns 0.1 (exploration) or 2048 (stage completion).
    """
    def __init__(self, eps: float = 1e-4):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = eps  # start non-zero to avoid /0 on first step

    def update(self, x: float) -> None:
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    def normalize(self, x: float) -> float:
        return x / (self.var ** 0.5 + 1e-8)


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
        """Run a full MAPPO update over the collected rollout.

        Parameters
        ----------
        neighbour_buffers : dict of {agent_id: RolloutBuffer}, optional
            Other agents' rollout buffers for social replay (Eq. 7).
            When provided alongside a Hebbian graph, transitions from
            strongly-bonded neighbours are mixed into each mini-batch.
        hebbian_graph : HebbianSocialGraph, optional
            Used to compute social replay sample indices via get_social_replay_indices().
        """
        if not self.config.enabled or not self.buffer.ready:
            return {}

        # Release any fragmented CUDA cache before the backward pass so the
        # activation tensors have room to allocate.
        torch.cuda.empty_cache()

        self.model.train()

        # Compute last value for GAE bootstrap.
        # In centralised mode we use the critic's evaluation of the last
        # transition's joint_state; in independent mode we fall back to the
        # per-agent value head on the last prompt.
        last_value = 0.0
        if len(self.buffer) > 0:
            last_tr = self.buffer.get_all()[-1]
            if not last_tr.done:
                if self._use_centralized and last_tr.joint_state is not None:
                    last_value = float(self.centralized_critic.evaluate(last_tr.joint_state))
                elif not self._use_centralized:
                    with torch.no_grad():
                        pooled = self._encode_prompt(last_tr.prompt_text)
                        last_value = self.value_head(pooled).squeeze(-1).item()

        self.buffer.compute_gae(
            self.config.gamma, self.config.gae_lambda, last_value,
            use_global_value=self._use_centralized,
        )

        # ── Social replay: collect neighbour transitions (Eq. 7) ──────────
        social_transitions = []
        if neighbour_buffers and hebbian_graph is not None:
            buffer_sizes = {
                aid: len(buf) for aid, buf in neighbour_buffers.items()
            }
            # get_social_replay_indices expects a list indexed by agent_id
            max_id = max(buffer_sizes.keys()) + 1 if buffer_sizes else 0
            sizes_list = [buffer_sizes.get(i, 0) for i in range(max_id)]
            indices = hebbian_graph.get_social_replay_indices(
                agent_i=self.agent_id,
                buffer_sizes=sizes_list,
                rho=hebbian_graph.config.social_replay_rho,
            )
            for buf_idx, agent_j in indices:
                buf_j = neighbour_buffers.get(agent_j)
                if buf_j is not None:
                    all_j = buf_j.get_all()
                    if buf_idx < len(all_j):
                        social_transitions.append(all_j[buf_idx])
            if social_transitions:
                logger.info(
                    "RLLayer agent %d: social replay — %d neighbour transitions from %d agents",
                    self.agent_id, len(social_transitions),
                    len({j for _, j in indices}),
                )
        # ─────────────────────────────────────────────────────────────────

        # ── Entropy annealing ──────────────────────────────────────────────
        if self.config.entropy_anneal_steps > 0:
            progress = min(self._update_count / self.config.entropy_anneal_steps, 1.0)
            entropy_coef = self.config.entropy_start + progress * (
                self.config.entropy_end - self.config.entropy_start
            )
        else:
            entropy_coef = self.config.entropy_coef
        # ─────────────────────────────────────────────────────────────────

        all_info = {}
        # GradScaler requires FP32 model weights; since the model is loaded
        # directly in FP16/BF16, gradients are already in that dtype and the
        # scaler would raise "Attempting to unscale FP16 gradients." Disable it.
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.sample_batches(
                self.config.mini_batch_size,
                extra_transitions=social_transitions,
            ):
                with torch.amp.autocast(self._device.type, dtype=self._dtype):
                    loss, info = action_level_ppo_step(
                        model=self.model,
                        action_head=self.action_head,
                        value_head=self.value_head,
                        tokenizer=self.tokenizer,
                        batch=batch,
                        clip_eps=self.config.clip_eps,
                        value_clip_eps=self.config.value_clip_eps,
                        entropy_coef=entropy_coef,
                        value_coef=self.config.value_coef,
                        device=self._device,
                        max_length=self.config.rl_prompt_max_tokens,
                        value_loss_enabled=not self._use_centralized,
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
        all_info["entropy_coef"] = entropy_coef

        logger.info(
            "RLLayer agent %d update #%d: %s",
            self.agent_id, self._update_count, all_info,
        )
        return all_info

    async def maybe_token_optimize(self, cancellation_token=None,
                                    hebbian_graph=None) -> Optional[Dict]:
        """Agent-decided token-level fine-tuning.

        The agent is shown its recent performance stats and decides for
        itself whether deeper training is needed.  The LLM call uses the
        learning_belief.txt prompt — the agent reasons about its own
        competence gap and outputs ``needs_training: true/false``.

        Hard guards prevent degenerate behaviour:
        - Cooldown: at least ``token_opt_window`` steps since last training
        - Minimum data: need ``token_opt_min_samples`` transitions
        - Enough history: need a full window of success/failure data

        Parameters
        ----------
        hebbian_graph : HebbianSocialGraph, optional
            When provided and enabled, bond strengths are included in the
            LLM decision prompt (social context) and transitions are
            prioritised by |advantage| so training focuses on the most
            informative steps.
        """
        if not self.config.enabled or not self.config.auto_token_opt:
            return None
        if LEARNING_BELIEF_PROMPT is None:
            return None

        window = self._recent_successes
        if len(window) < self.config.token_opt_window:
            logger.debug(
                "RLLayer agent %d: token-opt guard — window too small (%d/%d steps)",
                self.agent_id, len(window), self.config.token_opt_window,
            )
            return None

        # Cooldown — don't retrain too frequently
        steps_since_last = self.step_count - self._last_token_opt_step
        if steps_since_last < self.config.token_opt_window:
            logger.debug(
                "RLLayer agent %d: token-opt guard — cooldown (%d/%d steps since last opt)",
                self.agent_id, steps_since_last, self.config.token_opt_window,
            )
            return None

        # Need enough transitions to actually learn from
        transitions = self.buffer.get_all()
        if len(transitions) < self.config.token_opt_min_samples:
            logger.debug(
                "RLLayer agent %d: token-opt guard — not enough transitions (%d/%d)",
                self.agent_id, len(transitions), self.config.token_opt_min_samples,
            )
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

        # ── Social bond context (Hebbian) ──────────────────────────────────────
        # Adds bond strengths to the prompt so the agent can reason about
        # whether its failures are correlated with low social coordination.
        bond_context = ""
        if hebbian_graph is not None and hebbian_graph.config.enabled:
            n = hebbian_graph.W.shape[0]
            bond_parts = [
                f"agent_{j}: {float(hebbian_graph.W[self.agent_id, j]):.3f}"
                for j in range(n) if j != self.agent_id
            ]
            mean_bond = float(hebbian_graph.W[self.agent_id].mean())
            bond_context = (
                "\n\nSocial bond strengths (Hebbian weights with teammates):\n"
                f"- {', '.join(bond_parts)}\n"
                f"- Mean bond strength: {mean_bond:.3f}\n"
                "Consider whether your failures are linked to weak social coordination — "
                "training can improve how you communicate and cooperate with teammates."
            )

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
        ) + bond_context

        logger.info(
            "RLLayer agent %d: token-opt checking (step=%d, success_rate=%.1f%%, "
            "trend=%s, fail_streak=%d, transitions=%d, bond_context=%s)",
            self.agent_id, self.step_count, success_rate * 100,
            reward_trend, fail_streak, len(transitions),
            "yes" if bond_context else "no",
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
            logger.info(
                "RLLayer agent %d: token-opt SKIPPED by agent decision — '%s'",
                self.agent_id, reason,
            )
            return {"decision": "skip", "reason": reason}

        # ── Select transitions for training ────────────────────────────────────
        # 1. Skill-focus filter (by keyword in prompt text)
        if skill_focus:
            relevant = self.buffer.filter_by_keyword(skill_focus)
            if len(relevant) >= self.config.token_opt_min_samples:
                transitions = relevant
                logger.info(
                    "RLLayer agent %d: skill-focused selection — "
                    "%d transitions matching '%s'",
                    self.agent_id, len(transitions), skill_focus,
                )

        # 2. Hebbian advantage-weighted prioritisation: prefer high-|advantage|
        #    transitions (steps where the agent's policy had the most impact).
        if hebbian_graph is not None and hebbian_graph.config.enabled and \
                len(transitions) > self.config.token_opt_min_samples:
            n_top = max(self.config.token_opt_min_samples,
                        int(len(transitions) * 0.75))
            transitions = sorted(
                transitions, key=lambda t: abs(t.advantage), reverse=True
            )[:n_top]
            logger.info(
                "RLLayer agent %d: Hebbian advantage-weighted selection — "
                "top-%d high-impact transitions (of %d available)",
                self.agent_id, n_top, len(self.buffer),
            )

        # ── Agent decided to train — run token-level PPO ──────────────────────
        logger.info(
            "RLLayer agent %d: token-opt STARTED (step=%d) — "
            "reason='%s', skill='%s', transitions=%d, "
            "success_rate=%.1f%%, reward_trend=%s",
            self.agent_id, self.step_count, reason, skill_focus,
            len(transitions), success_rate * 100, reward_trend,
        )

        self.model.train()
        all_info = {"decision": "train", "reason": reason, "skill_focus": skill_focus}
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for epoch_idx in range(self.config.token_opt_epochs):
            epoch_losses = []
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
                epoch_losses.append(loss.item())
                all_info.update(info)

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            logger.info(
                "RLLayer agent %d: token-opt epoch %d/%d — "
                "avg_loss=%.4f (%d batches)",
                self.agent_id, epoch_idx + 1, self.config.token_opt_epochs,
                avg_loss, len(epoch_losses),
            )

        self._last_token_opt_step = self.step_count
        all_info["success_rate_at_trigger"] = round(success_rate, 4)
        all_info["transitions_used"] = len(transitions)

        # Log strong bonds so the caller can see which agents might benefit
        # from social propagation of the training signal.
        if hebbian_graph is not None and hebbian_graph.config.enabled:
            n = hebbian_graph.W.shape[0]
            strong = [
                f"agent_{j}({float(hebbian_graph.W[self.agent_id, j]):.3f})"
                for j in range(n)
                if j != self.agent_id
                and float(hebbian_graph.W[self.agent_id, j]) > 0.3
            ]
            if strong:
                logger.info(
                    "RLLayer agent %d: strong Hebbian bonds with [%s] — "
                    "social propagation may benefit these agents",
                    self.agent_id, ", ".join(strong),
                )

        logger.info(
            "RLLayer agent %d: token-opt DONE (step=%d) — %s",
            self.agent_id, self.step_count, all_info,
        )
        return all_info

    # ── Persistence ──

    def save(self, path: Optional[str] = None) -> None:
        """Save LoRA adapter, heads, optimizer, and running state to disk."""
        if not self.config.enabled:
            return
        save_dir = Path(path or self.config.lora_save_dir) / self._adapter_name
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_dir))
        torch.save(self.action_head.state_dict(), save_dir / "action_head.pt")
        torch.save(self.value_head.state_dict(), save_dir / "value_head.pt")

        # Extended state for checkpoint/resume
        rms = self._reward_rms
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "_update_count": self._update_count,
            "_last_token_opt_step": self._last_token_opt_step,
            "_recent_successes": list(self._recent_successes),
            "_recent_actions": list(self._recent_actions),
            "_recent_rewards": list(self._recent_rewards),
            "_current_task": self._current_task,
            "rms_mean": rms.mean if rms is not None else 0.0,
            "rms_var": rms.var if rms is not None else 1.0,
            "rms_count": rms.count if rms is not None else 1e-4,
        }, save_dir / "rl_state.pt")

        logger.info("RLLayer agent %d: saved to %s", self.agent_id, save_dir)

    def load(self, path: Optional[str] = None) -> None:
        """Load heads from disk (LoRA loaded at init).

        Handles action head size mismatches gracefully: if a checkpoint was
        saved with fewer actions than the current config (e.g. before macros
        were added), the action head is reinitialised from scratch rather than
        crashing.  The value head and LoRA weights are still restored.
        """
        if not self.config.enabled:
            return
        load_dir = Path(path or self.config.lora_save_dir) / self._adapter_name
        ah_path = load_dir / "action_head.pt"
        vh_path = load_dir / "value_head.pt"
        if ah_path.exists():
            saved_state = torch.load(ah_path, map_location=self._device, weights_only=True)
            saved_n_actions = saved_state["net.weight"].shape[0]
            current_n_actions = len(self.config.actions)
            if saved_n_actions != current_n_actions:
                logger.warning(
                    "RLLayer agent %d: action head size mismatch — "
                    "checkpoint has %d actions, config has %d. "
                    "Reinitialising action head (LoRA weights preserved).",
                    self.agent_id, saved_n_actions, current_n_actions,
                )
                hidden_size = self.model.config.hidden_size
                self.action_head = ActionHead(hidden_size, current_n_actions).to(
                    device=self._device, dtype=torch.float32
                )
                # Rebuild optimizer to include the new head parameters
                trainable = (
                    list(filter(lambda p: p.requires_grad, self.model.parameters()))
                    + list(self.action_head.parameters())
                    + list(self.value_head.parameters())
                )
                self.optimizer = torch.optim.Adam(trainable, lr=self.config.lr)
            else:
                self.action_head.load_state_dict(saved_state)
        if vh_path.exists():
            self.value_head.load_state_dict(torch.load(vh_path, map_location=self._device, weights_only=True))

        # Extended state for checkpoint/resume
        state_path = load_dir / "rl_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self._device, weights_only=False)
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError):
                logger.warning(
                    "RLLayer agent %d: optimizer state mismatch, reinitialising.", self.agent_id
                )
            self.step_count = state.get("step_count", 0)
            self._update_count = state.get("_update_count", 0)
            self._last_token_opt_step = state.get("_last_token_opt_step", 0)
            self._recent_successes = list(state.get("_recent_successes", []))
            self._recent_actions = list(state.get("_recent_actions", []))
            self._recent_rewards = list(state.get("_recent_rewards", []))
            self._current_task = state.get("_current_task", "Explore")
            if self._reward_rms is not None:
                self._reward_rms.mean = state.get("rms_mean", 0.0)
                self._reward_rms.var = state.get("rms_var", 1.0)
                self._reward_rms.count = state.get("rms_count", 1e-4)
            logger.info("RLLayer agent %d: restored rl_state (step=%d, updates=%d)",
                        self.agent_id, self.step_count, self._update_count)

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
