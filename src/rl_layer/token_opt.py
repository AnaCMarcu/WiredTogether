"""Agent-decided token-level fine-tuning (extracted from RLLayer.maybe_token_optimize).

Flow:
1. Hard guards (cooldown, min samples, full success/failure window).
2. Build a stats summary the agent can read (success rate, reward trend,
   failure description, optional Hebbian bond context).
3. LLM call against ``learning_belief.txt`` — the agent decides whether to train.
4. If yes, select transitions (skill-focus filter + |advantage| top-k when
   Hebbian is enabled) and run token-level PPO mini-batches.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from rl_layer.ippo import token_level_ppo_step

if TYPE_CHECKING:
    from rl_layer.rl_layer import RLLayer

logger = logging.getLogger(__name__)


# Load learning belief prompt once.
_prompt_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "mindforge", "prompts"
)
_learning_belief_path = os.path.join(_prompt_dir, "learning_belief.txt")
try:
    with open(_learning_belief_path, "r") as _f:
        LEARNING_BELIEF_PROMPT = _f.read()
except FileNotFoundError:
    LEARNING_BELIEF_PROMPT = None


async def maybe_token_optimize(
    rl: "RLLayer",
    cancellation_token=None,
    hebbian_graph=None,
) -> Optional[Dict]:
    """See ``RLLayer.maybe_token_optimize``. Returns None when guards reject."""
    if not rl.config.enabled or not rl.config.auto_token_opt:
        return None
    if LEARNING_BELIEF_PROMPT is None:
        return None
    if not _passes_guards(rl):
        return None

    transitions = rl.buffer.get_all()
    stats = _build_stats(rl)
    bond_context = _build_bond_context(rl, hebbian_graph)

    prompt = LEARNING_BELIEF_PROMPT.format(
        window_size=stats["total"],
        recent_outcomes=stats["recent_outcomes"],
        success_rate=stats["success_rate"],
        successes=stats["successes"],
        total=stats["total"],
        reward_trend=stats["reward_trend"],
        current_task=rl._current_task,
        failure_description=stats["failure_description"],
        min_samples=rl.config.token_opt_min_samples,
        recent_actions=", ".join(rl._recent_actions[-stats["total"]:])
            if rl._recent_actions else "none",
    ) + bond_context

    logger.info(
        "RLLayer agent %d: token-opt checking (step=%d, success_rate=%.1f%%, "
        "trend=%s, fail_streak=%d, transitions=%d, bond_context=%s)",
        rl.agent_id, rl.step_count, stats["success_rate"] * 100,
        stats["reward_trend"], stats["fail_streak"], len(transitions),
        "yes" if bond_context else "no",
    )

    response = await _ask_agent(rl, prompt, cancellation_token)
    if response is None:
        return None

    needs_training = response.get("needs_training", False)
    reason = response.get("reason", "")
    skill_focus = response.get("skill_focus", "")

    logger.info(
        "RLLayer agent %d: learning-belief → needs_training=%s, reason='%s', skill='%s'",
        rl.agent_id, needs_training, reason, skill_focus,
    )

    if not needs_training:
        logger.info(
            "RLLayer agent %d: token-opt SKIPPED by agent decision — '%s'",
            rl.agent_id, reason,
        )
        return {"decision": "skip", "reason": reason}

    selected = _select_training_transitions(rl, transitions, skill_focus, hebbian_graph)
    return _run_token_ppo(rl, selected, reason, skill_focus, stats, hebbian_graph)


# ─── Guards ────────────────────────────────────────────────────────────

def _passes_guards(rl: "RLLayer") -> bool:
    cfg = rl.config
    window = rl._recent_successes
    if len(window) < cfg.token_opt_window:
        logger.debug(
            "RLLayer agent %d: token-opt guard — window too small (%d/%d steps)",
            rl.agent_id, len(window), cfg.token_opt_window,
        )
        return False

    steps_since_last = rl.step_count - rl._last_token_opt_step
    if steps_since_last < cfg.token_opt_window:
        logger.debug(
            "RLLayer agent %d: token-opt guard — cooldown (%d/%d steps since last opt)",
            rl.agent_id, steps_since_last, cfg.token_opt_window,
        )
        return False

    if len(rl.buffer.get_all()) < cfg.token_opt_min_samples:
        logger.debug(
            "RLLayer agent %d: token-opt guard — not enough transitions (%d/%d)",
            rl.agent_id, len(rl.buffer.get_all()), cfg.token_opt_min_samples,
        )
        return False
    return True


# ─── Stats ─────────────────────────────────────────────────────────────

def _build_stats(rl: "RLLayer") -> dict:
    window = rl._recent_successes
    successes = sum(window)
    total = len(window)
    success_rate = successes / total

    if len(rl._recent_rewards) >= 4:
        mid = len(rl._recent_rewards) // 2
        first_half = sum(rl._recent_rewards[:mid]) / mid
        second_half = sum(rl._recent_rewards[mid:]) / (len(rl._recent_rewards) - mid)
        if second_half > first_half * 1.1:
            reward_trend = "improving"
        elif second_half < first_half * 0.9:
            reward_trend = "declining"
        else:
            reward_trend = "stable"
    else:
        reward_trend = "not enough data"

    recent_outcomes = ", ".join("pass" if s else "FAIL" for s in window)
    fail_streak = 0
    for s in reversed(window):
        if not s:
            fail_streak += 1
        else:
            break
    if fail_streak >= 3:
        failure_description = f"{fail_streak} consecutive failures on '{rl._current_task}'"
    else:
        failure_description = "no clear repeated failure pattern"

    return {
        "successes": successes,
        "total": total,
        "success_rate": success_rate,
        "reward_trend": reward_trend,
        "recent_outcomes": recent_outcomes,
        "fail_streak": fail_streak,
        "failure_description": failure_description,
    }


def _build_bond_context(rl: "RLLayer", hebbian_graph) -> str:
    if hebbian_graph is None or not hebbian_graph.config.enabled:
        return ""
    n = hebbian_graph.W.shape[0]
    bond_parts = [
        f"agent_{j}: {float(hebbian_graph.W[rl.agent_id, j]):.3f}"
        for j in range(n) if j != rl.agent_id
    ]
    mean_bond = float(hebbian_graph.W[rl.agent_id].mean())
    return (
        "\n\nSocial bond strengths (Hebbian weights with teammates):\n"
        f"- {', '.join(bond_parts)}\n"
        f"- Mean bond strength: {mean_bond:.3f}\n"
        "Consider whether your failures are linked to weak social coordination — "
        "training can improve how you communicate and cooperate with teammates."
    )


# ─── Decision LLM call ─────────────────────────────────────────────────

async def _ask_agent(rl: "RLLayer", prompt: str, cancellation_token):
    from agent_modules.llm_call import llm_call
    from agent_modules.util import create_model_client
    from pydantic import BaseModel

    class LearningBeliefResponse(BaseModel):
        needs_training: bool
        reason: str
        skill_focus: str

    try:
        return await llm_call(
            model_client=create_model_client(response_format=LearningBeliefResponse),
            user_prompt=prompt,
            cancellation_token=cancellation_token,
            log_prefix=f"RLLayer agent {rl.agent_id} learning-belief",
        )
    except Exception as e:
        logger.warning(
            "RLLayer agent %d: learning-belief LLM call failed: %s", rl.agent_id, e,
        )
        return None


# ─── Transition selection ──────────────────────────────────────────────

def _select_training_transitions(rl: "RLLayer", transitions, skill_focus, hebbian_graph):
    """Apply skill-focus filter (keyword in prompt) and Hebbian advantage-weighted top-k."""
    if skill_focus:
        relevant = rl.buffer.filter_by_keyword(skill_focus)
        if len(relevant) >= rl.config.token_opt_min_samples:
            transitions = relevant
            logger.info(
                "RLLayer agent %d: skill-focused selection — %d transitions matching '%s'",
                rl.agent_id, len(transitions), skill_focus,
            )
    if (hebbian_graph is not None and hebbian_graph.config.enabled
            and len(transitions) > rl.config.token_opt_min_samples):
        n_top = max(rl.config.token_opt_min_samples, int(len(transitions) * 0.75))
        transitions = sorted(
            transitions, key=lambda t: abs(t.advantage), reverse=True
        )[:n_top]
        logger.info(
            "RLLayer agent %d: Hebbian advantage-weighted selection — "
            "top-%d high-impact transitions (of %d available)",
            rl.agent_id, n_top, len(rl.buffer),
        )
    return transitions


# ─── PPO loop ──────────────────────────────────────────────────────────

def _run_token_ppo(rl: "RLLayer", transitions, reason, skill_focus, stats, hebbian_graph):
    logger.info(
        "RLLayer agent %d: token-opt STARTED (step=%d) — "
        "reason='%s', skill='%s', transitions=%d, "
        "success_rate=%.1f%%, reward_trend=%s",
        rl.agent_id, rl.step_count, reason, skill_focus,
        len(transitions), stats["success_rate"] * 100, stats["reward_trend"],
    )

    rl.model.train()
    all_info = {"decision": "train", "reason": reason, "skill_focus": skill_focus}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for epoch_idx in range(rl.config.token_opt_epochs):
        epoch_losses = []
        for start in range(0, len(transitions), rl.config.mini_batch_size):
            batch = transitions[start:start + rl.config.mini_batch_size]
            with torch.amp.autocast(rl._device.type, dtype=rl._dtype):
                loss, info = token_level_ppo_step(
                    model=rl.model,
                    tokenizer=rl.tokenizer,
                    batch=batch,
                    clip_eps=rl.config.clip_eps,
                    device=rl._device,
                )
            rl.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(rl.optimizer)
            nn.utils.clip_grad_norm_(rl.model.parameters(), rl.config.max_grad_norm)
            scaler.step(rl.optimizer)
            scaler.update()
            epoch_losses.append(loss.item())
            all_info.update(info)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        logger.info(
            "RLLayer agent %d: token-opt epoch %d/%d — avg_loss=%.4f (%d batches)",
            rl.agent_id, epoch_idx + 1, rl.config.token_opt_epochs,
            avg_loss, len(epoch_losses),
        )

    rl._last_token_opt_step = rl.step_count
    all_info["success_rate_at_trigger"] = round(stats["success_rate"], 4)
    all_info["transitions_used"] = len(transitions)

    _log_strong_bonds(rl, hebbian_graph)
    logger.info(
        "RLLayer agent %d: token-opt DONE (step=%d) — %s",
        rl.agent_id, rl.step_count, all_info,
    )
    return all_info


def _log_strong_bonds(rl: "RLLayer", hebbian_graph) -> None:
    if hebbian_graph is None or not hebbian_graph.config.enabled:
        return
    n = hebbian_graph.W.shape[0]
    strong = [
        f"agent_{j}({float(hebbian_graph.W[rl.agent_id, j]):.3f})"
        for j in range(n)
        if j != rl.agent_id
        and float(hebbian_graph.W[rl.agent_id, j]) > 0.3
    ]
    if strong:
        logger.info(
            "RLLayer agent %d: strong Hebbian bonds with [%s] — "
            "social propagation may benefit these agents",
            rl.agent_id, ", ".join(strong),
        )
