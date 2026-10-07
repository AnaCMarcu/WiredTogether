"""The MindForge cognitive stack as AgentWorld cognition.

Each agent runs MindForge's loop with MindForge's own modules —
``ActionSelection`` (action + communication), ``BeliefSystem`` (perception,
partner, interaction beliefs), ``Critic`` (sub-task check), an
auto-curriculum (next sub-task), skill and episode memory — driven by
AgentWorld text observations instead of WIRE frames:

    critic (every ``critic_interval`` rounds)
      → skill / episode memory update
      → new sub-task when done, missing, or failing (curriculum)
      → beliefs (every ``belief_interval`` rounds) ∥ memory retrieval
      → action selection (one tool call + optional DM + optional board post)

It is composed from the module classes rather than subclassing
``CustomAgent``: that class's ``on_messages`` is built around frames,
chambers and Lua milestones, and its constructor would build three ChromaDB
stores with their own GPU embedders per agent. Composition keeps the
modules, prompts-as-arguments and ``llm_call`` path, and swaps memory for the
run-wide :mod:`agentworld.memory`.

Scaling choices: the prompt names at most K contacts and K bonds; partner
beliefs are keyed by player name and refreshed for at most
``partner_updates`` senders per round (Hebbian arm: strongest bonds first;
base arm: most recent), which removes the O(N²) partner-belief cost.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentworld.actions import safe_text, tool_reference
from agentworld.llm import AccountingClient, LoopSemaphore, TokenLedger
from agentworld.memory import SharedMemory
from agentworld.scheduler import Decision, Turn

_PROMPTS = Path(__file__).parent / "prompts"


def _read(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


SYSTEM_PROMPT = _read("aw_system.txt").replace("{tool_reference}", tool_reference())
INSTRUCTION = _read("aw_instruction.txt")
CURRICULUM_SYSTEM = _read("aw_curriculum_system.txt")
CURRICULUM_INFO = _read("aw_curriculum_info.txt")
CRITIC_SYSTEM = _read("aw_critic_system.txt")
CRITIC_INFO = _read("aw_critic_info.txt")
PERCEPTION = _read("beliefs/perception.txt")
PARTNER = _read("beliefs/partner.txt")
INTERACTION = _read("beliefs/interaction.txt")


@dataclass
class AgentConfig:
    belief_interval: int = 5
    critic_interval: int = 3
    max_errors: int = 6            # failed critic checks before a new sub-task
    partner_updates: int = 2       # partner-belief refreshes per round (cap)
    skill_k: int = 3
    episode_k: int = 3
    text_limit: int = 400          # per message / belief, characters
    voyager: bool = False          # no beliefs, no episodes (ablation)


@dataclass
class ModelClients:
    """One inner client per response schema, shared by all agents."""
    action: Any
    belief: Any
    critic: Any
    task: Any

    @classmethod
    def from_env(cls) -> "ModelClients":
        from mindforge.agent_modules.util import (BeliefResponse, CriticResponse,
                                                  CurruliculumResponse, create_model_client)
        from agentworld.schemas import AWAgentResponse
        return cls(action=create_model_client(response_format=AWAgentResponse),
                   belief=create_model_client(response_format=BeliefResponse),
                   critic=create_model_client(response_format=CriticResponse),
                   task=create_model_client(response_format=CurruliculumResponse))


def _fmt_messages(msgs: List[Any], names: List[str], limit: int) -> str:
    if not msgs:
        return "(none)"
    return "\n".join(f"- from {names[m.sender]}: {safe_text(m.text, limit)}" for m in msgs)


def _fmt_board(msgs: List[Any], names: List[str], limit: int) -> str:
    if not msgs:
        return "(none)"
    return "\n".join(f"- [{m.post_kind}] {names[m.sender]}: {safe_text(m.text, limit)}"
                     for m in msgs)


class AgentWorldAgent:
    def __init__(self, index: int, name: str, clients: ModelClients, memory: SharedMemory,
                 ledger: TokenLedger, config: Optional[AgentConfig] = None,
                 semaphore: Optional[LoopSemaphore] = None, metric_log=None):
        from mindforge.agent_modules.action_selection import ActionSelection
        from mindforge.agent_modules.belief_system import BeliefSystem
        from mindforge.agent_modules.critic import Critic

        self.index = index
        self.name = name
        self.cfg = config or AgentConfig()
        self.log = metric_log or (lambda msg: None)

        def wrap(inner, module):
            return AccountingClient(inner, ledger, name, module, semaphore)

        self.action_selection = ActionSelection(system_prompt=SYSTEM_PROMPT,
                                                action_model_client=wrap(clients.action, "action"),
                                                user_prompt_template=INSTRUCTION)
        self.belief_system = BeliefSystem(number_of_agents=2,
                                          belief_model_client=wrap(clients.belief, "belief"),
                                          override_perception_prompt=PERCEPTION,
                                          override_partner_prompt=PARTNER,
                                          override_interaction_prompt=INTERACTION)
        self.critic = Critic(critic_model_client=wrap(clients.critic, "critic"),
                             override_critic_prompt=CRITIC_SYSTEM)
        self.task_client = wrap(clients.task, "curriculum")
        self.skills = memory.store(f"{name}/skills")
        self.episodes = memory.store(f"{name}/episodes")

        self.partner_beliefs: Dict[str, str] = {}
        self.current_task: Optional[str] = None
        self.completed: deque = deque(maxlen=12)
        self.failed: deque = deque(maxlen=12)
        self.error_count = 0
        self.critique = "(not checked yet)"
        self.last_action = "none"
        self.last_thoughts = ""
        self.rounds_seen = 0
        self.convo_log: Dict[str, deque] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    def on_reset(self) -> None:
        """New episode: drop working memory, keep skills and episodes."""
        self.current_task = None
        self.critique = "(not checked yet)"
        self.error_count = 0
        self.last_action = "none"
        self.last_thoughts = ""
        self.belief_system.reset()
        self.partner_beliefs.clear()
        self.convo_log.clear()
        self.completed.clear()
        self.failed.clear()

    # ── LLM sub-steps ──────────────────────────────────────────────────────
    async def _llm(self, client, system, template, label, **kw) -> Dict[str, Any]:
        from mindforge.agent_modules.llm_call import llm_call
        return await llm_call(client, user_prompt=template, cancellation_token=None,
                              system_prompt=system, log_prefix=label, **kw) or {}

    async def _check_task(self, turn: Turn) -> Optional[bool]:
        resp = await self._llm(self.critic.critic_model_client, CRITIC_SYSTEM, CRITIC_INFO,
                               "Critic check_task_success: ", task=self.current_task,
                               last_action=self.last_action,
                               last_result=turn.last_result, progress=turn.progress_text,
                               observation=turn.obs_text)
        if "success" not in resp:
            return None
        self.critique = safe_text(resp.get("critique", ""), self.cfg.text_limit)
        return bool(resp.get("success"))

    async def _new_task(self, turn: Turn, comms: str) -> None:
        resp = await self._llm(self.task_client, CURRICULUM_SYSTEM, CURRICULUM_INFO,
                               "Auto Curriculum get_new_task: ",
                               objective=safe_text(turn.objective),
                               game_context=safe_text(turn.game_context),
                               agent_name=self.name, progress=turn.progress_text,
                               observation=turn.obs_text, communications=comms,
                               last_task=self.current_task or "(none)", critique=self.critique,
                               completed_tasks="; ".join(self.completed) or "(none)",
                               failed_tasks="; ".join(self.failed) or "(none)")
        task = str(resp.get("task") or "").strip()
        self.current_task = safe_text(task, 300) if task else (
            self.current_task or "Work toward the team objective")
        self.error_count = 0
        self.critique = "(new sub-task)"

    def _partner_order(self, turn: Turn) -> List[str]:
        senders: List[str] = []
        for m in turn.inbox.dms:
            n = turn.names[m.sender]
            if n not in senders:
                senders.append(n)
        bond = dict(turn.bonds)
        if bond:   # Hebbian arm: strongest bonds first, then message order
            senders.sort(key=lambda n: -bond.get(n, 0.0))
        return senders[: self.cfg.partner_updates]

    async def _beliefs(self, turn: Turn, comms: str, partners_only: bool = False) -> None:
        bs = self.belief_system
        lim = self.cfg.text_limit

        async def perception():
            r = await self._llm(bs.belief_model_client, None, PERCEPTION,
                                "Belief System create_perception_beliefs: ",
                                task=self.current_task, observation=turn.obs_text,
                                communications=comms, error=turn.last_result)
            bs.perception_beliefs = safe_text(r.get("beliefs") or bs.perception_beliefs, lim)

        async def interaction():
            r = await self._llm(bs.belief_model_client, None, INTERACTION,
                                "Belief System update_interaction_beliefs: ",
                                task=self.current_task, conversations=comms,
                                previous_interaction_beliefs=bs.interaction_beliefs or "(none)")
            bs.interaction_beliefs = safe_text(r.get("beliefs") or bs.interaction_beliefs, lim)

        async def partner(n: str):
            convo = "\n".join(self.convo_log.get(n, [])) or "(nothing yet)"
            r = await self._llm(bs.belief_model_client, None, PARTNER,
                                "Belief System update_partner_beliefs: ", partner=n, convo=convo,
                                previous_partner_belief=self.partner_beliefs.get(n, "(none)"))
            text = r.get("beliefs")
            if text:
                self.partner_beliefs[n] = safe_text(text, lim)

        jobs = [] if partners_only else [perception(), interaction()]
        jobs += [partner(n) for n in self._partner_order(turn)]
        if jobs:
            await asyncio.gather(*jobs)

    def _remember_messages(self, turn: Turn) -> None:
        for m in list(turn.inbox.dms) + list(turn.inbox.board):
            n = turn.names[m.sender]
            log = self.convo_log.setdefault(n, deque(maxlen=6))
            log.append(f"r{m.round}: {safe_text(m.text, self.cfg.text_limit)}")

    # ── one round ──────────────────────────────────────────────────────────
    async def decide(self, turn: Turn) -> Decision:
        self.rounds_seen += 1
        names = turn.names
        lim = self.cfg.text_limit
        self._remember_messages(turn)
        comms = (_fmt_messages(turn.inbox.dms, names, lim) + "\n"
                 + _fmt_board(turn.inbox.board, names, lim))

        success = None
        if self.current_task and self.rounds_seen % self.cfg.critic_interval == 0:
            success = await self._check_task(turn)
            if success:
                self.completed.append(self.current_task)
                self.skills.add(f"{self.current_task} -> {self.last_action}",
                                {"task": self.current_task, "action": self.last_action})
            elif success is False:
                self.error_count += 1
            if success is not None and not self.cfg.voyager:
                self.episodes.add(
                    f"Sub-task: {self.current_task}. Action: {self.last_action}. "
                    f"{'Done' if success else 'Not done'}: {self.critique}",
                    {"success": success})
        if success or self.current_task is None or self.error_count > self.cfg.max_errors:
            if self.error_count > self.cfg.max_errors and self.current_task:
                self.failed.append(self.current_task)
            await self._new_task(turn, comms)

        if not self.cfg.voyager:
            full = self.rounds_seen == 1 or self.rounds_seen % self.cfg.belief_interval == 0
            if full or turn.inbox.dms:   # between intervals: only the DM senders
                await self._beliefs(turn, comms, partners_only=not full)

        skills = self.skills.query(self.current_task or "", k=self.cfg.skill_k)
        episodes = [] if self.cfg.voyager else self.episodes.query(
            self.current_task or "", k=self.cfg.episode_k)
        known = [n for n in turn.contacts]
        partner_text = "; ".join(f"{n}: {self.partner_beliefs[n]}" for n in known
                                 if n in self.partner_beliefs) or "(none yet)"
        bonds = ("Your strongest bonds (higher = you worked well together recently): "
                 + ", ".join(f"{n} ({w:.2f})" for n, w in turn.bonds)) if turn.bonds else ""

        beliefs = {
            "objective": safe_text(turn.objective),
            "game_context": safe_text(turn.game_context),
            "round": turn.round,
            "progress": turn.progress_text,
            "observation": turn.obs_text,
            "last_result": turn.last_result,
            "inbox": _fmt_messages(turn.inbox.dms, names, lim),
            "board": _fmt_board(turn.inbox.board, names, lim),
            "contacts": ", ".join(known) or "(none)",
            "bonds": bonds,
            "perception_beliefs": self.belief_system.perception_beliefs or "(none yet)",
            "partner_beliefs": partner_text,
            "interaction_beliefs": self.belief_system.interaction_beliefs or "(none yet)",
        }
        msg = type("Msg", (), {"content": ["", None]})()
        content = await self.action_selection.select_action(
            [msg], None, None, agent_name=self.name, task=self.current_task,
            last_action=self.last_action, critique=self.critique, error=turn.last_result,
            skill_memory="; ".join(t for _, t, _ in skills) or "(none yet)",
            episode_summary=" | ".join(t for _, t, _ in episodes) or "(none yet)",
            picked_object="", beliefs=beliefs,
            teammate_names=", ".join(known))
        content = content if isinstance(content, dict) else {}
        decision = Decision(
            action=str(content.get("action") or "wait()"),
            thoughts=str(content.get("thoughts") or ""),
            dm_target=(str(content.get("dm_target")).strip() or None)
            if content.get("dm_target") else None,
            dm_text=str(content.get("dm_text") or ""),
            board_post=str(content.get("board_post") or ""),
            post_kind=str(content.get("post_kind") or "status"),
        )
        self.last_action = decision.action
        self.last_thoughts = decision.thoughts
        self.log(f"{self.name} r{turn.round} task={self.current_task!r} -> {decision.action}")
        return decision


class MindForgeCognition:
    """Routes each Turn to its agent; implements the scheduler's Cognition."""

    def __init__(self, names: List[str], clients: ModelClients, *, memory: SharedMemory = None,
                 config: Optional[AgentConfig] = None, max_in_flight: int = 256,
                 metric_log=None):
        self.ledger = TokenLedger()
        self.memory = memory or SharedMemory()
        self._sem = LoopSemaphore(max_in_flight) if max_in_flight else None
        self.agents = [AgentWorldAgent(i, n, clients, self.memory, self.ledger, config,
                                       self._sem, metric_log) for i, n in enumerate(names)]

    async def decide(self, turn: Turn) -> Decision:
        return await self.agents[turn.agent].decide(turn)

    def on_reset(self) -> None:
        for a in self.agents:
            a.on_reset()
