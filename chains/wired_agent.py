"""Wired Together adapters for AgentWorld's reference runner (``agent.provider: custom``).

Each class subclasses one of AgentWorld's own provider agents, so the policy,
tool definitions, one-action-per-turn ``execute_single_tool_call`` and the
``[TOOL_CALL_INFO]`` / ``[TOOL_RESULT]`` records that ``agents/run.py`` reads
are inherited unchanged. What the adapter adds:

* identity — ``agent_id`` and ``role`` parsed from the username
  (``a3_weapon_smith`` -> ``a3``, ``weapon_smith``); any other username fails;
* hooks — ``on_task_start(agent_id, role, teammates)``,
  ``on_turn(round_num, interactions)``, ``on_skill(round_num, outcome)`` and
  ``on_task_end(agent_id, outcome)``,
  which read and write the agent's persistent state through the
  :class:`wired_state.AgentSession` that ``chains/run_chain.py`` attaches;
* prompt — the session's team-history block (earlier outcomes, past
  interactions per teammate, and in the ``hebbian`` arm the bond line) is
  appended to the system prompt, which the base class rebuilds on every call.
  Adding it to the user message instead would append a fresh copy of the
  whole task prompt to the conversation every time a bond moved.

Pick the class that matches the provider your config used before::

    agent:
      provider: custom
      llm:
        agent_class: wired_agent:WiredOpenAIAgent   # was provider: openai
        model: ...                                  # everything else unchanged

``chains/run_chain.py`` puts this directory on ``sys.path``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chain_spec import parse_username
from wired_state import AgentSession, ChainRunError, Teammate


class WiredMixin:
    """Persistent-identity hooks for a ``BaseAgent`` subclass (list it first)."""

    wired: Optional[AgentSession] = None
    agent_id: str = ""
    role: str = ""

    def _wired_setup(self, base: type, kwargs: Dict[str, Any]) -> None:
        username = kwargs.get("username") or ""
        self.agent_id, self.role = parse_username(username)
        self.wired = None
        accepted = inspect.signature(base.__init__).parameters
        base.__init__(self, **{k: v for k, v in kwargs.items() if k in accepted})

    # ── hooks ──────────────────────────────────────────────────────────────
    def attach_chain(self, session: AgentSession) -> None:
        if session.agent_id != self.agent_id or session.role != self.role:
            raise ChainRunError(f"session for {session.username} attached to {self.username}")
        self.wired = session

    def on_task_start(self, agent_id: str, role: str,
                      teammates: Sequence[Tuple[str, str, str]]) -> None:
        """teammates: (agent_id, role, username) of every other team member."""
        s = self._session(agent_id)
        if role != s.role:
            raise ChainRunError(f"{agent_id}: role {role!r} != session role {s.role!r}")
        s.start([Teammate(*t) for t in teammates])

    def on_turn(self, round_num: int, interactions: List[Dict[str, Any]]) -> None:
        """Log this agent's messages sent/received and item transfers (partner by agent id)."""
        if self.wired is None:
            return
        for item in interactions:
            self.wired.remember(round_num, item)

    def on_skill(self, round_num: int, outcome: Dict[str, Any]) -> None:
        """Record a craft/harvest that produced items, or a craft that failed."""
        if self.wired is not None:
            self.wired.learn(round_num, outcome)

    def on_task_end(self, agent_id: str, outcome: Dict[str, Any]):
        """Record the verified outcome and save the agent's state file."""
        return self._session(agent_id).finish(outcome)

    def _session(self, agent_id: str) -> AgentSession:
        if self.wired is None:
            raise ChainRunError(f"{self.username}: no chain session attached")
        if agent_id != self.agent_id:
            raise ChainRunError(f"{self.username}: hook called for {agent_id}")
        return self.wired

    # ── prompt ─────────────────────────────────────────────────────────────
    def _update_system_prompt(self):
        super()._update_system_prompt()
        block = self.wired.prompt_block() if self.wired is not None else ""
        history = getattr(self, "conversation_history", None)
        if block and history and history[0].get("role") == "system":
            history[0]["content"] = history[0]["content"] + block


def _wired(name: str, base: type, doc: str) -> type:
    def __init__(self, *, api_key=None, model=None, username=None, password=None,
                 base_url=None, dump_prompts=False, llm_params=None):
        self._wired_setup(base, dict(api_key=api_key, model=model, username=username,
                                     password=password, base_url=base_url,
                                     dump_prompts=dump_prompts, llm_params=llm_params))
    return type(name, (WiredMixin, base), {"__init__": __init__, "__doc__": doc,
                                           "__module__": __name__})


# Built on first access (PEP 562) so the hooks above import without AgentWorld's
# agents/ on sys.path; AgentFactory's getattr(module, "WiredOpenAIAgent") builds it.
_PROVIDERS = {
    "WiredOpenAIAgent": ("openai_agent", "OpenAIAgent",
                         "AgentWorld's OpenAI-compatible agent (vLLM, gateways) + hooks."),
    "WiredClaudeAgent": ("claude_agent", "ClaudeAgent", "AgentWorld's Claude agent + hooks."),
    "WiredGeminiAgent": ("gemini_agent", "GeminiAgent", "AgentWorld's Gemini agent + hooks."),
    "WiredDeepSeekAgent": ("deepseek_agent", "DeepSeekAgent", "AgentWorld's DeepSeek agent + hooks."),
}


def __getattr__(name: str):
    if name in _PROVIDERS:
        import importlib
        module, cls, doc = _PROVIDERS[name]
        klass = _wired(name, getattr(importlib.import_module(module), cls), doc)
        globals()[name] = klass
        return klass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
