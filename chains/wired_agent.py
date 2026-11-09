"""Wired Together adapters for AgentWorld's reference runner (``agent.provider: custom``).

Each class subclasses one of AgentWorld's own provider agents, so the policy,
the game tools, one-action-per-turn ``execute_single_tool_call`` and the
``[TOOL_CALL_INFO]`` / ``[TOOL_RESULT]`` records that ``agents/run.py`` reads
are inherited. What the adapter changes:

* identity — ``agent_id`` and ``role`` parsed from the username
  (``a3_weapon_smith`` -> ``a3``, ``weapon_smith``); any other username fails;
* communication — AgentWorld's global ``chat`` is replaced by
  ``direct_chat(target, message)``: one teammate per message, seen only by
  that teammate (in its system prompt, every arm). Global chat history is
  hidden. The name keeps AgentWorld's runner treating it as a chat (counted in
  the chats metric, not against the action cap); the experiment's
  communication switches (``--no-communication``, discussion rounds) still
  decide whether the tool is offered;
* hooks — ``on_task_start(agent_id, role, teammates)``,
  ``on_turn(round_num, interactions)``, ``on_skill(round_num, outcome)`` and
  ``on_task_end(agent_id, outcome)``, which read and write the agent's
  persistent state through the :class:`wired_state.AgentSession` that
  ``chains/run_chain.py`` attaches;
* prompt — the session's team-history block (earlier outcomes, past
  interactions, skill log and, in the ``hebbian`` arm, the bond line) and the
  direct messages are appended to the system prompt, which the base class
  rebuilds on every call.

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
from wired_state import DM_SENT, AgentSession, ChainRunError, Teammate

DIRECT_CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "direct_chat",
        "description": ("Send a private message to ONE teammate. Only that teammate sees it; "
                        "there is no group chat. To tell several teammates, message each one."),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string",
                           "description": "The teammate's username, e.g. a2_smelter."},
                "message": {"type": "string", "description": "What to tell them."},
            },
            "required": ["target", "message"],
        },
    },
}


def _tool_name(tool: Dict[str, Any]) -> str:
    return tool.get("function", {}).get("name", tool.get("name", ""))


class WiredMixin:
    """Persistent identity, direct messages and hooks for a ``BaseAgent`` subclass."""

    wired: Optional[AgentSession] = None
    agent_id: str = ""
    role: str = ""
    _comm_allowed: bool = True

    def _wired_setup(self, base: type, kwargs: Dict[str, Any]) -> None:
        username = kwargs.get("username") or ""
        self.agent_id, self.role = parse_username(username)
        self.wired = None
        accepted = inspect.signature(base.__init__).parameters
        base.__init__(self, **{k: v for k, v in kwargs.items() if k in accepted})
        self._rebuild_tools()

    # ── communication: direct messages only ───────────────────────────────
    # AgentWorld reads ``allow_chat`` to offer its global chat (tool list,
    # random-agent baseline). It is always False here; the value the experiment
    # sets is kept and decides whether direct_chat is offered instead.
    @property
    def allow_chat(self) -> bool:
        return False

    @allow_chat.setter
    def allow_chat(self, value: bool) -> None:
        self._comm_allowed = bool(value)

    def _rebuild_tools(self) -> None:
        super()._rebuild_tools()          # honours transfer and discussion-phase switches
        tools = [t for t in self.tools if _tool_name(t) != "chat"]
        if self._comm_allowed or getattr(self, "discussion_phase", False):
            tools.append(DIRECT_CHAT_TOOL)
        self.tools = tools

    def _get_current_chat_messages(self) -> str:
        return ""                          # no global chat; direct messages are in the prompt

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        name = tool_call.get("name", "")
        if name == "direct_chat":
            return self._direct_chat(tool_call.get("arguments") or {})
        if name == "chat":
            return ("Error: there is no group chat. Use direct_chat(target, message) "
                    "to message one teammate.")
        return super()._execute_tool_call(tool_call)

    def _direct_chat(self, args: Dict[str, Any]) -> str:
        if not (self._comm_allowed or getattr(self, "discussion_phase", False)):
            return "Error: communication is disabled in this run."
        if self.wired is None or self.wired.mailbox is None:
            return "Error: direct_chat needs a chain run (chains/run_chain.py)."
        target = str(args.get("target") or "").strip().lstrip("@")
        message = str(args.get("message") or "").strip()
        if not message:
            return "Error: message is empty."
        mate = self.wired.resolve(target)
        if mate is None:
            names = ", ".join(t.username for t in self.wired.teammates)
            return f"Error: {target!r} is not a teammate. Teammates: {names}."
        self.wired.mailbox.post(self.wired.agent_id, self.wired.username, mate.agent_id, message)
        return f"{DM_SENT} {mate.username}: {message}"

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
        s = self.wired
        extra = (s.prompt_block() + s.inbox_block()) if s is not None else ""
        history = getattr(self, "conversation_history", None)
        if extra and history and history[0].get("role") == "system":
            history[0]["content"] = history[0]["content"] + extra


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
