"""Per-agent, per-episode communication budget (the comm-budget sweep).

WHAT IT DOES
  Each agent gets a fixed budget of message TOKENS per episode. Every sent
  message is charged its tokenized length (counted with the served model's
  tokenizer; whitespace-words x 1.3 when no local model is loaded). A message
  longer than the per-message cap is cut to the cap; one longer than what is
  left is cut to the remainder and the agent is LOCKED; once locked, every
  later attempt is BLOCKED (the communication fields are blanked in code
  before routing, before the Hebbian co-firing events and before the comm
  reward tracker, so nothing downstream ever sees a phantom message).
  Receiving is free; charging is sender-side only. The ledger is rebuilt at
  every episode start, so the budget resets with the rest of the per-episode
  state.

  The agent is TOLD about it: a per-step ``{comm_budget}`` line in the action
  prompt (remaining tokens, ~messages, steps to go, what to spend it on), the
  system prompt's communication rule flipped from "REQUIRED EVERY STEP" to
  optional + budgeted, the JSON-schema hints relaxed from "never empty", and
  a matching line in the social module's deliberation prompt so it stops
  proposing ``ask_message`` once the budget is spent.

WHAT STAYS BYTE-IDENTICAL
  With ``--comm-budget-tokens`` unset (every legacy suite) there is no ledger,
  every placeholder below renders to its ORIGINAL pre-placeholder bytes
  (``COMM_RULE_LEGACY`` & co. are verbatim copies, same contract as
  ``team_scaling.LEGACY_PLACEHOLDERS``), ``{comm_budget}`` renders empty, and
  no new field is written to any log. Pinned by tests/test_comm_budget.py.

  The master switch is the ``WT_COMM_BUDGET`` environment variable, set by
  multi_agent_craftium.py when the flag is given (mirrors ``WT_TEAM_SCALING``),
  so prompt loaders that never see ``args`` (ActionSelection) agree with the
  training loop about which rule to render.

  A budget of 0 is the "zero" arm: same code path, same prompt structure,
  every message blocked from the first step.

Pure stdlib; safe to import anywhere (tests included).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

#: Master switch (set by the training loop; read by prompt loaders).
ENV_SWITCH = "WT_COMM_BUDGET"
#: Per-message cap forwarded to the static prompt text (same source of truth
#: as the ledger's cap; set next to ENV_SWITCH).
ENV_MSG_CAP = "WT_COMM_BUDGET_MSG_CAP"

#: Hard cap on the tokens ONE message may cost (~p99 of the observed
#: 12-word Gemma-E4B messages). Keeps tokens and message counts
#: interchangeable: budget / cap is the minimum number of messages an agent
#: can always afford.
DEFAULT_MSG_CAP = 32
#: Tokenizer-free estimate used when no local model is loaded (tests, API
#: backends): English runs ~1.3 model tokens per whitespace word.
FALLBACK_TOKENS_PER_WORD = 1.3
#: Typical cost of one short message, used only for the "about N messages"
#: rendering in the prompt (12.5 words x 1.3).
TOKENS_PER_SHORT_MESSAGE = 16


# ── Legacy prompt fragments (verbatim copies of the pre-placeholder bytes) ──
# Do NOT derive these from anything else: they must stay literal so legacy
# rendering can never drift. tests/test_comm_budget.py pins them.

COMM_RULE_LEGACY = (
    "TARGETED COMMUNICATION (REQUIRED EVERY STEP — there is NO broadcast channel):\n"
    "- Set \"communication_target\" to EXACTLY ONE teammate's name (form \"agent_N\", any agent other\n"
    "  than yourself). Never \"all\", never empty, never yourself.\n"
    "- Pick the recipient deliberately — whose situation does this message most help? Choose the\n"
    "  teammate your action affects or whose action affects yours (e.g. someone near you, or who\n"
    "  can act on what you just observed).\n"
    "- Make it ACTIONABLE for that recipient: what you observed, a scheme you just confirmed, what\n"
    "  you need from THEM, or what you commit to do. Share schemes a teammate lacks (e.g. \"digging\n"
    "  that block dropped wood\"). Vary messages — sub-5-char or repeated identical messages to the\n"
    "  same teammate are filtered as spam and never arrive."
)

COMM_RULE_BUDGET = (
    "TARGETED COMMUNICATION (OPTIONAL and BUDGETED — there is NO broadcast channel):\n"
    "- You have a fixed per-episode budget of message tokens; the amount left is shown in your\n"
    "  per-step context. Every message you send costs its length in tokens. When the budget\n"
    "  reaches 0 you cannot send anything for the rest of the episode.\n"
    "- Send a message ONLY when a specific teammate can act on it: a request for help, a\n"
    "  commitment about what you will do, or a discovery they lack. On most steps leave\n"
    "  \"communication\" and \"communication_target\" EMPTY — silence costs nothing.\n"
    "- When you do send: put EXACTLY ONE teammate's name (form \"agent_N\", never yourself,\n"
    "  never \"all\") in \"communication_target\"; keep the message short (it is cut at {cap}\n"
    "  tokens); never repeat a message you already sent."
)

COMM_TARGET_RULE_LEGACY = "never \"all\", never empty."
COMM_TARGET_RULE_BUDGET = (
    "never \"all\"; leave it empty on steps where you send no message."
)

COMM_FIELD_HINT_LEGACY = (
    " — an observation, request, or commitment relevant to THEM; never empty"
)
COMM_FIELD_HINT_BUDGET = (
    " — ONLY when you choose to spend communication budget this step (a request, "
    "commitment or discovery they can act on); otherwise the empty string"
)

STATIC_PLACEHOLDERS = ("comm_rule", "comm_target_rule", "comm_field_hint")


# ── Master switch ───────────────────────────────────────────────────────

def comm_budget_enabled() -> bool:
    """The master switch as seen via the environment."""
    return os.environ.get(ENV_SWITCH) == "1"


def msg_cap_from_env() -> int:
    try:
        return int(os.environ.get(ENV_MSG_CAP, DEFAULT_MSG_CAP))
    except ValueError:
        return DEFAULT_MSG_CAP


def set_env_switch(enabled: bool, msg_cap: int = DEFAULT_MSG_CAP) -> None:
    """Set (or clear) the master switch for this process."""
    if enabled:
        os.environ[ENV_SWITCH] = "1"
        os.environ[ENV_MSG_CAP] = str(int(msg_cap))
    else:
        os.environ.pop(ENV_SWITCH, None)
        os.environ.pop(ENV_MSG_CAP, None)


# ── Static prompt substitution ──────────────────────────────────────────

def static_placeholders(enabled: bool, msg_cap: int) -> Dict[str, str]:
    if enabled:
        return {
            "comm_rule": COMM_RULE_BUDGET.replace("{cap}", str(int(msg_cap))),
            "comm_target_rule": COMM_TARGET_RULE_BUDGET,
            "comm_field_hint": COMM_FIELD_HINT_BUDGET,
        }
    return {
        "comm_rule": COMM_RULE_LEGACY,
        "comm_target_rule": COMM_TARGET_RULE_LEGACY,
        "comm_field_hint": COMM_FIELD_HINT_LEGACY,
    }


def apply_comm_budget_static(text: str, enabled: Optional[bool] = None,
                             msg_cap: Optional[int] = None) -> str:
    """Literal-substitute the STATIC comm-budget placeholders in ``text``.

    Plain ``str.replace`` (no ``str.format``), so the JSON ``{{...}}`` escapes
    and the per-step placeholders (``{comm_budget}``, ``{task}``, ...) survive
    for the per-step formatting. ``enabled=None`` reads the master switch.
    """
    if enabled is None:
        enabled = comm_budget_enabled()
    if msg_cap is None:
        msg_cap = msg_cap_from_env()
    for key, value in static_placeholders(enabled, msg_cap).items():
        text = text.replace("{" + key + "}", value)
    return text


def apply_comm_budget_to_prompts(prompts: dict, enabled: Optional[bool] = None,
                                 msg_cap: Optional[int] = None) -> dict:
    """Substitute across the load_prompts() dict (one level of nesting)."""
    out = {}
    for key, value in prompts.items():
        if isinstance(value, str):
            out[key] = apply_comm_budget_static(value, enabled, msg_cap)
        elif isinstance(value, dict):
            out[key] = {
                k: (apply_comm_budget_static(v, enabled, msg_cap)
                    if isinstance(v, str) else v)
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out


# ── Token counting ──────────────────────────────────────────────────────

def fallback_token_count(text) -> int:
    """Tokenizer-free estimate: ceil(1.3 x whitespace words)."""
    words = len(str(text or "").split())
    return int(math.ceil(FALLBACK_TOKENS_PER_WORD * words)) if words else 0


def model_token_count(text) -> Optional[int]:
    """Count with the loaded local model's tokenizer, or None.

    Only consults ``local_model_client`` when it is ALREADY imported (i.e. a
    local model backs this run); never imports torch/transformers itself, so
    tests and API-backed runs stay light.
    """
    mod = sys.modules.get("mindforge.agent_modules.local_model_client")
    if mod is None:
        return None
    fn = getattr(mod, "count_text_tokens", None)
    if fn is None:
        return None
    try:
        return fn(text)
    except Exception:  # pragma: no cover - tokenizer misbehaviour
        return None


def make_token_counter(
    model_counter: Optional[Callable[[str], Optional[int]]] = None,
) -> Callable[[str], int]:
    """Counter that prefers the model tokenizer and falls back to the estimate."""
    counter = model_counter if model_counter is not None else model_token_count

    def _count(text) -> int:
        n = counter(text)
        return int(n) if n is not None else fallback_token_count(text)

    return _count


# ── Ledger ──────────────────────────────────────────────────────────────

STATUS_SENT = "sent"
STATUS_TRUNCATED = "truncated"
STATUS_BLOCKED = "blocked"


@dataclass
class ChargeResult:
    text: str              # what actually goes on the wire ("" when blocked)
    status: str            # sent | truncated | blocked
    tokens_model: int      # token count of the ORIGINAL text
    charged: int           # tokens deducted (0 when blocked)
    budget_left: int       # after this charge
    exhausted_now: bool    # True on the step the agent hits 0 (first time)


@dataclass
class AgentBudgetState:
    total: int
    spent: int = 0
    sent: int = 0
    truncated: int = 0
    blocked: int = 0
    exhausted_step: Optional[int] = None

    @property
    def left(self) -> int:
        return max(0, int(self.total) - int(self.spent))

    @property
    def locked(self) -> bool:
        return self.left <= 0

    def as_dict(self) -> dict:
        return {
            "budget": int(self.total), "spent": int(self.spent),
            "left": int(self.left), "sent": int(self.sent),
            "truncated": int(self.truncated), "blocked": int(self.blocked),
            "exhausted_step": self.exhausted_step,
        }


class CommBudgetLedger:
    """Per-agent token ledger for ONE episode.

    ``token_counter`` maps message text -> int tokens (see
    :func:`make_token_counter`); the default is the tokenizer-free estimate.
    """

    def __init__(self, agent_ids: Iterable[int], budget_tokens: int,
                 msg_cap: int = DEFAULT_MSG_CAP,
                 token_counter: Optional[Callable[[str], int]] = None):
        if budget_tokens is None or int(budget_tokens) < 0:
            raise ValueError("budget_tokens must be a non-negative integer")
        if int(msg_cap) <= 0:
            raise ValueError("msg_cap must be positive")
        self.budget_tokens = int(budget_tokens)
        self.msg_cap = int(msg_cap)
        self._count = token_counter or fallback_token_count
        self._state: Dict[int, AgentBudgetState] = {
            int(a): AgentBudgetState(total=self.budget_tokens) for a in agent_ids
        }

    # ── queries ──
    def state(self, agent_id: int) -> AgentBudgetState:
        return self._state[int(agent_id)]

    def left(self, agent_id: int) -> int:
        return self.state(agent_id).left

    def is_locked(self, agent_id: int) -> bool:
        return self.state(agent_id).locked

    def count_tokens(self, text) -> int:
        return int(self._count(text))

    def reset(self) -> None:
        """Fresh budgets for every agent (the training loop rebuilds the
        ledger per episode instead, but this keeps the class self-contained)."""
        for a in list(self._state):
            self._state[a] = AgentBudgetState(total=self.budget_tokens)

    # ── the one mutating operation ──
    def charge(self, agent_id: int, text, step: int) -> ChargeResult:
        st = self.state(agent_id)
        original = str(text or "")
        n = self.count_tokens(original)

        if st.locked:
            st.blocked += 1
            first = st.exhausted_step is None
            if first:
                st.exhausted_step = int(step)   # zero-budget arm: first attempt
            return ChargeResult(text="", status=STATUS_BLOCKED, tokens_model=n,
                                charged=0, budget_left=st.left,
                                exhausted_now=first)

        allowed = min(self.msg_cap, st.left)
        sent_text, charged, truncated = original, n, False
        if n > allowed:
            sent_text, charged = self._truncate(original, allowed)
            truncated = True
            if not sent_text:
                # Not even one word fits in what is left: the remainder is
                # unusable, so spend it and lock.
                st.spent = st.total
                st.blocked += 1
                st.exhausted_step = int(step)
                return ChargeResult(text="", status=STATUS_BLOCKED,
                                    tokens_model=n, charged=0, budget_left=0,
                                    exhausted_now=True)

        st.spent += int(charged)
        st.sent += 1
        if truncated:
            st.truncated += 1
        exhausted_now = st.locked and st.exhausted_step is None
        if exhausted_now:
            st.exhausted_step = int(step)
        return ChargeResult(
            text=sent_text,
            status=STATUS_TRUNCATED if truncated else STATUS_SENT,
            tokens_model=n, charged=int(charged), budget_left=st.left,
            exhausted_now=exhausted_now,
        )

    def _truncate(self, text: str, allowed: int):
        """Longest word-boundary prefix costing <= ``allowed`` tokens."""
        words = text.split()
        for k in range(len(words) - 1, 0, -1):
            cand = " ".join(words[:k])
            c = self.count_tokens(cand)
            if c <= allowed:
                return cand, c
        return "", 0

    # ── prompt rendering (each line carries its own leading newline so the
    # placeholder can sit at the end of an existing line and legacy renders
    # add NOTHING) ──
    def _approx_msgs(self, left: int) -> int:
        return max(1, int(round(left / TOKENS_PER_SHORT_MESSAGE)))

    def render_action_line(self, agent_id: int, steps_left: int) -> str:
        st = self.state(agent_id)
        if st.total <= 0:
            return (
                "\nCommunication budget: NONE this episode. Leave \"communication\" "
                "and \"communication_target\" empty; coordinate by moving to "
                "teammates and working the same target."
            )
        if st.locked:
            return (
                f"\nCommunication budget: EXHAUSTED (0 of {st.total} tokens left). "
                "You cannot send messages for the rest of this episode; leave "
                "\"communication\" and \"communication_target\" empty and "
                "coordinate by moving to teammates and working the same target."
            )
        return (
            f"\nCommunication budget: {st.left} of {st.total} tokens left this "
            f"episode (about {self._approx_msgs(st.left)} short messages), about "
            f"{max(0, int(steps_left))} steps to go. At 0 you cannot send anything "
            "for the rest of the episode. Spend it only on a request, commitment "
            "or discovery a specific teammate can act on; otherwise leave "
            "\"communication\" and \"communication_target\" empty."
        )

    def render_social_line(self, agent_id: int) -> str:
        st = self.state(agent_id)
        name = f"agent_{int(agent_id)}"
        if st.total <= 0:
            return (
                f"\nCommunication budget for {name}: NONE this episode — no "
                "message can be sent. Set ask_target and ask_message to null; "
                "help or seek help by co-locating and working the same target."
            )
        if st.locked:
            return (
                f"\nCommunication budget for {name}: EXHAUSTED (0 of {st.total} "
                "tokens). No message can be sent this episode. Set ask_target and "
                "ask_message to null; help or seek help by co-locating and working "
                "the same target."
            )
        return (
            f"\nCommunication budget for {name}: {st.left} of {st.total} tokens "
            f"left this episode (about {self._approx_msgs(st.left)} short "
            "messages). A message costs its length; suggest ask_message only when "
            "the request is worth spending budget on."
        )

    # ── summaries ──
    def summary(self) -> dict:
        return {
            "budget_tokens": self.budget_tokens,
            "msg_cap": self.msg_cap,
            "per_agent": {
                f"agent_{a}": st.as_dict() for a, st in sorted(self._state.items())
            },
        }
