"""Social-reasoning step sitting between the Hebbian graph and the action LLM.

The SocialModule consumes (per-teammate bond weights, recent bond deltas, this
agent's inbox of incoming messages, recent self-state) and produces a structured
SocialThought: who to ask for help, who to help in return, why the bonds are
moving the way they are, and which bonds the LLM claims to have used.

Architecturally a peer of BeliefSystem — same lifecycle, same LLM-call shape,
same response-format-enforced JSON. Lives in CustomAgent.on_messages, called
before action selection so its directive can be injected into the action prompt.

Couplings (controlled by --social-module CLI flag in multi_agent_craftium.py):
  - "prompt": directive rendered as text in the action prompt; action LLM may
    still ignore it
  - "bias": directive's ask_target literally overwrites the action's
    communication_target at the routing site
The "override" variant (task rewriting) was intentionally NOT implemented in
this version — added later if `bias` doesn't move the metric enough.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from agent_modules.llm_call import llm_call
from agent_modules.util import SocialThought, create_model_client

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

with open(os.path.join(_PROMPT_DIR, "social_module.txt"), "r", encoding="utf-8") as f:
    _social_prompt = f.read()


def _format_bond_table(
    bond_weights: Dict[str, float],
    bond_deltas: Dict[str, float],
) -> str:
    """Render the (w, Δw) table the LLM sees.

    Each line: 'agent_X:  w = 0.71  (Δ = +0.15)   ← STRENGTHENING/DECAYING/STABLE'.
    The verbal tag makes the dynamics legible without forcing the LLM to do
    sign-detection arithmetic on its own — small cognitive cost, smaller
    chance of mis-reading a "−0.08" as a positive number.
    """
    if not bond_weights:
        return "  (no teammates — you are alone)"
    lines = []
    for teammate in sorted(bond_weights.keys(), key=lambda k: -bond_weights[k]):
        w = bond_weights[teammate]
        d = bond_deltas.get(teammate, 0.0)
        if d > 0.02:
            tag = "← STRENGTHENING"
        elif d < -0.02:
            tag = "← DECAYING"
        else:
            tag = "← STABLE"
        sign = "+" if d >= 0 else "−"
        lines.append(f"  {teammate}:  w = {w:.2f}  (Δ = {sign}{abs(d):.2f})   {tag}")
    return "\n".join(lines)


def _format_incoming(incoming: List[str]) -> str:
    if not incoming:
        return "  (no messages this step)"
    return "\n".join(f"  - {msg}" for msg in incoming)


class SocialModule:
    """Per-agent social-reasoning module.

    One instance per agent. Holds the last produced SocialThought so the
    action prompt and the routing layer can both read the current directive
    without a re-call.
    """

    def __init__(
        self,
        agent_name: str,
        num_agents: int,
        social_model_client=None,
        override_prompt: Optional[str] = None,
        social_interval: int = 1,
    ):
        self.agent_name = agent_name
        self.num_agents = num_agents
        self.social_interval = social_interval
        self._call_count = 0
        self._client = (
            social_model_client
            if social_model_client
            else create_model_client(response_format=SocialThought)
        )
        self._prompt = override_prompt or _social_prompt
        self.last_thought: Optional[dict] = None  # last parsed SocialThought as dict

    async def deliberate(
        self,
        bond_weights: Dict[str, float],
        bond_deltas: Dict[str, float],
        incoming: List[str],
        teammate_names: str,
        last_action: str,
        last_reward: str,
        picked_object: str,
        position_text: str,
        cancellation_token,
    ) -> dict:
        """Run one deliberation step. Returns the parsed SocialThought dict.

        Caches the result in ``self.last_thought`` so the directive renderer
        and the routing-layer bias coupling can read the same thought without
        a duplicate LLM call.
        """
        self._call_count += 1
        if (
            self.social_interval > 1
            and self._call_count % self.social_interval != 0
            and self.last_thought is not None
        ):
            return self.last_thought

        bond_table = _format_bond_table(bond_weights, bond_deltas)
        incoming_block = _format_incoming(incoming)

        response = await llm_call(
            self._client,
            system_prompt=None,
            user_prompt=self._prompt,
            cancellation_token=cancellation_token,
            log_prefix=f"SocialModule[{self.agent_name}]: ",
            agent_name=self.agent_name,
            teammate_names=teammate_names,
            bond_table=bond_table,
            incoming_comms=incoming_block,
            last_action=last_action or "N/A",
            last_reward=last_reward or "N/A",
            picked_object=picked_object or "N/A",
            position_text=position_text or "Unknown",
        )

        if not response:
            response = {
                "bond_change_explanation": {},
                "reasoning": "(no thought — LLM call failed)",
                "referenced_bonds": {},
                "ask_target": None,
                "ask_message": None,
                "respond_to": [],
                "confidence": 0.0,
            }

        if not isinstance(response.get("bond_change_explanation"), dict):
            response["bond_change_explanation"] = {}
        if not isinstance(response.get("referenced_bonds"), dict):
            response["referenced_bonds"] = {}
        if not isinstance(response.get("respond_to"), list):
            response["respond_to"] = []

        self.last_thought = response
        return response

    def render_directive(self, thought: Optional[dict] = None) -> str:
        """Render the SocialThought as a text block for the action prompt.

        The action LLM sees this as a single section near the bottom of
        instruction_prompt_p2.txt, replacing the old raw 'Social bonds: ...'
        line. The deliberated directive — not raw weights — is what the
        action LLM now conditions on.
        """
        t = thought if thought is not None else self.last_thought
        if not t:
            return "Social directive: (none — social module disabled or not yet run)"

        ask_target = t.get("ask_target")
        ask_message = t.get("ask_message")
        respond_to = t.get("respond_to") or []
        if not isinstance(respond_to, list):
            respond_to = []
        reasoning = t.get("reasoning", "")
        bond_expl = t.get("bond_change_explanation") or {}
        if not isinstance(bond_expl, dict):
            bond_expl = {}

        outgoing = (
            f"Ask {ask_target} for help. Suggested message: \"{ask_message}\". "
            f"Put {ask_target} in your communication_target field and a help "
            f"request in your communication field."
            if ask_target
            else "No help to ask for this step."
        )
        incoming = (
            f"You have decided to help: {', '.join(respond_to)}. Your "
            f"communication this step should acknowledge them, and your "
            f"action should move toward or support their stated goal."
            if respond_to
            else "No pending requests warrant your help this step."
        )
        bond_notes = (
            "\n  ".join(f"- {k}: {v}" for k, v in bond_expl.items())
            if bond_expl
            else "(no significant bond changes)"
        )

        # Choice-mode act suggestion (absent in legacy thoughts → no line).
        suggest_act = t.get("suggest_act")
        suggest_target = t.get("suggest_target")
        suggest_line = (
            f"  Suggested social act: {suggest_act} toward {suggest_target} — "
            f"set social_act/social_target accordingly unless you have a "
            f"strong reason not to.\n"
            if suggest_act and suggest_target
            else ""
        )

        return (
            "Social directive (from your social-reasoning step):\n"
            f"  Reasoning: {reasoning}\n"
            f"{suggest_line}"
            f"  Outgoing: {outgoing}\n"
            f"  Incoming: {incoming}\n"
            f"  Bond changes noted:\n  {bond_notes}"
        )
