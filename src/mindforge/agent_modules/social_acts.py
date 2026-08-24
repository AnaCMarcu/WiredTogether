"""Choice-mode social acts (Experiment 2) — shared helpers.

One place for everything the per-step social-act choice needs: act/channel
names, CLI parsing, menu + schema rendering for the parallel choice-mode
prompt templates, guided-imitation delivery + adoption tracking, and
horizon clamping.

Legacy mode never imports the rendered templates: ``--social-act-mode legacy``
keeps loading the original prompt files byte-for-byte (see
``action_selection.py``), which is the non-regression contract.
"""

from __future__ import annotations

import os
import re
from typing import Sequence, Tuple

# ── Canonical names ────────────────────────────────────────────────────
# LLM-facing act verbs ↔ Hebbian channel tags.
ACT_COMMUNICATE = "communicate"
ACT_OBSERVE = "observe"
ACT_IMITATE = "imitate"
ACT_NONE = "none"

ACT_TO_CHANNEL = {
    ACT_COMMUNICATE: "comm",
    ACT_OBSERVE: "obs",
    ACT_IMITATE: "imit",
}
CHANNEL_TO_ACT = {v: k for k, v in ACT_TO_CHANNEL.items()}

ALL_CHANNELS = ("comm", "obs", "imit")

# Max actions one imitate() act may fetch (and thus the max adoption credit
# a single act can mint) — bounds imit's per-act credit against a message's
# one event.
IMITATE_MAX_HORIZON = 5

_ACT_ALIASES = {
    "communicate": ACT_COMMUNICATE, "comm": ACT_COMMUNICATE,
    "message": ACT_COMMUNICATE, "talk": ACT_COMMUNICATE, "say": ACT_COMMUNICATE,
    "observe": ACT_OBSERVE, "obs": ACT_OBSERVE, "watch": ACT_OBSERVE,
    "look_at": ACT_OBSERVE, "observation": ACT_OBSERVE,
    "imitate": ACT_IMITATE, "imit": ACT_IMITATE, "copy": ACT_IMITATE,
    "mimic": ACT_IMITATE, "imitation": ACT_IMITATE, "replay": ACT_IMITATE,
    "none": ACT_NONE, "null": ACT_NONE, "": ACT_NONE,
}


def parse_channels_csv(text: str) -> Tuple[str, ...]:
    """Parse a ``--social-acts`` / ``--cofiring-channels`` CSV into channels.

    Accepts channel tags ("comm,obs,imit") or act verbs ("communicate,...");
    "none" (or empty) → empty tuple. Order is normalised to ALL_CHANNELS
    order; unknown entries raise so a typo can't silently drop an arm's
    channel.
    """
    if text is None:
        return ()
    tokens = [t.strip().lower() for t in str(text).split(",") if t.strip()]
    if not tokens or tokens == ["none"]:
        return ()
    chans = set()
    for tok in tokens:
        if tok in CHANNEL_TO_ACT:            # already a channel tag
            chans.add(tok)
            continue
        act = _ACT_ALIASES.get(tok)
        if act in ACT_TO_CHANNEL:
            chans.add(ACT_TO_CHANNEL[act])
        elif act == ACT_NONE:
            continue
        else:
            raise ValueError(
                f"unknown social act/channel {tok!r} "
                f"(expected comm|obs|imit|none)"
            )
    return tuple(c for c in ALL_CHANNELS if c in chans)


def normalize_social_act(raw, enabled_channels: Sequence[str]) -> str:
    """Map an LLM-produced act string onto a canonical ENABLED act.

    Tolerant of case/decoration ("Observe agent_1" → "observe"). Anything
    unknown, or an act whose channel is not in the menu, degrades to "none" —
    the affordance ablation is enforced here, not trusted to the prompt.
    """
    if not isinstance(raw, str):
        return ACT_NONE
    tok = raw.strip().lower()
    tok = re.split(r"[\s(:,]", tok, 1)[0] if tok else ""
    act = _ACT_ALIASES.get(tok, ACT_NONE)
    if act == ACT_NONE:
        return ACT_NONE
    if ACT_TO_CHANNEL[act] not in enabled_channels:
        return ACT_NONE
    return act


def clamp_horizon(h) -> int:
    """Clamp the requested imitation horizon to [1, IMITATE_MAX_HORIZON]."""
    try:
        v = int(h)
    except (TypeError, ValueError):
        v = 1
    return max(1, min(IMITATE_MAX_HORIZON, v))


# ── Guided imitation (2026-08-07 redesign) ─────────────────────────────
# imitate(j, h) no longer force-replays: the agent RECEIVES j's last h
# actions + j's state (task, position) with instructions to judge whether
# its own situation matches, and stays in control of every action. The
# imit co-firing channel is credited on ADOPTION: each step the imitator's
# chosen action matches the next unconsumed element of the delivered
# sequence (in order, non-matching steps tolerated) within the window.

# A delivered sequence of h actions stays adoptable for h*FACTOR steps —
# slack for aiming/turning between copied actions.
ADOPTION_WINDOW_FACTOR = 3


class PendingImitation:
    """Tracks one delivered imitation sequence for adoption crediting."""

    def __init__(self, target: int, sequence: Sequence[str], start_step: int):
        self.target = int(target)
        self.sequence = [a for a in sequence if a]
        self.start_step = int(start_step)
        self.ptr = 0

    def done(self) -> bool:
        return self.ptr >= len(self.sequence)

    def expired(self, step: int) -> bool:
        window = ADOPTION_WINDOW_FACTOR * max(1, len(self.sequence))
        return step > self.start_step + window

    def note_action(self, action) -> bool:
        """True iff ``action`` matches the next expected element.

        Advances the pointer on a match (that step earns imit co-firing
        credit); a non-matching step neither advances nor resets — the
        agent may interleave aiming/turning between copied actions.
        """
        if self.done() or not action:
            return False
        if action == self.sequence[self.ptr]:
            self.ptr += 1
            return True
        return False


def render_imitation_payload(
    target_name: str,
    actions: Sequence[str],
    task,
    position_text,
    chamber,
) -> str:
    """The guided-imitation payload delivered to the initiator next step.

    Carries the target's action sequence + state AND the instructions for
    judging applicability — the agent decides whether re-enacting makes
    sense; nothing is auto-executed.
    """
    seq = " -> ".join(a for a in actions if a) or "(no recorded actions yet)"
    return (
        f"[imitation of {target_name}] Their task: {task or '(unknown)'} | "
        f"Position: {position_text or 'Unknown'} | Chamber: {chamber or '?'}\n"
        f"  Their last actions (oldest to newest): {seq}\n"
        f"  HOW TO USE THIS: compare their situation to yours. If your position and task\n"
        f"  are similar to theirs, RE-ENACT the sequence step by step through your own\n"
        f"  \"action\" choices, starting with the first action. If their situation does not\n"
        f"  match yours yet, keep it as a learned recipe — re-enact it when you find\n"
        f"  yourself in a similar situation."
    )


# ── Prompt rendering (choice mode only) ────────────────────────────────

_MENU_HEADER_BASE = (
    "═══════════════════════════════════════════\n"
    "YOUR SOCIAL ACT — CHOOSE AT MOST ONE PER STEP\n"
    "═══════════════════════════════════════════\n"
    "Alongside your game action you may take ONE social act aimed at ONE teammate\n"
    "(\"social_target\", form \"agent_N\", never yourself, never \"all\"). These acts are\n"
    "your ONLY window into what teammates are doing — nothing else reports their\n"
    "progress to you, and an act you never try teaches you nothing."
)

# Levels order the menu by engagement depth (obs=1, comm=2, imit=3) —
# independent of ALL_CHANNELS, which stays the canonical channel order.
_LEVEL_ORDER = ("obs", "comm", "imit")

# One gloss per level, mentioned ONLY when that act is in the arm's menu —
# an agent is never told about engagement levels it cannot trigger.
_LEVEL_GLOSS = {
    "obs": ("Level 1 gathers information ABOUT a teammate "
            "(they are unaffected)"),
    "comm": ("Level 2 exchanges information WITH a teammate "
             "(both of you are involved)"),
    "imit": ("Level 3 lets a teammate's behavior SHAPE YOUR OWN "
             "(their experience becomes your skill)"),
}
# The ladder summary presumes all three rungs — only shown for a full menu.
_LEVEL_LADDER_TAIL = ("Lower levels inform higher ones: know who matters, "
                      "align with them, learn from them.")

_MENU_LINES = {
    "obs": (
        "- LEVEL 1 — \"observe\": observe the behavior of the teammate in \"social_target\"\n"
        "  and what they have learnt. PURPOSE: an observation gives you, next step, their\n"
        "  position, health, inventory and current beliefs; they are not notified.\n"
        "  BENEFIT: knowing your teammates is the basis of working with them — it tells\n"
        "  you who is where, who needs help, and who is worth joining, so you act on what\n"
        "  they are actually doing rather than on guesses."
    ),
    "comm": (
        "- LEVEL 2 — \"communicate\": send a short targeted message. PURPOSE: coordinate\n"
        "  with a specific teammate — requests, warnings, commitments, discoveries. Put\n"
        "  the text in \"communication\" and the SAME teammate in \"communication_target\";\n"
        "  make it ACTIONABLE for them. BENEFIT: a team that talks aligns who does what,\n"
        "  gets help when stuck, and spreads what one agent learned to the whole team.\n"
        "  Sub-5-char or repeated identical messages to the same teammate are filtered\n"
        "  as spam."
    ),
    "imit": (
        "- LEVEL 3 — \"imitate\": learn from a teammate's behavior. PURPOSE: set\n"
        "  \"imitate_horizon\" (1-5) and next step you receive the last 1-5 actions the\n"
        "  teammate in \"social_target\" took (oldest to newest), plus their task and\n"
        "  position. HOW TO USE IT: compare their situation to yours — if it matches,\n"
        "  RE-ENACT their sequence through your own \"action\" choices, step by step; if\n"
        "  not, keep it as a learned recipe and use it when you encounter a similar\n"
        "  situation. BENEFIT: skills that already work for someone else become yours\n"
        "  without having to discover them alone."
    ),
}

_MENU_FOOTER = (
    "- \"none\": no social act this step. A real option — but picking \"none\" every step\n"
    "  is as much a habit as spamming one act. Early in an episode, TRY the acts\n"
    "  available to you to discover what they return; after that, engage whenever a\n"
    "  teammate is relevant to your task or you are stuck."
)


def render_social_act_menu(enabled_channels: Sequence[str]) -> str:
    """The system-prompt menu block: fixed header + one line per enabled act.

    Kept the same shape across arms (header/footer constant, one block per
    enabled act) so prompt length differs as little as the ablation allows.
    """
    enabled = [ch for ch in _LEVEL_ORDER if ch in enabled_channels]
    header = _MENU_HEADER_BASE
    if enabled:
        glosses = "; ".join(_LEVEL_GLOSS[ch] for ch in enabled)
        header += (
            "\nSocial acts are LEVELS of engagement — the higher the level, the"
            "\ndeeper the involvement with your teammate: " + glosses + "."
        )
        if len(enabled) == len(_LEVEL_ORDER):
            header += "\n" + _LEVEL_LADDER_TAIL
    header += "\nYour available act(s):"
    lines = [header]
    for ch in enabled:
        lines.append(_MENU_LINES[ch])
    lines.append(_MENU_FOOTER)
    return "\n".join(lines)


def render_social_act_schema(enabled_channels: Sequence[str]) -> str:
    """The instruction-prompt RESPONSE FORMAT line for choice mode.

    Doubled braces survive the per-step ``safe_format`` exactly like the
    legacy template's schema line.
    """
    acts = " | ".join(
        f'"{CHANNEL_TO_ACT[ch]}"' for ch in ALL_CHANNELS if ch in enabled_channels
    )
    acts = (acts + ' | "none"') if acts else '"none"'
    comm_enabled = "comm" in enabled_channels
    comm_field = (
        '"communication": "<message text IF social_act is communicate, else empty>", '
        '"communication_target": "<same teammate as social_target IF communicating, else empty>", '
        if comm_enabled else
        '"communication": "", "communication_target": "", '
    )
    return (
        "RESPONSE FORMAT — respond with EXACTLY this JSON, no extra text:\n"
        '{{"thoughts": "<spatial reasoning + predicted observation (\'I expect to see X next step\') '
        '+ check of last step\'s prediction vs reality>", '
        '"action": "<one action name>", '
        f"{comm_field}"
        f'"social_act": {acts}, '
        '"social_target": "<a teammate\'s name in the form agent_N IF social_act is not none, else empty>", '
        '"imitate_horizon": <1-5 IF social_act is imitate, else 0>}}'
    )


def render_enabled_acts_menu(enabled_channels: Sequence[str]) -> str:
    """Short act list for the social module's choice-mode deliberation."""
    if not enabled_channels:
        return '(none — no social acts are available this run; always suggest null)'
    descs = {
        "obs": ('- Level 1, "observe": look up the teammate\'s state and '
                'beliefs (information ABOUT them). Benefit: ground truth '
                'about what they are doing, without depending on what they '
                'say. Suggest it when you are unsure what a teammate is up '
                'to or whether joining them is worthwhile.'),
        "comm": ('- Level 2, "communicate": send the teammate a message (also '
                 'fill ask_target/ask_message) — information exchanged WITH '
                 'them. Benefit: coordination — requests, warnings, sharing '
                 'discoveries. Suggest it when the agent needs something FROM '
                 'a teammate or has something actionable to give.'),
        "imit": ('- Level 3, "imitate": fetch the teammate\'s recent action '
                 'sequence + their task/position so the agent can re-enact '
                 'it if its own situation matches — their behavior shaping '
                 'the agent\'s own. Benefit: transfers a working behavior. '
                 'Suggest it when the agent seems stuck while a teammate in '
                 'a SIMILAR situation (same room, similar task) is making '
                 'progress — the copied sequence only helps if the state it '
                 'worked in matches the agent\'s own.'),
    }
    return "\n".join(descs[ch] for ch in _LEVEL_ORDER if ch in enabled_channels)


# ── Template loading ───────────────────────────────────────────────────

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_choice_templates(enabled_channels: Sequence[str]):
    """Load the parallel choice-mode templates, pre-rendering the static parts.

    Returns (system_prompt_template, instruction_template):
    - system template still contains {environment_prompt} (filled by
      ActionSelection at construction, same as legacy);
      {social_act_menu} is pre-rendered here.
    - instruction template has {social_act_schema} pre-rendered (via replace,
      NOT format, so the per-step placeholders and the schema's doubled
      braces survive untouched for the per-step safe_format).
    """
    with open(os.path.join(_PROMPT_DIR, "system_prompt_choice.txt"), "r",
              encoding="utf-8") as f:
        system_txt = f.read()
    with open(os.path.join(_PROMPT_DIR, "instruction_prompt_p2_choice.txt"), "r",
              encoding="utf-8") as f:
        instruction_txt = f.read()
    system_txt = system_txt.replace(
        "{social_act_menu}", render_social_act_menu(enabled_channels)
    )
    instruction_txt = instruction_txt.replace(
        "{social_act_schema}", render_social_act_schema(enabled_channels)
    )
    return system_txt, instruction_txt


def load_social_module_choice_prompt(enabled_channels: Sequence[str]) -> str:
    """The SocialModule's choice-mode deliberation prompt, menu pre-rendered."""
    with open(os.path.join(_PROMPT_DIR, "social_module_choice.txt"), "r",
              encoding="utf-8") as f:
        txt = f.read()
    return txt.replace(
        "{enabled_acts_menu}", render_enabled_acts_menu(enabled_channels)
    )


def render_observed_notice(observer_name: str) -> str:
    """Notice delivered to the TARGET of an observe act (bidirectional mode).

    Makes the channel delivery-symmetric: the observed agent learns who is
    paying attention to it, mirroring how a message lands in the recipient's
    inbox. Kept short — it is information, not an instruction.
    """
    return (
        f"[social] {observer_name} observed you last step: they can now see "
        f"your position, status, inventory, and stated beliefs."
    )


def render_imitated_notice(imitator_name: str, action) -> str:
    """Notice delivered to the TARGET of an adopted imitation (bidirectional
    mode): its behavior was influential enough that a teammate re-enacted it."""
    act = action or "one of your recent actions"
    return (
        f"[social] {imitator_name} imitated you: they re-enacted your "
        f"action '{act}' after studying your recent behavior."
    )
