"""Choice-mode social acts (Experiment 2) — shared helpers.

One place for everything the per-step social-act choice needs: act/channel
names, CLI parsing, menu + schema rendering for the parallel choice-mode
prompt templates, the imitation act gate, and horizon clamping.

Legacy mode never imports the rendered templates: ``--social-act-mode legacy``
keeps loading the original prompt files byte-for-byte (see
``action_selection.py``), which is the non-regression contract.
"""

from __future__ import annotations

import math
import os
import re
from typing import Optional, Sequence, Tuple

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

# One imitate() act may commit the motor channel for at most this many steps:
# a single act must not skip 20 action calls and mint 20 co-firing events
# against a message's one.
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


def imitation_gate(
    pos_i: Optional[Sequence[float]],
    pos_j: Optional[Sequence[float]],
    chamber_i,
    chamber_j,
    radius: float = 5.0,
) -> bool:
    """The imitation act gate: within ``radius`` AND same chamber.

    This is what makes committed replay state-correspondent — replaying
    ``Dig`` from 8 blocks away facing the wrong wall is noise — and what
    couples imitation to proximity by construction. Missing positions or
    chambers fail the gate (degrade to the informational fallback).
    """
    if pos_i is None or pos_j is None:
        return False
    if chamber_i is None or chamber_j is None or chamber_i != chamber_j:
        return False
    try:
        d = math.dist(tuple(pos_i)[:3], tuple(pos_j)[:3])
    except (TypeError, ValueError):
        return False
    return d <= radius


# ── Prompt rendering (choice mode only) ────────────────────────────────

_MENU_HEADER = (
    "═══════════════════════════════════════════\n"
    "YOUR SOCIAL ACT — CHOOSE AT MOST ONE PER STEP\n"
    "═══════════════════════════════════════════\n"
    "Alongside your game action you may take ONE social act aimed at ONE teammate\n"
    "(\"social_target\", form \"agent_N\", never yourself, never \"all\"). These acts are\n"
    "your ONLY window into what teammates are doing — nothing else reports their\n"
    "progress to you, and an act you never try teaches you nothing:"
)

_MENU_LINES = {
    "comm": (
        "- \"communicate\": send a short targeted message. Put the text in \"communication\"\n"
        "  and the SAME teammate in \"communication_target\" — make it ACTIONABLE for them\n"
        "  (what you observed, what you need from THEM, or what you commit to do). Sub-5-char\n"
        "  or repeated identical messages to the same teammate are filtered as spam."
    ),
    "obs": (
        "- \"observe\": silently study the teammate in \"social_target\". Next step you receive\n"
        "  their position, health, inventory and current beliefs — cheap and always\n"
        "  available, a good first move whenever you are unsure what a teammate is doing.\n"
        "  They are NOT notified."
    ),
    "imit": (
        "- \"imitate\": copy the teammate in \"social_target\". Set \"imitate_horizon\" (1-5):\n"
        "  if you are CLOSE to them (within ~5 blocks, same room), your next steps will\n"
        "  REPLAY their last few actions instead of your own choices; otherwise you just\n"
        "  receive a report of what they did. Use it when a teammate is close by and making\n"
        "  progress, or when you are stuck and they are not — copying is how you learn\n"
        "  skills you lack."
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
    lines = [_MENU_HEADER]
    for ch in ALL_CHANNELS:
        if ch in enabled_channels:
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
        "comm": '- "communicate": send the teammate a message (also fill ask_target/ask_message)',
        "obs": '- "observe": silently look up the teammate\'s state and beliefs',
        "imit": '- "imitate": copy the teammate\'s recent actions (requires being near them)',
    }
    return "\n".join(descs[ch] for ch in ALL_CHANNELS if ch in enabled_channels)


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
