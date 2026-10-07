"""Hub-topology prompt rewrites (``--orchestrator-variant hmas2``).

In the hub topology agents cannot message teammates: every message an agent
writes is a report to the orchestrator, and the orchestrator's message is
the only message an agent receives. A handful of prompt lines state or
assume the peer channel ("communication_target must be one of your
teammates", "message a teammate") and would contradict the topology. They
are rewritten here — and ONLY in hub mode — as exact-substring replacements
on the RAW file text, before team scaling and the comm-budget placeholders
are resolved (both later stages use plain ``str.replace``, so placeholders
inside a replacement still resolve).

Legacy rendering is byte-identical by construction: with the rewrites
disabled nothing here touches any text. ``apply_hub_rewrites`` raises at
startup unless every source occurs exactly once, so a prompt edit that
breaks a rewrite fails loudly instead of silently leaving peer wording in a
hub run (pinned again by tests/test_hub_prompts.py).

The set is deliberately MINIMAL — every rewrite is a prompt difference
between the hub and the peer conditions:
- must change (false channel/target claims): the action template's target
  line and JSON hint, the critic's "message a teammate" advice;
- framing only: where incoming messages are described as coming from
  teammates.
Left alone: "Announce your cell" (announcing to the orchestrator is right in
hub mode), the generic environment prompt, and role files the launchers
never load.

Stdlib only.
"""

from __future__ import annotations

#: Inbox header in hub mode (replaces "Communications from other agents").
HUB_COMM_HEADER = "Message from the orchestrator"
PEER_COMM_HEADER = "Communications from other agents"

#: file name -> [(exact source substring, replacement)]
HUB_PROMPT_REWRITES = {
    "instruction_prompt_p2.txt": [
        ('"communication_target" must be EXACTLY one of {teammate_names} — '
         'never "{agent_name}", {comm_target_rule}',
         'You cannot message them directly: "communication_target" must be '
         '"orchestrator", {comm_target_rule}'),
        ('"communication": "<short message to the teammate in '
         'communication_target{comm_field_hint}>", "communication_target": '
         '"<a teammate\'s name in the form agent_N, never yourself, never '
         '\'all\'>"',
         '"communication": "<short report to the orchestrator'
         '{comm_field_hint}>", "communication_target": "orchestrator"'),
    ],
    "critic_prompt.txt": [
        ("(turn a different way, reposition, or message a teammate)",
         "(turn a different way, reposition, or report to the orchestrator)"),
        ("tell me to message a\n  specific teammate and act together",
         "tell me to tell the\n  orchestrator what I need and act together "
         "with the teammate"),
    ],
    "system_prompt.txt": [
        ("(useful info from teammate messages)",
         "(useful info from the orchestrator's messages)"),
    ],
    "perception_beliefs.txt": [
        ("Communications from teammates: {communications}",
         "Message from the orchestrator: {communications}"),
    ],
    "curriculum_info.txt": [
        ("Last communications received from other agents: {communications}",
         "Last message from the orchestrator: {communications}"),
    ],
    "partner_beliefs.txt": [
        ("exchanged messages with another agent.",
         "received a message from the orchestrator (agents cannot talk "
         "directly)."),
    ],
    "interaction_belief.txt": [
        ("exchanged messages with teammates.",
         "received a message from the orchestrator (agents cannot talk "
         "directly)."),
    ],
}

#: load_prompts() key -> prompt file it was read from.
PROMPT_KEY_FILES = {
    "system_template": "system_prompt.txt",
    "critic": "critic_prompt.txt",
    "perception": "perception_beliefs.txt",
    "partner": "partner_beliefs.txt",
    "interaction": "interaction_belief.txt",
}


def apply_hub_rewrites(text: str, file_name: str) -> str:
    """Apply the hub rewrites registered for ``file_name`` to ``text``.

    Raises RuntimeError unless every source occurs exactly once (drift
    guard)."""
    for source, replacement in HUB_PROMPT_REWRITES.get(file_name, ()):
        count = text.count(source)
        if count != 1:
            raise RuntimeError(
                f"hub prompt rewrite for {file_name} expected its source "
                f"exactly once, found {count}: {source[:60]!r}")
        text = text.replace(source, replacement)
    return text


def apply_hub_rewrites_to_prompts(prompts: dict, enabled: bool) -> dict:
    """Rewrite the load_prompts() dict in hub mode; identity otherwise."""
    if not enabled:
        return prompts
    out = dict(prompts)
    for key, file_name in PROMPT_KEY_FILES.items():
        if isinstance(out.get(key), str):
            out[key] = apply_hub_rewrites(out[key], file_name)
    return out
