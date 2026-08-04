"""SocialModule choice-mode generalisation (Experiment 2).

Pins the additive directive rendering: legacy SocialThought dicts (no
``suggest_act``) render byte-identically to before, while choice-mode
thoughts add exactly one suggestion line.
"""

import social_stubs  # noqa: F401  (installs autogen/chromadb stand-ins)

from agent_modules.social_module import SocialModule


def _module():
    # No LLM call in these tests — the client is never used by
    # render_directive, so a placeholder object suffices.
    return SocialModule(agent_name="agent_0", num_agents=3,
                        social_model_client=object())


_LEGACY_THOUGHT = {
    "bond_change_explanation": {"agent_1": "helped on the anvil"},
    "reasoning": "agent_1 is reliable",
    "referenced_bonds": {"agent_1": 0.7},
    "ask_target": "agent_1",
    "ask_message": "help me dig",
    "respond_to": [],
    "confidence": 0.8,
}


def test_legacy_thought_renders_without_suggestion_line():
    d = _module().render_directive(_LEGACY_THOUGHT)
    assert "Suggested social act" not in d
    assert "Ask agent_1 for help" in d


def test_choice_thought_adds_one_suggestion_line():
    t = dict(_LEGACY_THOUGHT, suggest_act="observe", suggest_target="agent_2")
    d = _module().render_directive(t)
    assert "Suggested social act: observe toward agent_2" in d
    # Everything else is unchanged relative to the legacy rendering.
    legacy = _module().render_directive(_LEGACY_THOUGHT)
    without = "\n".join(l for l in d.splitlines()
                        if "Suggested social act" not in l)
    assert without == legacy


def test_partial_suggestion_is_ignored():
    """suggest_act without a target (or vice versa) must not render."""
    for extra in ({"suggest_act": "observe"},
                  {"suggest_target": "agent_2"},
                  {"suggest_act": None, "suggest_target": None}):
        t = dict(_LEGACY_THOUGHT, **extra)
        assert "Suggested social act" not in _module().render_directive(t)
