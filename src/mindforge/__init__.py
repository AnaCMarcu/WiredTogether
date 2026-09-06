"""MindForge: the LLM-driven multi-agent runtime.

Entry point: ``multi_agent_craftium.py`` (the episode loop). Around it:

- ``cli``            — every flag, and the checks that reject useless combinations
- ``agent_factory``  — prompts, role configs, agent construction
- ``checkpointing``  — save/restore for chained cluster jobs
- ``agent_modules``  — the per-agent cognitive stack: action selection, beliefs,
  curriculum, critic, memories, the social module
- ``env``            — environment-side accounting: the Lua interface, message
  validity and payment, cooperation metrics, episode logs
- ``prompts``        — the text templates the modules render

The RL machinery (``rl_layer``) and the social graph (``hebbian``) are siblings of
this package and never import from it — keep it that way, so both stay testable
without the agent stack.
"""
