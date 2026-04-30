"""MindForge: LLM-driven multi-agent runtime.

Top-level entry point: ``multi_agent_craftium.py``. Subpackages:
- ``agent_modules`` — LLM-side: action selection, beliefs, critic, curriculum, …
- ``env``           — environment-side utilities (comms, cooperation metric, episode logger)
- ``prompts``       — text templates loaded by agent_modules

The RL machinery (``rl_layer``) and Hebbian graph (``hebbian``) live as siblings
of this package and are pure libraries — they don't import from mindforge.
"""
