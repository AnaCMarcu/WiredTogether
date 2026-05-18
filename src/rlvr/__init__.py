"""RLVR + GRPO training path (additive, parallel to the legacy MAPPO/IPPO stack).

See ``docs/rlvr_grpo_plan.md`` for the full design. New entry point lives at
``src/mindforge/multi_agent_craftium_grpo.py``. The legacy stack
(``rl_layer/``, ``mindforge/multi_agent_craftium.py``, ``rl_layer/token_opt.py``)
is read-only — this package never imports from it in a way that mutates state.
"""
