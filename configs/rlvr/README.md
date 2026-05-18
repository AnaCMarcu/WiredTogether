# `configs/rlvr/` — GRPO training configs

YAML configs for the GRPO path (`src/mindforge/multi_agent_craftium_grpo.py`).
The legacy MAPPO path uses argparse CLI flags, not YAML; do not mix the two.

Planned configs (added in their respective stages — see `docs/rlvr_grpo_plan.md`):

- `base_grpo.yaml` — shared defaults imported by the others (Stage 2)
- `verifier.yaml` — verifier scoring config fragment (Stage 1)
- `grpo_single_agent_ch3.yaml` — first training target (Stage 2)
- `grpo_multi_agent.yaml` — 3 agents, per-agent advantages (Stage 3)
- `grpo_hebbian_diffusion.yaml` — Stage 4a only
- `grpo_hebbian_composition.yaml` — Stage 4b only (Option 4b-i, clipped)
- `grpo_hebbian_composition_truncated_is.yaml` — Stage 4b only (Option 4b-ii)
- `grpo_hebbian_full.yaml` — Stage 4a + 4b
- `grpo_async.yaml` — Stage 5 (optional)

Launch pattern:

```bash
# From repo root, with PYTHONPATH=src per the project convention:
cd src/mindforge
PYTHONPATH=../ python multi_agent_craftium_grpo.py \
    --config ../../configs/rlvr/grpo_single_agent_ch3.yaml
```
