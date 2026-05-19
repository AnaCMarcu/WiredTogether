# hebbian-marl

Hebbian-modulated social plasticity on Level-Based Foraging.

This repo is the small-environment testbed for a thesis chapter on Hebbian social plasticity in cooperative multi-agent RL. The Hebbian mechanism is ported from a larger parent codebase (Craftium / LLM-conditioned RL) and isolated here so any negative result is interpretable. Full design and rationale: [HEBBIAN_MARL_PLAN.md](HEBBIAN_MARL_PLAN.md).

## What's inside

| Path | Role |
|---|---|
| [epymarl/](epymarl/) | Vendored fork of [uoe-agents/epymarl](https://github.com/uoe-agents/epymarl) at commit `cbc38c09` (see [VENDOR.md](VENDOR.md)) |
| [epymarl/src/hebbian_module/](epymarl/src/hebbian_module/) | Ported Hebbian social-plasticity module (numpy-only, ~800 lines) |
| [epymarl/src/envs/lbf_comm_wrapper.py](epymarl/src/envs/lbf_comm_wrapper.py) | Level-Based Foraging with a discrete targeted-signal comm channel |
| [epymarl/src/runners/hebbian_runner.py](epymarl/src/runners/hebbian_runner.py) | `EpisodeRunner` subclass — integration (a) reward diffusion + bond logging |
| [epymarl/src/learners/hebbian_seac_learner.py](epymarl/src/learners/hebbian_seac_learner.py) | PPO learner subclass — integration (b) IS-corrected Hebbian-weighted sharing |
| [scripts/](scripts/) | `run_ablation_grid.sh`, `plot_results.py`, `verify_baseline.sh` |
| [tests/](tests/) | 32 pytest tests covering all integration points |

## Quickstart

```bash
git clone <repo> hebbian-marl
cd hebbian-marl
python -m venv .venv
# Windows PowerShell:    .venv\Scripts\Activate.ps1
# Linux / macOS / Git Bash on Windows:
source .venv/Scripts/activate    # (on Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
pytest                            # 32 tests, ~3 s

# Smoke test: vanilla EPyMARL MAPPO on plain LBF, 50 k env steps (~1 min)
bash scripts/verify_baseline.sh

# Full ablation grid: 7 variants × 5 seeds × 5 M steps
# (4-10 h on a laptop depending on PARALLEL_CAP)
bash scripts/run_ablation_grid.sh
python scripts/plot_results.py --results-dir epymarl/results
```

## Ablation grid

| Variant | Sharing | (a) reward diff | (b) sharing | δ_comm | Spatial gate | Purpose |
|---|---|---|---|---|---|---|
| `ippo_baseline` | — | off | off | n/a | n/a | Baseline floor (per-agent networks, Hebbian module disabled) |
| `mappo` | param-shared | off | off | n/a | n/a | Standard cooperative baseline |
| `seac` | none | off | uniform | n/a | n/a | SEAC: uniform IS-corrected sharing — what `hebb_s` must beat |
| `hebb_r` | none | **on** | off | 0.5 | active | Tests (a) reward diffusion in isolation |
| `hebb_s` | none | off | **Hebbian-weighted** | 0.5 | active | **Headline:** Hebbian-weighted vs uniform sharing |
| `hebb_rs` | none | on | Hebbian-weighted | 0.5 | active | (a) + (b) combined |
| `hebb_s_nocomm` | none | off | Hebbian-weighted | **0.0** | active | Ablates the comm term of co-activity |
| `hebb_s_commonly` | none | off | Hebbian-weighted | 0.5 | **disabled** | Ablates the spatial gate |
| `hebb_rsp` *(stretch)* | none | on | Hebbian-weighted | 0.5 | active | Full system, integration (c) policy mixture — deferred |

All variants share the **same action space** (signal actions present), so "Hebbian helps" can't be confounded with "having a comm channel helps."

**Headline claims** (paired-seed Wilcoxon, 5 seeds, final 10% window):
1. `hebb_s` > `seac` on episode return, p < 0.01 — does Hebbian weighting beat uniform sharing?
2. `hebb_s` > `hebb_s_nocomm` on episode return — does the comm term of co-activity carry real signal?

## Adding a new variant

1. Create [epymarl/src/config/algs/your_variant.yaml](epymarl/src/config/algs/) — copy an existing `hebb_*` config and adjust the `hebbian.*` block.
2. Add the variant name to `VARIANTS` in [scripts/run_ablation_grid.sh](scripts/run_ablation_grid.sh) (or pass `VARIANTS="..." bash scripts/run_ablation_grid.sh`).
3. If the variant participates in a statistical claim, extend [scripts/plot_results.py](scripts/plot_results.py).

## Pinned environment

Versions in [pyproject.toml](pyproject.toml) are pinned exactly to what Phase 0's smoke test was validated against — Python 3.10–3.12, torch 2.12.0 (CPU), gymnasium 1.3.0, lbforaging 2.0.0, sacred 0.8.7, numpy 1.26.4. EPyMARL upstream pins PyYAML==5.3.1 which won't build on Python 3.10+, so this set deliberately diverges.

EPyMARL itself is **vendored**, not submoduled, so we add new files (runner, learner, controller, comm wrapper, hebbian_module) alongside upstream's. To re-sync with upstream, see [VENDOR.md](VENDOR.md).

## Citation

```bibtex
@misc{hebbian-marl,
  title  = {Hebbian-Modulated Social Plasticity on Level-Based Foraging},
  author = {Marcu, Ana},
  year   = {2026},
}
```

Builds on:
- Papoudakis et al. 2021, "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks" (NeurIPS) — EPyMARL framework
- Christianos et al. 2020, "Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning" (NeurIPS) — SEAC baseline
- Albrecht & Stone 2018 / Albrecht et al. 2024 — Level-Based Foraging environment
