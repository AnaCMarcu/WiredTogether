# RLVR + GRPO Training Path

This repository hosts **two parallel multi-agent training stacks** for the
Craftium Five-Chambers environment plus a token-level PPO variant of the
legacy stack. All three coexist; none subsumes the others.

| Stack | Entry point | Config | LoRA adapter | Algorithm |
|---|---|---|---|---|
| Legacy action-mode (MAPPO/IPPO) | `src/mindforge/multi_agent_craftium.py` | argparse CLI flags | per-agent (`rl_layer`) | PPO + centralised/independent critic |
| Legacy token-mode | `src/mindforge/multi_agent_craftium.py --rl --rl-mode token` | argparse CLI flags | per-agent (`rl_layer`) | token-level PPO (`rl_layer/token_opt.py`) |
| **New GRPO** | `src/mindforge/multi_agent_craftium_grpo.py` | YAML (`configs/rlvr/`) | shared `grpo_policy` | GRPO with verifiable rewards |

The GRPO path is **additive**: nothing in the legacy code is modified
beyond a one-line opt-in hook in `multi_agent_craftium.py` and a
small callback API added to `mindforge/env/episode_logger.py`. The two
trained LoRA adapters live side by side and are independently evaluable.
Full design: [`docs/rlvr_grpo_plan.md`](docs/rlvr_grpo_plan.md).

## Status

Implementation complete for Stages 0 – 4 + 6 (per the plan). Stage 5
(async serving/training) remains optional.

| Stage | What lands | Status |
|---|---|---|
| 0 | `src/rlvr/` scaffolding, `tests/rlvr/`, `configs/rlvr/`, `reward_table.py` | done |
| 1 | `EpisodeLogger` callback API, `FiveChambersVerifier`, `PassiveLoggerCallback`, `python -m rlvr.verifier` CLI | done |
| 2 | Single-agent GRPO (`grpo_buffer`, `rollout_sampler`, `reference_policy`, `grpo_trainer`, entry point) | done |
| 3 | Multi-agent GRPO (3A team-reward + 3B per-agent), `MultiAgentRolloutSampler`, dispatch in trainer | done |
| 4a | Hebbian reward diffusion (`HebbianGRPOBridge` + verifier hook) | done |
| 4b | Hebbian-weighted group composition (per-agent buffer + borrowing in trainer) | done |
| 5 | Async serving/training split | not implemented (optional) |
| 6 | `compare_modes.py` + metrics persistence + ablation configs | done |

189+ tests pass locally (1 skipped: a GRPOLanguageModel HF-load test that's
HPC-only).

## Running the legacy paths (unchanged)

```bash
cd src/mindforge

# Action-mode MAPPO with Hebbian
PYTHONPATH=../ python multi_agent_craftium.py \
    --num-agents 3 --episodes 1 --max-steps 100 \
    --rl --rl-mode action --hebbian --experiment-id E1a

# Token-mode PPO with Hebbian
PYTHONPATH=../ python multi_agent_craftium.py \
    --num-agents 3 --episodes 1 --max-steps 100 \
    --rl --rl-mode token --rl-auto-token-opt --hebbian

# Stage-1 passive observer (opt-in for the verifier sanity check)
RLVR_PASSIVE_LOG=1 PYTHONPATH=../ python multi_agent_craftium.py \
    --num-agents 3 --episodes 1 --max-steps 100 \
    --rl --hebbian --experiment-id rlvr_obs
# → runs/<id>/grpo_trajectories.jsonl gets written
# → score it with:  python -m rlvr.verifier --score-file runs/<id>/grpo_trajectories.jsonl --decompose
```

## Running the new GRPO path

### Local / interactive

```bash
cd src/mindforge

# Stage 2 single-agent on Ch3
PYTHONPATH=../ python multi_agent_craftium_grpo.py \
    --config ../../configs/rlvr/grpo_single_agent_ch3.yaml

# Stage 3 multi-agent, 3 agents trained, 3B per-agent reward (headline)
PYTHONPATH=../ python multi_agent_craftium_grpo.py \
    --config ../../configs/rlvr/grpo_multi_agent.yaml

# Stage 4a + 4b together (Hebbian full)
PYTHONPATH=../ python multi_agent_craftium_grpo.py \
    --config ../../configs/rlvr/grpo_hebbian_full.yaml
```

CLI overrides via `--set key.subkey=value` (repeatable):

```bash
PYTHONPATH=../ python multi_agent_craftium_grpo.py \
    --config ../../configs/rlvr/grpo_hebbian_full.yaml \
    --set seed=123 \
    --set llm.base_model_name=/scratch/models/Qwen3.5-2B \
    --set grpo.total_steps=2000 \
    --set grpo.learning_rate=1e-6
```

Each run writes to `<log_dir>/`:
- `grpo_metrics.jsonl` — one JSON record per GRPO step (consumed by `compare_modes.py`)
- `<checkpoint_dir>/step_NNNNNN/` — periodic LoRA adapter checkpoints
- Tensorboard / logging output (via the standard `logging` module)

### HPC / SLURM (DelftBlue)

The `scripts/experiments/G*.sh` files form the GRPO ablation grid. Each one
sources `_common.sh` (sets `PROJECT_DIR`, `MODEL_2B`, `MODEL_9B`, `SEED`
from `SLURM_ARRAY_TASK_ID`), then hands off to `scripts/grpo.sh` which
calls the entry point with per-job overrides.

| Script | What it runs | Config | Time |
|---|---|---|---|
| `G2_grpo_multi_agent.sh` | Stage 3 multi-agent, 3B per-agent reward (headline) | `grpo_multi_agent.yaml` | 18 h |
| `G2b_grpo_multi_agent_team_reward.sh` | Stage 3 multi-agent, 3A team reward (cooperation ablation) | `grpo_multi_agent.yaml` | 18 h |
| `G3a_grpo_hebbian_diffusion.sh` | Stage 4a only (reward diffusion, no composition) | `grpo_hebbian_diffusion.yaml` | 18 h |
| `G3b_grpo_hebbian_composition.sh` | Stage 4b only (composition, no diffusion) | `grpo_hebbian_composition.yaml` | 18 h |
| `G4_grpo_hebbian_full.sh` | Stage 4 full Hebbian (4a + 4b) — headline ablation | `grpo_hebbian_full.yaml` | 24 h |
| `G5_compare.sh` | Post-hoc `compare_modes.py` over the runs above | — | 30 min CPU |

Submit a single experiment (3-seed sweep is the convention):

```bash
sbatch --array=0-2 scripts/experiments/G4_grpo_hebbian_full.sh
```

Submit the whole ablation grid + comparison at once:

```bash
bash scripts/experiments/submit_grpo_ablation.sh
# → submits G2, G3a, G3b, G4 (parallel array jobs)
#   and G5_compare (depends on all four)
```

Run directories land at:
```
/scratch/$USER/WiredTogether/runs/grpo/<tag>/seed_<N>/
                                      ├── grpo_metrics.jsonl
                                      └── grpo_lora/step_NNNNNN/
```
Comparison figures:
```
/scratch/$USER/WiredTogether/reports/grpo_ablation/seed_<N>/*.png
```

**HPC smoke test before a long run** — override `grpo.total_steps` to a
tiny value to verify the env adapter, PEFT setup, and a few GRPO steps
end-to-end before kicking off a 24 h job:

```bash
sbatch --array=0 --time=00:30:00 scripts/experiments/G4_grpo_hebbian_full.sh \
    --export=ALL,EXTRA="--set grpo.total_steps=5"
```

(or edit the script's `bash scripts/grpo.sh ...` line directly to add
`--set grpo.total_steps=5`).

## Ablation grid

The full grid of YAML configs in [`configs/rlvr/`](configs/rlvr/):

| Config | Hebbian diffusion (4a) | Group composition (4b) | Mode | Used for |
|---|---|---|---|---|
| `grpo_single_agent_ch3.yaml` | — | — | single-agent | Stage-2 baseline |
| `grpo_multi_agent.yaml` | off | off | 3 agents, 3B | Stage-3 baseline / `grpo_only` |
| `grpo_hebbian_diffusion.yaml` | **on** | off | 3 agents, 3B | 4a isolation |
| `grpo_hebbian_composition.yaml` | off | **on** | 3 agents, 3B | 4b isolation |
| `grpo_hebbian_full.yaml` | **on** | **on** | 3 agents, 3B | headline (4a + 4b) |

For the thesis comparison, run each variant on HPC with a fixed seed and
collect the resulting `grpo_metrics.jsonl` files. Then:

```bash
python scripts/compare_modes.py \
    --grpo-metrics runs/grpo_only/grpo_metrics.jsonl \
                   runs/grpo_hebbian_diffusion/grpo_metrics.jsonl \
                   runs/grpo_hebbian_composition/grpo_metrics.jsonl \
                   runs/grpo_hebbian_full/grpo_metrics.jsonl \
    --labels base hebbian-4a hebbian-4b hebbian-full \
    --output-dir reports/grpo_ablation \
    --window 20 \
    --final-window 50
```

Produces in `reports/grpo_ablation/`:
- `summary.json` — per-run aggregate stats (total milestones, final-window
  mean reward, final fire rate, final KL loss, etc.)
- `group_mean_reward.png` — rolling reward over training, one line per run
- `milestone_fire_rate.png` — fraction of trajectories firing ≥ 1 milestone
- `kl_loss.png` — KL-to-reference over training
- `fraction_clipped.png` — how often the PPO surrogate clip activates
- `borrowed_fraction.png` — Stage-4b only; how often borrowed trajectories
  appear in each batch
- `final_milestone_rate_bar.png` — bar chart of end-of-training fire rate
  per run (the headline thesis figure)

### MAPPO baseline integration

The legacy stack writes metrics in a different schema (per-episode JSON
files, plot images). Bridging this into `compare_modes.py` is its own
work item. For the thesis figure, two pragmatic options:

1. **Run MAPPO eval separately**, plot end-of-training milestone-fire
   rate from `craftium_metric` output, overlay it as a horizontal line
   on the headline bar chart in postprocessing.
2. **Wrap the legacy eval loop** in a thin script that emits the same
   schema as `grpo_metrics.jsonl`, then feed both into
   `compare_modes.py` uniformly. See `docs/rlvr_grpo_plan.md` §10 for
   the expected schema bridge.

Both are unfinished — they're deferred to the HPC validation phase.

## Tests

```bash
pytest tests/rlvr/
# → 189+ passed, 1 skipped (HPC-only)
```

The local env doesn't have torch / gymnasium / autogen — anything
torch-dependent runs only on HPC. Pure-python logic (advantage math,
verifier scoring, sampler bucketing, batch assembly, metrics aggregation,
plotting) has full coverage.
