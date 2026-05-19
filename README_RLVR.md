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

## Thesis-grade results pipeline (n=5+ seeds)

For the full thesis tables, use `scripts/build_results.py` instead of the
older `compare_modes.py`. The new pipeline reads the multi-seed run tree,
aggregates with bootstrap CIs + Wilcoxon signed-rank, renders the five
canonical tables in both markdown and LaTeX, and produces the headline
2-panel figure.

### Required sidecars (per seed)

Every GRPO run writes four artifacts next to its `grpo_metrics.jsonl`:

| File | Source | Drives |
|---|---|---|
| `grpo_metrics.jsonl` | `GRPOTrainer.train(metrics_path=…)` per step | T1 (headline), T2 (per-chamber), learning curves |
| `time_to_first.json` | `GRPOTrainer._first_fire` dumped at end of `train()` | T5 (sample efficiency) |
| `hebbian_snapshots.jsonl` | `HebbianGRPOBridge.snapshot()` every K=10 steps | bond-strength evolution plot, T3 |
| `episode_summary.jsonl` | `CooperationMetric.episode_summary()` per joint rollout | T4 (coop_score / comm_efficacy / carry_imbalance) |

These four sidecars cover all the data the thesis tables read. They land
automatically when training runs through the entry point — no extra wiring
needed.

### Build the report

```bash
python scripts/build_results.py \
    --grpo runs/grpo \
    --out  results \
    --ablations G2,G2b,G3a,G3b,G4 \
    --baseline G2 \
    --bootstrap 10000 \
    --window 50 \
    --rolling-window 20
```

Output tree:
```
results/
├── per_ablation/
│   └── <tag>/summary.json          AblationSummary as JSON
├── cross_ablation/
│   ├── comparisons.json            list[PairwiseComparison]
│   └── plots/
│       ├── headline.png            2-panel: learning curves + final-rate bars
│       ├── learning_curves.png     all variants on one axis, median±p10-p90
│       ├── per_chamber_bars.png    grouped bars, x=chamber, color=method
│       ├── hebbian_axis_decomposition.png
│       └── bond_strength_evolution.png (only when ≥1 ablation has the sidecar)
└── tables/
    ├── T1_headline.{md,tex}        method × {fire rate, group reward}
    ├── T2_per_chamber.{md,tex}     method × {ch1..ch5, comm}
    ├── T3_hebbian_axis.{md,tex}    paired-bootstrap delta vs baseline
    ├── T4_coop_comm.{md,tex}       method × {coop_score, comm_efficacy, …}
    └── T5_sample_efficiency.{md,tex}  milestone × method, Kaplan-Meier censored
```

### Statistical conventions

- **Central tendency**: median + 10th/90th percentile across seeds
  (Agarwal et al., NeurIPS 2021).
- **Confidence intervals**: bootstrap 10k-resample percentile CI on every
  aggregated number; tables show `median [p10, p90]` for compactness.
- **Pairwise significance**: paired bootstrap of seed-aligned deltas (primary)
  + Wilcoxon signed-rank exact p-values (sanity check, **n≥5 only**).
  Stars: `*` p<0.05, `**` p<0.01, `***` p<0.001. When n<5, Wilcoxon
  reports `n.a.` and stars come from bootstrap-CI-excludes-zero alone.
- **Right-censoring**: in T5, if fewer than 50% of seeds reach a milestone,
  the median step is reported as `—` (no imputation).
- **Honest caveat**: every table caption notes the seed count. With n=3
  effects below ~5% absolute are unreliable; n=5+ tightens this.

### LaTeX integration

The `.tex` outputs use `booktabs` formatting (`\toprule` / `\midrule` /
`\bottomrule`) and are wrapped in a `table` float with caption + label.
Inputtable directly into the thesis:

```latex
\input{results/tables/T1_headline.tex}
\input{results/tables/T2_per_chamber.tex}
\input{results/tables/T4_coop_comm.tex}
\input{results/tables/T5_sample_efficiency.tex}
```

## Phase B+: cross-stack thesis comparison (11 methods)

The Phase A pipeline above handles **GRPO ablations only**. Phase B+
extends it to the full thesis grid — 11 methods covering GRPO, the
legacy RL stack (MAPPO / IPPO ± Hebbian), and LLM-as-policy variants
including reward propagation. New code lives in `src/rlvr/legacy_bridge.py`
+ `src/rlvr/reward_propagation.py`; new CLI flags are
`--reward-propagation` + `--interpretability` on the legacy entry point.

### Ablation grid (11 tags)

| Tag | Stack | Training | Hebbian | Heb in prompt | Reward prop in prompt |
|---|---|---|---|---|---|
| M1 | legacy | none | no | no | no |
| L1 | legacy | none | yes | yes | no |
| **L2** | legacy | none | yes | yes | **YES** (new) |
| M2 | legacy | MAPPO (centralized) | no | no | no |
| M3 | legacy | MAPPO + Hebbian | yes | yes | no |
| M4 | legacy | IPPO (independent) | no | no | no |
| M5 | legacy | IPPO + Hebbian | yes | yes | no |
| G2 | GRPO | per-agent (3B) | no | n/a | n/a |
| G2b | GRPO | team (3A) | no | n/a | n/a |
| G3a | GRPO | per-agent | 4a only | n/a | n/a |
| G3b | GRPO | per-agent | 4b only | n/a | n/a |
| G4 | GRPO | per-agent | 4a + 4b | n/a | n/a |

### Run-directory layout (Phase B++)

Legacy runs now write to **`runs/legacy/<tag>/seed_<N>/`** — matching
the GRPO `runs/grpo/<tag>/seed_<N>/` pattern. Activated by the new
`--tag <id>` flag on the legacy entry point (e.g. `--tag M3 --seed 0`).
Without `--tag`, the legacy entry point falls back to the old
`runs/<timestamp>_<experiment_id>_<uuid>/` layout for backwards
compatibility.

Gifs land at **`<run_dir>/gifs/`** by default — set
`--gif-dir auto` is the new default and resolves to the run-scoped
subdir. Pass `--gif-dir <absolute_path>` to override (the legacy
HPC pattern of writing to `/scratch/$USER/gifs` still works).

### Cooperation-metric parity across stacks (confirmation)

The T4 cooperation table reads the same 5 keys (`cooperation_score`,
`communication_efficacy`, `carry_imbalance`, `ch4_damage_gini`,
`ch5_damage_gini`) from **both** legacy and GRPO runs. The data lands
in `episode_summary.jsonl` next to `grpo_metrics.jsonl`, populated by:

- **GRPO runs** — `MultiAgentRolloutSampler` instantiates one
  `CooperationMetric` per joint rollout and dumps `episode_summary()`
  per joint (wired in Phase A §A.4).
- **Legacy runs** — `multi_agent_craftium.py` instantiates one
  `CooperationMetric` per episode; the schema bridge concatenates
  all `episodes/ep_*/episode_summary.json` files into the
  GRPO-shaped `episode_summary.jsonl` during translation.

The richer `comm_eval.py` / `coop_eval.py` post-hoc evaluators
(tokens_per_milestone, credit_gini, pair_message_count,
speaker_consistency, etc.) are legacy-only and don't enter the
cross-stack tables. If you need them for GRPO runs too, raise the
scope — it requires wiring an `EpisodeLogger`-style step/event/message
JSONL into the GRPO sampler.

### New CLI flags on the legacy entry point

`src/mindforge/multi_agent_craftium.py` gains three flags:

- `--tag <id>` — Phase B++ tagged-and-seeded run layout (see above).
- `--reward-propagation` — surfaces per-teammate reward deltas in the
  action-selection prompt:
  ```
  Propagated rewards this step: +2.50 from agent_1 (m17_switch_pressed),
  +0.30 from agent_2
  ```
  Requires `--hebbian` (the decomposition uses Hebbian's
  `diffuse_rewards`). The L2 row in the ablation grid is the only
  variant using this today.

- `--interpretability` — emits `runs/<run_id>/interpretability.jsonl`
  with one record per (env step, agent) capturing the bond row, chosen
  action, communication target, thoughts excerpt, and propagated-reward
  deltas the agent saw. Auto-on whenever `--hebbian` is on (cheap;
  ~250 B per record); pass explicitly to force on for non-Hebbian runs.

Existing `--hebbian` (no `--rl`) gives the L1 row — Hebbian weights in
the prompt without training. No new flag needed; the integration was
already there.

### SLURM scripts for the new variants

| Script | Tag | What it runs |
|---|---|---|
| `scripts/experiments/M1_plain_llm.sh` | M1 | No `--rl`, no `--hebbian` |
| `scripts/experiments/L1_llm_hebbian_prompt.sh` | L1 | `--hebbian` only |
| `scripts/experiments/L2_llm_hebbian_propagation.sh` | L2 | `--hebbian --reward-propagation` |
| `scripts/experiments/E2_mappo.sh` | M2 | already shipped |
| `scripts/experiments/E5_hebbian.sh` | M3 | already shipped (MAPPO + Hebbian) |
| `scripts/experiments/M4_ippo.sh` | M4 | `--rl --rl-critic-mode independent` |
| `scripts/experiments/M5_ippo_hebbian.sh` | M5 | `--rl --rl-critic-mode independent --hebbian` |
| `scripts/experiments/submit_full_grid.sh` | (orchestrator) | submits all 11 + post-hoc build_results |

### Legacy → GRPO schema bridge

The legacy stack writes `final_metrics.json` (rich, per-episode-summary
plus graph snapshots). The new translator turns that into the four
Phase A sidecars (`grpo_metrics.jsonl`, `time_to_first.json`,
`episode_summary.jsonl`, `hebbian_snapshots.jsonl`) so the same
`build_results.py` ingests legacy and GRPO uniformly.

```bash
# Single-run mode:
python scripts/legacy_to_grpo_schema.py \
    --input  runs/E5_seed_42 \
    --output runs/legacy_translated

# Directory-of-runs mode (recommended):
python scripts/legacy_to_grpo_schema.py \
    --input  runs                      # every subdir with final_metrics.json
    --output runs/legacy_translated    # auto-tagged + auto-seeded per run
```

Auto-tag classification reads `final_metrics.json::config.cli_args` and
maps it to M1/L1/L2/M2/M3/M4/M5 per the grid above.

### Build the cross-stack report

```bash
python scripts/build_results.py \
    --grpo   runs/grpo \
    --legacy runs                            # legacy runs (translator runs first)
    --out    results \
    --ablations M1,L1,L2,M2,M3,M4,M5,G2,G2b,G3a,G3b,G4 \
    --baseline M1 \
    --bootstrap 10000 --window 50 --rolling-window 20
```

Either `--grpo` or `--legacy` may be supplied alone; both is the full
thesis run. `build_results.py` runs the translator before any
aggregation, drops translated sidecars under
`<out>/legacy_translated/`, then aggregates both stacks into the same
`AblationSummary` dict.

Output additions vs. Phase A:
- `results/legacy_translated/<tag>/seed_<N>/` — translated sidecars
  per legacy run (re-runnable, idempotent)
- `results/cross_ablation/plots/cross_stack_grouped.png` — learning
  curves with line styles by stack family (GRPO solid, legacy-RL
  dashed, LLM-only dotted) + warm/cool palette by Hebbian on/off
- All five tables (T1-T5) now show 11 rows with no schema changes —
  the renderers handle any number of ablations the dict carries.

### Interpretability sidecar (raw exports for external analysis)

Hebbian-enabled runs (L1, L2, M3, M5, and any other variant with
`--interpretability`) emit per-(step, agent) records:

```json
{
  "step": 247, "agent_id": 0, "chamber": "ch3",
  "bond_row": [0.0, 0.42, 0.18],
  "chosen_action": "dig",
  "communication_target": 1,
  "thoughts_excerpt": "agent_1 is closer to the switch...",
  "propagated_delta_by_teammate": {"1": 2.5, "2": 0.3},
  "propagated_source_events": {"1": "m17_switch_pressed"}
}
```

Phase B+ does not auto-render a T6 interpretability table; the raw
JSONL is intended for analysis in a notebook. Typical questions:

- Does `communication_target` correlate with the maximum entry of
  `bond_row`? (Does the agent prefer to message its strongest bond?)
- Does the bond row at step N predict the agent's action at step N+1?
- Do mentions of teammates in `thoughts_excerpt` scale with bond
  strength?

### MAPPO baseline integration (Phase B+, **now shipped**)

The MAPPO bridge originally deferred from Phase A is implemented as the
legacy → GRPO schema translator above. The same `build_results.py` call
ingests MAPPO (M2), MAPPO+Hebbian (M3), IPPO (M4), IPPO+Hebbian (M5),
plain LLM (M1), and the two new LLM-as-policy variants (L1, L2) without
schema-specific code paths.

## Tests

```bash
pytest tests/rlvr/
# → 189+ passed, 1 skipped (HPC-only)
```

The local env doesn't have torch / gymnasium / autogen — anything
torch-dependent runs only on HPC. Pure-python logic (advantage math,
verifier scoring, sampler bucketing, batch assembly, metrics aggregation,
plotting) has full coverage.
