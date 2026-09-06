# Configuration

Three layers of settings, in increasing precedence: dataclass defaults (`HebbianConfig`,
`RLConfig`), the CLI (`mindforge/cli.py`), and the launcher overrides in
`hpc/daic/experiments/*.sbatch`. The CLI defaults reproduce the zero-shot LLM condition; a run
becomes an experiment arm by adding flags. `--help` lists everything; this page covers the knobs
that carry meaning in the paper.

`validate_args` rejects combinations that would run for a day and produce nothing — a social module
without a graph, choice-mode social acts under `--rl`, an init file together with a preset, an
orchestrator together with `--hebbian`.

## Run shape

| Flag | Default | Meaning |
|---|---|---|
| `--num-agents` | 3 | Team size. Propagates into the Lua world as `FC_NUM_AGENTS` |
| `--episodes` | 1 | Episodes per run. `W`, skills and episodic memory persist across them; the environment and working memory reset |
| `--max-steps` | 1500 | Steps per episode; each chamber gets 20% of it before the rescue teleport |
| `--simultaneous` | on | All living agents act on the same state. `--no-simultaneous` falls back to round-robin |
| `--seed` | None | Seeds Python, numpy, torch and the world generator |
| `--run-group`, `--tag` | None | Output layout: `runs/<group>/<tag>/seed_<N>/` |
| `--start-chamber`, `--max-chamber` | None | Restrict the curriculum (Phase B of the transplant starts at Ch3) |

## Hebbian graph

Paper symbols in the first column; `HebbianConfig` field names match the flag names.

| Symbol | Flag | Default | Meaning |
|---|---|---|---|
| — | `--hebbian` | off | Master switch; required by every coupling below |
| — | `--hebbian-mode` | `reward_modulated` | `reward_modulated` \| `three_factor` \| `coactivity` \| `legacy` |
| `d` | `--hebbian-radius` | 5.0 | Interaction radius, world units |
| `α` | `--hebbian-alpha` | 0.5 | Weight of reward vs social activity in engagement `g_i` |
| `δ_k` | `--hebbian-delta` | None → 0.5 | Co-activity value of one social act, all channels alike |
| `η₀` | `--hebbian-eta-0` | 0.01 | Association floor: growth on a zero-reward step |
| `η₊` | `--hebbian-eta-plus` | 0.05 | Reward-modulated growth rate |
| `η₋` | `--hebbian-eta-minus` | 0.025 | Failure-gated decay rate |
| `λ` | `--hebbian-decay` | 0.005 | Homeostatic decay |
| `R` | `--hebbian-reward-norm` | 300.0 | Salience normaliser (the paper's runs use 50) |
| `ε` | `--hebbian-coop-eps` | 0.05 | Co-activity floor for the failure gate |
| `n` | `--hebbian-coop-window` | 50 | Co-activity / loss window, steps |
| `θ` | `--hebbian-neg-theta` | 5.0 | Windowed loss marking a negative outcome |
| `ρ_e` | `--hebbian-eligibility-rho` | 0.9 | Eligibility-trace decay (`three_factor` only) |
| `W₀` | `--hebbian-init-weight` | 0.1 | Initial off-diagonal bond |

Couplings:

| Symbol | Flag | Default | Meaning |
|---|---|---|---|
| `γ_d` | `--hebbian-gamma` | 0.2 | Reward-diffusion strength |
| `ρ` | `--hebbian-rho` | 0.0 | Weight-gated experience sharing; 0.3 in the opt-in arms |
| — | `--social-module` | `none` | `prompt` renders the directive; `bias` also overrides routing |
| — | `--social-interval` | 8 | Steps between deliberations; the directive is cached in between |

Imposed and transplanted graphs: `--hebbian-freeze`, `--hebbian-preset {none,uniform,star,ring,pair}`,
`--hebbian-bond-strong` (0.8) / `--hebbian-bond-weak` (0.1) / `--hebbian-hub`,
`--hebbian-init-file`, `--agent-state-init`.

## Co-firing channels

| Flag | Default | Meaning |
|---|---|---|
| `--social-act-mode` | `legacy` | `choice` gives each agent at most one social act per step |
| `--social-acts` | `comm,obs,imit` | Which acts the agent may choose from (`none` for the Null arm) |
| `--cofiring-channels` | = `--social-acts` | Which of them the wiring rule credits |
| `--social-bidirectional` | off | Credit both directions of an observation/imitation event |
| `--comm-distance-free` | off | Count a message as co-firing regardless of distance |
| `--social-act-rewards` | off | Pay for observation/imitation acts as well as messages |

## RL layer

| Flag | Default | Meaning |
|---|---|---|
| `--rl` | off | Enable PPO |
| `--rl-model-path` | None | Base checkpoint; the same weights serve the cognitive stack |
| `--rl-critic-mode` | `centralized` | `centralized` = MAPPO, `independent` = IPPO |
| `--rl-lora-rank` | 8 | LoRA rank (α = 2r, dropout 0.05) |
| `--rl-lr` | 1e-4 | Actor learning rate; the critic uses 3e-4 |
| `--rl-update-interval` | 256 | Transitions per PPO update (the paper's runs use 64/128) |
| `--rl-update-stagger` | off | Spread per-agent updates; required for Gemma runs |
| `--rl-prompt-max-tokens` | 512 | Prompt cap for the policy pass |

The remaining PPO constants — `γ`=0.995, `λ_GAE`=0.95, clip 0.2, value clip `ξ`, 2 epochs,
mini-batch 4, entropy 0.05 → 0.001 — live in `RLConfig` and are pinned by
`tests/test_paper_defaults.py`.

## Orchestrator baselines

| Flag | Default | Meaning |
|---|---|---|
| `--orchestrator` | off | Enable the central coordinator (excludes `--hebbian`) |
| `--orchestrator-variant` | `task` | `task` \| `social` \| `plan` \| `villager` |
| `--orchestrator-mode` | `advisory` | `advisory` writes a directive; `bias` also routes messages |
| `--orchestrator-cadence` | 8 | Minimum steps between decompositions — matched to `--social-interval` |
| `--orchestrator-node-timeout-steps` | 60 | Steps before an unfinished subtask is failed |

## Environment variables

| Variable | Used for |
|---|---|
| `CRAFTIUM_ENV_DIR` | The world to load — must point at `five-chambers` |
| `CRAFTIUM_LUANTI_DIR` | Engine directory, if not the one inside the craftium install |
| `PYTHONPATH` | Must contain `src/` |
| `LLM_MODEL_PATH` | Local checkpoint served in-process |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | OpenAI-compatible endpoint instead |
| `LLM_VISION_MODE` | `auto` \| `text` \| `vision`; must match across runs that are pooled |
| `ST_MODEL_NAME` | Sentence-transformer used by memory and the critic encoder |
| `WIREDTOGETHER_RUN_GROUP`, `WIREDTOGETHER_RUNS_ROOT` | Output layout, equivalent to `--run-group` |
| `FC_NUM_AGENTS`, `FC_CH4_MOB_COUNT`, `WT_TEAM_SCALING` | Set by the CLI; read on the Lua side |
