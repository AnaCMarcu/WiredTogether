# 10 — Configuration Reference

**Source files:** `src/hebbian/config.py`, `src/rl_layer/config.py`, `src/mindforge/multi_agent_craftium.py` (parse_args :44-324, run() :975-1125), `src/mindforge/agent_modules/util.py`, `src/mindforge/run_layout.py`, `hpc/daic/experiments/_common.sh`
**Paper sections:** Table 6 (RL/PPO/LoRA hyper-parameters), Table 7 (Hebbian hyper-parameters), Eq. 2/6/7/8, App. C.1, App. D
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

Three configuration layers exist: **dataclass defaults** (what `tests/test_paper_defaults.py` pins against Tables 6/7), **CLI defaults** (what a bare `python multi_agent_craftium.py` run gets — a few differ from the dataclass), and **launcher overrides** (`hpc/daic/experiments/*.sbatch`, what the reported runs actually used). All three are tabulated below.

## 1. HebbianConfig (`src/hebbian/config.py:10-97`)

`HebbianConfig()` is a no-op when `enabled=False` (the default). `mode` selects the update rule: `reward_modulated` (Variant B, paper Eq. 6), `coactivity` (Variant A), `legacy` (pre-thesis advantage-modulated rule) — see 02-hebbian-graph.md.

| Field | Default | Symbol | Table 7 | Match |
|---|---|---|---|---|
| `enabled` | `False` | — | — | no-op switch |
| `mode` | `"reward_modulated"` | — | Variant B | yes |
| `num_agents` | 3 | N | 3 | yes |
| `interaction_radius` | 5.0 | d (Eq. 2 gate for c_ij) | 5.0 | yes |
| `engagement_reward_weight` | 0.5 | alpha (g_i blend) | 0.5 | yes |
| `communication_coactivity_bonus` | 0.5 | delta_comm (c_ij bonus) | 0.5 | yes |
| `eta_plus` | 0.05 | eta_plus | 0.05 | yes |
| `eta_0` | 0.01 | eta_0 | 0.01 | yes (#10 folding consistent) |
| `eta_minus` | 0.025 | eta_minus | 0.025 | yes |
| `coop_eps` | 0.05 | epsilon (activity floor) | 0.05 | yes |
| `coop_window` | 50 | n (rolling window) | 50 | yes |
| `neg_theta` | 5.0 | theta (neg-reward gate) | 5.0 | yes |
| `reward_norm_R` | 300.0 | R (salience normaliser = max milestone m27) | 300 | yes |
| `decay` | 0.0003 | lambda (homeostatic) | absent | **no** — see below |
| `social_replay_rho` | 0.0 | rho (Eq. 7) | disabled | yes — see below |
| `reward_diffusion_gamma` | 0.2 | gamma_d (Eq. 8) | 0.2 | yes |
| `init_weight` | 0.1 | W_0 | 0.1 | yes |
| `freeze_weights` | `False` | — | — | frozen-W ablation |
| `init_preset` / `preset_bond_strong` / `preset_bond_weak` / `preset_hub` | `"none"` / 0.8 / 0.1 / 0 | — | — | hardcoded-topology ablation |
| `init_matrix` | `None` | — | — | explicit W override |
| `ltp_lr` / `ltd_lr` / `ltd_threshold` / `base_ltp` | 0.01 / 0.005 / 0.1 / 0.005 | legacy eta_+/eta_- | — | legacy mode only |
| `modulation_beta` | 1.0 | beta (legacy) | — | legacy mode only |
| `ltd_sustained_lr` / `failure_memory_window` | 0.002 / 50 | lambda_F / F_ij window | — | legacy mode only |
| `failure_grace_enabled` / `_threshold` / `failure_ltp_lr` | `True` / 0.3 / 0.015 | — | — | legacy mode only |
| `log_graph_every` | 50 | — | — | snapshot cadence |

> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #4 (unconditional `-lambda*W` term, lambda=0.0003, absent from Eq. 6/Table 7).
> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #9 (social replay implemented, rho=0 dataclass default consistent with App. C.1 — **but** the CLI default is 0.3, see §3 caveats).

Table-7 rows pinned by `tests/test_paper_defaults.py::test_hebbian_table7_defaults`.

## 2. RLConfig (`src/rl_layer/config.py:7-78`)

No-op when `enabled=False`. `mode`: `"action"` (discrete MAPPO over the candidate set) or `"token"` (sequence PPO only). Rows marked **T6** are paper Table 6 entries; all match (#13 CONSISTENT), pinned by `tests/test_paper_defaults.py::test_rlconfig_table6_defaults`.

| Group | Field | Default | Notes |
|---|---|---|---|
| Model/LoRA | `model_path` | `None` | base model path (#8: config-driven) |
| | `lora_rank` / `lora_alpha` / `lora_dropout` | 8 / 16 / 0.05 | **T6** |
| | `lora_per_role` | `True` | adapter per role |
| | `lora_save_dir` | `"rl_checkpoints"` | overridden to `<run>/rl_live` (§5) |
| | `dtype` | `"float16"` | **T6** |
| PPO core | `gamma` / `gae_lambda` | 0.995 / 0.95 | **T6** |
| | `clip_eps` / `value_clip_eps` | 0.2 / 1.0 | **T6**; value clip on normalised returns |
| | `entropy_coef` / `value_coef` / `max_grad_norm` | 0.01 / 0.5 / 0.5 | **T6** (value_coef, max_grad_norm) |
| | `ppo_epochs` / `mini_batch_size` / `lr` | 2 / 4 / 1e-4 | **T6**; lr = actor LoRA rate |
| | `entropy_start` / `entropy_end` / `entropy_anneal_steps` | 0.05 / 0.001 / 500 | **T6**; anneal over PPO updates |
| | `normalize_rewards` / `death_penalty` | `True` / 0.0 | RunningMeanStd before buffer |
| Critic | `critic_mode` | `"centralized"` | MAPPO; `"independent"` = IPPO value head |
| | `critic_hidden` / `critic_lr` | 256 / 3e-4 | **T6** |
| | `critic_value_clip_eps` | 10.0 | **T6** — xi (critic value clip) |
| | `value_hidden` | 256 | IPPO head only |
| Buffer/cadence | `buffer_size` / `update_interval` | 2048 / 128 | **T6**; CLI default differs (§3) |
| | `rl_prompt_max_tokens` | 256 | CLI default 512 (§3) |
| | `gradient_checkpointing` | `True` | shared-A100 memory tradeoff |
| Token-opt | `auto_token_opt` | `False` | agent-triggered token PPO |
| | `token_opt_min_samples` / `_window` / `_success_threshold` / `_epochs` | 32 / 10 / 0.3 / 2 | trigger: success < 0.3 |
| Action space | `actions` | 22-tuple | must equal `ACTION_MAP` keys in `custom_environment_craftium.py:17-42` (`VALID_ACTIONS = list(ACTION_MAP)`); pinned by `tests/test_paper_defaults.py::test_rlconfig_actions_tuple` |
| | `mask_slot_actions` | `True` | drops the 8 `Slot*` entries -> 14 RL candidates (#7 CONSISTENT) |

## 3. CLI arguments (`multi_agent_craftium.py:44-324`)

Only flags that feed configs or change run semantics; `-> field` means the value is forwarded at the instantiation sites in §5. **Bold defaults differ from the dataclass default** — the CLI value wins on every CLI launch.

| Group | Flag | Default | Maps to |
|---|---|---|---|
| Env/episodes | `--num-agents` | 3 | env + both configs `num_agents` |
| | `--episodes` / `--max-steps` | 1 / 1500 | loop bounds; chamber timer = 20% of max-steps |
| | `--obs-width` / `--obs-height` | 320 / 180 | `CraftiumEnvironmentInterface` |
| | `--seed` | `None` | torch/numpy/random + world seed |
| | `--simultaneous` | `True` | `step_all()` vs legacy round-robin |
| | `--no-communication` | off | disables messaging |
| | `--voxel-obs` | off | per-agent voxel grid in prompt |
| | `--warmup-time` / `--ch1-timeout-steps` | 60 / 400 | media-load wait; Lua backstop ticks |
| Agents/roles | `--team-mode` | `homogeneous-agent` | role assignment |
| | `--homogeneous-role` / `--roles` | `agent` / `None` | role names (agent/hunter/harvester/scouter) |
| | `--belief-interval` / `--critic-interval` / `--sleep-time` | 5 / 20 / 0.0 | cognitive cadence (see 08-cognitive-agent.md) |
| RL | `--rl` | off | `RLConfig.enabled` |
| | `--rl-mode` / `--rl-critic-mode` | `action` / `centralized` | `mode` / `critic_mode` |
| | `--rl-model-path` / `--rl-lora-rank` / `--rl-lr` | `None` / 8 / 1e-4 | `model_path` / `lora_rank` / `lr` |
| | `--rl-update-interval` | **256** | `update_interval` (dataclass 128) |
| | `--rl-prompt-max-tokens` | **512** | `rl_prompt_max_tokens` (dataclass 256) |
| | `--rl-auto-token-opt` | off | `auto_token_opt` |
| Hebbian | `--hebbian` | off | `HebbianConfig.enabled` |
| | `--hebbian-mode` | `reward_modulated` | `mode` |
| | `--hebbian-eta-plus` / `--hebbian-eta-0` / `--hebbian-eta-minus` | 0.05 / 0.01 / 0.025 | eta_plus / eta_0 / eta_minus |
| | `--hebbian-coop-eps` / `--hebbian-coop-window` / `--hebbian-neg-theta` | 0.05 / 50 / 5.0 | coop_eps / coop_window / neg_theta |
| | `--hebbian-reward-norm` | 300.0 | `reward_norm_R` |
| | `--hebbian-alpha` / `--hebbian-radius` | 0.5 / 5.0 | `engagement_reward_weight` / `interaction_radius` |
| | `--hebbian-ltp` / `--hebbian-ltd` / `--hebbian-beta` | 0.01 / 0.005 / 1.0 | legacy `ltp_lr` / `ltd_lr` / `modulation_beta` |
| | `--hebbian-decay` | **0.005** | `decay` (dataclass 0.0003) |
| | `--hebbian-rho` | **0.3** | `social_replay_rho` (dataclass 0.0) |
| | `--hebbian-gamma` | 0.2 | `reward_diffusion_gamma` |
| | `--hebbian-init-weight` | 0.1 | `init_weight` |
| | `--hebbian-no-comm-bond` | off | sets `communication_coactivity_bonus=0.0` (else 0.5) |
| | `--hebbian-freeze` / `--hebbian-preset` / `--hebbian-bond-strong` / `--hebbian-bond-weak` / `--hebbian-hub` | off / `none` / 0.8 / 0.1 / 0 | frozen/preset-W ablation fields |
| Social module | `--social-module` | `none` | `none` / `prompt` / `bias` (needs `--hebbian`) |
| | `--social-interval` | 8 | deliberation cadence (steps) |
| | `--interpretability` | off (auto-on with `--hebbian`) | interpretability.jsonl sidecar |
| Logging/wandb | `--wandb` / `--wandb-project` / `--wandb-entity` | off / `wired-together` / `None` | W&B init |
| | `--wandb-tags` / `--wandb-id` / `--wandb-upload-artifacts` | `""` / `None` / off | run id defaults to sanitised run_id (chunk-resume) |
| | `--log-interval` / `--no-gif` / `--gif-dir` / `--gif-interval` | 10 / off / `auto` / 300 | console + media cadence |
| | `--experiment-id` / `--tag` | `None` / `None` | metrics traceability; `--tag` -> `runs/legacy/<tag>/seed_<seed>/` layout |
| Checkpoint | `--checkpoint-dir` / `--checkpoint-interval` | `None` (-> `<run>/checkpoints/`) / 500 | save cadence |
| | `--resume` / `--resume-skip-warmup` / `--checkpoint-frames` | `None` / off / off | restore cognitive+RL+Hebbian state |

**CLI-vs-dataclass caveats** (every CLI launch forwards these flags unconditionally, so the CLI default — not the dataclass — is the effective value):

| Flag | CLI default | Dataclass | Effective in reported runs |
|---|---|---|---|
| `--hebbian-decay` | 0.005 | 0.0003 | 0.005 — DAIC final suite passes `--hebbian-decay 0.005` explicitly (e.g. `exp05_mappo_hebbian.sbatch:39`) |
| `--hebbian-rho` | 0.3 | 0.0 | 0.3 — no launcher overrides it; with `--rl --hebbian` neighbour buffers ARE passed (`multi_agent_craftium.py:2236-2246`), so Eq. 7 replay was live in MAPPO+Hebbian runs despite #9 |
| `--rl-update-interval` | 256 | 128 (Table 6) | 64 in exp05 (`exp05_mappo_hebbian.sbatch:34`) |
| `--rl-prompt-max-tokens` | 512 | 256 | 512 |

The DAIC final suite also overrides `--hebbian-eta-0 0.005` and `--hebbian-reward-norm 50` (vs Table-7 defaults 0.01 / 300) — so #13's "consistent" applies to the dataclass defaults, not to the launched-run values. Per-run truth is in each run's `config.json` (§5).

## 4. Environment variables

| Variable | Read by | Purpose |
|---|---|---|
| `LLM_MODEL_PATH` | `util.py:22`; `create_model_client` (`util.py:316-327`) | non-empty -> in-process `LocalModelClient` (HPC path). #8: model is config-driven; launchers set Qwen weights (`_common.sh:188`) |
| `LLM_BASE_URL` / `LLM_MODEL` | `util.py:23-24` | remote fallback client; defaults OpenRouter + `google/gemini-2.5-flash` (#8) |
| `LLM_API_KEY` | `util.py:306` (`_resolve_api_key`) | remote auth; falls back to `api.key` file |
| `LLM_ENABLE_THINKING` | `local_model_client.py:345` | `"1"` enables Qwen thinking mode; launcher pins 0 |
| `ST_MODEL_NAME` | `util.py:26` | sentence-transformers codebook model (default `all-MiniLM-L6-v2`) |
| `SENTENCE_TRANSFORMERS_HOME` | sentence-transformers library | offline model cache; set in `_common.sh:191` with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| `CRAFTIUM_ENV_DIR` | `openworld_multi_agents.py:108` | five-chambers world dir (see 04-environment-interface.md) |
| `WIREDTOGETHER_RUNS_ROOT` | `run_layout.py:43,146` | absolute anchor for `runs/`; tagged layout uses `<root>/legacy` fallback |
| `WIREDTOGETHER_INTERMEDIATE_GIF_DIR` | `multi_agent_craftium.py:892` | redirects checkpoint media off the runs tree (node /tmp on DAIC) |
| `CH1_TIMEOUT_TICKS` | Lua mod (set by Python, `multi_agent_craftium.py:953`) | Lua-side chamber-timeout backstop |
| `WANDB`, `WANDB_PROJECT`, `WANDB_EXTRA_TAGS`, `WANDB_MODE` | `_common.sh:45-48` (shell) | gate + compose `--wandb*` flags; auth via `~/.netrc` |
| `WANDB_DIR`, `WANDB_SILENT` | wandb library (`_common.sh:205-206`) | offline-run location (salvaged + auto-synced post-job) |
| `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy` | SDL inside Luanti (`_common.sh:196-197`) | headless rendering |
| `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`, `MESA_*`, `EGL_PLATFORM=surfaceless` | Mesa/EGL (`_common.sh:198-203`) | force CPU llvmpipe; DAIC GPUs are compute-only, `/dev/dri` is masked (see 11-operations.md) |

## 5. Config flow

1. **CLI** — `parse_args()` (`multi_agent_craftium.py:44-324`).
2. **RLConfig** — built in `run()` at `multi_agent_craftium.py:989-1000`; only 9 fields are forwarded from args (enabled, mode, model_path, lora_rank, update_interval, lr, auto_token_opt, rl_prompt_max_tokens, critic_mode) plus `lora_save_dir=<run>/rl_live` (`:988`); everything else keeps the dataclass default. Consumed by `CentralizedCritic` (`:1013-1017`) and per-agent `RLLayer`s via `build_agents` (`:1021-1028`) — see 03-rl-layer.md.
3. **HebbianConfig** — built at `multi_agent_craftium.py:1031-1059`; forwards ~23 fields (all `--hebbian-*` flags; `communication_coactivity_bonus` is 0.0/0.5 from `--hebbian-no-comm-bond`). Fields with no flag (`ltd_threshold`, `base_ltp`, `failure_grace_*`, `log_graph_every`, `init_matrix`) keep dataclass defaults. Consumed by `HebbianSocialGraph(hebbian_config, agent_roles=...)` (`:1061`) — see 02-hebbian-graph.md.
4. **Run-dir dump** — `config.json` written at `multi_agent_craftium.py:1116-1125` to `run_paths.config_json` (`run_layout.py:59`): run_id, start timestamp, `git rev-parse HEAD`, the **full `vars(args)`** (so the effective CLI values, including the §3 caveats, are recorded per run), num_agents, communication mode.
5. **Checkpoints** — default dir `<run>/checkpoints/` (`:1105`, `run_layout.py:65`); `--resume` reads `run_state.json` from the old checkpoint to recover the original run_id and re-points `rl_config.lora_save_dir` at the original run's `rl_live` (`:978-987`, `:1157-1159`). See 07-orchestrator.md for the save/restore cycle and 11-operations.md for the chunked-SLURM pattern.
