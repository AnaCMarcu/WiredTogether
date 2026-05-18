# RLVR + GRPO Additive Integration Plan — Craftium Track

**Audience:** Claude Code (autonomous coding agent).
**Goal:** Add a parallel RLVR (Reinforcement Learning with Verifiable Rewards) + GRPO (Group Relative Policy Optimization) training path to the existing Craftium / Five-Chambers codebase, **without deleting or modifying any existing code**.
**Strategy in one line:** Every new component lives in new files; existing components (`rl_layer/`, `multi_agent_craftium.py`, action head, centralised critic, dense reward shaping, token-level PPO) are read-only references and continue to work unchanged.
**Expected effort:** 3–5 weeks to first end-to-end GRPO training run on a single chamber; further stages optional.

> **Document status.** This is the implementation-ready version, written after auditing the codebase to verify the facts the plan depends on. See [§12 — Codebase verification facts](#12-codebase-verification-facts) for the grounded file paths, line numbers, and symbol shapes. Corrections to a prior draft are absorbed in place rather than tracked separately.

---

## 1. The non-deletion principle (read this first)

The user has explicitly required that **nothing in the existing codebase be deleted or modified**. This is a hard constraint, not a stylistic preference. The reasons:

- The existing MAPPO / IPPO / centralised-critic stack is debugged and works mechanically, even if it doesn't learn well — deleting it loses information.
- Mode-A vs. Mode-B comparisons (legacy MAPPO vs. new GRPO) need both stacks live simultaneously.
- The thesis benefits from having the old and new approaches side by side as an ablation: "MAPPO with dense shaping" vs. "GRPO with verifiable rewards" is a real comparison once both exist.
- The implementer should not have to make judgment calls about what "counts" as obsolete — every existing file stays.

**Operational rule for the implementer:** the only changes to existing files are *additive imports* (so the new entry point can read from existing modules) and *new config files* (in new YAML files, not by editing existing configs). If you find yourself wanting to refactor an existing file, **stop and ask the user**. The cost of asking is small; the cost of an unwanted refactor is large.

Concretely, the following are **forbidden** without explicit user approval:

1. Deleting any file in `src/`.
2. Removing or renaming any function, class, or method in any existing file.
3. Modifying the body of any existing function or method.
4. Changing the default value of any existing configuration / argparse field.
5. Adding fields to existing dataclasses or pydantic models (instead, subclass them in a new file).
6. Editing `src/mindforge/multi_agent_craftium.py` for any reason.
7. Editing any file in `src/rl_layer/` for any reason (this **includes** `src/rl_layer/token_opt.py` — see §3.4).
8. Editing any file in `src/hebbian/` for any reason — these are read-only imports.

The following **are** allowed:

- Adding new files anywhere in `src/`, ideally under a new top-level package `src/rlvr/`.
- Adding a new entry-point script alongside `multi_agent_craftium.py`.
- Adding new YAML config files under `configs/rlvr/`.
- Adding new tests under `tests/rlvr/`.
- Reading any existing file (the Hebbian graph, the milestone tables, the cooperation metrics, the LocalModelClient, the episode logger) and importing it.

---

## 2. Architectural shape: two parallel paths sharing a substrate

The existing system:

```
                ┌─────────────────────────────────────────────────────┐
                │  multi_agent_craftium.py (legacy entry point)        │
                │                                                      │
   env step ────┼─→ LLM (LocalModelClient + LoRA) ─→ pooled hidden    │
                │       │                                  │           │
                │       │                                  ▼           │
                │       │                          ActionHead (23-way: │
                │       │                          NOP + 22 named)     │
                │       ▼                                  │           │
                │   JSON output                            ▼           │
                │   (parsed for comm,               action index       │
                │    target, etc.)                        │            │
                │                                         ▼            │
                │                                   env.step()         │
                │                                         │            │
                │                                         ▼            │
                │   reward shaping (proximity, futile,                  │
                │   action-repetition, milestones, comm bonus,         │
                │   Hebbian diffusion + social-replay at ρ=0.3)        │
                │                                         │            │
                │                                         ▼            │
                │   CentralizedCritic → GAE → IPPO/MAPPO update        │
                │   (action-mode) OR token_opt.maybe_token_optimize    │
                │   → token_level_ppo_step (token-mode)                │
                └─────────────────────────────────────────────────────┘
```

The added GRPO path:

```
                ┌─────────────────────────────────────────────────────┐
                │  multi_agent_craftium_grpo.py (NEW entry point)      │
                │                                                      │
                │  for each "decision episode":                        │
                │                                                      │
                │    1. Sample G trajectories from current policy      │
                │         (re-uses LocalModelClient, NO ActionHead,    │
                │          action = parsed JSON output directly)       │
                │                                                      │
                │    2. Score each trajectory:                         │
                │         FiveChambersVerifier(traj) → scalar reward   │
                │         (re-uses TRACKS reward table + reads from    │
                │          milestone_events.jsonl AND event_log.jsonl  │
                │          which live in different directories — see   │
                │          §4.2 — read-only)                            │
                │                                                      │
                │    3. Group-relative advantage:                      │
                │         A_i = (r_i - mean(r)) / (std(r) + ε)         │
                │                                                      │
                │    4. GRPO update on LoRA weights only               │
                │         (NEW LoRA adapter named "grpo", separate     │
                │          from MAPPO's adapter)                        │
                │                                                      │
                │    5. (Optional) Hebbian hooks:                      │
                │         - Reward diffusion in verifier               │
                │         - Hebbian group composition (with explicit   │
                │           off-policy correction — see §5.4 Stage 4b) │
                └─────────────────────────────────────────────────────┘
```

Both paths share the same Craftium env, the same milestone-event stream, the same base LLM, and the same Hebbian module. They differ in: trainer, buffer, advantage estimator, reward function, and action representation. The new entry point starts from the base model (or a user-specified checkpoint) — it does **not** load the MAPPO-trained LoRA by default, so the two trained policies are independently evaluable.

---

## 3. Why RLVR + GRPO is the right additive shape

Four properties of the Craftium setup make this almost over-determined:

### 3.1 Verifiable rewards already exist.

`milestone_events.jsonl` is a deterministic, programmatic stream of which milestones fired when. M1–M28 (across tracks `ch1_solo`, `ch2_anvils`, `ch3_switches`, `ch4_combat`, `ch5_boss`) plus comm milestones (`m_comm_ch1`–`m_comm_ch5`) are all binary events. They are exactly what RLVR papers (DeepSeek-R1, o-series) call "verifiable rewards." The reward values are tabulated in `TRACKS` ([`src/mindforge/agent_modules/craftium_metric.py:61-90`](../src/mindforge/agent_modules/craftium_metric.py)) and the milestone → track mapping in `MILESTONE_TRACK` ([same file, line 24-58](../src/mindforge/agent_modules/craftium_metric.py)). **Both are imported read-only.**

### 3.2 GRPO eliminates the critic, which is the weak component.

Audit notes show `critic_explained_variance` (computed at [`src/rl_layer/centralized_critic.py:365`](../src/rl_layer/centralized_critic.py)) is low on this env — the critic is approximating the mean. GRPO uses `(r_i - mean(r)) / std(r)` over a group of G trajectories as the advantage. No critic, no GAE, no value-clip-calibration issues. Failure modes of GRPO are documented and tractable.

### 3.3 The existing Hebbian social-replay (ρ=0.3) is mathematically uncomfortable; GRPO offers a cleaner home.

The legacy stack already runs Hebbian-weighted social replay with `--hebbian-rho` defaulting to **0.3** (see [`src/mindforge/multi_agent_craftium.py:138`](../src/mindforge/multi_agent_craftium.py) and [`src/rl_layer/ppo_update.py:128`](../src/rl_layer/ppo_update.py)). It is not dead; it is running. However, the implementation reuses transitions from teammate j's buffer without explicit importance correction — this is a known approximation that the audit notes flag as a source of bias.

GRPO does not make this issue vanish, but it changes its shape: in GRPO the advantage is **group-relative**, not policy-ratio-relative, so the bias from cross-agent borrowing is more cleanly isolated to a single off-policy correction term on the surrogate ratio (see §5.4 Stage 4b for the honest math). The thesis story is therefore: "PPO's social replay is approximate; GRPO can make the same idea explicit and either correct or bound the bias."

### 3.4 Token-level PPO already exists; GRPO is a parallel sibling.

The codebase already has a token-level RL pathway in [`src/rl_layer/token_opt.py`](../src/rl_layer/token_opt.py) + `token_level_ppo_step` in [`src/rl_layer/ippo.py`](../src/rl_layer/ippo.py). This is *agent-decided* token-level PPO: the LLM is asked whether to train, and if yes, runs PPO mini-batches on its own outputs. The plan's GRPO path is a **sibling**, not a replacement:

| | `token_opt` (existing) | GRPO (this plan) |
|---|---|---|
| Algorithm | PPO with value baseline | GRPO with group-relative advantage |
| Adapter | shared `rl_layer` LoRA | new `grpo` LoRA |
| Trigger | agent self-decides via `learning_belief.txt` | training-loop driven, every K steps |
| Reward | dense shaping + milestone bonuses | verifier-only (milestone + format) |
| Critic | needed | not needed |
| Entry point | `multi_agent_craftium.py --rl --rl-mode token` | `multi_agent_craftium_grpo.py` |

The new entry point must not import or call `token_opt`. They are independent experiments that happen to share a base LLM.

---

## 4. Repository layout: what's added, what's untouched

### 4.1 New files only

```
src/
├── rlvr/                                       # NEW PACKAGE — all RLVR/GRPO code
│   ├── __init__.py
│   ├── verifier.py                             # FiveChambersVerifier — pure function
│   ├── trajectory.py                           # GRPOTrajectory dataclass
│   ├── grpo_buffer.py                          # group-of-G storage
│   ├── grpo_trainer.py                         # GRPO update loop
│   ├── rollout_sampler.py                      # samples G trajectories per "prompt"
│   ├── reference_policy.py                     # frozen reference policy + KL term
│   ├── action_parser.py                        # parses LLM JSON output → env action
│   ├── hebbian_grpo_bridge.py                  # Hebbian hooks for GRPO mode
│   ├── passive_logger.py                       # Stage-1 jsonl tail-reader
│   ├── reward_table.py                         # flattens TRACKS → {mid: float}
│   ├── async_runner.py                         # (optional, Stage 5) async serving/training
│   └── metrics_grpo.py                         # GRPO-specific logging
│
├── mindforge/
│   └── multi_agent_craftium_grpo.py            # NEW entry point (sibling of multi_agent_craftium.py)
│                                               # Existing multi_agent_craftium.py untouched.
│
├── hebbian/                                    # UNCHANGED — read-only import
├── rl_layer/                                   # UNCHANGED — read-only reference (includes token_opt.py)
├── mindforge/agent_modules/                    # UNCHANGED — TRACKS / MILESTONE_TRACK / critic / etc. imported read-only
└── ...everything else unchanged

configs/
├── rlvr/                                       # NEW config directory
│   ├── base_grpo.yaml
│   ├── verifier.yaml                           # verifier scoring config
│   ├── grpo_single_agent_ch3.yaml              # Stage 2 first target
│   ├── grpo_multi_agent.yaml                   # Stage 3
│   ├── grpo_hebbian_diffusion.yaml             # Stage 4a
│   ├── grpo_hebbian_composition.yaml           # Stage 4b
│   └── grpo_async.yaml                         # Stage 5
│
└── ...existing configs unchanged

tests/
├── rlvr/                                       # NEW test directory
│   ├── test_verifier.py
│   ├── test_reward_table.py
│   ├── test_action_parser.py
│   ├── test_grpo_trainer.py
│   ├── test_rollout_sampler.py
│   ├── test_reference_policy.py
│   ├── test_hebbian_grpo_bridge.py
│   ├── test_passive_logger.py
│   └── test_grpo_smoke.py                      # end-to-end smoke test
│
└── ...existing tests unchanged

scripts/
├── run_grpo.sh                                 # NEW
├── compare_modes.py                            # NEW — plots MAPPO vs GRPO trained policies
└── ...existing scripts unchanged
```

### 4.2 Read-only imports from the existing codebase

The new code reads these existing modules without modifying them:

| Existing module | What we read from it | Why |
|---|---|---|
| `src/hebbian/config.py::HebbianConfig` | Construct Hebbian graph | Shared module |
| `src/hebbian/graph.py::HebbianSocialGraph` | The graph itself; `.diffuse_rewards(...)` at line 496; `.get_social_replay_indices(...)` for parity tests | Shared module |
| `src/mindforge/agent_modules/craftium_metric.py::TRACKS` | **Per-milestone reward table** — list of `(mid, reward)` per track | Verifier oracle (this is the right symbol; `MILESTONE_TRACK` maps to **track names**, not rewards) |
| `src/mindforge/agent_modules/craftium_metric.py::MILESTONE_TRACK` | milestone → track-name mapping | Used for grouping milestone events by chamber/track |
| `src/mindforge/agent_modules/craftium_metric.py::_plot_hebbian_asymmetry` | Plotting | Reuse plot code |
| `src/mindforge/env/cooperation_metric.py` | Patterns for joint-kill, pair-interaction tensors | Inspiration only |
| `src/mindforge/env/episode_logger.py` | Where episode-level summaries are written; entry point for the passive observer (§5.1) | Stage-1 hook |
| `src/marl_craftium/openworld_multi_agents.py` | Env wrapper interface; `_PatchedMarlCraftiumEnv` (luanti binary patch) | Used as the env |
| `src/marl_craftium/_actions.py::_DISCRETE_ACTIONS` | The 22 named actions; `Discrete(23)` = NOP + 22 | Action vocabulary for the verifier's format-reward |
| `src/mindforge/agent_modules/local_model_client.py::LocalModelClient` | LLM call interface | Same model loader |
| PEFT/LoRA machinery in [`src/rl_layer/rl_layer.py:77-119`](../src/rl_layer/rl_layer.py) — `LoraConfig`, `get_peft_model`, `adapter_name=` | LoRA pattern, multi-adapter support | The ReferencePolicy (§5.2) uses the same `adapter_name=` mechanism to keep policy + reference adapters loaded simultaneously |
| `runs/{run_id}/world/.../milestone_events.jsonl` | Lua-written milestone event stream (**in the Minetest world directory**, not the run dir) | One half of the verifiable reward source |
| `runs/{run_id}/episodes/ep_*/event_log.jsonl` | Python-written episode event log (**in the run dir's episode subdir**) | Other half of the verifiable reward source |

> **Note on the two jsonl files.** The plan's prior draft conflated them. They live in different places, are written by different code paths, and have different schemas. The verifier reads both and merges them by `(step, agent_id)` keys. See `passive_logger.py` design in §5.1.

---

## 5. Implementation stages with verification gates

**Hard rule:** do not move to the next stage until the current stage's verification gate is green.

### Stage 0 — Scaffolding and shared infrastructure (2 hours)

Tasks:

1. Create `src/rlvr/` with empty `__init__.py` and stub files for each module listed in §4.1.
2. Create `tests/rlvr/` and `configs/rlvr/` directories.
3. Verify `src/rlvr/` is picked up by the project's package discovery. Read `pyproject.toml` (it exists at the repo root). If it uses `setuptools.find_packages()` or Poetry's auto-discovery, the new package will be discovered automatically. If it uses an explicit `packages = [...]` list, **stop and ask the user** before adding `src/rlvr/` to it.
4. Run `pytest tests/rlvr/` and confirm it discovers zero tests but runs cleanly.
5. Write a `README_RLVR.md` at the repo root that explains: "This is a parallel RLVR/GRPO training path. To use the existing MAPPO path, run `multi_agent_craftium.py` with the existing CLI flags (see `python -m mindforge.multi_agent_craftium --help`). To use this path, run `multi_agent_craftium_grpo.py --config configs/rlvr/<file>.yaml`. Both are supported and produce different LoRA checkpoints." Two paragraphs is enough.
6. Add a flat-reward-table helper in `src/rlvr/reward_table.py`:

   ```python
   """Flatten TRACKS into a single {milestone_id: reward} dict.

   TRACKS is shaped as {track_name: [(milestone_id, reward), ...]}; the
   verifier wants a flat lookup. This helper centralises the flattening so
   future code never imports MILESTONE_TRACK by mistake (that maps to
   track_name strings, NOT to reward floats).
   """
   from mindforge.agent_modules.craftium_metric import TRACKS

   def build_milestone_rewards() -> dict[str, float]:
       out: dict[str, float] = {}
       for _track, entries in TRACKS.items():
           for mid, reward in entries:
               out[mid] = float(reward)
       return out
   ```

   With a single-line test in `tests/rlvr/test_reward_table.py`:
   ```python
   from rlvr.reward_table import build_milestone_rewards
   def test_known_milestones():
       r = build_milestone_rewards()
       assert r["m17_switch_pressed"] == 40.0
       assert r["m22_all_mobs_killed"] == 150.0
       assert r["m27_boss_defeated"] == 300.0
       assert r["m_comm_ch1"] == 40.0
   ```

**Verification gate:** `pytest -v` runs all existing tests + 1 new test (`test_reward_table`) with no errors. Importing `src.rlvr` works. Existing `python -m mindforge.multi_agent_craftium --help` is unaffected.

### Stage 1 — Verifier as a passive observer (3–5 days)

This stage builds the `FiveChambersVerifier` and runs it **alongside** the existing pipeline, scoring trajectories that the legacy MAPPO pipeline already produces. The verifier's output is logged but does **not** drive training yet. This lets the implementer (and user) inspect verifier output against the existing reward stream for a few days before committing to it.

Tasks:

1. **`src/rlvr/trajectory.py`.** Define a minimal trajectory record:

   ```python
   @dataclass(frozen=True)
   class GRPOTrajectory:
       """A trajectory in the GRPO sense: one rollout from a starting state,
       up to a horizon or until termination.
       """
       prompt_id: str                  # identifier for the starting condition
       agent_id: int
       chamber: str                    # "ch1", "ch3", etc.
       start_step: int                 # global env step when the trajectory began
       end_step: int                   # global env step when the trajectory ended
       actions: list[dict]             # the LLM's parsed JSON outputs, in order
       env_outputs: list[dict]         # per-step env info (positions, milestone fires)
       milestone_events: list[dict]    # slice of milestone_events.jsonl (world dir) in [start_step, end_step]
       event_log: list[dict]           # slice of event_log.jsonl (episode dir) in [start_step, end_step]
       termination_reason: str         # "horizon", "milestone_fired", "death", "chamber_transition"
   ```

   Use `frozen=True` because trajectories are immutable once collected and we want hashability for caching.

2. **`src/rlvr/verifier.py`.** The pure-function verifier:

   ```python
   class FiveChambersVerifier:
       """Maps a GRPOTrajectory to a scalar reward.

       Pure function. Idempotent: scoring the same trajectory twice gives
       the same result. No env state, no LLM calls, no learned components.

       Composes a configurable set of verifiable signals:
         - milestone fires (read from trajectory.milestone_events,
           rewarded via build_milestone_rewards() — not MILESTONE_TRACK)
         - boss-damage events (from event_log)
         - all-alive bonus (no agent died in this trajectory)
         - format reward (the LLM's JSON output had valid required fields)

       Optionally applies Hebbian reward diffusion if a HebbianSocialGraph is
       provided AND config.hebbian_reward_diffusion is True. This is the
       Stage-4a integration; the verifier is the natural place because it's
       the single point where rewards become scalars.
       """
       def __init__(self, config: VerifierConfig, hebbian: HebbianSocialGraph | None = None):
           self.config = config
           self.hebbian = hebbian
           from rlvr.reward_table import build_milestone_rewards
           self.milestone_rewards = build_milestone_rewards()

       def score(self, trajectory: GRPOTrajectory) -> float:
           """Per-trajectory scalar reward."""
           ...

       def score_group(self, trajectories: list[GRPOTrajectory]) -> list[float]:
           """Score a group of trajectories. Identical to mapping `score` over
           them, EXCEPT when reward diffusion is enabled — diffusion needs the
           group to compute the team-level signal.
           """
           ...

       def explain(self, trajectory: GRPOTrajectory) -> dict[str, float]:
           """Decomposed reward for logging: {milestone: x, format: y, alive: z}.
           Critical for diagnosing what the verifier is rewarding.
           """
           ...
   ```

3. **`src/rlvr/action_parser.py`.** Parses an LLM JSON output into a format-validity score:

   ```python
   from marl_craftium._actions import _DISCRETE_ACTIONS

   _VALID_ACTION_NAMES = frozenset(_DISCRETE_ACTIONS) | {"nop"}

   def parse_action_json(text: str, n_agents: int) -> tuple[dict | None, float]:
       """Returns (parsed_action, format_reward).

       format_reward ∈ [0, 1]:
         - 1.0 if JSON parses, has 'action' (str in _VALID_ACTION_NAMES),
           'communication_target' (int in [0, N) or null), 'thoughts' (str).
         - 0.5 if JSON parses but is missing one optional field.
         - 0.0 if it doesn't parse or 'action' is missing/invalid.
       """
       ...
   ```

   This is what enables the "format reward" component of RLVR — the verifier rewards the LLM for producing well-formed output before it can be rewarded for completing milestones.

4. **Passive observer hook.** Use the existing `EpisodeLogger` ([`src/mindforge/env/episode_logger.py`](../src/mindforge/env/episode_logger.py)) as the integration point. Two options, both fully non-invasive:

   - **(a) Disk-tail approach (preferred):** `src/rlvr/passive_logger.py` runs as a separate process or a thread, tails `milestone_events.jsonl` (world dir, path resolved via the run's `run_layout`) and `event_log.jsonl` (run dir / episode subdir), and writes reconstructed trajectories to `runs/{run_id}/grpo_trajectories.jsonl` after each episode boundary. No changes to existing code; latency on the order of seconds.
   - **(b) Callback approach:** if `EpisodeLogger` exposes a public hook (read its source — do not modify), register a callback that emits the reconstructed trajectory inline.

   **Default to (a).** Confirm with the user before switching to (b), because (b) may require adding a hook-registration API to `EpisodeLogger`, which would be a modification.

5. **Tests.** `tests/rlvr/test_verifier.py`:
   - Construct a trajectory with one milestone fire (`m22_all_mobs_killed`, reward 150.0 per `TRACKS`). Assert `verifier.score(traj) ≈ 150.0`.
   - Score the same trajectory twice — assert exact equality (idempotence).
   - Construct a trajectory with no milestone fires but valid JSON throughout — assert score equals format_reward sum.
   - Construct a trajectory with malformed JSON outputs — assert format_reward is 0 for those steps.
   - Construct two trajectories that differ only in JSON validity — assert the well-formed one scores strictly higher.

   `tests/rlvr/test_action_parser.py`:
   - Valid JSON with `action: "dig"` → (dict, 1.0).
   - Missing `thoughts` field → (dict, 0.5).
   - Malformed JSON → (None, 0.0).
   - `action: "fly"` (not in `_DISCRETE_ACTIONS`) → (None or partial dict, low score).

   `tests/rlvr/test_passive_logger.py`:
   - Synthesise a small `milestone_events.jsonl` + `event_log.jsonl` pair in a tmpdir, run the tail-reader, assert that the emitted trajectory contains the expected milestone events and episode boundaries.

**Verification gate:** A full legacy training run completes with the passive observer enabled (run it as a sidecar process: `python -m rlvr.passive_logger --run-dir runs/<id>`). `runs/{run_id}/grpo_trajectories.jsonl` is populated. `python -m rlvr.verifier --score-file runs/{run_id}/grpo_trajectories.jsonl` produces scores for all trajectories. The user inspects the score distribution — milestones fire at expected rates, format reward correlates with stable LLM output. **The verifier produces no surprises before it becomes a training signal.**

### Stage 2 — Single-agent GRPO training on Ch3 (1–2 weeks)

Now the verifier feeds a real training loop. Start single-agent on Ch3 specifically because Ch3's switch puzzle fires milestones (`m17_switch_pressed`, reward 40.0; `m18_door_opened`, reward 60.0) within ~30 steps and gives the fastest learning signal.

Tasks:

1. **`src/rlvr/rollout_sampler.py`.** Samples G trajectories per "decision episode":

   ```python
   class RolloutSampler:
       """Produces G trajectories for GRPO.

       In a fast env (e.g. with save/restore), G trajectories would re-roll
       from the same starting state. In Craftium WE CANNOT DO THAT cheaply.
       Instead, this sampler treats G as a batched-policy-improvement budget:
       collect G episodes (or G fixed-length segments) under the current
       policy, group them by some equivalence (default: same chamber +
       similar starting condition), and compute group-relative advantages
       within each equivalence class.

       This is more like REINFORCE++ / generalized advantage normalization
       than strict on-policy GRPO. The thesis should note this as a
       practical departure from the GRPO paper.
       """
       def __init__(self, env, policy, n_per_group: int = 8,
                    horizon: int = 50, chamber_filter: str | None = None):
           ...

       def sample_group(self) -> list[GRPOTrajectory]:
           """Sample G trajectories. Returns a list of length G."""
           ...
   ```

   Design choice the implementer must surface to the user: how to define "equivalence class" for grouping. Default proposal: same chamber + agent starts within 2 blocks of a fixed reference. This is **Open Question #2 in §9** — confirm before implementing.

2. **`src/rlvr/reference_policy.py`.** Manages the frozen reference policy for the KL penalty:

   ```python
   class ReferencePolicy:
       """A frozen snapshot of the base LoRA-adapted LLM at training start.
       Used to compute the KL penalty term in the GRPO loss.

       Implementation: PEFT supports multiple adapters per model via
       adapter_name=. The legacy stack already uses this at
       src/rl_layer/rl_layer.py:77-119. We register the trainable adapter as
       'grpo_policy' and a frozen copy of the same initial weights as
       'grpo_reference'. KL is computed by running both adapters on the same
       prompt+response pair and taking the KL between their token distributions.

       The 'grpo_policy' adapter is independent of the legacy stack's
       adapter (typically named per-agent or 'rl_layer'). Do not load
       the MAPPO LoRA into 'grpo_reference' — that would warm-start GRPO
       from a partially-trained policy and contaminate the comparison.
       """
       def __init__(self, model, initial_lora_state_dict):
           ...
       def compute_kl(self, prompt: str, response_tokens: torch.Tensor) -> torch.Tensor:
           ...
   ```

3. **`src/rlvr/grpo_trainer.py`.** The trainer:

   ```python
   class GRPOTrainer:
       def __init__(self, config: GRPOConfig,
                    policy: LLMPolicy,
                    reference: ReferencePolicy,
                    verifier: FiveChambersVerifier,
                    sampler: RolloutSampler):
           ...

       def step(self) -> dict:
           """One GRPO update step:
             1. sampler.sample_group() → G trajectories
             2. verifier.score_group() → G scalar rewards
             3. compute group advantages
             4. for each (prompt, response, advantage) tuple,
                compute the clipped surrogate loss + KL penalty
             5. backprop, update LoRA weights
           Returns a dict of logged metrics.
           """
           ...

       def train(self, total_steps: int):
           """Outer loop. Saves LoRA checkpoint every K steps."""
           ...
   ```

   GRPO loss (token-level, per the DeepSeek-R1 formulation):

   ```
   ratio_t = π(a_t | s_t, history) / π_old(a_t | s_t, history)
   surrogate = min(ratio_t · A, clip(ratio_t, 1-ε, 1+ε) · A)
   kl_t = KL(π(·|s_t) || π_ref(·|s_t))
   loss = -E_t[surrogate] + β · E_t[kl_t]
   ```

   where `A` is the trajectory-level group-relative advantage (same for every token in the trajectory — whole-trajectory credit assignment), `ε` is the clip parameter (default 0.2), `β` is the KL coefficient (default 0.05 — tune carefully).

4. **`src/rlvr/grpo_buffer.py`.** A simple buffer for groups of G:

   ```python
   @dataclass
   class ScoredTrajectory:
       trajectory: GRPOTrajectory
       reward: float
       advantage: float = 0.0  # populated after group normalization
       prompt_text: str = ""
       response_tokens: torch.Tensor | None = None
       response_logprobs: torch.Tensor | None = None
       origin_agent: int | None = None  # for Stage-4b cross-agent borrowing

   class GroupBuffer:
       """Holds the most recent G trajectories for one decision episode.
       After scoring, normalizes advantages within the group.
       """
       def __init__(self, group_size: int):
           ...
       def add_group(self, trajectories: list[GRPOTrajectory],
                     rewards: list[float],
                     prompts_responses: list[tuple]):
           ...
       def get_minibatch(self, batch_size: int) -> list[ScoredTrajectory]:
           ...
   ```

5. **`src/mindforge/multi_agent_craftium_grpo.py`.** The new entry point. Skeleton:

   ```python
   """GRPO + RLVR training entry point.

   Parallel to multi_agent_craftium.py (legacy MAPPO entry point).
   Existing legacy code is unchanged; this is an additive sibling.

   Launch:  python -m mindforge.multi_agent_craftium_grpo \
                --config configs/rlvr/grpo_single_agent_ch3.yaml

   The legacy path is launched with CLI flags (no --config), e.g.:
       python -m mindforge.multi_agent_craftium --rl --hebbian ...
   The new path's --config pattern is a deliberate departure — GRPO has
   enough hyperparameters that a YAML config is more practical than ~50
   argparse flags. See README_RLVR.md.
   """
   import argparse
   from pathlib import Path

   def main(config_path: str):
       config = load_config(config_path)

       # Same env, same LLM client as the legacy path
       env = build_env(config.env)         # reuse marl_craftium.openworld_multi_agents
       llm = build_llm_client(config.llm)  # reuse LocalModelClient

       # NEW: separate LoRA adapter from the MAPPO one
       policy = LLMPolicy(llm, lora_config=config.lora,
                          adapter_name="grpo_policy",
                          checkpoint_path=config.policy.checkpoint_path)
       reference = ReferencePolicy.from_initial(policy, adapter_name="grpo_reference")

       hebbian = build_hebbian(config.hebbian) if config.hebbian.enabled else None
       verifier = FiveChambersVerifier(config.verifier, hebbian=hebbian)
       sampler = RolloutSampler(env, policy, **config.sampler)
       trainer = GRPOTrainer(config.grpo, policy, reference, verifier, sampler)

       trainer.train(total_steps=config.total_steps)

   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", type=str, required=True,
                           help="Path to GRPO YAML config")
       args = parser.parse_args()
       main(args.config)
   ```

6. **Single-agent restriction for Stage 2.** The simplest workable setup: train **one** agent in Ch3, fixed LLM for the other two agents (they execute scripted/fixed-policy behaviour). This isolates whether GRPO can learn at all in this env before adding multi-agent complications. The fixed-policy behaviour for the other agents can be the existing prompt-driven LLM behaviour without any training — they're effectively scenery.

7. **Tests.** `tests/rlvr/test_grpo_trainer.py`:
   - Synthetic group of G=4 trajectories with hand-set rewards `[1.0, 1.0, 0.0, 0.0]`. Assert group advantages are `[+1, +1, -1, -1]` after normalization (the std of `[1,1,0,0]` is 0.5 so advantages are `±1`, give or take the `ε` term).
   - Run a single `trainer.step()` with mocked sampler/policy and assert the LoRA weights changed.
   - Same call twice with the same RNG seed and frozen sampler → identical weight delta.

   `tests/rlvr/test_grpo_smoke.py`:
   - Tiny config (G=2, horizon=10, 5 GRPO steps) using a stub env that fires milestone `m17_switch_pressed` deterministically after 3 steps. Assert the policy's milestone-fire rate increases over the 5 GRPO steps.

**Verification gate:** A 24-hour single-agent GRPO run on Ch3 produces visible learning: milestone-fire rate for `m17_switch_pressed` (40.0) and `m18_door_opened` (60.0) increases over the run. Tensorboard / JSONL logs show group-mean reward trending up. The LoRA checkpoint is saved to `runs/{run_id}/grpo_lora/`. Critically: the legacy MAPPO entry point still works unchanged — `pytest` over the existing test suite passes, and a sanity-check legacy training run (`python -m mindforge.multi_agent_craftium --num-agents 3 --episodes 1 --max-steps 50 --rl --hebbian`) completes without error.

### Stage 3 — Multi-agent GRPO (1 week)

Once Stage 2 is solid, extend to multiple agents trained simultaneously.

Two options, both worth trying:

**Option 3A — Joint MAGRPO.** Each "trajectory" is a joint trajectory: all N agents acting in one rollout. The group of G consists of G joint trajectories. Reward is the team-level verifier reward. Advantages are computed once per joint trajectory and applied to every agent's policy update on that trajectory. Cleanest cooperative signal; most expensive per group.

**Option 3B — Per-agent GRPO with shared env rollouts.** Each agent has its own GRPO group, but the env is rolled out jointly (we can't separate them). Each agent gets its own per-trajectory reward (the verifier supports this — milestone fires can be credited to the firing agent based on `agent_id` in milestone events). Group-relative advantage is per-agent. Multi-agent learning happens because each agent independently learns from its own trajectory of joint experience.

Both options can be implemented in the same trainer with a config flag (`grpo.team_reward: bool`). Start with 3B (simpler) and add 3A as an ablation.

Tasks:

1. Generalize `RolloutSampler.sample_group` to return groups of joint trajectories (length G, each containing N agents' actions).
2. Generalize `FiveChambersVerifier.score_group` to return either a per-trajectory scalar (3A) or a per-(trajectory, agent) matrix (3B).
3. Generalize `GRPOTrainer.step` to compute either team-shared or per-agent advantages.
4. Update the entry point to start N policies (or one shared LoRA, configurable via `policy.share_params`).

**Verification gate:** A 48-hour multi-agent GRPO run on Ch3 with 3 agents shows joint milestone fire rate (`m_comm_ch3` reward 30.0 + `m17_switch_pressed` 40.0 + `m19_all_in_communal` 100.0) trending up. Compare against Stage 2 single-agent baseline — multi-agent should not be drastically worse.

### Stage 4 — Hebbian integration (2 weeks)

Two integration points, both independently toggleable. They are **additive** to the GRPO trainer — when both flags are off, GRPO runs as in Stage 3.

#### 5.4 (Stage 4a) — Hebbian reward diffusion in the verifier

This is the easier of the two. When `config.hebbian.reward_diffusion = True`, the verifier's `score_group` method, after computing per-(trajectory, agent) rewards, applies `hebbian.diffuse_rewards(...)` ([`src/hebbian/graph.py:496`](../src/hebbian/graph.py)) to spread reward across the Hebbian-bonded teammates. The Hebbian graph is updated as the training progresses, using positions and comm_events extracted from trajectory metadata.

Tasks:
1. Add `hebbian_reward_diffusion` flag to `VerifierConfig`.
2. In `FiveChambersVerifier.__init__`, accept an optional `HebbianSocialGraph`.
3. In `score_group`, if the flag is on and the graph is provided, apply diffusion after computing raw per-agent rewards.
4. Add a `HebbianGRPOBridge` callback (in `src/rlvr/hebbian_grpo_bridge.py`) that updates the Hebbian graph each time a trajectory is added to the buffer — feeding positions, step_rewards, and comm_events from the trajectory.

Test: `tests/rlvr/test_hebbian_grpo_bridge.py`:
- With diffusion off, group rewards are identity-equal to raw rewards.
- With diffusion on and a manually-set W = identity, group rewards are unchanged (`diffuse_rewards` with zero off-diagonal is identity).
- With diffusion on and a manually-set W with non-zero off-diagonal, group rewards differ from raw rewards in the expected direction.

#### 5.4 (Stage 4b) — Hebbian-weighted group composition (with explicit off-policy correction)

This is the scientifically interesting integration and the one that justifies the thesis claim that Hebbian directly shapes LLM training.

**Concept:** when assembling agent i's group of G trajectories for the GRPO update, sample some fraction of them from teammate buffers with probability proportional to W̄[i, j]. The rest come from agent i's own buffer.

**Honest math: this IS off-policy, and the off-policy correction matters.**

The prior draft of this plan claimed that GRPO's group-relative advantage makes the importance correction unnecessary. That is not quite right. The standard GRPO surrogate is:

```
L_i = E_t [ min( r_t · A_g , clip(r_t, 1±ε) · A_g ) ]
where r_t = π_i(a_t | s_t) / π_old(a_t | s_t)
```

If trajectory `g` was generated by teammate j's policy π_j, then evaluating the surrogate ratio under agent i's policy gives:

```
r_t = π_i(a_t | s_t) / π_j(a_t | s_t)
```

This is a legitimate importance ratio. The advantage `A_g` being group-relative (rather than value-baseline-relative) does **not** remove this. What changes is:

- The advantage no longer requires a critic, so the variance from a bad critic is gone — good.
- The clip on `r_t` still bounds the off-policy distortion, but at the cost of biased gradients when `r_t` is far from 1. With teammate-borrowed samples, `r_t` can easily exit `[1-ε, 1+ε]` because π_i and π_j differ structurally — bad.

**Two honest options for the implementer:**

**Option 4b-i — Clipped off-policy (cheap, biased).** Treat teammate-borrowed trajectories exactly like own trajectories: `π_old` is "agent i's policy at the time the group was assembled" for own samples, and "teammate j's policy at the time trajectory was collected" for borrowed samples. Use the same clip range. Bias is bounded by clipping but real. Use a small `hebbian_borrow_fraction` (e.g. 0.25) to limit exposure. This is the pragmatic default.

**Option 4b-ii — Rejection sampling (correct, expensive).** When borrowing trajectory g from teammate j, evaluate the trajectory tokens under agent i's policy. Compute the importance weight `w_g = ∏_t π_i(a_t|s_t) / π_j(a_t|s_t)` (the trajectory-level IS weight). If `w_g` exceeds a threshold (e.g. 0.5), accept the trajectory with weight `min(w_g, c)` for a clip ceiling `c`; otherwise reject and resample. This is closer to the truncated-IS estimator used in V-trace / IMPALA.

**Plan recommendation:** implement Option 4b-i first as a Stage-4b baseline. Add Option 4b-ii as an ablation if Stage-4b-i shows the clip term is firing too often (which Tensorboard should reveal — log `fraction_clipped` per minibatch).

**Concept revision.** The plan's prior framing — "no IS needed because GRPO" — is replaced by: "borrowing teammate trajectories is off-policy; we acknowledge it and either accept the clipped bias (cheap) or apply explicit truncated IS (expensive)." This is more honest and more defensible in the thesis.

Tasks:
1. Add `hebbian_group_composition` flag to `GRPOConfig`.
2. Add a per-agent recent-trajectory buffer (separate from `GroupBuffer`; this one persists across GRPO steps).
3. In `GRPOTrainer.step`, when constructing agent i's group:
   - K of the G trajectories come from agent i's own recent rollouts (typical default: K = ⌈0.75 · G⌉).
   - G − K come from teammate buffers, sampled with `np.random.choice(N, p=W̄[i, :])`.
   - For borrowed samples, tag `origin_agent` in `ScoredTrajectory` so the loss function knows which `π_old` to use.
4. Group-relative advantage is computed across the assembled group of G.
5. Log `fraction_clipped` and `borrow_advantage_mean` separately from own-trajectory metrics.

Test: `tests/rlvr/test_hebbian_group_composition.py`:
- With composition off, agent 0's group contains only its own trajectories.
- With composition on and W̄[0, :] = [0.0, 0.9, 0.1], agent 0's group contains many more trajectories from agent 1 than from agent 2.
- With borrowed trajectories tagged `origin_agent`, the loss function applies the teammate's `π_old` for the ratio, not agent 0's.
- Verify that advantages within the assembled group still sum to ~0 (group-relative property is preserved).

**Verification gate (end of Stage 4):** A full multi-agent + Hebbian-diffusion + Hebbian-composition GRPO run completes 1M training steps. Hebbian asymmetric bonds form and are logged. The headline plot — bond evolution alongside milestone-fire rate over training — shows interpretable patterns (e.g. bonds strengthen before milestones start firing reliably). `fraction_clipped` for borrowed trajectories is below 30% (or Option 4b-ii is activated if higher).

### Stage 5 (optional) — Async serving/training split (2 weeks)

Only attempt if Stages 1–4 are working and you need throughput. This decouples the rollout worker (running env + frozen policy) from the trainer (consuming scored trajectories + updating weights), in the verl / OpenRLHF / TRL pattern.

Tasks (sketch only — flag this to the user as a separate work item):
1. `src/rlvr/async_runner.py`: a process that runs the env continuously with a frozen LoRA snapshot, dumps trajectories to a queue.
2. Trainer reads from the queue, scores with the verifier, runs GRPO updates.
3. Periodic weight sync: trainer pushes a new LoRA checkpoint to the rollout worker every K minutes.

Effective sample utilization should roughly double on slow Craftium envs. **Defer to user discussion**; this stage has its own design decisions worth a separate conversation.

### Stage 6 — Comparison and ablation tooling (3 days)

Tasks:
1. `scripts/compare_modes.py`: loads checkpoints from both the legacy MAPPO run and the new GRPO run, evaluates each on the same fixed eval set (e.g. 100 seeded Ch3 episodes), produces side-by-side milestone-fire rate plots, cooperation_score comparison, and time-to-milestone violin plots.
2. Add ablation configs covering the combinations:
   - `grpo_only.yaml` (no Hebbian)
   - `grpo_hebbian_diffusion.yaml` (4a only)
   - `grpo_hebbian_composition.yaml` (4b only, Option 4b-i)
   - `grpo_hebbian_composition_truncated_is.yaml` (4b only, Option 4b-ii)
   - `grpo_hebbian_full.yaml` (4a + 4b)
3. Document the ablation grid in `README_RLVR.md`.

**Verification gate:** `compare_modes.py` produces a publication-quality figure showing the MAPPO baseline vs. GRPO variants on Ch3.

---

## 6. Configuration: mode toggle and parallel configs

The two paths have different launch conventions because of the codebase's history:

```bash
# Legacy MAPPO/IPPO path (unchanged) — uses CLI argparse flags
python -m mindforge.multi_agent_craftium \
    --num-agents 3 --episodes 10 --max-steps 1000 \
    --rl --rl-mode action --rl-critic-mode centralized \
    --hebbian --hebbian-rho 0.3 \
    --experiment-id E1a

# Legacy token-PPO path (unchanged) — same entry point, different flags
python -m mindforge.multi_agent_craftium \
    --rl --rl-mode token --rl-auto-token-opt \
    --hebbian

# New GRPO path — YAML config
python -m mindforge.multi_agent_craftium_grpo \
    --config configs/rlvr/grpo_single_agent_ch3.yaml
```

There is **no shared top-level "mode" flag** that switches between them. The three configurations are siblings:
- legacy action-mode PPO (`--rl --rl-mode action`)
- legacy token-mode PPO (`--rl --rl-mode token`)
- new GRPO (`multi_agent_craftium_grpo.py`)

Each loads its own config. This avoids any temptation to add a giant `if mode == "mappo": ...` switch in shared code.

A representative GRPO config skeleton (`configs/rlvr/grpo_single_agent_ch3.yaml`):

```yaml
# Inherits nothing from legacy configs — fully self-contained
env:
  type: craftium_five_chambers
  num_agents: 3                  # but only agent 0 is trained in Stage 2
  chamber_subset: [ch3]          # train only on Ch3 starting conditions
  max_episode_steps: 100
  ch1_timeout_steps: 400

llm:
  base_model: ...                # same base as legacy
  lora_config:
    r: 16
    target_modules: [q_proj, v_proj]

policy:
  share_params: false            # one LoRA per agent (Stage 3+)
  adapter_name: grpo_policy
  checkpoint_path: null          # start from base; alternative: load a Stage-2 single-agent checkpoint
  trained_agents: [0]            # Stage-2 single-agent restriction

reference:
  adapter_name: grpo_reference
  initialise_from: policy        # frozen copy of policy at training start

verifier:
  use_milestone_rewards: true
  use_format_reward: true
  format_reward_weight: 0.1
  use_alive_bonus: true
  hebbian_reward_diffusion: false   # Stage-4a toggle

sampler:
  n_per_group: 4                 # G
  horizon: 50                    # H
  equivalence_class: chamber_plus_position

grpo:
  team_reward: false             # per-agent rewards (option 3B)
  hebbian_group_composition: false   # Stage-4b toggle
  hebbian_borrow_fraction: 0.25
  hebbian_borrow_method: clipped_off_policy   # 4b-i; alt: truncated_is for 4b-ii
  clip_epsilon: 0.2
  kl_coefficient: 0.05
  learning_rate: 5e-6
  total_steps: 1000

hebbian:
  enabled: false                 # off until Stage 4
  # When enabled, re-use the same HebbianConfig schema as the legacy code
  radius: 5.0
  ltp: 0.01
  ltd: 0.005
  decay: 0.005
  beta: 1.0
  init_weight: 0.1

logging:
  jsonl: true
  tensorboard: true
  log_interval: 10               # GRPO steps, not env steps
  checkpoint_interval: 100
```

---

## 7. Hard constraints (read before every commit)

1. **No deletions.** Nothing in `src/rl_layer/`, `src/hebbian/`, `src/mindforge/multi_agent_craftium.py`, or any existing file is deleted or modified. Period.
2. **No silent modifications.** If a change to an existing file feels necessary, stop and ask the user.
3. **GRPO LoRA is a separate adapter.** Use `adapter_name="grpo_policy"`. Never load the MAPPO-trained LoRA into this adapter, and vice versa. They are independently evaluable models.
4. **The verifier is pure.** No mutable state, no env calls, no LLM calls inside `score`. Idempotence is tested.
5. **Tests come first.** Each new module ships with its test in the same commit. Smoke tests use synthetic data, not Craftium.
6. **Read-only imports only.** When importing from existing modules (`TRACKS`, `MILESTONE_TRACK`, `HebbianSocialGraph`, etc.), never call mutating methods on the imported objects without explicit user approval.
7. **Comments explain why, not what.** Look at `src/hebbian/graph.py` for the right tone.
8. **Reward table comes from `TRACKS`, never `MILESTONE_TRACK`.** `MILESTONE_TRACK` maps milestone_id → track_name (string). Use `rlvr.reward_table.build_milestone_rewards()` to get the flat reward dict.
9. **Two different jsonl files, two different locations.** `milestone_events.jsonl` lives in the Minetest world dir (Lua writes it). `event_log.jsonl` lives in `runs/{run_id}/episodes/ep_*/` (Python writes it). The verifier reads both.
10. **The action space is `Discrete(23)`.** 22 named actions (`_DISCRETE_ACTIONS` in `src/marl_craftium/_actions.py`) + 1 NOP at index 0. Use `_DISCRETE_ACTIONS` as the source of truth for action-name validation in the format reward.
11. **`token_opt.py` is the legacy token-PPO sibling.** GRPO does not subsume it. Do not import from it. Document the parallel in `README_RLVR.md`.
12. **Verify the LBF track is independent.** The LBF plan (separate document, separate repo) is unaffected by this work. They are parallel thesis chapters, not branches of one effort.
13. **Pin every dependency.** Any new dep added for GRPO (e.g. `trl`, `verl`, `transformers_lm_eval`) is pinned in `pyproject.toml`.
14. **Commit frequently with descriptive messages.** "Add FiveChambersVerifier with milestone + format rewards (Stage 1, pure function, 16 unit tests)" — not "wip".

---

## 8. Open questions for the user to resolve

Each of these should be raised explicitly before silently picking an answer. Ordered by when they will be encountered:

- **Stage 0 — Package discovery.** Does `pyproject.toml` use auto-discovery (`find_packages`) or an explicit `packages = [...]` list? If explicit, do we add `src/rlvr/` to it? (One-line invasive change; otherwise blocking.)
- **Stage 1 — Passive observer hook.** Disk-tail (preferred, fully non-invasive) vs. `EpisodeLogger` callback (one new hook in `EpisodeLogger`, instant)? Plan defaults to disk-tail.
- **Stage 2 — RolloutSampler equivalence class.** What defines "comparable starting conditions" for grouping trajectories? Default: same chamber + position within 2 blocks of a fixed reference. Alternatives: same chamber only (looser groups, more variance), exact same env seed (tighter, but artificial).
- **Stage 2 — Horizon H.** Plan assumes H=50 (Ch3 milestones fire fast). What if early-stage runs show the policy needs longer horizons (e.g. coordinating before pressing a switch)?
- **Stage 2 — Whether to load an existing checkpoint.** Start GRPO from the base LLM (clean comparison) or from a partially-trained MAPPO LoRA (warm start, but contaminates the comparison)? Recommend the former for the headline result.
- **Stage 3 — Team-reward vs. per-agent reward (3A vs 3B).** Plan implements both, recommends 3B for first pass. User decides which becomes the headline ablation.
- **Stage 4b — Off-policy correction method.** Option 4b-i (clipped, biased, cheap) or Option 4b-ii (truncated IS, correct, expensive)? Plan defaults to 4b-i first, 4b-ii as an ablation.
- **Stage 4 — Hebbian-group-composition vs. policy mixture.** This plan only includes the composition variant of Hebbian, not the policy mixture from earlier discussions. Should the policy mixture be added as Stage 4c? Most likely yes, but worth confirming.
- **Stage 5 — Async serving/training.** Whether to attempt this at all. Probably yes for thesis-quality results; probably no for thesis-deadline reasons. User decides.
- **Format-reward weight.** Plan defaults to 0.1. RLVR literature shows this matters — too high and the model optimizes for valid JSON instead of milestone completion; too low and it never learns to produce parseable output. Likely needs tuning.
- **KL coefficient β.** Plan defaults to 0.05. This is the most sensitive hyperparameter in GRPO; expect to spend time tuning.

---

## 9. What this plan does NOT cover

- **LBF / EPyMARL track.** Separate repo, separate plan, separate thesis chapter. The two efforts share only the Hebbian module (numpy-only, imported read-only in both).
- **Pre-training on Minetest wiki text** (the LARM Path 3 idea). Useful but orthogonal; can run in parallel without affecting this plan.
- **Repurposing the existing Critic LLM as a per-step referee** (the LARM Path 1 idea). Verifier-based RLVR replaces the referee idea entirely. If the user later wants both, they can be composed (verifier reward + referee bonus).
- **Architectural simplification** (the "delete the centralised critic" idea). Explicitly out of scope per the non-deletion constraint. The legacy stack stays as-is.
- **Replacing the ActionHead with JSON parsing in the legacy path.** Tempting, but a modification to existing code — explicitly out of scope. The GRPO path uses JSON parsing; the legacy path keeps the ActionHead.
- **Subsuming `token_opt.py` into GRPO.** They remain parallel siblings. Token-PPO stays; GRPO is additive.

---

## 10. Success criteria

After approximately 4–5 weeks of focused implementation, the user should be able to:

```bash
# Legacy MAPPO path still works (regression check)
python -m mindforge.multi_agent_craftium \
    --num-agents 3 --episodes 1 --max-steps 100 \
    --rl --rl-mode action --hebbian --experiment-id regression_check
# → completes, milestones fire, MAPPO LoRA checkpoint saved

# Legacy token-PPO path still works (regression check)
python -m mindforge.multi_agent_craftium \
    --num-agents 3 --episodes 1 --max-steps 100 \
    --rl --rl-mode token --rl-auto-token-opt --hebbian
# → completes, token-opt fires at least once

# New GRPO path works
python -m mindforge.multi_agent_craftium_grpo \
    --config configs/rlvr/grpo_hebbian_full.yaml
# → completes, milestones fire, GRPO LoRA checkpoint saved, Hebbian bonds form

# Side-by-side comparison
python scripts/compare_modes.py \
    --mappo-ckpt runs/mappo_latest/lora/ \
    --grpo-ckpt runs/grpo_latest/grpo_lora/ \
    --eval-episodes 100
# → publication-quality figure: MAPPO vs GRPO vs GRPO+Hebbian
```

And get a clean answer to the thesis question:

> **Does GRPO with verifiable rewards, optionally augmented with Hebbian-modulated group composition, learn more sample-efficiently than the legacy MAPPO + dense-shaping baseline on the Craftium Five-Chambers environment?**

Strong positive result: GRPO+Hebbian outperforms MAPPO baseline on milestone-fire rate and cooperation metrics. This becomes the Craftium chapter's headline.

Strong negative result: GRPO doesn't beat MAPPO in this env. Also publishable — characterise why (sparse rewards too sparse, env too long-horizon, Hebbian composition bias from clipped IS, etc.) and present alongside the LBF result.

Either way, the parallel-tracks design means the GRPO/Craftium outcome doesn't gate the LBF/Hebbian-MARL outcome. The thesis remains defensible from the LBF chapter alone if Craftium fails to learn, and the Craftium chapter strengthens the story if it succeeds.

---

## 11. Glossary

- **RLVR**: Reinforcement Learning with Verifiable Rewards. Rewards come from deterministic programmatic checks (milestone fires, format validity, unit-test passes), not learned reward models or human preferences.
- **GRPO**: Group Relative Policy Optimization (Shao et al., DeepSeek-Math 2024 / DeepSeek-R1 2025). Computes advantages by normalising within a sampled group of G trajectories, replacing a value-function critic.
- **MAGRPO / M-GRPO**: Multi-agent extensions of GRPO (Hong et al. 2025, Wei et al. 2025).
- **Verifier**: A pure function from trajectory to scalar reward. In this plan, `FiveChambersVerifier` reads milestone events and format-validity scores.
- **Group / G**: The number of trajectories sampled per "decision episode" for group-relative advantage. Typical values 4–8.
- **Horizon / H**: Maximum trajectory length in env steps. Trajectories may terminate earlier on milestone fire / death / chamber transition.
- **Reference policy**: A frozen snapshot of the LoRA-adapted LLM at training start, registered as a separate PEFT adapter (`grpo_reference`). Used in the KL penalty term to prevent the trained policy from drifting too far.
- **Format reward**: A small reward (≤ 0.1 of milestone reward magnitude) for the LLM producing well-formed JSON output with an `action` field in `_DISCRETE_ACTIONS`. Stabilises early training.
- **Equivalence class**: The grouping used to decide which G trajectories form one GRPO group. Practical concession to slow envs that don't support save/restore.
- **Hebbian reward diffusion (Stage 4a)**: After scoring G trajectories, redistribute reward across agents via the Hebbian-bonded teammate graph, before computing advantages.
- **Hebbian group composition (Stage 4b)**: When assembling agent i's group, some trajectories come from teammate j's buffer, with selection probability proportional to W̄[i, j]. Treated explicitly as off-policy (Option 4b-i clipped, or 4b-ii truncated IS).
- **Legacy path**: The existing MAPPO/IPPO/centralised-critic stack + the token-PPO stack. Stays alive and unmodified.
- **GRPO path**: The new stack added by this plan. Independent LoRA adapter, independent training loop, independent entry point.

---

## 12. Codebase verification facts

These facts are confirmed by reading the codebase. They ground the plan against drift.

### Confirmed module paths and symbols

| Claim | Verified at | Notes |
|---|---|---|
| `MILESTONE_TRACK` dict | `src/mindforge/agent_modules/craftium_metric.py:24-58` | **Maps to track-name strings, NOT rewards.** Used for grouping only. |
| `TRACKS` dict | `src/mindforge/agent_modules/craftium_metric.py:61-90` | **The reward source.** Flatten via `rlvr.reward_table.build_milestone_rewards()`. |
| `_plot_hebbian_asymmetry` | `src/mindforge/agent_modules/craftium_metric.py:743` | Reusable for plots. |
| `HebbianConfig` | `src/hebbian/config.py` | Imported via `from hebbian import HebbianConfig` ([__init__.py:11](../src/hebbian/__init__.py)). |
| `HebbianSocialGraph` | `src/hebbian/graph.py:46` | Class definition. |
| `HebbianSocialGraph.diffuse_rewards()` | `src/hebbian/graph.py:496` | Stage 4a integration point. |
| `HebbianSocialGraph.update()` | `src/hebbian/graph.py:309` | Called per step in legacy; bridge replicates this in GRPO. |
| `HebbianSocialGraph.get_social_replay_indices()` | called from `src/rl_layer/ppo_update.py:125` | Used in legacy social-replay; **not** used by GRPO. |
| `LocalModelClient` | `src/mindforge/agent_modules/local_model_client.py` | LLM call interface. |
| PEFT/`LoraConfig`/`get_peft_model` with `adapter_name=` | `src/rl_layer/rl_layer.py:77-119` | Confirms multi-adapter support — ReferencePolicy approach is feasible. |
| `EpisodeLogger` | `src/mindforge/env/episode_logger.py` | Stage-1 passive-observer integration point candidate. |
| `_PatchedMarlCraftiumEnv` (luanti binary patch) | `src/marl_craftium/openworld_multi_agents.py` | Env wrapper; see CLAUDE.md memory note. |
| `_DISCRETE_ACTIONS` (22 named actions) | `src/marl_craftium/_actions.py:8-16` | Action space is `Discrete(23)` = NOP + 22. |
| `critic_explained_variance` metric | `src/rl_layer/centralized_critic.py:365` | Confirms the "weak critic" audit claim. |
| `social_replay_rho` (legacy Hebbian PPO) | `src/rl_layer/ppo_update.py:128`, default `--hebbian-rho 0.3` at `src/mindforge/multi_agent_craftium.py:138` | **NOT zero by default.** Plan's prior "ρ=0" claim corrected here. |
| `token_opt.maybe_token_optimize` | `src/rl_layer/token_opt.py:41` | Existing token-PPO pathway. **Not subsumed by GRPO.** |
| `token_level_ppo_step` | `src/rl_layer/ippo.py` | Existing token-PPO update function. |

### File locations of the two verifiable-reward streams

| Stream | Path | Writer | Schema |
|---|---|---|---|
| `milestone_events.jsonl` | `<minetest_world_dir>/milestone_events.jsonl` (resolved via `run_layout.py`) | Lua mod (`emit_milestone()` in five-chambers Lua mod) | `{step, agent_id, milestone_id, ...}` |
| `event_log.jsonl` | `runs/{run_id}/episodes/ep_*/event_log.jsonl` | Python (`EpisodeLogger`) | `{step, type, agent_id, ...}` |

The verifier reads both and joins on `(step, agent_id)`. Do not assume they share a directory.

### Legacy entry-point convention

The legacy entry point is **argparse-based**, not config-file-based:

```
python -m mindforge.multi_agent_craftium [--num-agents N] [--episodes N] [--max-steps N]
                                          [--rl --rl-mode {action,token}]
                                          [--rl-critic-mode {centralized,independent}]
                                          [--hebbian --hebbian-rho R --hebbian-radius D ...]
                                          [--experiment-id ID]
                                          ...  (~40 flags total)
```

There is no `--config` argument. The new GRPO entry point's `--config` pattern is a deliberate departure documented in `README_RLVR.md`.

---

*End of plan.*
