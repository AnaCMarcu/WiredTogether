# Hebbian Social Plasticity on Level-Based Foraging — Implementation Plan

## Context

The user is writing a thesis chapter on Hebbian-modulated social plasticity in cooperative multi-agent RL. The parent codebase (this repo, `WiredTogether`) entangles three load-bearing experimental variables — the Hebbian mechanism, LLM-conditioned RL training, and a custom Craftium/Five-Chambers environment — and any negative result there is uninterpretable.

This sub-project isolates the Hebbian variable in a **small, fast, well-instrumented environment** (Level-Based Foraging) using a **maintained MARL framework** (EPyMARL), so that the result speaks cleanly to the thesis claim. The Hebbian implementation already exists at [src/hebbian/config.py](src/hebbian/config.py) and [src/hebbian/graph.py](src/hebbian/graph.py) (807 lines total, numpy-only, no torch dependency) and is the mechanism the thesis defends — it is **ported, not rewritten**.

**Two questions this sub-project must answer:**
1. Does Hebbian-weighted experience sharing improve cooperative MARL over uniform shared experience on LBF? (`hebb_s` vs. `seac`)
2. Does the communication term of the Hebbian co-activity signal carry real learning signal, or is the spatial term sufficient? (`hebb_s` vs. `hebb_s_nocomm`)

The work lives in a **new sibling repo** (recommended: `../hebbian-marl/`), not inside `WiredTogether`. Parent codebase has zero EPyMARL or lbforaging dependencies; this is greenfield.

---

## Verification findings folded into the plan

These were checked against the parent repo before finalizing:

- `HebbianConfig` (dataclass, 83 lines): all expected fields present, including `enabled`, `num_agents`, `interaction_radius` (default `5.0`), `engagement_reward_weight` (= α), `communication_coactivity_bonus` (= δ_comm, default `0.5`), `reward_diffusion_gamma`. `agent_roles` is a **constructor kwarg** on `HebbianSocialGraph`, not a config field. `modularity_proxy` is internal state, not a config field.
- `HebbianSocialGraph.update(positions, step_rewards, advantages=None, comm_events=None)` — signatures verified at [src/hebbian/graph.py:309](src/hebbian/graph.py#L309).
- **`HebbianSocialGraph.diffuse_rewards(raw_rewards, co_activity_matrix=None)`** — takes no `gamma` kwarg; reads `self.config.reward_diffusion_gamma`. Test in Phase 1 reformulated accordingly.
- `get_normalized_weights(i: int) -> np.ndarray` returns a row of W̄ for one agent. Used in the learner sampler.
- `_compute_coactivity` in [src/hebbian/graph.py:92](src/hebbian/graph.py#L92) matches the spec: engagement `g_i = α · |r_i|/(max_r + ε) + (1-α) · comm_i`, spatial co-activity `cij_spatial = I[dist ≤ d] · g_i · g_j`, comm bonus `cij_comm = δ_comm · comm_pair_ij · (1 - I[dist ≤ d])`.
- `comm_events` is `List[Tuple[int, int]]` (sender, receiver) — call site at [src/mindforge/multi_agent_craftium.py:1563](src/mindforge/multi_agent_craftium.py#L1563).
- Plotting function `_plot_hebbian_asymmetry` is at [src/mindforge/agent_modules/craftium_metric.py:743](src/mindforge/agent_modules/craftium_metric.py#L743). It reads from `self._graph_snapshots` (an instance attribute), takes no args. Phase 6 must adapt it to read from `bonds.jsonl` — slightly more than a copy-paste.

---

## Scope decisions resolved

- **Headline environment: `lbforaging:Foraging-10x10-3p-3f-v3`** with `force_coop=False` (cooperation incentive from food levels, not constraint) and `time_limit=50`. The 10x10/3-food variant is the middle ground: richer foraging dynamics than 8x8 but still tractable for a 45-run ablation grid on one laptop in ≤10h.
- **Integration point (c) — Hebbian policy mixture — is a stretch goal.** Headline result rests on (a) reward diffusion + (b) Hebbian-weighted sharing. Phase 5 is attempted only after Phases 3 and 4 are green; `hebb_rsp` is reported as future work if (c) destabilizes.
- **Comm channel uses the opportunity-cost design** (signal action consumes the agent's movement that step). Rationale in §3.3.
- **IS correction in shared learner is required for the headline run.** First-pass commit may land without it for plumbing reasons, but it must be in place before any reported ablation result.

---

## 1. The communication channel

### Why it's required, not optional

The Hebbian co-activity computation has three components, and communication appears in two of them:
```
gi(t)         = α · |ri(t)|/(max_r + ε) + (1-α) · comm_i(t)        # engagement
cij_spatial   = I[dist ≤ d] · gi(t) · gj(t)                         # spatial co-activity
cij_comm      = δ_comm · comm_pair_ij · (1 - I[dist ≤ d])           # cross-distance comm
```
Running on stock LBF means `comm_i(t) ≡ 0` and `comm_pair_ij ≡ 0` always — that's a valid ablation (`hebb_s_nocomm`) but not the headline experiment, because it tests only the spatial half of the mechanism the thesis defends.

### Design: discrete targeted signal as action-space extension

Each agent's action space grows from LBF's 6 primitives to `6 + (N − 1)`:

| Index | Action |
|---|---|
| 0–5 | LBF native: NoOp, North, South, East, West, Load |
| 6 .. 6 + (N − 2) | Signal to teammate j (one action per non-self agent) |

For N=3 the per-agent action space is size 8. The wrapper:
1. When agent picks signal action k≥6, computes recipient index from k, records `(sender, receiver)` in `comm_events`, and issues NoOp to the underlying env for that agent (opportunity cost).
2. Augments each agent's observation with `(N − 1)` binary flags "was I signalled by teammate j on the previous step."
3. Returns `info['comm_events']` (list of pairs) and `info['positions']` (list of `(x, y, 0.0)` triples — z=0 because LBF is 2D, but the Hebbian spatial gate is 3D-shaped) on every step.

### Design rationale

- **Communication must cost something.** A free signal saturates `comm_events` and destroys signal-to-noise in the comm-bond term. Mutual exclusion with movement gives signalling real opportunity cost.
- **Communication must be targeted.** Asymmetric W[i,j] degenerates to symmetric if comm is broadcast. Forcing recipient choice produces per-pair events that map onto W's asymmetric structure.
- **Communication must be available to every variant.** Otherwise "Hebbian helps" is confounded with "having a comm channel helps." Baselines have access to the same action space and simply don't learn to use it — no training signal rewards signalling for its own sake. The mechanism by which Hebbian creates the learning signal for comm (Hebbian update → bond strengthens → reward diffusion / shared sampling weights j higher → more on-policy signal arriving) is itself part of the thesis claim.

### Limitation to acknowledge in the thesis

The comm channel is designed to interact cleanly with the Hebbian mechanism. A stronger robustness test repeats the headline on PressurePlate (implicit comm via plate-pressing creates cross-distance coordination without an explicit signal action). Noted in §10 as future work, not in scope for this implementation.

---

## 2. The three Hebbian integration points

Each is **independently toggleable** via CLI / YAML, so each can be ablated in isolation.

| Tag | Name | Location | Mechanism |
|---|---|---|---|
| (a) | Reward diffusion | Custom runner subclass | After each env step, call `hebbian_graph.diffuse_rewards()` on the per-agent reward tuple before it enters the batch |
| (b) | Hebbian-weighted shared experience | Custom learner subclass | When training agent i, sample fraction `lambda_share` of off-policy transitions from teammate j with probability proportional to `W̄[i, j]` |
| (c) | Hebbian policy mixture (**stretch**) | Custom controller | At action selection, agent i's effective distribution is a W-weighted mixture of its own and teammates' policies |

Invasiveness: (a) is ~30 lines of wrapper code; (b) is the headline contribution and needs care around importance sampling; (c) is highest-risk and deferred to Phase 5.

---

## 3. The ablation grid

**Headline environment: `lbforaging:Foraging-10x10-3p-3f-v3`**, comm-wrapped, `time_limit=50`.

| Variant | Param sharing | (a) reward diffusion | (b) sharing | (c) mixture | δ_comm | Purpose |
|---|---|---|---|---|---|---|
| `ippo` | none | off | off | off | n/a | Baseline floor |
| `mappo` | full | off | off | off | n/a | Standard cooperative baseline |
| `seac` | none | off | uniform | off | n/a | Published SEAC baseline; what `hebb_s` must beat |
| `hebb_r` | none | on | off | off | 0.5 | Tests (a) in isolation |
| `hebb_s` | none | off | Hebbian-weighted | off | 0.5 | **Headline:** Hebbian-weighted vs. uniform sharing |
| `hebb_rs` | none | on | Hebbian-weighted | off | 0.5 | (a) + (b) combined |
| `hebb_rsp` | none | on | Hebbian-weighted | on | 0.5 | Full system, stretch |
| `hebb_s_nocomm` | none | off | Hebbian-weighted | off | **0.0** | Ablates comm term; spatial-only |
| `hebb_s_commonly` | none | off | Hebbian-weighted | off | 0.5 (spatial gate disabled) | Ablates spatial term; comm-only |

All variants share the **same action space** (signal actions present, regardless of whether they're useful to the variant). 5 seeds per variant, 5M env steps per run. Single laptop, ~4-10h depending on parallelism.

**Headline statistical claims** (paired-seed Wilcoxon signed-rank, 5 seeds):
1. `hebb_s` > `seac` on final episode return, p < 0.01.
2. `hebb_s` > `hebb_s_nocomm` on final episode return.

Null results on either are publishable when accompanied by the clean ablation grid; what's not acceptable is an inconclusive result driven by infrastructure bugs.

---

## 4. Repository layout

```
hebbian-marl/                          # NEW REPO, sibling to WiredTogether
├── README.md
├── HEBBIAN_MARL_PLAN.md               # copy of this plan
├── pyproject.toml                     # deps; pin EPyMARL commit
├── epymarl/                           # vendored fork of uoe-agents/epymarl
│   └── src/
│       ├── main.py                    # not modified
│       ├── config/
│       │   ├── algs/                  # add: hebb_*.yaml, seac.yaml configs
│       │   ├── envs/                  # add: lbf_comm.yaml
│       │   └── default.yaml           # add `hebbian:` and `comm:` blocks
│       ├── envs/
│       │   └── lbf_comm_wrapper.py    # NEW: comm-augmented LBF wrapper
│       ├── runners/
│       │   ├── episode_runner.py      # not modified
│       │   ├── parallel_runner.py     # not modified
│       │   └── hebbian_runner.py      # NEW: subclass with Hebbian step hook
│       ├── learners/
│       │   ├── ppo_learner.py         # reference, not modified
│       │   └── hebbian_seac_learner.py # NEW: PPO with W-weighted sharing
│       ├── controllers/
│       │   └── hebbian_mixture_mac.py # NEW (Phase 5, stretch)
│       └── hebbian_module/            # PORTED from parent's src/hebbian/
│           ├── __init__.py
│           ├── config.py
│           └── graph.py
├── configs/                           # outer-level ablation configs
│   └── ablations/
│       ├── ippo.yaml
│       ├── mappo.yaml
│       ├── seac.yaml
│       ├── hebb_r.yaml
│       ├── hebb_s.yaml
│       ├── hebb_rs.yaml
│       ├── hebb_rsp.yaml
│       ├── hebb_s_nocomm.yaml
│       └── hebb_s_commonly.yaml
├── scripts/
│   ├── run_ablation_grid.sh
│   ├── plot_results.py
│   └── verify_baseline.sh             # reproduces a Papoudakis 2021 LBF number
├── tests/
│   ├── test_hebbian_port.py
│   ├── test_comm_wrapper.py
│   ├── test_runner_hook.py
│   ├── test_learner_sharing.py
│   ├── test_ablation_flags_passthrough.py
│   └── test_baseline_smoke.py
├── runs/                              # gitignored
└── .github/workflows/test.yml         # pytest on push
```

Vendor EPyMARL by copying source into `epymarl/` and committing; **do not submodule** — we add files alongside the existing source. Pin upstream commit hash in `README.md`.

---

## 5. Implementation phases with verification gates

**Hard rule:** do not move to the next phase until the current phase's verification gate is green. The parent codebase accumulated several bugs that took multi-day runs to surface; small steps with explicit checks prevent that here.

### Phase 0 — Environment + EPyMARL bootstrap (~1h)

1. Create the new sibling repo `../hebbian-marl/` with the layout in §4. Initialize git.
2. Fork `uoe-agents/epymarl` at current `main`. Record the commit hash in `README.md`.
3. Vendor the source under `epymarl/`.
4. Install in a fresh venv:
   ```bash
   pip install -r epymarl/requirements.txt
   pip install lbforaging
   ```
5. Pin verified versions in `pyproject.toml`.

**Gate:** EPyMARL quickstart works end-to-end on plain LBF (no wrapper yet):
```bash
python epymarl/src/main.py --config=mappo --env-config=gymma \
    with env_args.time_limit=50 \
         env_args.key="lbforaging:Foraging-10x10-3p-3f-v3" \
         t_max=50000
```
Run completes without error; sacred output written; episode return non-zero after 50k steps. If this fails, stop — every later phase depends on it.

### Phase 1 — Port the Hebbian module (~1h)

1. Copy [src/hebbian/config.py](src/hebbian/config.py) and [src/hebbian/graph.py](src/hebbian/graph.py) from `WiredTogether` to `epymarl/src/hebbian_module/`. Add `__init__.py` re-exporting `HebbianConfig` and `HebbianSocialGraph`.
2. **Strip Craftium-specific docstring fragments** from `HebbianConfig` (references to "Ch1", "Five Chambers", "VoxeLibre", "survival phase"). The mechanism stays.
3. **Do not pass `agent_roles`** to the constructor for LBF (LBF has no role concept). Leave the constructor kwarg available for future re-introduction.
4. **Adjust `interaction_radius` default for LBF grid scale** in the new YAML (not in Python). LBF positions are integer grid cells; suggest `interaction_radius=2.0` (Chebyshev distance ≤ 2). Document in the config docstring.
5. Write `tests/test_hebbian_port.py`:
   - Construct `HebbianSocialGraph(HebbianConfig(enabled=True, num_agents=3))`.
   - Step 100 iterations with random positions, rewards, and `comm_events`.
   - Assert `W` stays in `[0, 1]`, diagonal is zero, mean off-diagonal weight > 0.05 after 50 cooperative-like updates.
   - Assert `enabled=False` returns sensible no-op values.
   - **Assert `diffuse_rewards` is identity when constructed with `HebbianConfig(reward_diffusion_gamma=0.0)`** (the actual signature takes no `gamma` kwarg).
   - Assert `to_dict` → `from_dict` roundtrip preserves `W` exactly.
   - **Assert non-empty `comm_events` produces a different W than `comm_events=None` after 50 steps with identical RNG** — confirms the comm path is wired.

**Gate:** `pytest tests/test_hebbian_port.py -v` passes.

**Hard constraint:** do not modify the Hebbian update equations in `graph.py`. The math in `_compute_coactivity`, `_compute_modulator`, `update`, `diffuse_rewards`, `get_normalized_weights`, `_update_failure_window` is the thesis contribution. Adapt I/O, naming, and comments only.

### Phase 2 — Communication-augmented LBF wrapper (~2-3h)

1. Create `epymarl/src/envs/lbf_comm_wrapper.py`. The wrapper:
   - Wraps an underlying `lbforaging:Foraging-*` gymma env.
   - Expands per-agent action space from 6 to `6 + (N − 1)`.
   - On `step(actions)`:
     - For each agent with action ≥ 6: compute recipient from the action index (action 6 → first non-self teammate), record `(sender, receiver)` in `comm_events`, issue NoOp to underlying env for that agent.
     - For other agents: pass through unchanged.
   - Augments each agent's observation with `(N − 1)` binary "signalled-by-j on previous step" flags. Compute in `step`, persist to be read on the next `step` / `reset`.
   - Returns `info['comm_events']: list[(sender, receiver)]` every step.
   - Returns `info['positions']: list[(x, y, 0.0)]` per agent every step, read from `self.env.unwrapped.players[i].position`. (z=0 because LBF is 2D; the Hebbian spatial gate is shaped for 3D positions.)
2. Register the wrapper as a gymma env. Create `epymarl/src/config/envs/lbf_comm.yaml` pointing at it.
3. Write `tests/test_comm_wrapper.py`:
   - Actions 0–5 forwarded unchanged; agent moves as normal; `comm_events` empty.
   - Agent 0 picks action 6 (N=3): `comm_events == [(0, 1)]`; agent 0's position unchanged; agent 1's next-step observation has the signalled-by-0 flag set.
   - Agent 0 picks action 7: `comm_events == [(0, 2)]`; agent 2's flag set on next step.
   - Reset clears comm flags.
   - Underlying env reward dynamics identical to base LBF when no signal action is ever picked (regression guard).
   - `info['positions']` matches `env.unwrapped.players[i].position` for all agents.

**Gate:** test passes AND a 1000-step random-policy run on the comm-augmented env produces non-empty `comm_events` lists, signal flags appear in observations, and an all-zero-comm random run yields returns statistically indistinguishable from base LBF.

### Phase 3 — Hebbian runner subclass — integration (a) (~2-3h)

1. Read `epymarl/src/runners/episode_runner.py` end-to-end. Understand the env step loop, where rewards enter the batch, how `info` flows through.
2. Create `epymarl/src/runners/hebbian_runner.py`:
   - Subclass `EpisodeRunner`.
   - In `__init__`, construct `HebbianSocialGraph` from a new `args.hebbian` config section.
   - Override the inner step loop to, after each env step:
     - `positions = info['positions']`.
     - `comm_events = info['comm_events']`.
     - `self.hebbian.update(positions=positions, step_rewards=raw_rewards, advantages=None, comm_events=comm_events)`.
     - If `args.hebbian.reward_diffusion`: `rewards = self.hebbian.diffuse_rewards(raw_rewards)` before appending to the batch.
3. Register in `epymarl/src/runners/__init__.py`.
4. Add `hebbian:` block to `epymarl/src/config/default.yaml` with all `HebbianConfig` fields plus the three toggle flags (`reward_diffusion`, `weighted_sharing`, `policy_mixture`). All default `False`.
5. Add `runner: hebbian` to select the new runner.
6. Write `tests/test_runner_hook.py`:
   - Env + runner with `hebbian.enabled=True, reward_diffusion=False`.
   - Run 100 steps with a policy that occasionally emits signal actions.
   - Assert `runner.hebbian._step_count == 100`, W has changed from initial, and W differs from a parallel run where no signal actions were emitted (comm path active).
   - With `reward_diffusion=True`: rewards entering the batch differ from raw env rewards when W has non-zero off-diagonals.
   - With `hebbian.enabled=False`: runner behaves bitwise-identically to `EpisodeRunner` for the same seed.

**Gate:** test passes AND a 100k-step run with `runner=hebbian`, `hebbian.enabled=true`, `hebbian.reward_diffusion=true`, env = comm-augmented LBF, completes and logs non-zero return.

### Phase 4 — Hebbian-weighted experience sharing — integration (b) (~4-5h)

This is the scientifically interesting integration and needs the most care.

1. Read `epymarl/src/learners/ppo_learner.py` to understand EPyMARL's PPO update.
2. Create `epymarl/src/learners/hebbian_seac_learner.py` subclassing `PPOLearner`.
3. In `train()`, for each agent i:
   - Compute standard on-policy loss on agent i's own batch (unchanged).
   - **Additionally:** sample fraction `lambda_share` (default 0.5) of off-policy transitions from other agents' batches.
     - `args.hebbian.weighted_sharing=True`: sample teammate j with probability proportional to `W̄[i, j]` (call `hebbian.get_normalized_weights(i)`).
     - `args.uniform_sharing=True` (`seac` variant): sample uniformly across teammates.
     - Both off: standard IPPO.
   - For shared transitions, compute IS-corrected PPO loss: `ratio = π_i(a|s) / π_j(a|s)`, then standard clipped objective.
   - **Document the IS correction explicitly** in the learner's docstring. The first commit may land without IS for plumbing reasons, but **no reported result may come from a non-IS-corrected run.** Flag to user before merging the final implementation.
4. Register in `epymarl/src/learners/__init__.py`.
5. Add YAML configs `hebb_s.yaml`, `seac.yaml`, `hebb_rs.yaml`, `hebb_s_nocomm.yaml` (sets `hebbian.communication_coactivity_bonus: 0.0`), `hebb_s_commonly.yaml` (sets `hebbian.interaction_radius: 0.0` to disable the spatial gate).
6. Write `tests/test_learner_sharing.py`:
   - 3-agent setup with stub buffers containing distinguishable transitions per agent.
   - With `weighted_sharing=True` and manually-set `W = [[0, 0.9, 0.1], [0.9, 0, 0.1], [0.1, 0.1, 0]]`: agent 0's training batch contains many more transitions from agent 1 than agent 2.
   - With `uniform_sharing=True`: agent 0's batch has roughly equal counts from agents 1 and 2 (within 2σ over 100 samples).
   - Both off: agent 0's batch contains only its own transitions.

**Gate:** test passes AND `hebb_s_seed0` training run on comm-augmented LBF for 1M steps completes, logs non-trivial bond evolution, reaches non-zero episode return.

### Phase 5 — Policy mixture — integration (c), stretch (~3-4h)

Only attempt if Phases 3 and 4 are stable. Highest-risk integration.

1. Read `epymarl/src/controllers/basic_controller.py`.
2. Create `epymarl/src/controllers/hebbian_mixture_mac.py` subclassing the basic MAC.
3. Override `select_actions` so agent i's distribution is `(1 − α) π_i + α Σ_{j≠i} W̄[i,j] π_j`.
4. **Critical:** the log-prob stored in the batch must be the log-prob of the action under the **sampled mixture distribution**, not under π_i. This keeps the PPO ratio well-defined. Document in the controller with a short comment explaining why.
5. Register the controller. Add `hebb_rsp.yaml`.
6. Write `tests/test_mixture_controller.py`:
   - α = 0 → matches basic controller exactly.
   - α = 1 with one-hot W (agent 0 fully borrows from agent 1) → agent 0's distribution matches agent 1's exactly.

**Gate:** `hebb_rsp_seed0` completes 1M steps without diverging (KL bounded). If KL diverges, **fall back to (a)+(b) as the headline result** and document (c) as future work. Do not let this phase block thesis progress.

### Phase 6 — Metrics, logging, ablation tooling (~3h)

1. Hebbian-specific logger callback, every K env steps (default 5000):
   - Writes current W to `runs/{run_id}/bonds.jsonl` (one JSON object per snapshot, including step index, full W matrix, and per-agent normalized rows).
   - Writes derived metrics: mean off-diagonal weight, asymmetry `‖W − W.T‖_F`, top-3 pairs, per-agent out-strength.
2. **Adapt** the asymmetric-bond plot from [src/mindforge/agent_modules/craftium_metric.py:743](src/mindforge/agent_modules/craftium_metric.py#L743). The parent function `_plot_hebbian_asymmetry` takes no args and reads from `self._graph_snapshots`. Adaptation: refactor it into a free function `plot_hebbian_asymmetry(snapshots: list[dict], out_path: str)` that reads from `bonds.jsonl`. Keep the same visual output (asymmetry curve, heatmap snapshots).
3. Cooperation- and comm-specific per-episode metrics:
   - **Time-to-first-cooperative-load:** step index when ≥2 adjacent agents jointly harvest a food whose level exceeds any single agent's.
   - **Joint-load count per episode.**
   - **Per-pair joint-load matrix N×N.**
   - **Signal-action count per agent per episode.**
   - **Per-pair signal matrix N×N** (sender × receiver counts).
   - **Bond-comm correlation:** at episode end, Pearson correlation between agent i's "signals to j" count and `W[i, j]`. **Headline comm-side plot.**
4. Write `scripts/run_ablation_grid.sh`:
   ```bash
   #!/usr/bin/env bash
   set -e
   VARIANTS=(ippo mappo seac hebb_r hebb_s hebb_rs hebb_rsp hebb_s_nocomm hebb_s_commonly)
   SEEDS=(0 1 2 3 4)
   T_MAX=5000000
   PARALLEL_CAP=4

   for v in "${VARIANTS[@]}"; do
       for s in "${SEEDS[@]}"; do
           python epymarl/src/main.py \
               --config="$v" --env-config=lbf_comm \
               with env_args.key="lbforaging-comm:Foraging-10x10-3p-3f-v3" \
                    env_args.time_limit=50 \
                    seed="$s" t_max="$T_MAX" \
                    name="${v}_seed${s}" &
           while [ "$(jobs -r | wc -l)" -ge "$PARALLEL_CAP" ]; do sleep 1; done
       done
   done
   wait
   ```
5. Write `scripts/plot_results.py`:
   - Loads `runs/*/metrics.jsonl` and `runs/*/bonds.jsonl`.
   - Groups by variant. Mean ± 95% CI of episode return across seeds.
   - Produces:
     - Episode return curves (mean ± 95% CI, one line per variant).
     - Time-to-first-cooperative-load violin plot.
     - Bond evolution: N×N heatmap over time for `hebb_rsp` (or `hebb_s` if (c) skipped).
     - Bond asymmetry over time, one line per Hebbian variant.
     - Pair joint-load matrix at 100k vs. 4M steps.
     - **Signal-count per variant over training** (baselines never learn to signal; Hebbian variants do).
     - **Bond-comm correlation scatter for `hebb_s`**: x = `W[i,j]` at episode end, y = signals from i to j.
   - Writes PNGs to `runs/plots/`.
   - Paired-seed Wilcoxon signed-rank tests:
     - `hebb_s` vs. `seac` final returns.
     - `hebb_s` vs. `hebb_s_nocomm` final returns.
   - Prints p-values.

**Gate:** 3-seed × 3-variant mini-grid (`ippo`, `mappo`, `hebb_s`) completes; `plot_results.py` produces non-empty PNGs.

### Phase 7 — Documentation and reproducibility (~1h)

1. `README.md`:
   - One-paragraph problem statement.
   - Quickstart: 5 shell commands from `git clone` to a running training job.
   - Ablation grid table (copy from §3).
   - How to add a new variant.
   - Citation block.
2. Pin exact EPyMARL commit, lbforaging version, Python version, PyTorch version in `README.md` and `pyproject.toml`.
3. `scripts/verify_baseline.sh` reproduces one Papoudakis 2021 published number on **plain LBF** (no comm wrapper) — sanity check. Run in CI on every PR.

**Gate:** fresh clone on a different machine completes the quickstart. `verify_baseline.sh` matches published within 10%.

---

## 6. Critical files to be created or modified

| Path | Action | Phase |
|---|---|---|
| `../hebbian-marl/epymarl/src/hebbian_module/{__init__.py, config.py, graph.py}` | NEW (ported from [src/hebbian/](src/hebbian/)) | 1 |
| `../hebbian-marl/epymarl/src/envs/lbf_comm_wrapper.py` | NEW | 2 |
| `../hebbian-marl/epymarl/src/config/envs/lbf_comm.yaml` | NEW | 2 |
| `../hebbian-marl/epymarl/src/runners/hebbian_runner.py` | NEW | 3 |
| `../hebbian-marl/epymarl/src/runners/__init__.py` | MODIFY (register runner) | 3 |
| `../hebbian-marl/epymarl/src/config/default.yaml` | MODIFY (add `hebbian:` / `comm:` blocks) | 3 |
| `../hebbian-marl/epymarl/src/learners/hebbian_seac_learner.py` | NEW | 4 |
| `../hebbian-marl/epymarl/src/learners/__init__.py` | MODIFY (register learner) | 4 |
| `../hebbian-marl/epymarl/src/config/algs/{hebb_*.yaml, seac.yaml}` | NEW (one per variant) | 4 |
| `../hebbian-marl/epymarl/src/controllers/hebbian_mixture_mac.py` | NEW (stretch) | 5 |
| `../hebbian-marl/scripts/{run_ablation_grid.sh, plot_results.py, verify_baseline.sh}` | NEW | 6, 7 |
| `../hebbian-marl/tests/test_*.py` | NEW (one per phase) | each phase |

**Reused without modification from `WiredTogether`:** [src/hebbian/config.py](src/hebbian/config.py) (copied, docstrings trimmed), [src/hebbian/graph.py](src/hebbian/graph.py) (copied verbatim — no equation changes).

---

## 7. Hard constraints

Non-negotiable. Each exists because of a specific failure mode in the parent codebase.

1. **Do not modify the Hebbian update equations.** Adapt I/O, naming, and comments only. (Exception: the additive *failure-grace LTP bonus* introduced in the `hebb_s_grace` / `hebb_rs_grace` variants is an opt-in extension gated by `HebbianConfig.failure_grace_enabled` — legacy variants' math is bit-identical when the flag is False, satisfying the spirit of constraint #3.)
2. **No LLM dependencies.** No `transformers`, `peft`, `sentence-transformers`, `chromadb`, `torch.cuda` requirement. This sub-project isolates Hebbian from the LLM variable.
3. **Every Hebbian integration must be a no-op when its flag is False.** Verified by `test_ablation_flags_passthrough.py`. With all three flags off + `runner: hebbian`, behaviour is bitwise-identical to vanilla EPyMARL (up to seeded RNG).
4. **The comm channel is available to every variant.** All ablations use the comm-augmented LBF with the same action space.
5. **Do not vectorize environments inside one process.** Parallelism comes from running multiple seeds as separate processes (`PARALLEL_CAP` in the ablation script).
6. **Do not silently change Hebbian defaults.** If a default must change for LBF (e.g. `interaction_radius=2.0`), surface in YAML, not Python. Document the rationale.
7. **Log to disk in JSONL** in addition to sacred/tensorboard. JSONL is for post-hoc analysis and plot regeneration.
8. **Pin every dependency version.** EPyMARL commit, lbforaging, gymnasium, torch, numpy, sacred.
9. **`pytest` runs green before each commit.**
10. **When in doubt about EPyMARL or LBF API, read the source.** Both are small and readable.
11. **Comments explain why, not what.** Look at [src/hebbian/graph.py](src/hebbian/graph.py) for the right tone.
12. **No reported ablation number may come from a non-IS-corrected shared-learner run.** First commit may land without IS; final result may not.

---

## 8. Verification — end-to-end

After implementation, the user runs:
```bash
git clone <repo> hebbian-marl
cd hebbian-marl
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest                                    # all tests green
bash scripts/verify_baseline.sh           # reproduces Papoudakis 2021 LBF number
bash scripts/run_ablation_grid.sh         # 4-10h on a laptop
python scripts/plot_results.py runs/      # publication-quality plots
```

Each phase has its own verification gate (see §5). The end-to-end run verifies the system is reproducible from scratch and produces:
1. A statistically-defensible comparison of `hebb_s` vs. `seac` (paired-seed Wilcoxon, p < 0.01 for the positive claim).
2. A statistically-defensible comparison of `hebb_s` vs. `hebb_s_nocomm` (validates the comm term of the Hebbian co-activity signal).
3. Bond-evolution and bond-comm-correlation plots showing the **mechanism** is doing what the thesis says, not just that the numbers came out.

A null result on (1) or (2) is publishable when accompanied by the clean ablation grid. What's not acceptable is an inconclusive result driven by infrastructure bugs — the per-phase gates and the `verify_baseline.sh` CI check are specifically designed to prevent that.

---

## 9. Deferred / out-of-scope

These were considered and deliberately set aside:

- **Integration point (c)** — implemented as stretch in Phase 5 only after (a)+(b) are stable. `hebb_rsp` reported as future work if KL diverges.
- **PressurePlate as a second env** — strong robustness follow-up. Out of scope for the first pass; noted in the thesis as future work.
- **PyMARLzoo+ migration** — EPyMARL+ AAMAS 2025 extension with more algorithms. Drop-in upgrade if EPyMARL becomes limiting; not needed for the headline experiments.
- **GPU acceleration** — LBF runs at thousands of steps/sec on one CPU thread. GPUs add complexity without throughput gain at this scale.
- **Repairing `scripts/mappo_train.py` reference in the parent's auto-memory** — minor drift noted during verification; the parent's training is in `src/rl_layer/`, not `scripts/mappo_*.py`. Outside this sub-project's scope.

---

## 10. Glossary

- **EPyMARL** — Extended PyMARL, `uoe-agents/epymarl`. Academic standard cooperative MARL training framework.
- **LBF** — Level-Based Foraging. Small grid-world MARL env requiring heterogeneous-level agents to cooperate on foraging.
- **MAPPO** — Multi-Agent PPO with parameter sharing and centralised critic.
- **IPPO** — Independent PPO; per-agent independent learners.
- **SEAC** — Shared Experience Actor-Critic; independent learners sampling off-policy transitions from teammate buffers with uniform weighting + IS correction.
- **Hebbian social graph** — Learned matrix `W ∈ [0,1]^{N×N}` updated by a reward-modulated Hebbian rule, encoding which teammates an agent depends on. Implemented at [src/hebbian/graph.py](src/hebbian/graph.py).
- **W̄ (W-bar)** — Row-normalised Hebbian matrix used for weighted sampling. `W̄[i,j] = W[i,j] / Σ_k W[i,k]`.
- **`comm_events`** — `List[Tuple[int, int]]`. Sender, receiver pairs emitted by the env wrapper each step. Consumed by `HebbianSocialGraph.update`.
- **Signal action** — Discrete action in the comm-augmented LBF targeting one teammate. Consumes the agent's movement that step (opportunity cost).
- **Integration points (a), (b), (c)** — Three independently-toggleable mechanisms by which the Hebbian graph influences RL training. See §2.
- **δ_comm** — Comm-co-activity coefficient (`communication_coactivity_bonus` in `HebbianConfig`). `δ_comm = 0` disables cross-distance comm term, reducing co-activity to spatial-only.

---
*End of plan.*
