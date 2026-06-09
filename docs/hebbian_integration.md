# Hebbian Social Plasticity: mechanism & integrations

The Hebbian Social Plasticity module is a learned `N×N` social graph over
agents, updated by a reward-modulated Hebbian rule. It lives in
[src/hebbian/](../src/hebbian/) (parent project) and is vendored verbatim
under [hebbian-marl/epymarl/src/hebbian_module/](../hebbian-marl/epymarl/src/hebbian_module/)
for the LBF testbed. The math is identical in both copies — divergence
is enforced by mirror commits.

This document covers:

1. [What the graph is](#1-what-the-graph-is)
2. [The update rule](#2-the-update-rule)
3. [The outputs other code consumes](#3-the-outputs-other-code-consumes)
4. [Integration A — hebbian-marl (LBF + EPyMARL + IPPO/SEAC)](#4-integration-a--hebbian-marl-lbf--epymarl--ippoSeac)
5. [Integration B — Craftium MAPPO/IPPO (multi_agent_craftium.py)](#5-integration-b--craftium-mappoippo-multi_agent_craftiumpy)
6. [Integration C — GRPO (rlvr bridge)](#6-integration-c--grpo-rlvr-bridge)
7. [Side-by-side: what each pipeline modulates](#7-side-by-side-what-each-pipeline-modulates)

---

## 1. What the graph is

A directed weighted graph `W ∈ [0, 1]^{N×N}` (diagonal forced to zero)
where `W[i, j]` represents agent *i*'s learned attention or trust toward
agent *j*. The asymmetric form means agent *i* can value agent *j* even
if the reverse is weak — useful when roles or skills differ.

Two things to keep in mind:

- **The graph is numpy-only.** Zero torch dependency. Same code runs in
  any RL pipeline that can call a Python function per env step.
- **The graph has no parameters to train.** Updates come from a fixed
  rule with hyperparameters in [HebbianConfig](../src/hebbian/config.py).
  Nothing about `W` is back-propagated.

Initialization: `W` starts at `init_weight = 0.1` on every off-diagonal
entry (warm start so the early-training reward-diffusion signal has
something to flow through).

---

## 2. The update rule

Each `update(positions, step_rewards, advantages, comm_events)` call
performs one step of plasticity. The update has four additive components:

```
ΔW_ij = m_ij · c_ij · (1 - W_ij)       advantage-gated LTP/LTD
      + base_ltp · c_ij · (1 - W_ij)   unconditional co-activity LTP
      - decay · W_ij                    passive decay
      - ltd_sustained_lr · F_ij · W_ij sustained LTD from repeated co-failure
      + (grace_bonus_ij)                additive grace LTP on team failure
W_ij(t+1) = clip(W_ij(t) + ΔW_ij, 0, 1)
```

### 2.1 Co-activity `c_ij` ([graph.py:_compute_coactivity](../src/hebbian/graph.py))

```
g_i  = clip(α · |r_i| / max_r  +  (1 - α) · comm_i, 0, 1)    soft engagement
c_ij_spatial = I[dist(p_i, p_j) ≤ d] · g_i · g_j
c_ij_comm    = δ_comm · comm_pair_ij · (1 - I[dist ≤ d])     cross-distance bonus
c_ij         = clip(c_ij_spatial + c_ij_comm, 0, 1)
```

- `g_i` is the agent's "engagement score" — high when rewarded or signaling.
- Two pairs only count as co-active if they're spatially close **AND** both
  engaged, OR if they communicated across distance.
- The comm bonus only fires when agents are NOT already in spatial range —
  prevents double-counting.

### 2.2 Modulator `m_ij` ([graph.py:_compute_modulator](../src/hebbian/graph.py))

Asymmetric per-agent: `m_ij` depends only on agent *i*'s own signal
(advantage if available, else normalized reward). The bond *i→j*
strengthens when *i* succeeds — agent *i* learns "teammates I was with
when I did well are valuable."

```
m_i_ltp = ltp_lr · tanh(β · max(A_i, 0))
m_i_ltd = ltd_lr · tanh(β · max(-A_i - ltd_threshold, 0))
m_ij    = m_i_ltp - m_i_ltd              (same for all j; per-pair differentiation comes from c_ij)
```

The `ltd_threshold` exists because zero-reward steps with a positive value
estimate produce small-negative advantage from exploration noise; without
the threshold every quiet step would slowly depress bonds.

### 2.3 Sustained LTD from co-failure ([graph.py:_compute_sustained_ltd](../src/hebbian/graph.py))

A rolling window of `failure_memory_window` snapshots tracks which pairs
were co-active during *team-level* failures (`At_team < -ltd_threshold`).
`F_ij = failure_coactivation[i,j] / window_size` is the fraction of recent
failure-steps the pair shared.

```
ΔW_sustained_ij = -ltd_sustained_lr · F_ij · W_ij
```

Pairs that *repeatedly* co-fail get depressed proportionally to their
current bond strength.

### 2.4 Failure-grace LTP bonus *(canonical default since this commit)*

Counter-pressure to the sustained-LTD term in the early phase: when the
team is failing AND the pair's `F_ij` is still under
`failure_grace_threshold = 0.3`, an additive LTP bonus is added that
fades linearly to zero as `F_ij` approaches the threshold.

```
if failure_grace_enabled and At_team < -ltd_threshold:
    grace_factor = clip(1 - F_ij / failure_grace_threshold, 0, 1)
    grace_bonus  = failure_ltp_lr · grace_factor · c_ij · (1 - W_ij)
```

Semantic: *"we tried something together (even though it failed) → strengthen
the partnership; we keep failing together → eventually give up on it."*
Reverses the early collapse observed without the bonus.

Set `failure_grace_enabled = False` in HebbianConfig to reproduce the
pre-grace math for ablation.

### 2.5 Reward diffusion ([graph.py:diffuse_rewards](../src/hebbian/graph.py))

A separate output computed on demand (not part of the W update):

```
diffused_r_i = (1 - γ) · r_i  +  γ · Σ_{j≠i} W̄[i, j] · r_j
```

Each agent's reward is blended with a `W`-weighted average of teammate
rewards. Used downstream to influence the RL signal.

### 2.6 Normalised weights `W̄` ([graph.py:get_normalized_weights](../src/hebbian/graph.py))

Row-normalised: `W̄[i, j] = W[i, j] / (Σ_{k≠i} W[i, k] + ε)`. Used by
sharing-based integrations to distribute attention across teammates so it
sums to 1.

---

## 3. The outputs other code consumes

| Output | Method | Used by |
|---|---|---|
| Raw `W` | `get_all_weights()` | logging, plots, snapshots |
| Row-normalised `W̄` | `get_normalized_weights(i)` | weighted experience sharing (SEAC / GRPO group composition) |
| Diffused rewards | `diffuse_rewards(per_agent_rewards)` | MAPPO/IPPO reward modulation; GRPO trajectory-reward modulation |
| Metrics bundle | `get_graph_metrics()` | sacred / TensorBoard / bonds.jsonl |
| Co-activity snapshot | `_last_coactivity` | reward-attribution decomposition (Craftium MAPPO) |

---

## 4. Integration A — hebbian-marl (LBF + EPyMARL + IPPO/SEAC)

The smallest, most isolated integration. Lives entirely in the vendored
[hebbian-marl/](../hebbian-marl/) sub-project. Base RL alg is per-agent
**IPPO** (independent PPO) via EPyMARL.

### Configuration

Set in the alg YAML, e.g. [hebb_s.yaml](../hebbian-marl/epymarl/src/config/algs/hebb_s.yaml):

```yaml
runner: "hebbian"
learner: "hebbian_seac_learner"
weighted_sharing: True
lambda_share: 0.5

hebbian:
  enabled: True
  reward_diffusion: False    # toggle integration (a)
  weighted_sharing: True     # toggle integration (b)
  interaction_radius: 2.0
  communication_coactivity_bonus: 0.5
  # (grace fields default to enabled now; can be set False here for ablation)
```

### Update site — per env step

[HebbianRunner.run()](../hebbian-marl/epymarl/src/runners/hebbian_runner.py)
overrides `EpisodeRunner.run()`. After every `env.step()`:

```python
self.hebbian.update(
    positions=info["positions"],       # from LBFCommWrapper
    step_rewards=raw_per_agent_rewards,
    advantages=None,                   # advantage not yet computed at runner-time
    comm_events=info["comm_events"],   # from LBFCommWrapper
)
if self.reward_diffusion:
    reward = self.hebbian.diffuse_rewards(raw_per_agent_rewards)
```

The `LBFCommWrapper` ([envs/lbf_comm_wrapper.py](../hebbian-marl/epymarl/src/envs/lbf_comm_wrapper.py))
adds the 6+(N-1)-action comm channel and emits `info['comm_events']` and
`info['positions']` that feed `update()`.

### Sharing site — per PPO update

[HebbianSEACLearner](../hebbian-marl/epymarl/src/learners/hebbian_seac_learner.py)
subclasses `PPOLearner`. Inside `train()` it adds an IS-corrected cross-agent
loss term:

```python
L_i = L_i_own + λ · Σ_{j≠i} w_ij · L_ij_cross
where  ratio = π_i_new(a_j | o_j) / π_j_old(a_j | o_j)
       w_ij  = W̄[i, j]  (Hebbian-weighted)  OR  1/(N-1)  (uniform SEAC)
```

The W̄ weights are pulled by `get_normalized_weights(i)` from the runtime
singleton in [hebbian_module/runtime.py](../hebbian-marl/epymarl/src/hebbian_module/runtime.py)
— the runner registers the graph; the learner reads it.

### What gets modulated

- (a) `reward_diffusion: True` → per-agent reward fed into the PPO buffer
- (b) `weighted_sharing: True` → cross-agent term in the PPO loss

### Where to read more

- Design rationale: [hebbian-marl/HEBBIAN_MARL_PLAN.md](../hebbian-marl/HEBBIAN_MARL_PLAN.md)
- Manifest of experimental variants: [hebbian-marl/scripts/experiments.yaml](../hebbian-marl/scripts/experiments.yaml)

---

## 5. Integration B — Craftium MAPPO/IPPO ([multi_agent_craftium.py](../src/mindforge/multi_agent_craftium.py))

The original integration. Lives in `src/mindforge/multi_agent_craftium.py`.
Base RL alg is **MAPPO or IPPO** (selectable via `--rl-critic-mode`)
running on the Craftium / Voxel environment with LLM-conditioned agents.

### Configuration — CLI flags

Every Hebbian hyperparameter is a CLI arg (`--hebbian-ltp`, `--hebbian-ltd`,
`--hebbian-decay`, `--hebbian-beta`, `--hebbian-rho`, `--hebbian-gamma`,
`--hebbian-radius`, `--hebbian-no-comm-bond`, `--hebbian-init-weight`).
Constructed at startup:

```python
hebbian_config = HebbianConfig(
    enabled=args.hebbian,
    num_agents=num_agents,
    interaction_radius=args.hebbian_radius,
    ltp_lr=args.hebbian_ltp,
    ltd_lr=args.hebbian_ltd,
    decay=args.hebbian_decay,
    modulation_beta=args.hebbian_beta,
    social_replay_rho=args.hebbian_rho,
    reward_diffusion_gamma=args.hebbian_gamma,
    communication_coactivity_bonus=0.0 if args.hebbian_no_comm_bond else 0.5,
    init_weight=args.hebbian_init_weight,
)
hebbian_graph = HebbianSocialGraph(hebbian_config, agent_roles=agent_roles)
```

Example invocation in [scripts/experiments/E5_hebbian.sh](../scripts/experiments/E5_hebbian.sh):

```bash
python multi_agent_craftium.py --num-agents 3 --episodes 5 \
    --rl --rl-model-path "$MODEL_2B" --rl-update-interval 64 --rl-lr 3e-4 \
    --hebbian \
    --hebbian-ltp 0.01 --hebbian-ltd 0.005 \
    --hebbian-decay 0.005 --hebbian-beta 1.0 \
    --hebbian-rho 0.3 --hebbian-gamma 0.2 \
    --seed "$SEED" --experiment-id "E5"
```

### Update site — per env step

In the main episode loop in `multi_agent_craftium.py`:

```python
hebbian_graph.update(
    positions=positions,
    step_rewards=step_rewards_raw,
    advantages=step_advantages if _any_advantage else None,
    comm_events=comm_events if communication else None,
)
diffused_rewards = hebbian_graph.diffuse_rewards(step_rewards_raw)
```

### What gets modulated

- **Per-agent step reward** fed into the RL buffer is `diffused_rewards[i]`
  (not the raw reward). The MAPPO/IPPO advantage estimate is computed from
  these diffused rewards. This is the main pathway.
- **Reward decomposition**: the delta `diffused_rewards[i] - step_rewards_raw[i]`
  is logged separately as the "Hebbian contribution" stream alongside task /
  communication / proximity rewards (interpretability).
- **Prompt-side propagation** (optional, `--reward-propagation`): in the
  next step's LLM prompt, the *teammates whose rewards diffused into agent
  i this step* are listed with their contribution magnitudes
  (`per_teammate_contributions()`). Lets the LLM verbalise the social
  influence at decision time.

### M4 vs M5 (IPPO ablation)

- [scripts/experiments/M4_ippo.sh](../scripts/experiments/M4_ippo.sh) —
  IPPO with `--rl-critic-mode independent`, no Hebbian
- [scripts/experiments/M5_ippo_hebbian.sh](../scripts/experiments/M5_ippo_hebbian.sh) —
  same IPPO setup PLUS the `--hebbian-*` flags

Isolates the Hebbian effect from the critic architecture choice.

---

## 6. Integration C — GRPO ([rlvr bridge](../src/rlvr/hebbian_grpo_bridge.py))

The newest integration. Base RL alg is **GRPO** (Group Relative Policy
Optimization) for LLM rollouts. Bridge architecture: a thin wrapper
mediates between GRPO's joint-trajectory sampling and the per-step
graph update API.

### Configuration — YAML

Set in [configs/rlvr/grpo_hebbian_*.yaml](../configs/rlvr/):

```yaml
hebbian:
  enabled: true
  num_agents: 3
  interaction_radius: 2.0
  ltp_lr: 0.01
  ltd_lr: 0.005
  reward_diffusion_gamma: 0.2
  # … plus the standard HebbianConfig fields
```

Constructed in [multi_agent_craftium_grpo.py](../src/mindforge/multi_agent_craftium_grpo.py):

```python
def build_hebbian(hebbian_cfg: dict):
    if not hebbian_cfg or not hebbian_cfg.get("enabled", False):
        return None
    from hebbian import HebbianConfig, HebbianSocialGraph
    hc = HebbianConfig(**{k: v for k, v in hebbian_cfg.items() if k != "enabled"})
    return HebbianSocialGraph(hc)
```

### Update site — per env step

Same cadence as the other two integrations, but routed through the
bridge in [rollout_sampler.py](../src/rlvr/rollout_sampler.py):

```python
self.hebbian_bridge.observe_step(
    positions=positions,
    step_rewards=step_rewards,
    comm_events=comm_evs,
)
```

The bridge ([hebbian_grpo_bridge.py](../src/rlvr/hebbian_grpo_bridge.py))
forwards to `graph.update()` and additionally maintains a snapshot history
for `hebbian_snapshots.jsonl` logging.

### What gets modulated

**Two independent toggles**, called *Stage 4a* and *Stage 4b*:

**Stage 4a — reward diffusion (`hebbian_reward_diffusion: true`)**
The reward verifier in [verifier.py](../src/rlvr/verifier.py) intercepts
the joint trajectory's reward vector BEFORE advantage computation:

```python
diffused = self.hebbian.diffuse_rewards(ordered)
# … then GRPO computes advantages from `diffused`
```

So GRPO's advantages and policy updates see the Hebbian-blended reward
across teammates within a joint trajectory.

**Stage 4b — group composition (`hebbian_group_composition: true`)**
GRPO computes its policy update over a *group* of trajectories. By default
each agent's group is its own buffer. With group composition enabled, the
group is augmented with teammate trajectories sampled proportionally to
`W̄[i, j]`:

```python
w_bar = hebbian_bridge.normalized_weights(agent_id)
batch = assemble_composed_multi_agent_batch(
    own_buffer=...,
    teammate_buffers=...,
    weights=w_bar,
    borrow_fraction=cfg.hebbian_borrow_fraction,   # default 0.25
    capacity=cfg.hebbian_buffer_capacity,
)
```

Bounded by `borrow_fraction` so own-data still dominates; the
clipped-surrogate loss provides the off-policy correction.

### Where to read more

- GRPO design: [docs/grpo_explained.md](grpo_explained.md)
- Launch scripts: `scripts/experiments/G3*_grpo_hebbian_*.sh`,
  `scripts/experiments/G4_grpo_hebbian_full.sh`

---

## 7. Side-by-side: what each pipeline modulates

| | hebbian-marl (LBF) | Craftium MAPPO/IPPO | GRPO |
|---|---|---|---|
| **Base RL alg** | IPPO + SEAC sharing | MAPPO or IPPO | GRPO |
| **Config source** | YAML (`hebbian.*` block) | CLI flags (`--hebbian-*`) | YAML (`hebbian:` block) |
| **`update()` called** | per env step (in runner) | per env step (main loop) | per env step (in bridge) |
| **`positions` source** | `info['positions']` from `LBFCommWrapper` | from Craftium env state | from Craftium env state (via sampler) |
| **`comm_events` source** | `info['comm_events']` from wrapper | LLM-emitted utterances, parsed | LLM-emitted utterances, parsed |
| **`advantages` passed in?** | No (runner doesn't have them) | Yes (computed in loop) | No (graph is decoupled from GRPO loss) |
| **Reward diffusion target** | per-agent reward into PPO buffer | per-agent reward into RL buffer | joint trajectory reward in verifier (Stage 4a) |
| **Weighted sharing target** | cross-agent term in PPO loss (W̄ weights) | none | group composition: teammate trajectories sampled with W̄ (Stage 4b) |
| **Snapshot output** | `bonds/<label>/seed_<n>.jsonl` (per HebbianRunner) | sacred metrics + plot-side history | `hebbian_snapshots.jsonl` (per bridge) |
| **Ablation knobs** | `reward_diffusion`, `weighted_sharing`, `δ_comm`, `interaction_radius`, `failure_grace_*` | same fields, exposed as CLI flags | `hebbian_reward_diffusion`, `hebbian_group_composition`, `hebbian_borrow_fraction` |

## 8. What's the same everywhere

- The graph object itself ([HebbianSocialGraph](../src/hebbian/graph.py)) —
  same code, same math, same hyperparameters.
- `update()` runs on every env step. Never per RL step.
- Reward diffusion is the most common modulation path; weighted-sharing
  variants exist for SEAC-style (hebbian-marl) and GRPO group-composition.
- Snapshots are JSONL files written at fixed step intervals, suitable
  for post-hoc time-series plots.

## 9. What this commit standardised

- `failure_grace_enabled` defaults to **True** in both
  [src/hebbian/config.py](../src/hebbian/config.py) and the vendored
  [hebbian-marl/epymarl/src/hebbian_module/config.py](../hebbian-marl/epymarl/src/hebbian_module/config.py).
- The grace term `+ failure_ltp_lr · grace_factor · c_ij · (1 - W_ij)`
  is added to the main update in [src/hebbian/graph.py](../src/hebbian/graph.py)
  and the vendored copy.
- All three integrations pick up the change automatically — no per-integration
  edits needed because they all construct `HebbianConfig` from defaults
  (or YAML/CLI fields that don't override the new ones).
- To run with the legacy pre-grace math, set `failure_grace_enabled: false`
  in the YAML (hebbian-marl, GRPO) or add a `--hebbian-no-grace` flag wrapper
  in Craftium MAPPO/IPPO (not yet exposed as a CLI arg; pass via Python if
  needed).
