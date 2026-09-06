# The Hebbian social graph

`src/hebbian/` — `HebbianSocialGraph` (the rule) and `HebbianConfig` (its constants). numpy only:
the graph carries no gradient and never imports torch, so it can be replayed offline from logs,
which is what `analysis/replay_hebbian_terms.py` does.

## State

`W ∈ [0,1]^{N×N}`, zero diagonal, initialised to `init_weight` (0.1) off-diagonal. `W[i,j]` is the
bond *from* i *to* j. The rule is directed — the modulator is egocentric and the observation and
imitation channels are one-sided — but co-firing from proximity and messages is mutual, so in
practice the two directions stay within a fraction of a percent of each other except when a
one-sided channel is the only social channel enabled.

## Co-firing

Two things have to be true for a pair to be eligible: both agents must be *engaged*, and something
must couple them this step.

**Engagement** `g_i ∈ [0,1]` mixes outcome salience with social activity:

```
g_i = clip( α·|r_bond_i| / r_max  +  (1−α)·1[i acted socially] , 0, 1)
```

`r_bond` is the bond-eligible reward stream, which is not the same as the total reward:

- it is gated to the cooperative chambers (Ch2–Ch5); Ch1 progress never wires anything;
- deaths are excluded by construction, so a death spike can never *raise* engagement;
- `r_max` is a running maximum over observed magnitudes, so the ratio is a bounded gate.

**Coupling** is the sum of a spatial term and the enabled social channels, clipped to `[0,1]`:

```
c_spat_ij = 1[‖p_i − p_j‖ ≤ d] · g_i · g_j
c_k_ij    = δ_k · 1[event_k_ij]          k ∈ social_act_channels
c_ij      = clip(c_spat_ij + Σ_k c_k_ij, 0, 1)      and 0 if below coop_eps
```

Proximity alone is an opportunity, not a social event: without engagement `c_spat` is zero. Which
channels count is `social_act_channels` (`--cofiring-channels`); the default `("comm",)` credits
messages only, which is the historical rule. Two flags change the geometry of the channels:
`social_bidirectional` credits both directions of an observation/imitation event, and
`comm_distance_free` drops the "messages only count across distance" gate.

## The update

`ΔW` has three terms — reward-modulated growth, failure-gated decay, and a homeostatic pull:

```
ΔW_ij = (η₀ + η₊·|r_bond_i|/R) · c_ij · (1 − W_ij)     growth
      −  η₋ · φ_ij · W_ij                              failure-gated decay
      −  λ  · W_ij                                     homeostatic decay
φ_ij  = 1[ coop_ij < ε  ∧  neg_i ]
```

`|r_bond|` is used rather than the signed reward: a salient shared event strengthens the
association whether it went well or badly. `η₀` is the association floor, so repeated co-firing
wires a pair even on zero-reward steps — which, given how sparse rewards are in WIRE, is what most
of the learned structure comes from. `coop_ij` is the max co-activity over the last `coop_window`
steps and `neg_i` marks a windowed loss below `−θ`; the two branches are mutually exclusive by
construction, because the window includes the current step.

Four rules are selectable with `--hebbian-mode`:

| Mode | Growth term | Notes |
|---|---|---|
| `reward_modulated` | `(η₀ + η₊·\|r_bond\|/R)·c·(1−W)` | The paper's Eq. 7. Reward acts on the step it arrives. |
| `three_factor` | `η₀·c·(1−W) + η₊·(\|r_bond\|/R)·e·(1−W)` | Eligibility trace `e ← ρ_e·e + c` (~10-step memory), so a milestone credits the work that preceded it. Optionally routes death blame through the same trace (`eta_minus_death`). |
| `coactivity` | `η₊·c·(1−W)` | Ablation: no reward modulation at all. |
| `legacy` | advantage-modulated LTP/LTD with a failure-grace window | The first implementation; kept so old runs stay reproducible. |

Because the always-on `−λW` term never switches off, the rule has an interior fixed point
`W* = κ·c / (κ·c + λ)` rather than saturating at the clip. The term-by-term decomposition of an
actual run is `analysis/replay_hebbian_terms.py`; the eligibility-trace comparison is
`analysis/prototype_three_factor_rule.py`.

## Couplings back to behaviour

**Reward diffusion** (Eq. 9, `diffuse_rewards`) — every mode, on by default at `γ_d = 0.2`:

```
r'_i = (1 − γ_d)·r_i + γ_d · Σ_{j≠i} W̄_ij · c_ij · r_j
```

`W̄` is the row-normalised bond and `c_ij` gates diffusion to *currently* co-active pairs, so a
strong bond to an agent doing something unrelated moves no reward. Diffused rewards are what the
metric recorder and the PPO buffers see; the undiffused stream is kept for reporting, which is why
task return is comparable across arms.

**Weight-gated experience sharing** (Eq. 10, `get_social_replay_indices`, `--hebbian-rho`) — each
agent's PPO batch is a `(1−ρ)/ρ` mixture of its own transitions and neighbours' transitions drawn
in proportion to `W̄_ij`, above a sparsity floor. Shared transitions keep their source policy's
behaviour log-probability, so PPO's clipped ratio is the importance correction. Off by default
(`ρ = 0`).

**Social module** (`--social-module`, `mindforge/agent_modules/social_module.py`) — the LLM
coupling. Each deliberation the agent receives its bond row plus the change in each bond over a
fixed window, tagged rising/decaying/stable, and returns a structured decision: request help from
a named teammate, offer help to specific senders, or stay on task. `prompt` renders that decision
into the action prompt; `bias` additionally lets its `ask_target` override message routing. The
module is queried every `--social-interval` steps, not every step; the last directive is cached in
between.

## Imposing and transplanting a graph

`--hebbian-freeze` stops all updates, and `--hebbian-preset {uniform,star,ring,pair}` writes a
hand-set topology (strong/weak values from `--hebbian-bond-strong/-weak`) — the frozen-topology
ablation. `--hebbian-init-file` loads a `W` from JSON instead, which is how a learned graph is
carried into a new team composition; `src/mindforge/tools/pair_transplant.py` assembles that matrix
and the matching per-agent cognitive state from a set of dyad runs.

## What gets logged

`hebbian_snapshots.jsonl` holds one end-of-episode `W`; `final_metrics.json` carries a full `W`
every `log_graph_every` (50) steps plus the graph metrics (`get_graph_metrics`), the per-channel
attribution of bond growth (`get_channel_attribution`) and the LTD heatmap. Everything the analysis
scripts report about bonds is derived from those two files.
