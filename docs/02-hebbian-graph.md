# Hebbian Social Graph (W)

**Source files:** `src/hebbian/config.py`, `src/hebbian/graph.py`, `src/hebbian/__init__.py`; call sites in `src/mindforge/multi_agent_craftium.py`, `src/rl_layer/ppo_update.py`
**Paper sections:** §3 (Hebbian social plasticity), Eq. 2 (co-activity), Eq. 6 (bond update), Eq. 7 (social replay), Eq. 8 (reward diffusion), Table 7 (hyperparameters), App C.1
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

`HebbianSocialGraph` is numpy-only (no torch) and fully no-op when `HebbianConfig(enabled=False)` (`config.py:18`, `graph.py:70-71,634-635`). The module is independent of the RL layer and LLM stack by design (`__init__.py:8`).

## 1. State

| Item | Definition | Anchor |
|---|---|---|
| W | directed bond matrix in [0,1]^{NxN}; W[i,j] = "i's trust toward j", W[i,j] != W[j,i] in general | `graph.py:252-255` |
| Init | uniform `init_weight=0.1` unless `init_matrix` / `init_preset` (uniform/star/ring/pair) given | `graph.py:75-86,128-154`; `config.py:86-94` |
| Diagonal | forced to 0 at init, after every update, and in every loader | `graph.py:86,126,153,576,693` |
| Clip | `np.clip(W, 0, 1)` after every update | `graph.py:575,692` |
| Warm start | gated modes require `init_weight > 0`; constructor raises otherwise (code rationale: "W=0 is a fixed point of the gated rule, so a zero warm-start can never grow") | `graph.py:98-103`; pinned by `tests/test_hebbian_update.py::test_init_weight_zero_raises_for_gated_modes` |

### Engagement g_i (gated path, `_engagement`, `graph.py:368-395`)

```
g_i = clip( alpha * |r_bond_i| / (r_max + eps) * 1[ch_i >= 2]  +  (1-alpha) * 1[i sent/received a message], 0, 1 )
```

- alpha = `engagement_reward_weight = 0.5` (`config.py:30`); eps = `_EPS = 1e-8` (`graph.py:27`).
- The chamber gate `1[ch>=2]` is applied to `bond_rewards` *before* the call (`graph.py:519-526`), so g_i is built from Ch2-5, death-excluded reward only — "a death spike can never raise g_i" (`graph.py:377-380`).
- r_max = running max of |r_bond| seen (`_max_bond_reward_seen`, `graph.py:384-388`); monotone drift is harmless because g_i is a bounded gate.

### Co-activity c_ij (gated path, `_coactivity_gated`, `graph.py:397-443`)

```
c_spat[i,j] = 1[ ||p_i - p_j|| <= d ] * g_i * g_j                      # close AND both engaged
c_comm[i,j] = delta_comm * 1[i<->j message] * (1 - 1[||p_i-p_j|| <= d]) # talk across distance only
c_ij        = clip(c_spat + c_comm, 0, 1);  c_ij[c_ij < eps_coop] = 0;  diag = 0
```

| Term | Value | Anchor |
|---|---|---|
| Spatial radius d | `interaction_radius = 5.0` world units; vectorised pairwise distance, missing position => no gate (NaN) | `config.py:27`; `graph.py:417-425` |
| delta_comm | `communication_coactivity_bonus = 0.5`; **suppressed when co-located** (factor `1 - spatial_gate`); a single directed message activates both directions | `config.py:33`; `graph.py:430-437`; pinned by `tests/test_hebbian_coactivity.py::test_comm_bonus_suppressed_when_colocated` |
| eps floor | `c_ij < coop_eps (0.05)` floored to exactly 0 ("no co-activity") — makes growth/decay branches strictly mutually exclusive | `config.py:75`; `graph.py:441-443` |

The legacy mode has its own older implementations (`_compute_coactivity`, `graph.py:157-241`; engagement normalised by a max over *all* step rewards, no chamber gate).

## 2. Update rule — three modes

Dispatch is on `config.mode` in `update()` (`graph.py:634-645`; default `"reward_modulated"`, `config.py:21`). Legacy requires `step_rewards` (`graph.py:647-648`).

| Mode | Growth coefficient (per agent i) | Gates / extra terms | Anchor | Paper status |
|---|---|---|---|---|
| `legacy` | advantage modulator m_i = ltp_lr*tanh(beta*max(A_i,0)) - ltd_lr*tanh(beta*max(-A_i-thr,0)), plus unconditional `base_ltp` | failure window + sustained LTD (lambda_F=0.002), failure-grace LTP bonus, passive `-lambda*W` | `graph.py:244-304,307-348,650-727` | superseded; failure modes in legacy runs (see project notes) |
| `coactivity` (Variant A) | coeff_i = eta_plus (flat) — eta_plus plays the role of B's floor eta_0; A is **not** B with eta_plus=0 | same gated growth/decay as B | `graph.py:451-457,463-464` | ablation |
| `reward_modulated` (Variant B) | coeff_i = eta_0 + eta_plus * \|r_bond_i\| / R | salience term is death-excluded by construction | `graph.py:465-467` | **the paper's evaluated rule** |

Gated step (`_update_gated`, `graph.py:492-590`), per directed pair (i,j):

```
growth_ij = (eta_0 + eta_plus*|r_bond_i|/R) * c_ij * (1 - W_ij)            # graph.py:552-553
decay_ij  = eta_minus * W_ij,  fired iff  coop_ij < eps_coop  AND  neg_i    # graph.py:555-556
dW_ij     = where(decay_mask, -decay_ij, growth_ij) - lambda * W_ij         # graph.py:565-566
```

> CONSISTENT with paper Eq. 6 (eta_0 folded into the growth coefficient) — see PAPER_INCONSISTENCIES.md #10.

Window statistics (`_windowed_stats`, `graph.py:470-490`), both deques `maxlen = coop_window = 50` (`config.py:77`; `graph.py:104-105`):

- `coop_ij = max over window of c_ij(s)` — peak co-activity, includes current step, so coop_ij >= c_ij(t): decay (coop < eps) implies c_ij(t)=0 implies zero growth — the mutual-exclusivity invariant (`graph.py:478-481,510-512`).
- `neg_i = 1[ sum over window of total_reward_i < -neg_theta ]`, theta = 5.0 (`config.py:79`; `graph.py:488-489`); broadcast over row i (`graph.py:555`).

`freeze_weights=True` computes deltas but never writes W (`config.py:84`; `graph.py:573`) — used for the frozen-pair preset runs. Realised per-branch deltas are kept in `_last_growth`/`_last_decay` for analysis (`graph.py:568-571`).

### Decision tree for one bond (i,j), per step

```
c_ij >= eps_coop ?
├── yes ── GROWTH  (eta_0 + eta_plus|r_bond_i|/R) * c_ij * (1-W_ij)   [decay_mask false: coop_ij >= c_ij >= eps]
└── no ─── window-max coop_ij < eps  AND  sum r_total_i < -theta ?
           ├── yes ── FAILURE DECAY  -eta_minus * W_ij
           └── no ─── neither branch (delta = 0)
... then ALWAYS:  - lambda * W_ij   (homeostatic), clip to [0,1], diag=0
```

## 3. Homeostatic decay

Unconditional `-lambda*W` every gated step, lambda = `decay = 0.0003` dataclass default (`config.py:48`; `graph.py:558-566`) — the CLI default and the final launch scripts use **0.005** (see PAPER_INCONSISTENCIES.md #15). Without it the growth branch is monotone and co-present bonds saturate at the W=1 clip (the "saturate-to-1" legacy failure); with it the rule has a stable interior fixed point

```
W* = coeff*c / (coeff*c + lambda)
```

(`graph.py:505-508,561-564`). Added in commit `ecb6d24`. Pinned by `tests/test_hebbian_update.py::test_homeostatic_decay_geometric` and `::test_homeostatic_fixed_point`.

> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #4 (the -lambda*W term is absent from paper Eq. 6 and lambda is missing from Table 7).

## 4. Reward channels: bond_rewards vs total_rewards

Two per-agent streams enter `update()` (`graph.py:599-602,624-628`); both are chamber-gated to Ch2-5 by `_chamber_gate` (`graph.py:351-366,519-537`; chambers=None => all cooperative, the unit-test path).

| Stream | Contents | Death? | Used for | Why |
|---|---|---|---|---|
| `bond_rewards` | milestone drain + comm rewards + pitch ("futile") penalty | **EXCLUDED** | Variant B growth salience + engagement normaliser | shared deaths must never *strengthen* bonds — exclusion is by construction, not by sign filtering (`graph.py:458-459,624-626`) |
| `total_rewards` | full step reward incl. drained death / would-die penalties (`= step_rewards_raw`) | **INCLUDED** | only the `neg_i` failure-decay gate | a death *can* trip sustained-loss decay (`graph.py:530-537,627-628`) |

Call site: `multi_agent_craftium.py:2061-2097` — `_bond_rewards = milestone_drain + comm + pitch_penalty` (`:2083-2088`), `total_rewards=step_rewards_raw` (`:2096`), chambers via `_CHAMBER_TO_INT` from `environment.get_chamber` (`:2078-2082`). Death penalties arrive via the death-events JSONL drain (see 06-rewards.md); pinned by `tests/test_hebbian_update.py::test_death_exclusion`.

## 5. Reward diffusion (Eq. 8)

`diffuse_rewards` (`graph.py:847-894`):

```
r'_i = (1 - gamma_d) * r_i + gamma_d * sum_{j != i} Wbar_ij * c_ij * r_j
```

- gamma_d = `reward_diffusion_gamma = 0.2` (`config.py:66`); shared by all modes.
- Wbar = row-normalised W: `Wbar_ij = W_ij / (sum_{k!=i} W_ik + eps)` (`get_normalized_weights`, `graph.py:730-745`).
- c_ij defaults to `_last_coactivity` from the current step (`graph.py:876-880`).
- Consumed by the orchestrator immediately after `update()` (`multi_agent_craftium.py:2098`); `diffused - raw` is logged as the `hebbian_diffuse` stream in the reward decomposition (`:2106-2118`). See 07-orchestrator.md and 06-rewards.md.
- Pinned by `tests/test_hebbian_graph_api.py::test_diffuse_rewards_hand_case` and the gamma=0 identity test.

## 6. Social replay (Eq. 7)

`get_social_replay_indices` (`graph.py:784-844`) samples (buffer_idx, agent_j) pairs from neighbour buffers proportional to Wbar_ij * rho, excluding weak bonds W_ij < 0.05 (`graph.py:831`). Consumer: `_collect_social_replay` in `ppo_update.py:141-169`; rho <= 0 returns `[]` (`graph.py:815-816`). The dataclass default is rho = 0.0 ("was 0.3 — disabled until IS correction is added", `config.py:63`), **but** the entry point's CLI default `--hebbian-rho 0.3` (`multi_agent_craftium.py:202`) overrides it, and the final RL+Hebbian launchers pass no rho flag — so replay was ACTIVE (rho=0.3) in the evaluated MAPPO/IPPO+Hebbian runs.

> PAPER MISMATCH — App C.1 presents replay as non-evaluated, but the CLI default made it active in the RL+Hebbian runs; see PAPER_INCONSISTENCIES.md #9 and #15.

## 7. Introspection & persistence

| API | Returns / does | Anchor |
|---|---|---|
| `get_graph_metrics()` | `mean_bond_strength` (off-diag mean), `sparsity` (frac off-diag < 0.1), `top_3_pairs`, `per_agent_out_strength` (row sums), `modularity_proxy` (within-role minus cross-role mean, needs `agent_roles`), `ltd_heatmap` (F_ij), **full `W`** (added so per-pair plots don't drop out of top-3) | `graph.py:897-982` |
| `get_ltd_heatmap()` | F_ij = failure co-activation / window (legacy diagnostics) | `graph.py:984-998` |
| `bond_delta_row(i)` | {j: W_ij(now) - W_ij(50 steps ago)} from `_W_history` (window fixed at 50, `graph.py:111-113`); zeros if <2 snapshots. Feeds the SocialModule / prompt bond strings via the orchestrator's bond cache (`multi_agent_craftium.py:1445-1473`); see 08-cognitive-agent.md | `graph.py:753-772` |
| `get_weight(i,j)` / `get_all_weights()` / `get_normalized_weights(i)` | raw w_ij / copy of W / Wbar row | `graph.py:730-781` |
| `to_dict()` / `from_dict()` | JSON round-trip of W, failure state, running maxima, step count; `from_dict` clears all deque windows (they are not serialised) | `graph.py:1001-1022,1034-1055` |
| `snapshot()` | compact per-episode record for `hebbian_snapshots.jsonl` (W, step, max_reward_seen, N) | `graph.py:1024-1032` |
| `reset()` | W back to `init_weight`, clears windows/counters incl. `_W_history` (cleared + re-seeded with the fresh W — a post-commit fix; previously stale snapshots leaked into post-reset bond deltas, see PAPER_INCONSISTENCIES.md #14, pinned by `tests/test_hebbian_graph_api.py::test_reset_clears_w_history`); preserves config and roles. Episode resets keep bonds alive via `CustomAgent.on_reset` instead — see 08-cognitive-agent.md | `graph.py:1057-1083` |

All reward inputs pass `_sanitize_reward` (NaN/inf/non-numeric -> 0, clamp to +/-1e6, `graph.py:30-43`).

## 8. Hyperparameters (config field -> paper symbol)

Paper Table 7 agreement is recorded centrally in PAPER_INCONSISTENCIES.md #13 (all match except lambda); defaults pinned by `tests/test_paper_defaults.py`.

| Config field (`config.py`) | Symbol | Default | Paper Table 7 | Match |
|---|---|---|---|---|
| `eta_0` (:71) | eta_0 | 0.01 (dataclass) / **0.005** in final runs | 0.01 | dataclass yes — runs no, #15 |
| `eta_plus` (:69) | eta_plus | 0.05 | 0.05 | yes |
| `eta_minus` (:73) | eta_minus | 0.025 | 0.025 | yes |
| `decay` (:48) | lambda | 0.0003 (dataclass) / **0.005** in final runs | **not in paper** | no — #4, #15 |
| `coop_eps` (:75) | eps | 0.05 | 0.05 | yes |
| `coop_window` (:77) | n (window) | 50 | 50 | yes |
| `neg_theta` (:79) | theta | 5.0 | 5.0 | yes |
| `reward_norm_R` (:81) | R | 300.0 (dataclass) / **50** in final runs | 300 | dataclass yes — runs no, #15 |
| `interaction_radius` (:27) | d | 5.0 | 5.0 | yes |
| `engagement_reward_weight` (:30) | alpha | 0.5 | 0.5 | yes |
| `communication_coactivity_bonus` (:33) | delta_comm | 0.5 | 0.5 | yes |
| `init_weight` (:94) | w_0 | 0.1 | 0.1 | yes |
| `reward_diffusion_gamma` (:66) | gamma_d | 0.2 | 0.2 | yes |
| `social_replay_rho` (:63) | rho | 0.0 (dataclass) / **0.3** (CLI default, used in runs) | 0 ("not evaluated") | no — #9, #15 |

Legacy-only fields (`ltp_lr`, `ltd_lr`, `ltd_threshold`, `base_ltp`, `modulation_beta`, `ltd_sustained_lr`, `failure_*`, `config.py:36-60`) are not part of the evaluated Variant-B rule and are not in Table 7. Topology presets (`init_preset`, `preset_bond_*`, `freeze_weights`, `config.py:84-91`) configure the frozen-graph baselines — see 10-configuration.md.
