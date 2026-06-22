# 06 — Rewards: The Economy Ledger

**Source files:** `src/mindforge/multi_agent_craftium.py` (per-step assembly, lines ~1433–2312), `src/mindforge/env/communication_rewards.py`, `src/mindforge/custom_environment_craftium.py` (JSONL drains), `src/marl_craftium/craftium-envs/five-chambers/mods/five_chambers/milestones.lua`, `.../deaths.lua`, `src/hebbian/graph.py` (diffusion), `src/hebbian/config.py`
**Paper sections:** Table 2 (milestone ladder + comm milestones), Table 3 (auxiliary rewards), Eq. 6 (bond growth inputs eta_0/eta_plus/R), Eq. 8 (reward diffusion gamma_d), App. C.1 (social replay rho)
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

This is the single place to answer "where did this reward come from". Per step the orchestrator
builds `step_rewards_raw` from five accumulators (`multi_agent_craftium.py:1433-1437`), in order:

1. Env step-reward channel drain (`:1625-1628` simultaneous / `:1691-1694` turn-based)
2. Pitch-futility penalty (`:1885-1893`)
3. Communication rewards (`:1898-1916`)
4. Phase 1c: Lua milestone JSONL drain (`:1979-2009`)
5. Phase 1d: Lua death JSONL drain (`:2024-2036`)
6. Phase 2: Hebbian `update()` + `diffuse_rewards()` (`:2089-2098`)
7. Phase 3: decomposition + `record_reward` + RL `store_reward` (`:2119-2224`)

## 1. Master ledger

"r_bond" = enters the bondable stream fed to Hebbian growth (`:2083-2088`). "total" = enters
`step_rewards_raw`, which is both the diffusion input and the Hebbian neg-gate. Everything in
"total" reaches the RL buffer and `metric.record_reward` **after diffusion** (Section 3).

| Source | Value | Emitted / drained at | r_bond | total | Paper |
|---|---|---|---|---|---|
| Milestone ladder m1–m28 (full value to **every** contributor) | +10…+300 | values in `milestones.lua` (e.g. `m8_anvil_A1=40` at `:38`); drained `multi_agent_craftium.py:1987-2009` | yes | yes | Table 2; ladder table in 05-five-chambers-world.md |
| Comm base, per valid targeted message | +0.5, cap 50 msgs (= 25.0) | `communication_rewards.py:17-18,82-85`; added `:1903-1904` | yes | yes | Table 3 — > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #3 |
| Comm chamber milestones `m_comm_ch1..5` | +10/10/20/10/10 at >= 4 msgs | `communication_rewards.py:22-28,88-95` | yes | yes | > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #1 (also #11) |
| Would-die (virtual-HP lethal hit) | −10, **Ch4 only**, capped 1/agent/episode by the Python drain | `deaths.lua:121-150`; cap in `custom_environment_craftium.py:1550-1611` (`poll_death_events`); applied `:2024-2036` | **no** | yes | Table 3 — > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #6 |
| Ch5 permadeath | −50, terminal | `deaths.lua:36,157+`; same drain; terminal booking `:2187-2205` | **no** | yes | Table 3 (consistent) |
| Pitch-futility (LookUp/LookDown redirected to NoOp at camera limit) | −1 × count | `consume_futile` (`custom_environment_craftium.py:947-956`); applied `:1885-1893` | yes | yes | not in inconsistencies record |
| Env step-reward channel | ~0 in practice — server-side `craftium.reward()` never reaches `env.step()` in multi-agent five-chambers (hence the JSONL drains) | `get_step_reward` (`custom_environment_craftium.py:753-756`) | no (only via `total_rewards`) | yes | — |
| ~~Proximity bonus +0.3/pair~~ | **removed** (kept as zero-valued `proximity` stream for back-compat) | comment `:2055-2059`, `:2134` | — | — | — |
| ~~Action-repetition penalty~~ | **removed** | comment `:1872-1878` | — | — | — |
| RL-internal `config.death_penalty` added on `done=True` inside `store_reward` | default **0.0** (no-op) | `rl_layer/config.py:51`, `rl_layer/rl_layer.py:311-312` | — | — (post-diffusion, buffer-only) | — |

Notes: milestone drain parses both `agent0` and `agent_0` contributor forms (`:1991-2009`) — the
old underscore-only parser silently dropped **all** Lua milestone rewards. Comm milestones are also
recorded as milestone events for metrics (`:1905-1916`), see 09-metrics-and-evaluation.md.

## 2. Communication rewards in detail

`CommunicationTracker` (`communication_rewards.py:31-100`) is rebuilt **per episode**
(`multi_agent_craftium.py:1237`), so all caps and fired-flags reset each episode.

**Validity gate** (`_is_valid`, `communication_rewards.py:48-55`) — all three must pass:

| Rule | Constant | Effect |
|---|---|---|
| Min length | `MIN_MSG_LEN = 5` (after strip) | blocks empty/trivial strings |
| No duplicate | message != agent's own last valid message | blocks verbatim repeats |
| Rate limit | `RATE_LIMIT_STEPS = 2` | < 2 steps since own last valid message ⇒ invalid |

**Base reward:** +0.5 (`BASE_MSG_REWARD`) per valid message, paid only while
`total_valid_msgs < BASE_MSG_CAP = 50` — i.e. max **25.0 points** per agent per episode
(`:82-85`). > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #3. Pinned by
`tests/test_paper_defaults.py::test_comm_reward_constants`.

**Chamber milestones** (`CHAMBER_COMM_THRESHOLDS`, `:22-28`): fires once per agent per episode
when that agent's valid-message count **in that chamber** reaches the threshold (`>= 4`, `:91`).
> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #1 (values) and #11 (">= 4" wording).

| Chamber | Threshold | Reward | Milestone id |
|---|---|---|---|
| ch1 | 4 | 10.0 | m_comm_ch1 |
| ch2 | 4 | 10.0 | m_comm_ch2 |
| ch3 | 4 | 20.0 | m_comm_ch3 |
| ch4 | 4 | 10.0 | m_comm_ch4 |
| ch5 | 4 | 10.0 | m_comm_ch5 |

**Bad-target semantics** (routed but unpaid): if the model's `communication_target` was self /
"all" / unparseable, routing rescues the message via Hebbian/random fallback (see
07-orchestrator.md); the sender lands in `_bad_target_speakers`
(`multi_agent_craftium.py:1854-1862`, detected by routing prefix `hebbian_fallback:` /
`random_fallback:`). The receiver still gets the message and the message still counts toward
chamber comm milestones, but the sender is **not** paid the +0.5 base — otherwise self-talk would
be positively reinforced (`communication_rewards.py:64-69,82-85`).

**Chamber attribution is a z-band lookup**: `CHAMBER_BOUNDS` (`:9-15`) maps the sender's position
`p[2]` (world z) to ch1..ch5; a position outside every band earns base reward only, no milestone
counting. Per-message accounting (`valid`, `rewarded_base`, `rewarded_milestone`, sender/receiver
chambers) is stamped into `messages.jsonl` (`multi_agent_craftium.py:1918-1941`).

## 3. The diffusion transform and stream decomposition

**Hebbian inputs** (`hebbian_graph.update(...)`, `:2089-2097`; consumed in `graph.py:492-549`):

- `bond_rewards` (r_bond) = milestone drain + comm total + pitch penalty (`:2083-2088`).
  Death/would-die are **excluded by construction** — Variant B never bonds on shared deaths.
  Chamber-gated: Ch1/unknown chamber zeroes the contribution (`graph.py:519-528`).
  Growth coefficient = eta_0 + eta_plus·|r_bond|/R with R = `reward_norm_R = 300`
  (`graph.py:445-466`; eta_0 = 0.01, eta_plus = 0.05, `config.py:69-81`).
- `total_rewards` = `step_rewards_raw` (death **included**) — used only for the neg_i decay gate
  (eta_minus = 0.025), so a death can still trip bond decay (`graph.py:530-537`).
  The unconditional homeostatic term lambda·W (lambda = 0.0003) also applies every step.
  > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #4

**Diffusion** (`diffuse_rewards`, `graph.py:847-894`) implements Eq. 8:
`r'_i = (1 − gamma_d)·r_i + gamma_d · Σ_{j≠i} w̄_ij · c_ij · r_j`, with gamma_d =
`reward_diffusion_gamma = 0.2` (`config.py:66`), w̄_ij row-normalised
(`get_normalized_weights`, `graph.py:730-745`), c_ij the current-step co-activity, and inputs
clamped to ±1e6 (`_sanitize_reward`, `graph.py:30-43`). Diffusion is **not conservative**: the
team sum shrinks unless bonds/co-activity fully reciprocate (see worked example).

**Decomposition dict** (`:2119-2142`) — five streams that sum exactly to `diffused_rewards[i]`:

| Stream | Contents |
|---|---|
| `task` | env-step reward + pitch penalty + milestone drain + **death drain** |
| `comm_base` | +0.5 per valid message (comm total minus milestones) |
| `comm_milestone` | Tier-2 chamber comm milestones |
| `proximity` | always 0.0 (removed bonus; field kept for 5-tuple back-compat) |
| `hebbian_diffuse` | `diffused[i] − raw[i]` (can be negative) |

Note: death is folded into `task`; there is no separate `death` stream.

**Who consumes what:**

| Consumer | Receives | Where |
|---|---|---|
| `metric.record_reward` / `record_reward_decomposed` | **diffused** + stream dict | `:2208-2210` (live), `:2187-2191` (death-step booking for terminated agents) |
| RL `store_reward` | **diffused** (+`done`); kwargs `reward_task`/`reward_comm` are diagnostics — `reward_task` is the pre-pitch, pre-drain env-channel snapshot (`:1846`), NOT the decomposition's `task` | `:2220-2224`; terminal path `:2200-2205` |
| Centralized critic | team reward = mean of **alive** agents' diffused | `:2292-2307` |
| Hebbian growth | death-excluded bondable r_bond | `:2083-2095` |
| Hebbian neg-gate | raw totals (death included) | `:2096` |
| stdout `[REWARD ...]` audit line | non-zero streams + diffused total | `:2155-2171` |

Social replay does not change rewards — it mixes neighbour transitions at update time; see
03-rl-layer.md. Note the dataclass default rho = 0.0 (`config.py:63`) is overridden by the CLI
default `--hebbian-rho 0.3`, so it was active in the RL+Hebbian runs
(PAPER_INCONSISTENCIES.md #9).

## 4. Worked example

Setup: 2 agents in Ch2, gamma_d = 0.2, bonds W[1][0] = 0.5, W[0][1] = 0, co-activity
c_01 = c_10 = 1.0. This step: both agents break anvil A1 → milestone `m8_anvil_A1` fires with
contributors `[agent0, agent1]`, +40 **each** (`milestones.lua:38`; drain `:1991-2009`);
agent_0 also sends a valid, well-targeted message (+0.5). No deaths, no futile actions.

| Quantity | agent_0 | agent_1 |
|---|---|---|
| raw `step_rewards_raw` | 40.0 + 0.5 = **40.5** | **40.0** |
| r_bond (bondable, Ch2 gate = 1) | 40.5 | 40.0 |
| growth coeff eta_0 + eta_plus·\|r_bond\|/R | 0.01 + 0.05·40.5/300 ≈ 0.0168 | ≈ 0.0167 |
| w̄ toward the other agent | 0/(0+eps) = 0.0 | 0.5/(0.5+eps) ≈ 1.0 |
| diffused r' (Eq. 8) | 0.8·40.5 + 0.2·0 = **32.40** | 0.8·40.0 + 0.2·(1.0·1.0·40.5) = **40.10** |
| decomposition | task 40.0, comm_base 0.5, hebbian_diffuse **−8.10** | task 40.0, hebbian_diffuse **+0.10** |
| RL `store_reward` / `record_reward` | 32.40 | 40.10 |

agent_0's unreciprocated bond row (w̄ = 0) means it "donates" 20% of its reward into a social
term that nobody returns; agent_1's strong bond to agent_0 recovers almost all of its own 20%
plus a share of agent_0's comm bonus. Team sum drops 80.5 → 72.5 — diffusion redistributes and
shrinks, it does not conserve. Both agents' r_bond drives bond growth this step; had one died
instead, the −50 would appear only in `total`/`task` and the neg-gate, never in r_bond.
