# Paper ↔ Code Consistency Record

**Paper:** *Wired Together: Reward-Modulated Hebbian Social Plasticity for Emergent Social
Intelligence in Multi-Agent Systems* (Ana Cristiana Marcu, thesis draft of June 8, 2026).
**Code verified at commit:** `52bb302` (wired_final, 2026-06-10) + the two bug fixes listed in §3.
**Method:** every entry pairs a paper claim (section / table / equation) with the code reality
(file:line), checked by reading the source and — where marked — pinned by a regression test in
`tests/`.

## Status legend

| Status | Meaning |
|---|---|
| **MISMATCH** | Paper and code genuinely disagree; recommendation says which side to change. |
| **PAPER-INTERNAL** | The paper contradicts itself; code is a bystander. |
| **AMBIGUOUS** | Paper wording admits two readings; code implements one of them. |
| **CONSISTENT** | Claim verified true against source (recorded because it was previously in doubt). |

## Summary table

| # | Item | Status | Severity | Fix on |
|---|---|---|---|---|
| 1 | Communication-milestone reward values | MISMATCH | Affects reported reward numbers | Paper |
| 2 | Milestone numbering M1–M24 vs m1–m28 | PAPER-INTERNAL | Presentation only | Paper |
| 3 | Message-reward cap "50 total" | AMBIGUOUS | Presentation only | Paper |
| 4 | Homeostatic decay λ missing from Eq. 6 | MISMATCH | Affects stated dynamics | Paper |
| 5 | Sampling temperature 0.7 | CONSISTENT (local path) | — | — |
| 6 | Would-die −10 scope and frequency | MISMATCH | Affects stated reward scheme | Paper |
| 7 | Slot masking → 14 RL candidates | CONSISTENT | — | — |
| 8 | Base model Qwen3.5 vs model-agnostic code | NEEDS FRAMING | Reproducibility wording | Paper |
| 9 | Social replay active (ρ=0.3) in RL+Hebbian runs | MISMATCH | Affects evaluated-system claim | Paper |
| 10 | η₀ folded into growth coefficient (Eq. 6) | CONSISTENT | — | — |
| 11 | Comm milestone "more than a threshold" | AMBIGUOUS | Wording | Paper |
| 12 | Milestone ladder rewards (Table 2, M1–M24) | CONSISTENT | — | — |
| 13 | RL & Hebbian hyperparameters (Tables 6, 7) | CONSISTENT (defaults) | See #15 for run-time overrides | — |
| 14 | Repo hygiene: declared tests/ did not exist | FIXED | — | Repo (this work) |
| 15 | HPC launch scripts override Table-7 values (η₀, R, λ, ρ) | MISMATCH | Affects reported hyperparameters | Paper |
| 16 | Stale world-layout coordinates (zombie spawns; Python chamber bands) | CODE BUG (flagged, not fixed) | Affects env behaviour / comm attribution | Code (before new runs) |

---

## 1. Communication-milestone reward values — MISMATCH

**Paper** (Table 2, communication track): M_comm_ch1 = 40, ch2 = 20, ch3 = 30, ch4 = 15, ch5 = 20.
**Code** (`src/mindforge/env/communication_rewards.py:22-28`):

```python
CHAMBER_COMM_THRESHOLDS = {
    "ch1": (4, 10.0, "m_comm_ch1"),
    "ch2": (4, 10.0, "m_comm_ch2"),
    "ch3": (4, 20.0, "m_comm_ch3"),
    "ch4": (4, 10.0, "m_comm_ch4"),
    "ch5": (4, 10.0, "m_comm_ch5"),
}
```

Every chamber's value differs (10/10/20/10/10 vs 40/20/30/15/20). All collected runs used the
code values. **Recommendation:** update paper Table 2 to 10/10/20/10/10. Pinned by
`tests/test_paper_defaults.py::test_comm_reward_constants`.

## 2. Milestone numbering — PAPER-INTERNAL

Paper Table 2 renumbers milestones contiguously M1–M24 (+M_door1), but the running text uses the
*code's* mod IDs: §A.3 cites "M19, M22, M23, M27, M28" as team milestones and §A.2 says Chamber 5
spans "(M24–M28)". In code (`milestones.lua`) the IDs are non-contiguous (`m9` reuses a legacy
slot; `m14/m15` are gear equips; Ch5 = `m24`–`m28`), so the text and table disagree with each
other: e.g. Table-2's "boss defeated" is M23 while the text's is M27 (= code `m27_boss_defeated`).
**Recommendation:** renumber the in-text references to the Table-2 scheme, or add a footnote that
code IDs differ. The code-side mapping is documented in `docs/05-five-chambers-world.md`.

## 3. Message-reward cap — AMBIGUOUS

**Paper** (Table 3): valid targeted message "+0.5, capped at 50 total per agent per episode".
**Code** (`communication_rewards.py:18`): `BASE_MSG_CAP = 50` caps the number of *rewarded
messages* at 50, i.e. **25.0 reward points** maximum. If "50 total" meant 50 reward points, the
code delivers half that. **Recommendation:** reword to "capped at 50 rewarded messages
(= 25 reward) per agent per episode".

## 4. Homeostatic decay missing from Eq. 6 — MISMATCH

**Paper** (Eq. 6): ΔW has exactly two terms — reward-modulated growth and failure-gated decay.
**Code** (`src/hebbian/graph.py`, gated update; `HebbianConfig.decay = 0.0003`,
`src/hebbian/config.py:48`): a third, *unconditional* homeostatic term `−λ·W` is applied every
step in addition to the two paper terms. It prevents saturation at W = 1 and gives the bond an
interior fixed point W* = coeff·c / (coeff·c + λ). This term was added in commit `ecb6d24`
("Add decay for hebbian") and is active in all current runs. Note on the value: the dataclass
default is 0.0003, but the CLI default is `--hebbian-decay 0.005`
(`multi_agent_craftium.py:198`) and the final HPC scripts pass `--hebbian-decay 0.005`
explicitly — so the **effective λ in collected runs is 0.005** (see #15).
**Recommendation:** add the −λ·W term to Eq. 6 and λ = 0.005 to Table 7. Pinned by
`tests/test_hebbian_update.py` (homeostatic-decay tests).

## 5. Sampling temperature 0.7 — CONSISTENT (for the experiment path)

**Paper** (§D, run protocol): "LLM action and message sampling remains stochastic at temperature
0.7". **Code:** the in-process local model client used when `LLM_MODEL_PATH` is set (the HPC /
thesis configuration) defaults to `temperature = 0.7`
(`src/mindforge/agent_modules/local_model_client.py:312`). Caveat: the remote OpenAI-compatible
fallback client (`util.py:328`, default model `google/gemini-2.5-flash` via OpenRouter) does
*not* pin a temperature — only relevant for non-thesis debugging runs.

## 6. Would-die penalty scope and frequency — MISMATCH

**Paper** (Table 3 + §A.4): would-die hit −10, "once per agent per episode; Ch1–4".
**Code** (`deaths.lua:121-150`): the −10 fires **only in Ch4** — Ch1–Ch3 near-deaths are
explicitly free ("the refill is silent and free", `deaths.lua:15-16`) — and fires **per
would-death event**: the virtual-HP pool refills after each would-death (`deaths.lua:149`) and
the penalty can fire again, with a per-agent counter (`would_die_count`) rather than a once-flag.
**Recommendation:** update Table 3 to "−10 per would-die event; Ch4 only (Ch1–Ch3 near-deaths
are penalty-free)". The −50 Ch5 permadeath entry is correct as written
(`deaths.lua:157-184`; episode ends when all agents are dead, `_signal_episode_over`).

## 7. Slot masking → 14 RL candidates — CONSISTENT

**Paper** (§B, Table 4): the 8 `Slot*` actions are masked from the RL candidate set, leaving 14
candidates. **Code:** `RLConfig.mask_slot_actions = True` (`src/rl_layer/config.py:78`);
`RLLayer._build_candidate_actions` (`src/rl_layer/rl_layer.py:467-479`) filters
`a.startswith("Slot")` from the 22-action tuple → 14 candidates. Pinned by
`tests/test_action_space.py`.

## 8. Base model naming — NEEDS FRAMING

**Paper:** experiments use Qwen3.5-2B / Qwen3.5-9B. **Code:** model-agnostic — the reasoning core
is selected via environment variables (`LLM_MODEL_PATH` for the in-process local model;
`LLM_MODEL`/`LLM_BASE_URL` for remote, defaulting to `google/gemini-2.5-flash` via OpenRouter;
`src/mindforge/agent_modules/util.py:22-24,316-331`). The HPC launchers
(`hpc/daic/experiments/_common.sh`) set `LLM_MODEL_PATH` to the Qwen weights, so the paper claim
holds for the reported runs, but nothing in the code enforces it. **Recommendation:** phrase as
"the reasoning core is configurable; all reported experiments used Qwen3.5-2B/9B via
`LLM_MODEL_PATH`".

## 9. Social replay was ACTIVE in the RL+Hebbian runs — MISMATCH

**Paper** (§C.1): weight-gated experience sharing (social replay, Eq. 17–18) is "not part of
the evaluated system", included only as a design sketch.
**Code:** the *dataclass* default is indeed off — `social_replay_rho = 0.0` with the comment
"was 0.3 — disabled until IS correction is added" (`src/hebbian/config.py:63`). **But the
entry point overrides it:** `--hebbian-rho` defaults to **0.3**
(`multi_agent_craftium.py:202-203`) and is passed into `HebbianConfig(social_replay_rho=...)`
at `multi_agent_craftium.py:1040`. The final RL+Hebbian launch scripts
(`hpc/daic/experiments/exp05_mappo_hebbian.sbatch`, `exp06_ippo_hebbian.sbatch`) pass no
`--hebbian-rho` flag, so those runs executed with ρ = 0.3: `ppo_update._collect_social_replay`
→ `HebbianSocialGraph.get_social_replay_indices` returns neighbour transitions whenever
ρ > 0, and they enter the PPO mini-batch pool. LLM-only Hebbian conditions (exp07/exp08) are
unaffected (no PPO updates). **Recommendation:** either state in the paper that social replay
was active at ρ = 0.3 in the RL+Hebbian conditions, or re-run those conditions with
`--hebbian-rho 0` before claiming it was not evaluated.

## 10. η₀ in the growth coefficient — CONSISTENT

**Paper** (Eq. 6): growth = (η₀ + η₊·|r_bond|/R) · c_ij · (1 − W_ij). **Code:** `_growth_coeff`
returns exactly `eta_0 + eta_plus * |r_bond| / R` in `reward_modulated` mode and the update
multiplies by `c · (1 − W)`. The folding of η₀ into the coefficient matches the equation as
printed. Pinned by `tests/test_hebbian_update.py::test_growth_coeff_variant_B`.

## 11. Comm-milestone threshold wording — AMBIGUOUS

**Paper** (§A.3): the chamber comm milestone fires "when an agent emits *more than* a threshold
number of valid messages". **Code** (`communication_rewards.py:91`): fires at `count >= 4`, i.e.
*at* the 4th message. Minor wording; recommend "at least 4 valid messages in a chamber".

## 12. Milestone ladder rewards — CONSISTENT

All 25 milestone rewards in `milestones.lua` match paper Table 2 value-for-value under the
renumbering of item #2 (ch1 10/30/30/50/50/80/60 + door 50; ch2 40/40/50/30; ch3 20/40/60/100;
ch4 30/60/150/100; ch5 50/80/120/300/250). Pinned by
`tests/test_lua_spec.py::test_milestone_rewards_match_paper_table2`.

## 13. RL and Hebbian hyperparameters — CONSISTENT at the dataclass level

Every value in paper Table 6 (PPO/LoRA/optimisation) and Table 7 (Hebbian, except the missing λ
of item #4) matches the dataclass defaults in `src/rl_layer/config.py` and
`src/hebbian/config.py`. Pinned by `tests/test_paper_defaults.py`. **However**, the values the
final experiments actually ran with come from the CLI and the sbatch scripts, which override
four of them — see #15.

## 14. Repo hygiene — FIXED

`pyproject.toml` declared `testpaths = ["tests"]` but no tests existed. This verification work
added the `tests/` suite (194 tests); see `tests/README.md` for scope. Six genuine bugs were
fixed in the process — all in the metrics/analysis layer or guarding numerical edge cases;
none changes the reward stream delivered during any collected run:

1. `cooperation_metric.py:230` — stale `m11_` anvil prefix → `("m8_", "m9_")` (Ch2 cooperation
   score could never count the second anvil).
2. `trajectory_buffer.py` — GAE advantage standardisation NaN'd a single-transition buffer
   (torch unbiased std); now skipped below 2 transitions.
3. `hebbian/graph.py reset()` — did not clear `_W_history`, so post-reset bond deltas were
   computed against stale pre-reset snapshots; now cleared and re-seeded.
4. `coop_eval.py _credit_damage_share` — damage keys ('0'/'1') never matched contributor names
   ('agent0'/'agent_0'), so damage-share credit always degraded to an equal split; ids are now
   normalised before matching.
5. `craftium_metric.py` — `m_door1_open` (50) was missing from `MILESTONE_TRACK`/`TRACKS`, so
   its reward was silently dropped from every ch1_solo aggregate; added.
6. `make_results.py` — same `m_door1_open` gap in its mirrored table; added.

## 15. HPC launch scripts override Table 7 — MISMATCH

Paper Table 7 documents the dataclass defaults, but the **final experiment launch scripts**
(`hpc/daic/experiments/exp05_mappo_hebbian.sbatch`, `exp06_ippo_hebbian.sbatch`,
`exp07_llm_2b_social_prompt.sbatch`, `exp08_llm_9b_social_prompt.sbatch`) and the CLI defaults
of `multi_agent_craftium.py` override four Hebbian values:

| Symbol | Paper Table 7 | Dataclass default | Effective in final runs | Source |
|---|---|---|---|---|
| η₀ (association floor) | 0.01 | 0.01 | **0.005** | `--hebbian-eta-0 0.005` in exp05–exp08 |
| R (salience normaliser) | 300 | 300 | **50** | `--hebbian-reward-norm 50` in exp05–exp08 |
| λ (homeostatic decay) | — (absent, #4) | 0.0003 | **0.005** | `--hebbian-decay 0.005` in exp05–exp08 (also the CLI default, `multi_agent_craftium.py:198`) |
| ρ (social replay) | — ("not evaluated", #9) | 0.0 | **0.3** (RL runs) | CLI default `multi_agent_craftium.py:202`, not overridden by exp05/exp06 |

All other gated-rule CLI defaults (η₊ 0.05, η₋ 0.025, ε 0.05, n 50, θ 5.0, α 0.5, d 5.0,
γ_d 0.2, W₀ 0.1) match Table 7 (`multi_agent_craftium.py:173-207`).
**Recommendation:** update Table 7 to the values actually used (η₀ = 0.005, R = 50, λ = 0.005,
ρ = 0.3 for RL+Hebbian), or align the sbatch flags and re-run.

## 16. Stale world-layout coordinates — CODE BUG (flagged, NOT fixed)

The five-chamber world was re-laid-out at some point (Lua `config.lua:150,155`: CH4 z∈[47,57],
CH5 z∈[59,67]), and two consumers still carry the old layout:

1. **Zombie spawns** (`mobs.lua:81-85`): `CH4_SPAWN_POSITIONS` z = 54/57/**60** — with the
   current bounds the third zombie spawns *inside Chamber 5*. Consequences: the Ch4 arena-clear
   milestones (`m22_all_mobs_killed`, `m23_all_alive_ch4`) cannot be completed by normal play
   while a zombie sits in the next chamber.
2. **Python chamber bands** (`communication_rewards.py:9-15`): `CHAMBER_BOUNDS` still maps
   ch4 to z∈[52,62] and ch5 to z∈[64,72]. Positions in z∈[47,51] (Lua ch4) attribute comm
   messages to ch3/none, and z∈[59,63] (Lua ch5) to ch4/none — wrong-chamber comm-milestone
   attribution near boundaries.

Not fixed here because both changes alter environment/reward behaviour relative to the
collected runs (the same reason paper-alignment changes were excluded). **Recommendation:**
fix both before any new data collection, and audit other coordinate consumers
(`world_gen.lua` comments are also stale per `docs/05-five-chambers-world.md` §5).
