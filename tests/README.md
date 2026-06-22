# WiredTogether test suite

Verification suite for the `src/` implementation, written as part of the pre-submission
audit (see `docs/PAPER_INCONSISTENCIES.md` for the paper-consistency record this suite
underpins). 194 tests, all runnable locally on Windows/CPU with no game binary, no LLM
weights, and no network.

## Running

```powershell
# from the repo root — conftest handles sys.path; PYTHONPATH not required
python -m pytest tests -q          # full suite, ~5 s
python -m pytest tests -q -x       # stop at first failure
python -m pytest tests/test_hebbian_update.py -q -k decay   # focused
```

Markers (registered in `pyproject.toml`; the default `addopts` excludes the heavy ones):

| Marker | Meaning |
|---|---|
| `slow` | >5 s; included in the default run (currently only the critic learning-sanity test) |
| `needs_game` | requires the Luanti/Craftium binary — never run locally; reserved for future integration tests |
| `needs_llm` | requires model weights or network — reserved for future tests |

## Layout

| File | Covers |
|---|---|
| `conftest.py` | offline env guards, `sys.path`, stubs for `pettingzoo`/`craftium` (not importable on Windows), `seed_all`, `hcfg` HebbianConfig factory, `fake_sentence_transformers`, `lua_root` |
| `test_paper_defaults.py` | RLConfig == paper Table 6; HebbianConfig == paper Table 7; comm-reward constants (pins inconsistency #1) |
| `test_hebbian_coactivity.py` | engagement g_i, spatial gate (radius d, inclusive boundary), comm bonus δ_comm, ε-floor, chamber gate |
| `test_hebbian_update.py` | hand-computed growth/decay, `_growth_coeff` variants, directed asymmetry, failure-gated decay windows, homeostatic λ decay + fixed point, death exclusion, clip/diag invariants |
| `test_hebbian_graph_api.py` | presets, init matrices, reward-diffusion hand cases, replay sampling, metrics, to_dict/from_dict, reset, bond deltas, legacy mode, disabled no-ops |
| `test_trajectory_buffer.py` | GAE hand cases (incl. done-truncation, global-value fallback), advantage standardisation + single-transition guard, pending mechanics, reward sanitisation, batching |
| `test_heads_and_anneal.py` | RunningMeanStd (Welford), ValueHead/ActionHead structure, entropy anneal endpoints + `anneal_steps<=0` quirk |
| `test_centralized_critic.py` | joint-state encoding layout (uses the fake sentence-transformer), critic-buffer GAE, update()/save/load, learning sanity |
| `test_communication_rewards.py` | message validity gates, rate limit, cap (50 msgs = 25.0), chamber z-bands, comm milestones, bad-target semantics |
| `test_cooperation_metric.py` | 5-plane pair tensor thresholds, damage routing, joint-kill lookback, Gini, comm efficacy, chamber performance/fairness, cooperation score |
| `test_coop_comm_eval.py` | post-hoc credit splitting (equal + damage share with id normalisation), credit Gini, cross-episode aggregation, comm entropy/MI |
| `test_action_space.py` | raw/named action spaces, alias recovery, canonicalize fallbacks, pitch/sustain/idle constants, RL candidate masking (22−8 Slot = 14) |
| `test_lua_spec.py` | regex spec tests over the Lua mod: milestone ids/rewards vs paper Table 2, switch rotation, boss HP, death penalties, anvil constants, Python↔Lua milestone-id drift guard |

## Conventions

- Tests pin **actual code behaviour** with hand-computed expected values; where code and the
  thesis paper diverge, the test asserts the code and the divergence is recorded in
  `docs/PAPER_INCONSISTENCIES.md` — never silently reconciled.
- The Lua game logic is not executed; `test_lua_spec.py` parses the `.lua` sources and pins
  constants/ids, which doubles as a cross-language drift guard (it caught two of the bugs
  below).
- Heavy dependencies are faked at the `sys.modules` level (`pettingzoo`, `craftium`) or via
  fixture (`sentence_transformers`); `autogen_*`/`chromadb`/`wandb` are deliberately NOT
  stubbed globally so accidental heavy imports fail loudly.

## Source fixes that came out of this suite

All six are in the metrics/analysis layer or numerical guards — none changes the reward
stream delivered during any collected run (full entries: `docs/PAPER_INCONSISTENCIES.md` #14):

1. `src/mindforge/env/cooperation_metric.py` — `_CH2_ANVIL_PREFIXES` `("m8_", "m11_")` →
   `("m8_", "m9_")`: the Lua mod renamed the second anvil to `m9_anvil_B1`, so Ch2
   cooperation performance could never exceed 0.5.
2. `src/rl_layer/trajectory_buffer.py` — GAE advantage standardisation NaN'd a
   single-transition buffer (torch unbiased std); now skipped below 2 transitions.
3. `src/hebbian/graph.py` — `reset()` now clears and re-seeds `_W_history`; previously
   post-reset bond deltas were computed against stale pre-reset snapshots.
4. `src/mindforge/agent_modules/coop_eval.py` — damage-share credit ids normalised
   (`_agent_key`): '0'/'1' damage keys never matched 'agent0'/'agent_0' contributors, so
   damage-share always silently degraded to an equal split.
5. `src/mindforge/agent_modules/craftium_metric.py` — `m_door1_open` (reward 50) added to
   `MILESTONE_TRACK`/`TRACKS`; it was silently dropped from every ch1_solo aggregate.
6. `make_results.py` — same `m_door1_open` gap closed in the mirrored table.

Known issues found but deliberately **not** fixed (they change environment/reward behaviour
relative to collected runs): stale Ch4 zombie spawn positions and stale Python chamber
z-bands — see `docs/PAPER_INCONSISTENCIES.md` #16.
