# WiredTogether test suite

Verification suite for `src/`: 600 tests, all runnable locally on CPU with no game binary,
no model weights and no network. They pin the hyperparameter defaults against the paper's
tables, the Hebbian update arithmetic against hand computations, the reward ledger, and the
Lua and Python milestone tables against each other.

## Running

```powershell
# from the repo root — conftest handles sys.path; PYTHONPATH not required
python -m pytest tests -q          # full suite, ~25 s
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
| `test_paper_defaults.py` | RLConfig and HebbianConfig against the paper's hyperparameter tables; comm-reward constants |
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
| `test_gemma4_compat.py` | Gemma 4 shims: `LLM_VISION_MODE` override, multimodal auto-class order, `dtype`/`torch_dtype` rename, system-role merge for templates that reject it, text-tower hidden size (stubs `autogen_core` module-locally) |
| `test_lua_spec.py` | regex spec tests over the Lua mod: milestone ids/rewards vs paper Table 2, switch rotation, boss HP, death penalties, anvil constants, Python↔Lua milestone-id drift guard |

## Conventions

- Tests pin **actual code behaviour** with hand-computed expected values. Where the code and
  the paper diverge, the test asserts the code — never the other way round.
- The Lua game logic is not executed; `test_lua_spec.py` parses the `.lua` sources and pins
  constants/ids, which doubles as a cross-language drift guard.
- Heavy dependencies are faked at the `sys.modules` level (`pettingzoo`, `craftium`) or via
  fixture (`sentence_transformers`); `autogen_*`/`chromadb`/`wandb` are deliberately NOT
  stubbed globally so accidental heavy imports fail loudly.

## Known divergences

Two issues are deliberately left unfixed, because changing either would alter the environment
relative to every run already collected: stale Ch4 zombie spawn positions, and Python-side chamber
z-bands that lag the Lua geometry.
