# CLI flag audit (ICLR release)

Scope: all 99 flags of `src/mindforge/cli.py` on branch `iclr-anon`, after the orchestrator was
reduced to the villager variant. For each flag the audit checked whether a launcher behind a v6
result sets it, whether the code reads it, which tests touch it, and whether `analysis/` reads its
value back from a run's `config.json`.

Two facts frame every recommendation below.

- **Removing a flag cannot change a reported number.** Every analysis read of a flag value uses
  `cfg.get(key, default)` with the same default as the CLI (`analysis/replay_hebbian_terms.py:151-162`,
  `analysis/qualitative/qual_lib/episode_io.py:168-169`). The only cost is that future
  `config.json` files stop recording that value.
- **Removing a flag changes what a reviewer can vary.** Flags that carry a Table 7 or Table 8
  hyperparameter stay, even when every launcher leaves them at their default.

Prefix matching is now off (`allow_abbrev=False`). With it on, the stale
`--orchestrator-mode bias` parsed silently as `--orchestrator-model bias`.

## Summary

| Class | Flags | Recommendation |
|---|---|---|
| A. Set by a paper launcher | 48 | Keep |
| B. Set through an environment variable | 2 | Keep |
| C. Paper hyperparameter left at its default | 12 | Keep |
| D. Communication budget (kept feature) | 4 | Keep |
| E. Rule internals not in the paper | 3 | Keep, document |
| F. Operational knobs | 15 | Keep |
| G. Gate code that is being deleted | 15 | Remove with the code, plus two choices of `--hebbian-mode` |
| **Total** | **99** | 15 removals |

## A. Set by a paper launcher — keep (48)

`--num-agents`, `--team-scaling`, `--ch4-mob-count`, `--episodes`, `--max-steps`,
`--simultaneous`/`--no-simultaneous`, `--warmup-time`, `--wandb`, `--wandb-project`, `--wandb-tags`,
`--wandb-upload-artifacts`, `--seed`, `--rl`, `--rl-model-path`, `--rl-update-interval`, `--rl-lr`,
`--rl-critic-mode`, `--hebbian`, `--hebbian-mode`, `--hebbian-eta-0`, `--hebbian-eligibility-rho`,
`--hebbian-coact-floor`, `--hebbian-death-ltd`, `--hebbian-death-cap`, `--hebbian-reward-norm`,
`--hebbian-decay`, `--hebbian-rho`, `--hebbian-gamma`, `--hebbian-freeze`, `--hebbian-preset`,
`--hebbian-bond-strong`, `--hebbian-bond-weak`, `--max-chamber`, `--start-chamber`,
`--hebbian-init-file`, `--agent-state-init`, `--social-module`, `--social-interval`,
`--social-act-mode`, `--social-acts`, `--cofiring-channels`, `--social-bidirectional`,
`--comm-distance-free`, `--social-act-rewards`, `--orchestrator`,
`--orchestrator-node-timeout-steps`, `--experiment-id`, `--tag`.

## B. Set through an environment variable — keep (2)

| Flag | How it is set |
|---|---|
| `--rl-update-stagger` | Defaults from `RL_UPDATE_STAGGER`, which the Gemma RL submissions export |
| `--run-group` | Defaults from `WIREDTOGETHER_RUN_GROUP`, which `_common.sh` forwards into the container |

## C. Paper hyperparameter left at its default — keep (12)

| Flag | Paper symbol | Default |
|---|---|---|
| `--hebbian-radius` | d | 5.0 |
| `--hebbian-alpha` | α | 0.5 |
| `--hebbian-delta` | δ_k | 0.5 (None → 0.5) |
| `--hebbian-coop-eps` | ε_c | 0.05 |
| `--hebbian-init-weight` | W₀ | 0.1 |
| `--hebbian-eta-plus` | η₊ | 0.05 |
| `--rl-lora-rank` | LoRA r | 8 |
| `--rl-prompt-max-tokens` | actor encoding cap | 512 |
| `--orchestrator-decompose-min-interval` | T_soc for the orchestrator | 8 |
| `--orchestrator-variant` | — (records `villager` in the config) | villager |
| `--belief-interval` | belief refresh; read back by the qualitative pipeline | 5 |
| `--critic-interval` | "a critic evaluates task success every twenty steps" | 20 |

## D. Communication budget — keep (4)

`--comm-budget-tokens`, `--comm-budget-msg-cap`, `--comm-reward-scale`, and `--no-communication`
(the budget validator rejects a budget on a muted channel). Kept with the module, its launchers
and `tests/test_comm_budget.py`.

## E. Rule internals not in the paper — keep, document (3)

`--hebbian-eta-minus`, `--hebbian-coop-window`, `--hebbian-neg-theta` configure a failure-gated
decay inside the three-factor rule (`src/hebbian/graph.py:564,675-676`) that applies to pairs that
are both quiet and failing. It is live code, but earlier analysis found it never fires in practice.
Keep the flags, and either add the three values to Table 8 or say in `docs/hebbian-graph.md` that
the term exists and is inactive at these settings.

## F. Operational knobs — keep (15)

`--obs-width`, `--obs-height`, `--sleep-time`, `--no-gif`, `--gif-dir`, `--gif-interval`,
`--log-interval`, `--wandb-entity`, `--wandb-id`, `--orchestrator-max-open-tasks`,
`--orchestrator-model`, `--orchestrator-log-dir-name`, `--checkpoint-dir`, `--checkpoint-interval`,
`--ch1-timeout-steps`.

None carries experimental meaning, and removing them buys nothing. `--checkpoint-interval` must
stay: the periodic save writes the `agent_state/` the transplant experiment consumes.
`--wandb-entity` has no default, so it identifies nobody.

## G. Gate code that is being deleted — remove with the code (15)

| Flags | Code they gate |
|---|---|
| `--interpretability` | Nothing — parsed and never read |
| `--rl-auto-token-opt`, `--rl-mode` | `src/rl_layer/token_opt.py` and `prompts/learning_belief.txt` |
| `--voxel-obs` | Voxel observations, never used |
| `--resume`, `--resume-skip-warmup`, `--checkpoint-frames` | `checkpointing.load_checkpoint` (old chained cluster jobs) |
| `--team-mode`, `--homogeneous-role`, `--roles` | OpenWorld role prompts (hunter, harvester, scouter) |
| `--hebbian-ltp`, `--hebbian-ltd`, `--hebbian-beta` | The `legacy` Hebbian mode (`graph.py:176-368`) |
| `--hebbian-hub` | The `star` / `ring` topology presets (only `uniform` and `pair` are used) |
| `--hebbian-no-comm-bond` | δ_comm = 0 ablation, not reported |
| `--hebbian-mode` choices `legacy`, `coactivity` | Keep the flag; drop these two choices |

`--roles` still appears in three tests as a stub attribute; remove it there too.

## Defaults do not match the paper

Every paper launcher passes the rule's settings explicitly, so no reported result is affected. But
the defaults are the older single-timescale rule, so `multi_agent_craftium.py --hebbian` with no
other flags does **not** run the rule in Table 8.

| Setting | CLI default | `HebbianConfig` default | Paper (Table 8) |
|---|---|---|---|
| `--hebbian-mode` | reward_modulated | reward_modulated | three_factor |
| `--hebbian-eta-0` (η₀) | 0.01 | 0.01 | 0.001 |
| `--hebbian-reward-norm` (R) | 300 | 300 | 50 |
| `--hebbian-decay` (λ) | 0.005 | 0.0003 | 0.001 |
| `--hebbian-death-ltd` (η₋ᵈ) | 0.0 | 0.0 | 0.05 |
| `--hebbian-rho` (ρ, RLFT only) | 0.0 | 0.0 | 0.3 |

`tests/test_paper_defaults.py` pins the `HebbianConfig` column, so it guards an older paper
version, and the CLI and config disagree on λ.

Recommendation: make the paper's rule the default. That means flipping the six defaults above in
`cli.py` and `HebbianConfig`, updating `test_paper_defaults.py` to Table 8, and adding
`--hebbian-mode reward_modulated` plus its old η₀/R/λ explicitly to the launchers that relied on
the old defaults: `exp09`–`exp11`, `expA_pair_bonding`, `new_exp_0_gemma.sbatch` (HEBBIAN=1 path),
`new_exp_pareto.sbatch`. Without those launcher edits, flipping the defaults would change what
those launchers run.
