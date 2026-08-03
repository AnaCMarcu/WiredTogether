# Metrics and Evaluation

**Source files:** `src/mindforge/agent_modules/craftium_metric.py`, `src/mindforge/agent_modules/coop_eval.py`, `src/mindforge/agent_modules/comm_eval.py`, `src/mindforge/agent_modules/_metric_summary.py`, `src/mindforge/agent_modules/_metric_plots.py`, `src/mindforge/env/cooperation_metric.py`, `src/mindforge/env/episode_logger.py`, `make_results.py`
**Paper sections:** §Results (tab:main_results, tab:steps_to_milestone, tab:topology_ablation, tab:graph_stats; fig:milestone_progression, fig:milestone_timeline, fig:bond_evolution), §A.3 milestones, App. comm-metric references (Lowe 2019, Eccles 2019)
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

## 1. Online vs post-hoc split

| Stage | Component | Records / reads | Output |
|---|---|---|---|
| Online (per step) | `CraftiumMetric` (craftium_metric.py:239) | rewards, milestones, comm events, RL updates, W snapshots | in-memory; serialised to `final_metrics.json` |
| Online (per step) | `CooperationMetric` (cooperation_metric.py:21) | positions/actions/messages/damage — observer only, no reward effect | per-episode `summary.json` via `episode_summary()` |
| Online (per step) | `CommunicationTracker` (env/communication_rewards.py:31) | message validity, comm rewards (see 06-rewards.md) | rewards + `m_comm_*` milestone fires |
| Online (per step) | `EpisodeLogger` (episode_logger.py:46) | raw step/event/message streams | `episodes/ep_*/` files (§6) |
| Post-hoc (run end) | `coop_eval.py`, `comm_eval.py` | the `episodes/ep_*/` files | `coop_metrics` / `comm_metrics` keys in `final_metrics.json` |
| Post-hoc (run end) | `_metric_summary.py`, `_metric_plots.py` | CraftiumMetric in-memory state | `summary.txt`, `plots/*.png` |
| Post-hoc (cross-run) | `make_results.py` (repo root) | `runs/final/<cond>/seed_*/final_metrics.json` | LaTeX rows, CSV, PDFs (§7) |

All three online recorders plus the logger are wired in the orchestrator loop (multi_agent_craftium.py:1237-1239; see 07-orchestrator.md). Post-hoc evaluators run inside `save_run_metrics()` via `_run_posthoc_evaluators` (craftium_metric.py:593-616); failures degrade to `{}` with a warning.

**Caveat — diffused returns.** `metric.record_reward()` receives the *post-diffusion* reward: the orchestrator computes `diffused_rewards = hebbian_graph.diffuse_rewards(step_rewards_raw)` and books that value (multi_agent_craftium.py:2098, 2208-2209). So `cumulative_returns` / `per_episode_returns` include the gamma_d Hebbian transfer term. `record_reward_decomposed` (craftium_metric.py:370-395) stores the five streams (task, comm_base, comm_milestone, proximity [vestigial, always 0], hebbian_diffuse) that sum to the recorded value — `make_results.py` uses this to report task return *excluding* hebbian_diffuse (§7). Cross-link: 06-rewards.md.

## 2. CooperationMetric (online, per episode)

### 2.1 Five-plane NxN pair-interaction tensor

Built in `observe_step` / `observe_kill` (cooperation_metric.py:57-165), serialised by `episode_summary` as `pair_interaction` (cooperation_metric.py:368-374) via `_pair_to_matrix` (canonical agent_ids ordering, cooperation_metric.py:338-349).

| Plane | Trigger | Threshold | Symmetric? | Anchor |
|---|---|---|---|---|
| `messages` | routed message sender->receiver (from `infos["routed_messages"]`) | — | no (directed) | :137-141 |
| `joint_dig` | both agents `Dig` same step | distance < 3.0 blocks | yes | :83-93 |
| `proximity` | pair within range, any step | distance < 4.0 blocks | yes | :62-73 |
| `joint_kill` | both damaged the dead target within lookback | <= 5 steps before kill | yes | :143-157 |
| `ch5_damage_overlap` | same as joint_kill, target == "boss" | same 5-step lookback | yes | :159-164 |

Also tracked per agent: per-chamber `dwell_steps` and per-chamber `action_hist` (cooperation_metric.py:95-103), chamber bounds = z-slices (cooperation_metric.py:12-18); ch4/ch5 per-agent damage from `infos["damage_events"]` (:121-135).

### 2.2 Gini

`_gini` (cooperation_metric.py:201-208): `G = 2*sum_i(i*v_(i)) / (n*sum v) - (n+1)/n` on sorted values; returns 0 on empty/all-zero input. Single earner among N agents gives G = (N-1)/N (e.g. 0.667 for N=3), not 1.0.

### 2.3 Per-chamber performance and fairness

`_chamber_performance` (cooperation_metric.py:255-272), `_chamber_fairness` (:274-290). Fairness = 1 − Gini of the per-agent contribution dict (all agents keyed, zeros included, :241-253).

| Chamber | Performance (clipped to [0,1]) | Fairness input |
|---|---|---|
| ch2 | count of `m8_`/`m9_` anvil milestones / 2 | anvil milestone contributor counts |
| ch3 | count of `m17_` + `m18_` + `m19_` milestones / 7 | `m17_` switch-press contributor counts |
| ch4 | sum ch4_damage / 60 (3 zombies x 20 HP) | per-agent ch4 damage |
| ch5 | sum ch5_damage / 60 (boss HP) | per-agent ch5 damage |

`_CH2_ANVIL_PREFIXES = ("m8_", "m9_")` (cooperation_metric.py:230) — the m9 prefix is the post-commit fix (was stale `m11_`, which silently zeroed ch2 performance); pinned by tests/ (see PAPER_INCONSISTENCIES.md #14).

### 2.4 Cooperation score (5 components)

`_cooperation_score` (cooperation_metric.py:314-336) = mean of 5 components, each in [0,1]:
1. comm_efficacy (§3);
2-5. per cooperative chamber ch2..ch5: `performance x fairness`, but **0 if the chamber was never entered** (entry tracked at :114-119) — unreached chambers cannot inflate the score via vacuous fairness. Ch1 is excluded (solo chamber). `_cooperation_breakdown` (:292-312) exposes the per-component values; `sum/5 == cooperation_score`.

`episode_summary` (cooperation_metric.py:351-380) also emits proximity/co-action/joint-dig totals, messages_per_agent, chamber_entry_steps, ch4/ch5 damage Ginis, the milestone log, carry_imbalance (max-min contributions, :216-225), and the final Hebbian bond matrix W as `hebbian_W`.

## 3. Communication efficacy (online)

`_comm_efficacy` (cooperation_metric.py:210-214): of milestones with `contributor_count >= 2`, the fraction whose `comm_before_coop` flag is set — i.e. some contributor sent a message of >= 5 stripped chars (:107) within the 10 steps before the milestone fired (rolling buffer pruned at `step - s <= 10`, :110-112; flag set in `observe_milestone`, :177-188). Returns 0.0 if no multi-contributor milestones occurred.

## 4. coop_eval (post-hoc)

`compute_coop_metrics(run_root, num_agents)` (coop_eval.py:98-206) walks `episodes/ep_*/`: sums the five pair-tensor planes, dwell, action histograms, and per-agent ch4/ch5 damage across episodes, then credits milestone rewards from `event_log.jsonl` events of type `milestone`/`comm_milestone`.

Credit rules (`_CREDIT_RULES`, coop_eval.py:87-95; default = equal split among contributors):

| Milestone | Rule |
|---|---|
| m22_all_mobs_killed | ch4 damage share |
| m25_first_boss_dmg, m26_boss_half_hp, m27_boss_defeated | boss (ch5) damage share |
| m19_all_in_communal, m23_all_alive_ch4, m28_all_alive_bonus | equal |
| all others | equal |

Damage-share credit (`_credit_damage_share`, coop_eval.py) splits by *run-total* damage for that target class, renormalised over contributors; falls back to equal split when no damage info exists. Heterogeneous agent ids are normalised via `_agent_key` ('agent0' Lua names, 'agent_0' Python names, '0' JSON-stringified damage keys → int index) — a post-commit fix; before it the lookup never matched and damage-share silently degraded to equal split (PAPER_INCONSISTENCIES.md #14, pinned by `tests/test_coop_comm_eval.py::test_damage_share_real_pipeline_keys_match`). Outputs `milestone_credit` (per agent per milestone), `milestone_credit_total`, and `credit_gini` (same Gini formula) — the fairness column of tab:topology_ablation (§7).

## 5. comm_eval (post-hoc)

`compute_comm_metrics(run_root, num_agents)` (comm_eval.py:129-233) reads `messages.jsonl` + `step_log.csv` across episodes.

| Key | Measures | Anchor |
|---|---|---|
| `speaker_consistency[agent].entropy` | H(M_i) over message clusters — low = consistent vocabulary | :177-191 |
| `speaker_consistency[agent].mi_chamber` | I(M_i; chamber_i) — are messages situation-dependent (positive signaling) | :177-191 |
| `instantaneous_coord["i->j"]` | I(M_t^sender; A_{t+1}^receiver) per directed pair (Lowe 2019 IC) | :193-212 |
| `pair_message_count` | asymmetric NxN routed who-talks-to-whom counts | :162-171 |
| `routing_breakdown` | counts per routing mode: model / hebbian_fallback / random_fallback | :164-172 |
| `total_tokens`, `tokens_per_milestone` | token cost; total tokens / number of milestone events (None if zero) | :214-222 |

Clustering (`_cluster_messages`, comm_eval.py:78-100): K-means (K=16) on sentence-transformer embeddings (`ST_MODEL_NAME`, default all-MiniLM-L6-v2); **offline fallback** hashes (first word, length//5 bucket) into K bins when ST/sklearn are unavailable — same metric definitions, cruder bins.

## 6. EpisodeLogger file schemas

Written to `runs/<run_id>/episodes/ep_{N:04d}/` (episode_logger.py:46-64). Supports passive callbacks (`on_step`/`on_event`/`on_finalize`, exceptions swallowed, :137-159).

| File | Format | Schema | Anchor |
|---|---|---|---|
| `step_log.csv` | one row per (step, agent) | step, agent_id, chamber, pos_x, pos_y, pos_z, action, reward_task, reward_comm, wielded_item, hp, message | :52-94 |
| `event_log.jsonl` | one JSON object per event | free-form; milestone events carry type, milestone, contributors, reward (consumed by coop_eval) | :99-103 |
| `messages.jsonl` | one JSON object per message | t, sender, receiver, text, tokens, valid, rewarded_base, rewarded_milestone, chamber, routing | :105-121 |
| `summary.json` + `episode_summary.json` | identical copies | the `CooperationMetric.episode_summary()` dict (§2), incl. `pair_interaction` and `hebbian_W` | :123-135 |

## 7. Results pipeline: final_metrics.json -> make_results.py -> paper assets

`CraftiumMetric.save_run_metrics` (craftium_metric.py:575-591) writes `final_metrics.json` (`_build_metrics_dict`, :618-743: config + git stamp, per-episode returns/milestones/comm counts, track rewards (TRACKS, :58-86), milestone_events, steps_to_milestone, graph_snapshots with W, reward_history_decomposed, comm_metrics, coop_metrics), plus `summary.txt` (_metric_summary.py:16-32), `plots/*.png` (_metric_plots.py:36-49), and `communication_log.json`. Co-completion events use a 5-step window (craftium_metric.py:182, 436-447).

`make_results.py` reads `runs/final/<dir>/seed_*/final_metrics.json` per condition (`CONDITIONS`, make_results.py:57-70): 8 `main` rows (LLM-2B/9B x {plain, +Heb}; IPPO/MAPPO x {plain, +Heb}) + 3 `topo` rows (No-bonds, Allied-pair, Allied-all), `heb` flag selecting tab:graph_stats rows.

Conventions (make_results.py:21-32, `aggregate` :246-303):
- **Coop. milestones** = distinct team milestones in Ch2-Ch5 per episode (union over agents; Ch1 + comm track excluded; `COOP_TRACKS`, :97).
- **Task return** = team sum of task + comm_base + comm_milestone from `reward_history_decomposed` — excludes hebbian_diffuse; falls back to `per_episode_returns` with a `[WARN ... includes hebbian_diffuse!]` flag (:163-181, :539-541).
- **Steps-to-milestone** = within-episode step of first fire; median + n over completing episodes, for `KEY_MILESTONES` (:101-106) m8_anvil_A1 / m19_all_in_communal / m22_all_mobs_killed / m27_boss_defeated, labelled with paper numbers M8/M15/M18/M23.
  > PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #2 (table M-numbers vs code m-ids).
- **Aggregation** pooled over all episodes of all seeds; per-seed means also emitted in CSV.
- **Graph stats** from the last W snapshot per episode: mean bond, max off-diagonal W_ij, sparsity (fraction of off-diagonal W < warm-start 0.1), asymmetry mean |W_ij - W_ji| (:205-234; cf. 02-hebbian-graph.md).

| Output | Content | Paper target |
|---|---|---|
| `table_rows.tex` (:314-358) | rows for all four tables | tab:main_results, tab:steps_to_milestone, tab:topology_ablation (uses credit_gini + excluded-agent `agent_2` credit share), tab:graph_stats |
| `summary.csv` (:361-388) | every number, machine-readable | — |
| `milestone_progression.pdf` (:392-431) | mean +/- std cumulative distinct Ch2-5 milestones vs within-episode step | fig:milestone_progression |
| `milestone_timeline.pdf` (:434-479) | raster of median/min/max first-fire step per milestone, chamber-banded | fig:milestone_timeline |
| `bond_evolution.pdf` (:482-514) | W_ij vs W_ji per pair over one Hebbian run, asymmetry shaded | fig:bond_evolution |

Comm-milestone rewards counted into the `communication` track come from `CHAMBER_COMM_THRESHOLDS` (fires at >= 4 valid messages, communication_rewards.py:22-28, 91).
> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #1 (values), #3 (cap), #11 (threshold wording). Note craftium_metric.py's own `TRACKS["communication"]` tuples (:81-85) carry the *paper* values 40/20/30/15/20 — but no consumer ever reads the reward element of TRACKS (all use `for mid, _ in ...`); booked track rewards come from each event's own `reward` field (craftium_metric.py:428-432).

See 05-five-chambers-world.md for milestone semantics, 06-rewards.md for the reward streams being measured, 11-operations.md for where runs/ land on HPC.

## 8. Qualitative pipeline: analysis_qualitative/ (post-hoc, local)

One CLI, `python analysis_qualitative/run.py <stage>`, stages
`parse | metrics | sample | validate | cases | report` over
`runs_from_daic/medium_runs/`. See `analysis_qualitative/README.md` for
mechanics. Non-obvious facts discovered while building it:

- `final_metrics.json["coop_metrics"]` pair/dwell/damage totals are ZERO for
  all real runs: coop_eval.py reads episode summaries at top level but the
  pipeline nests them under `"cooperation_metrics"`
  (multi_agent_craftium.py finalize call). Read episode summaries directly.
- llm_logs are APPEND-mode: relaunched jobs leave the aborted attempt's lines
  in place (parse marks them `stale`); log.txt can even interleave a
  concurrent different-config job's step markers (exp03/seed_42,
  exp11/seed_456). The parse stage's alignment_report.json records how each
  run was aligned (text-ordinal vs log.txt step-clock) and its cross-checks.
- Step alignment for LLM runs is exact (message text matches step_log 100%);
  RL runs need the log.txt step-marker clock because the routed message text
  is NOT the rl_thoughts unit's communication field.
