# Medium suite — extended-seed aggregation (PORTED to the paper 2026-07-17)

**Generated:** 2026-07-17, from local `runs_from_daic/medium_runs/` (updated sync).
**Command:** `python make_results.py --runs-root runs_from_daic/medium_runs --out paper_assets_medium_ext --max-steps 1000`
**Baseline compared against:** `paper_assets_medium/summary.csv` (the 3-seed
aggregation the paper's results section was filled from).

> **STATUS: 65/66 runs pooled — PORTED to the paper 2026-07-17** (results.tex
> tables/text updated, new PDFs copied to paper figures/, setup text now says
> 6 seeds, n=18, IPPO+Heb n=15).
> Pre-registered extension (2026-07-07): seeds 789/1011/1213 added to all 11
> conditions + exp07/seed_42 backfill → target 66 runs. Only 1 run is still
> missing: **exp06_ippo_hebbian/seed_1213** (resubmitted 2026-07-17, in
> flight) — when it lands, re-run make_results.py and re-port the IPPO+Heb
> row + steps-table column. Unlike the 2026-07-14 snapshot, **both RL+Hebbian
> conditions now have extension data**, so the decisive RL vs RL+Heb
> comparison can be evaluated.

## Coverage

| Condition (label) | dir | seeds pooled | n | missing |
|---|---|---|---|---|
| LLM-2B | exp01_llm_2b | all 6 | 6 | — |
| LLM-9B | exp02_llm_9b | all 6 | 6 | — |
| LLM-2B+Heb | exp07_llm_2b_social_prompt | all 6 | 6 | — |
| LLM-9B+Heb | exp08_llm_9b_social_prompt | all 6 | 6 | — |
| IPPO | exp04_ippo | all 6 | 6 | — |
| MAPPO | exp03_mappo | all 6 | 6 | — |
| IPPO+Heb | exp06_ippo_hebbian | 42,123,456,789,1011 | 5 | 1213 (in flight) |
| MAPPO+Heb | exp05_mappo_hebbian | all 6 | 6 | — |
| No-bonds | exp11_llm_9b_allied_none | all 6 | 6 | — |
| Allied-pair | exp10_llm_9b_allied_pair | all 6 | 6 | — |
| Allied-all | exp09_llm_9b_allied_all | all 6 | 6 | — |

Episodes pooled = 3 per run (medium config: 3 eps × ≤1000 steps).

## How the metrics are computed (`make_results.py`)

All numbers come from each run's `final_metrics.json` (schema:
`src/mindforge/agent_modules/craftium_metric.py`). Global steps are mapped to
episodes via `episode_lengths` (cumulative `[start, end)` bounds); "within-episode
step" = global step − episode start.

**Aggregation.** Unless noted, statistics are **pooled at the episode level over
all seeds** (6 seeds × 3 eps = 18 episodes), reported as mean ± *population* std
(`statistics.pstdev`). The CSV additionally has per-seed means
(`coop_seedmean/seedstd` = mean over seeds of each seed's episode mean).

- **Coop. milestones** — per episode, the **distinct team milestone set** (union
  over the 3 agents of `milestones_per_episode`), counting only milestones in the
  cooperative tracks ch2_anvils, ch3_switches, ch4_combat, ch5_boss. Chamber 1
  (solo) and the communication track are excluded.
- **All milestones** — size of the same distinct team set, all tracks included
  (Ch1 + comm too).
- **Task return** — per episode, summed over agents and steps from
  `reward_history_decomposed`: `task + comm_base + comm_milestone` per record,
  assigned to episodes by the record's global step. **Excludes
  `hebbian_diffuse`** so Hebbian and non-Hebbian conditions are comparable
  (`task_is_decomposed=True` for every run in this aggregation; the
  `per_episode_returns` fallback, which would include diffuse, was never used).
- **Steps-to-milestone (M8/M15/M18/M23 …)** — from `milestone_events`: the
  within-episode step of the **first** fire of that milestone in each episode;
  reported as median over completing episodes, with n = number of completing
  episodes (subscript in the LaTeX table). Paper ids are renumbered from repo
  ids: M8=`m8_anvil_A1`, M15=`m19_all_in_communal`, M18=`m22_all_mobs_killed`,
  M23=`m27_boss_defeated`.
- **Furthest-chamber mode** — per episode, the highest chamber number whose
  track contains any fired team milestone; the reported value is the **mode**
  across pooled episodes.
- **Entry-milestone honesty filter (added 2026-07-17)** — `m20_enter_ch4` /
  `m24_enter_ch5` are position-triggered in Lua and leak through
  episode-reset / rescue-teleport races (suppression flags cleared while
  agents still stand in Ch4/Ch5; in some RL runs the Lua reset desyncs
  entirely — `lua_step` ≫ episode length). An entry event is kept only if its
  episode also fired the enabling milestone (`m18_door_opened` for Ch4,
  `m22_all_mobs_killed` for Ch5); otherwise it is removed from the event list,
  the per-episode milestone sets, and the episode task return (its +30/+50
  per-contributor rewards). In this suite **no episode has m18/m22**, so all
  m20/m24 fires (RL conditions only, 13 event-episodes) were artifacts and are
  excluded. LLM and topology conditions are unaffected.
- **Boss defeats** — count of pooled episodes containing `m27_boss_defeated`.
- **Credit Gini** — per run, from `coop_metrics.milestone_credit_total`.
  NB: the stored `credit_gini` is wrong (the log mixes `agent_0`/`agent0` key
  styles for the same agent → 2N phantom agents), so keys are merged by agent
  index and the Gini recomputed: Σᵢⱼ|cᵢ−cⱼ| / (2n·Σc). Mean ± std over runs with
  positive total credit.
- **Excl. share** — same merged credit vector: `agent_2`'s fraction of total
  milestone credit (agent_2 is the excluded agent in Allied-pair; 1/3 ≈ fair).
- **Graph stats** (Hebbian conditions) — from the **last `graph_snapshots` entry
  within each episode**, pooled over episodes × seeds: *bond mean* = logged
  `mean_bond_strength`; *bond max* = max off-diagonal Wᵢⱼ; *sparsity* = fraction
  of off-diagonal Wᵢⱼ below the warm-start weight 0.1; *asymmetry* = mean
  |Wᵢⱼ − Wⱼᵢ| over unordered pairs.

## Main results — old (paper baseline) → new (extended seeds)

Coop. milestones = distinct team Ch2–Ch5 milestones per episode (pooled mean ± pooled
std); Task return = team task+comm return, excl. hebbian_diffuse (decomposed in all runs).

| Condition | seeds | Coop. milestones | All milestones | Task return |
|---|---|---|---|---|
| LLM-2B | 3→6 | 1.78 ± 0.63 → **1.56 ± 0.83** | 10.33 → 9.78 | 438 → **358** |
| LLM-9B | 3→6 | 3.00 ± 0.94 → **3.00 ± 1.05** | 12.22 → 12.00 | 613 → **622** |
| LLM-2B+Heb | 2→6 | 1.00 ± 1.00 → **1.17 ± 0.96** | 9.00 → 9.28 | 354 → **363** |
| LLM-9B+Heb | 3→6 | 2.78 ± 0.92 → **2.39 ± 0.95** | 11.56 → 10.94 | 588 → **574** |
| IPPO | 3→6 | 1.33 ± 0.94 → **0.78 ± 0.97** | 8.78 → 6.17 | 442 → **255** |
| MAPPO | 3→6 | 1.11 ± 1.10 → **0.78 ± 0.97** | 9.33 → 8.72 | 416 → **390** |
| IPPO+Heb | 3→5 | 1.00 ± 1.15 → **0.53 ± 0.88** | 8.78 → 5.47 | 409 → **200** |
| MAPPO+Heb | 3→6 | 0.67 ± 0.94 → **0.56 ± 0.83** | 9.22 → 7.61 | 454 → **360** |

(RL rows include the entry-milestone honesty filter — the old 3-seed paper
baseline and the pre-filter 2026-07-17 numbers both still contained artifact
m20/m24 fires.)

Furthest-chamber modes (new): LLM-9B 4; all other LLM/topology conditions 3; all
four RL conditions 1 (paper baseline had IPPO at 3 — the new-seed RL runs are
weaker). Boss defeats: still **0** in every condition. Of the four key milestones
only M8 (first anvil) is ever reached: LLM-9B median 201 (n=2), LLM-9B+Heb 301
(n=1), No-bonds 269 (n=1) — unchanged from the 55-run snapshot.

## RL vs RL+Hebbian — the decisive comparison (now evaluable)

| Pair | Coop. milestones | All milestones | Task return |
|---|---|---|---|
| IPPO 0.78 ± 0.97 vs IPPO+Heb 0.53 ± 0.88 | Heb −0.24 | 6.17 vs 5.47 | 255 vs 200 |
| MAPPO 0.78 ± 0.97 vs MAPPO+Heb 0.56 ± 0.83 | Heb −0.22 | 8.72 vs 7.61 | 390 vs 360 |

On the extended seeds, **Hebbian bonds do not help either RL backbone** — both
+Heb cells sit below their plain-RL counterparts on every headline metric, with
overlapping (large) stds. Note IPPO+Heb is still 5/6 seeds; seed_1213 could
move its row.

## Topology ablation — old → new

| Condition | seeds | Coop. milestones | Task return | Credit Gini | Excl. share |
|---|---|---|---|---|---|
| LLM-9B (ref.) | 3→6 | 3.00 → 3.00 | 613 → 622 | 0.08 → 0.08 | 0.31 → 0.32 |
| No-bonds | 3→6 | 2.33 → **2.17** | 580 → 572 | 0.06 → 0.07 | 0.35 → 0.36 |
| Allied-pair | 3→6 | 1.89 → **1.89** | 549 → 542 | 0.05 → 0.06 | 0.30 → 0.30 |
| Allied-all | 3→6 | 2.11 → **2.28** | 557 → 573 | 0.06 → 0.07 | 0.32 → 0.31 |

Ordering preserved: Allied-all ≈ No-bonds > Allied-pair, all below the plastic-graph
reference. Allied-pair's excluded-agent share (0.30) still ≈ its 1/3 fair share.

## Graph stats (Hebbian conditions)

| Condition | bond mean | bond max | sparsity | asymmetry |
|---|---|---|---|---|
| LLM-2B+Heb (6 seeds) | 0.23 | 0.28 | 0.11 | 0.004 |
| LLM-9B+Heb (6 seeds) | 0.24 | 0.29 | 0.04 | 0.005 |
| IPPO+Heb (5 seeds) | 0.13 | 0.18 | 0.56 | 0.001 |
| MAPPO+Heb (6 seeds) | 0.19 | 0.24 | 0.26 | 0.001 |

The RL+Heb graphs are markedly weaker and sparser than the LLM+Heb ones
(bond mean 0.13–0.19 vs 0.23–0.24; sparsity up to 0.56 for IPPO+Heb).

## Observations

1. **LLM-9B is rock-stable at 3.00 coop milestones** across 3→6 seeds; the paper's
   top-line condition is robust to the seed extension.
2. **LLM-9B+Heb declined (2.78 → 2.39)**: the gap to plain LLM-9B widened from
   −0.22 to −0.61 pooled. The new seeds weaken the "social prompt helps 9B"
   reading. LLM-2B+Heb, conversely, ticked up slightly (1.00 → 1.17) and its
   furthest-chamber mode recovered to 3.
3. **All four RL conditions got worse and noisier on the new seeds** (and the
   honesty filter removed 13 artifact entry-milestone episodes, all RL): IPPO
   1.33→0.78 with task return −42% (442→255); IPPO+Heb collapsed hardest
   (1.00→0.53, task 409→200, all-milestones 8.78→5.47). All RL furthest-chamber
   modes are now 1.
4. **The decisive comparison now reads negative**: on pooled extended seeds,
   IPPO+Heb < IPPO and MAPPO+Heb < MAPPO on coop milestones, all milestones,
   and task return alike (see table above). Pending exp06/seed_1213 for the
   final IPPO+Heb row.
5. Topology ablation moved little; Allied-all slightly up, No-bonds slightly
   down — the ranking argument is unchanged.
6. Nothing new fires beyond Chamber 3 in any condition (M15/M18/M23 all n=0;
   0 boss defeats) — consistent with the 1000-step medium budget.

## Missing runs (1) blocking the full 66-run aggregation

- `exp06_ippo_hebbian/seed_1213` — resubmitted 2026-07-17 (RUN_GROUP=legacy,
  3 ep × 1000), in flight. See `runs_from_daic/medium_runs/STILL_TO_COPY.md`
  for copy commands.

## Regenerating

```bash
# after copying newly finished runs into runs_from_daic/medium_runs/:
python make_results.py --runs-root runs_from_daic/medium_runs \
    --out paper_assets_medium_ext --max-steps 1000
```

Outputs in this directory: `summary.csv` (machine-readable, per-condition),
`table_rows.tex` (LaTeX rows), `milestone_progression.pdf`,
`milestone_timeline.pdf`, `bond_evolution.pdf`. The paper's current numbers
remain sourced from `paper_assets_medium/` (untouched).
