---
name: pareto-plots
description: Regenerate the model-size Pareto figures (compute / perception vs performance, base vs Hebbian, Gemma / Qwen / both) in the CoDe dashed-line style, plus the results tables. Use whenever new pareto runs land, a metric or axis needs changing, or a new model size/family is added.
---

# Pareto plots for the model-size sweep

Two scripts, both read the SAME run directories and agree with the paper's
`make_results.py` definitions (they import its helpers rather than re-deriving):

| script | produces |
|---|---|
| `scripts/make_pareto_grid.py` | the grid: {gemma (default; qwen/both on request)} × {reward, milestone_pct, coop_pct, completions, coop_completions} × {flops, grounding, partner_loc}, PNG singles + one composite per family, `points.csv` |
| `scripts/make_pareto_fig.py` | the two log-x headline figures (`--metric`, default `milestone_pct`) + `pareto_results.tex/.md` tables; ALSO the home of `run_metrics()` — the single source of truth the grid imports |

## Standard invocation (from the repo root)

```bash
python scripts/make_pareto_grid.py runs_from_daic/pareto_gemma4 \
    runs_from_daic/new_exp_0_gemma --out-dir paper_assets_pareto/grid

python scripts/make_pareto_fig.py runs_from_daic/pareto_gemma4 \
    runs_from_daic/new_exp_0_gemma \
    --out-dir paper_assets_pareto --csv paper_assets_pareto/points.csv
# Qwen is excluded from the paper's plots by decision. To get the Qwen variants
# anyway, add runs_from_daic/medium_runs as a root and --families gemma,qwen,both.
```

Roots are run-group directories holding `<condition>/seed_<N>/`. The three
above are: the pareto suite (e2b, 12b), the E4B point (`new_exp_0_gemma_*`),
and the Qwen points (`exp01/02` base, `exp07/08` hebbian — same protocol and
flags as the suite, verified against the sbatch files).

Subset flags: `--families gemma,qwen,both`, `--ys coop_pct,reward`, `--xs flops`,
`--no-point-labels`. FLOPs are cached in `<out-dir>/flops_cache.json`
(keyed on run path + log.txt mtime), so only the first run is slow.

## Metric definitions — do not change silently

**Aggregation = the paper's**: every metric is computed PER EPISODE, the
episodes of all seeds are POOLED (6 seeds × 3 = n=18), and mean ± POPULATION
std (`statistics.pstdev`, divide by n) is reported — exactly
`make_results.aggregate`/`mean_std`. Not "mean of seed means", not sample SD.
Verified 2026-08-28: the pareto table reproduces
`paper_assets_final_comparison/final_comparison.md` to the decimal
(Gemma-E4B 521 ± 117 / 16.4 ± 4.8 / 8.5 ± 4.9; LLM-9B 622 ± 150 / 31.1 ± 5.1 /
17.6 ± 6.2). The `n` column is seeds; error bars are over the 18 episodes.

- **reward**: team task return per episode via `make_results.episode_task_returns`
  (task + comm streams, `hebbian_diffuse` excluded, chamber-entry honesty
  filter applied), mean over episodes then seeds.
- **milestone_pct**: distinct team milestones outside the social-act tracks
  / `NONCOMM_MAX` (25) × 100, per episode.
- **coop_pct**: distinct team milestones in Ch2–Ch5 / `COOP_MAX` (17) × 100 —
  the paper's "Coop. milestones".
- **completions / coop_completions** (ATTAINMENT, vs the COVERAGE above):
  per-agent milestone completions summed over agents, per episode. Legitimate
  because `five_chambers.fire_milestone` credits only the agents who earned or
  took part in a milestone (`{name}` for solo/gear/entry, participant lists for
  m22/m25–m28), so this counts agents that got there, not one event ×3.
  Coverage = how far the team got; attainment = how many agents got there.
- NEVER use `final_metrics.mean_milestone_count_per_agent` as a performance
  axis: it INCLUDES the communication milestones (m_comm_ch1–5, ~12–13
  per-agent completions/episode in every arm). The apparent 12B Hebbian
  advantage in the first pareto figure was entirely that comm term — task
  completions were a 9.00 vs 9.00 tie.
- **flops**: `2·N_eff·(prefill+decode tokens)` over all agents/modules/retries
  (`scripts/compute_flops.py`); linear axis in units of 1e17. `N_eff` per size
  lives in `make_pareto_fig.SIZES` — effective params for the Gemma E-series
  (PLE tables are looked up, not multiplied), ACTIVE params for MoE.
- **grounding / partner_loc** (the perception x-axes): the paper's
  `tab:belief_quality` metrics from the QUALITATIVE pipeline —
  *perception-grounding rate* = fraction of perception statements naming no
  object impossible for the agent's current chamber; *partner-location
  accuracy* = fraction of partner-location claims matching the partner's true
  chamber. Read from `analysis_qualitative/out_*/tables/beliefs.csv` (one row
  per run; quarantined rows dropped); mean ± sd across seeds, drawn as x
  error bars. They are PER-ARM values. A new size needs the pipeline run:
  `python analysis_qualitative/run.py parse --runs-root <root> --out analysis_qualitative/out_<name>`
  then `metrics` with the same flags — and its condition dirs registered in
  `make_results.CONDITIONS` (group "pareto") + hebbian dirs in
  `registry.SOCIAL_PROMPT_DIRS`, or the registry will not discover them.
  (MMMU-Pro from the model cards was used in an earlier version and dropped:
  a backbone property, identical for both arms.)

## Strict perception metric + extra figure sets (2026-08-31)

- `perception_grounding_strict` (beliefs_metrics.py): nothing impossible AND
  >=1 CHAMBER-DIAGNOSTIC object named (`lexicons.diagnostic_mentions` — the
  IMPOSSIBLE_OBJECTS table read positively). Fixes the permissive rate's
  degenerate optimum where vague/empty statements pass, which made terse
  Gemma-2B look best. Strict is MONOTONIC in size (base): 2B 0.056 < 4B 0.103
  < 12B 0.232; `perception_specificity` is the diagnostic-mention rate alone.
  These are per-RUN values (n=6, sd across seeds), usable as x AND y axes.
- Perception-as-outcome set: `--out-dir .../grid_perception
  --ys grounding_strict,specificity,grounding --xs flops`.
- Perception-as-X pareto (the thesis's original framing, x=perception,
  y=performance): `--out-dir .../grid_perception_x --ys milestone_pct,reward
  --xs grounding_strict,specificity,grounding,partner_loc`. THE pareto is
  `milestone_pct vs grounding_strict` (both arms rise left-to-right;
  specificity is its robustness twin). The permissive-grounding and
  partner_loc x-axes produce inverted/garbled frontiers — the earlier
  "perception axes dropped from the paper" verdict applied to THOSE; the
  strict metric rehabilitates the axis. Return as y stays non-monotone on any
  perception axis (it peaks at 4B) — use milestones.
- `--join-families`: ONE line per arm through every family's points —
  perception axes only (the flag hard-refuses the compute axis, where a
  joined line would fake a cross-architecture scaling law). On a measured-
  capability x the joined line is the legitimate cross-model correlation.
  THE story figure: grid_perception_x_qwen/
  pareto_both_milestone_pct_vs_grounding_strict.png — milestones vs strict
  grounding, Gemma-2B/4B/12B + Qwen-9B on one line per arm, family kept
  visible via open (Gemma) vs filled (Qwen) markers.
- With-Qwen set (labels "Name-<N>B", e.g. Gemma-4B / Qwen-9B via SHORT_LABEL):
  `--out-dir .../grid_with_qwen --families both --sizes e2b,e4b,12b,qwen9b
  --ys milestone_pct,reward --paper` (needs medium_runs root). `--sizes`
  filters points; the "qwen 8B" the user names is Qwen3.5-9B (9B params).
- After touching beliefs_metrics.py, re-run ONLY the metrics stage per out
  dir (parse is cached): `python analysis_qualitative/run.py metrics
  --runs-root <root> --out analysis_qualitative/out_<name>`.

## Style rules — the house compute-figure look

Identical to `make_pareto_social_fig.fig_pareto_paper` (the social-interval
frontier), so every compute figure in the paper reads as one family:

- OPEN markers in the house colours, shared with `make_scaling_fig`:
  base `#2c7fb8` squares (ms 8, mew 1.9), +Hebbian `#d95f0e` circles
  (ms 7, mew 1.7). Colour AND marker carry the arm.
- Solid connecting line (lw 1.6) within an arm; families never joined.
- No grid, linear compute axis, framed legend lower-right (fontsize 8.5,
  framealpha 0.95), figsize 5.0×3.5, dpi 200, PNG only.
- NO error bars by default — `--errorbars` adds ±1 sd over pooled episodes;
  the spread is large and the table carries it. `--normalize` divides by the
  base arm's cheapest point.
- Each size direct-labelled once (E2B/E4B/12B) in `#555555`, fontsize 8.5,
  11 pt above the higher arm — the same treatment as the frontier's Δ labels.
- After generating, OPEN the PNGs: nothing checks label collisions but you.

## The paper figures

**Consistency rule:** the model-size Pareto and the social-interval Pareto
(`make_pareto_social_fig.py`) plot the SAME headline metric with the SAME
label — "Task milestones (% of 25)". Verified identical computations:
its `total_pct` = `100 * task_milestone_count(ep) / (n_eps * TASK_MAX=25)`
is exactly this script's `milestone_pct`. Do not diverge them.

A bare run emits, all vs compute:

| file | y-axis |
|---|---|
| `pareto_gemma_milestone_pct_vs_flops.png` | Task milestones (% of 25) — **headline, matches the social frontier** |
| `pareto_gemma_reward_vs_flops.png` | Task return |
| `pareto_gemma_coop_pct_vs_flops.png` | Coop. milestone completion [%] |

plus `pareto_gemma_grid.png` (stacked) and, with `--paper`,
`pareto_gemma_paper.png` = task milestones + task return side by side, one
legend. `grid/normalized/` holds the `--normalize` variants.

Use COVERAGE (`milestone_pct`, `coop_pct`), not attainment (`completions*`,
`*_pa`) — coverage is the paper's own definition, so figure and table
cross-check. Attainment earns its place only for the redundancy point: at
12B, Hebbian reaches fewer distinct milestones (5.61 vs 6.06) with more
agents each (1.60 vs 1.49), so completions tie 9.00 while coverage favours
base. Per-agent axes exist (`completions_pa`) and are honest, but note 12B
saturates at 3.00/3 agents in both arms.

NEVER label any per-agent axis "cooperative": true coop/agent is 0.31–1.06,
whereas `final_metrics.mean_milestone_count_per_agent` is 6.4–7.2 because
~2/3 of it is the communication track — that would contradict
tab:final_comparison (1.44 distinct coop milestones for the whole team at
E4B) on the same page.

Perception axes (`--xs grounding,partner_loc`) still work but are DROPPED
from the paper: grounding is a precision-style rate that rewards terseness
(E2B 0.92 > 12B 0.87 > E4B 0.85) — a verbosity axis, not a capability one.

## Adding a size or family

1. `make_pareto_fig.SIZES`: add `key: dict(family=..., n_eff=..., label=...)`.
2. If its runs predate the suite, map the directory name in
   `make_pareto_fig.LEGACY_RUNS`; suite runs are auto-mapped from
   `pareto_<size>_<arm>`.
3. Run the qualitative pipeline on its runs (see the perception bullet) so
   `beliefs.csv` has its rows.
4. New family → add a marker in `FAMILY_MARKER` (both scripts) and a label in
   `make_pareto_grid.FAMILY_LABEL`.

## Guards already built in (do not bypass)

- Runs that generated ZERO tokens are excluded — the 26b/31b sharding failure
  wrote a plausible `final_metrics.json` while every LLM call failed.
- `make_pareto_fig.py` refuses to silently pool mixed protocols (a 1×50 smoke
  tree with a 3×1000 production tree is ~100× apart on the compute axis).
  Never point either script at a `*_smoke` root together with production.

## Known caveats to carry into the text

- Qwen points ran on `wiredtogether.sif`; Gemma points on
  `wiredtogether_gemma4.sif` (newer torch/transformers). Tolerable only because
  families are separate series — say so in the methods.
- 26b / 31b are absent: no DAIC card exceeds 44.3 GiB and multi-GPU Gemma-4
  vision inference is broken upstream (transformers #45823 fixed only the
  text path). See memory `pareto-flops-accounting` for the full probe ladder.
