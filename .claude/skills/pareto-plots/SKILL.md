---
name: pareto-plots
description: Regenerate the model-size Pareto figures (compute / perception vs performance, base vs Hebbian, Gemma / Qwen / both) in the CoDe dashed-line style, plus the results tables. Use whenever new pareto runs land, a metric or axis needs changing, or a new model size/family is added.
---

# Pareto plots for the model-size sweep

Two scripts, both read the SAME run directories and agree with the paper's
`make_results.py` definitions (they import its helpers rather than re-deriving):

| script | produces |
|---|---|
| `scripts/make_pareto_grid.py` | the 18-panel grid: {gemma, qwen, both} × {reward, milestone_pct, coop_pct} × {flops, perception}, PNG+PDF singles + one composite per family, `points.csv` |
| `scripts/make_pareto_fig.py` | the two log-x headline figures (`--metric`, default `milestone_pct`) + `pareto_results.tex/.md` tables; ALSO the home of `run_metrics()` — the single source of truth the grid imports |

## Standard invocation (from the repo root)

```bash
python scripts/make_pareto_grid.py runs_from_daic/pareto_gemma4 \
    runs_from_daic/new_exp_0_gemma runs_from_daic/medium_runs \
    --out-dir paper_assets_pareto/grid

python scripts/make_pareto_fig.py runs_from_daic/pareto_gemma4 \
    runs_from_daic/new_exp_0_gemma runs_from_daic/medium_runs \
    --out-dir paper_assets_pareto --csv paper_assets_pareto/points.csv
```

Roots are run-group directories holding `<condition>/seed_<N>/`. The three
above are: the pareto suite (e2b, 12b), the E4B point (`new_exp_0_gemma_*`),
and the Qwen points (`exp01/02` base, `exp07/08` hebbian — same protocol and
flags as the suite, verified against the sbatch files).

Subset flags: `--families gemma`, `--ys coop_pct,reward`, `--xs flops`,
`--no-point-labels`. FLOPs are cached in `<out-dir>/flops_cache.json`
(keyed on run path + log.txt mtime), so only the first run is slow.

## Metric definitions — do not change silently

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
- **perception**: MMMU-Pro from the official model cards,
  `paper_assets_pareto/perception_scores.csv` (columns `size,score,mode,source`).
  Prefer NON-thinking mode — the agents run with thinking off. It is a backbone
  property: identical for both arms of a size.

## Style rules (from the dataviz skill + CoDe Fig. 10)

- Colour = arm: base `#2a78d6`, hebbian `#eb6834` (the two palette slots that
  validate all-pairs CVD in both modes). Marker = family: circle Gemma, square
  Qwen. Identity is never colour-alone.
- Dashed lines connect points WITHIN a (family, arm) series only. Families are
  never joined or fitted into one frontier — across families size is confounded
  with architecture and training data. Two series on one canvas is legitimate
  only because the FLOPs axis is cross-model-valid.
- Boxed legend inside the axes, thin ±1 sd error bars over seeds, every size
  direct-labelled once (above the taller arm).
- After generating, OPEN the PNGs and look: the validator checks colour, not
  label collisions or clipping.

## Adding a size or family

1. `make_pareto_fig.SIZES`: add `key: dict(family=..., n_eff=..., label=...)`.
2. If its runs predate the suite, map the directory name in
   `make_pareto_fig.LEGACY_RUNS`; suite runs are auto-mapped from
   `pareto_<size>_<arm>`.
3. Add a row to `perception_scores.csv` with the model-card MMMU-Pro and its
   source URL.
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
