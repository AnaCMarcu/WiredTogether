# Experiments

## The arms

Every condition in the paper is one launcher under `hpc/daic/experiments/`, and every launcher
sources `_common.sh` (container, model paths, headless rendering, `CRAFTIUM_ENV_DIR`, seeds, W&B)
and calls `multi_agent_craftium.py` with a fixed configuration. Seeds come from a SLURM array.
Results land in `runs/<group>/<arm>/seed_<N>/`; the group name is what the analysis scripts look
for under `runs_from_daic/`.

| Question | Launchers | Run group |
|---|---|---|
| RQ1 — LLM baselines and RL arms | `exp01`–`exp08` (2B/9B, MAPPO, IPPO, ±Hebbian, ±social module) | `final` (synced as `medium_runs`) |
| RQ1 — Gemma anchors | `new_exp_0_gemma` (base, +Hebbian, three-factor) | `new_exp_0_gemma` |
| RQ1 — centralised orchestration | `new_exp_orchestrator` | `orchestrator` |
| RQ2 — co-firing channels | `exp20`–`exp29` (`prc`, `pro`, `pri`, `prco`, `prcoi`, anchor, null) | `cofiring_final` |
| RQ3 — imposed topology | `exp09`–`exp11` (allied-all, allied-pair, no-bonds) | `final` |
| RQ3 — transplant | `expA_pair_bonding`, then `expB_merged_transplant` / `expB_merged_shuffled` | `pair_bonding` |
| Compute — deliberation interval | `new_exp_0_gemma_si` | `pareto_social` |
| Compute — model size | `new_exp_pareto` | `pareto_gemma4` |
| Compute — team size | `scale_gemma` | `agent_scaling` |
| Experience sharing (ρ = 0.3) | `exp30`, `exp31` | `social_replay` |

`submit_*.sh` launches a whole family across seeds; a single arm is `sbatch <launcher>`. The `_3f`
variants are the same arms under `--hebbian-mode three_factor`.

Two operational rules the suite depends on:

- Move a failed run's directory aside before re-running it. `log.txt` and `llm_logs/*.log` are
  opened in append mode, so a rerun in place silently doubles the token counts that
  `analysis/compute_flops.py` reports.
- Keep `LLM_VISION_MODE` identical across runs that will be pooled. A text-only run and a vision
  run of the same "arm" are not the same condition.

## The transplant

Two phases. Phase A runs isolated dyads (`expA_pair_bonding`) until they have a bond history.
`src/mindforge/tools/pair_transplant.py` then merges chosen dyads into one six-agent population:
it builds the block `W` matrix, equalises the within-pair bonds so magnitude cannot explain the
result, and assembles each agent's cognitive state. Phase B (`expB_merged_transplant`) starts that
population in Chamber 3 with `--hebbian-init-file` and `--agent-state-init`;
`expB_merged_shuffled` is the control that pairs agents with strangers from other Phase-A runs.
`merge_pair_runs.py` picks the input runs, `transplant_report.py` and `make_transplant_table.py`
produce the numbers.

## Analysis

`analysis/` reads run directories and writes into `paper_assets/`. `make_results.py` is the shared
aggregation layer — condition registry, milestone accounting, episode slicing — and every other
script imports it rather than re-deriving the numbers, so a change to how a milestone counts
propagates everywhere at once. `analysis/README.md` maps each script to the table or figure it
produces.

Milestone conventions, applied everywhere:

- percentages, never counts: 25 task milestones, 17 cooperative (Ch2–Ch5), team union per episode;
- the communication track is stripped from comparable counts — it pays for social acts, not task
  progress;
- task return is the undiffused environment reward, so Hebbian arms are not credited with reward
  that diffusion merely moved between agents;
- every episode of every seed is pooled, reporting mean ± population SD — not a mean of seed means.

The qualitative pipeline (`analysis/qualitative/`) is a separate seven-stage CLI over the same
logs: it parses `llm_logs` into per-run tables, computes the communication / interpretability /
failure / belief dimensions, samples stratified batches for annotation, and emits the report and
CSVs behind the appendix.

## Compute accounting

`analysis/compute_flops.py` estimates inference FLOPs per run as `2·N_eff·tokens`. Decode tokens
are exact; prefill is exact for runs whose logs carry `[LocalModel usage]` lines and
char-estimated otherwise, with a per-run chars/token calibration. The Pareto figures deliberately
use the estimate everywhere, including runs that have exact totals, so that anchors and sweep runs
sit on a bias-consistent axis. Per-module attribution (social module vs the rest) comes from the
per-module log files.
