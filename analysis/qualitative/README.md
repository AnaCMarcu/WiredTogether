# Qualitative pipeline

Extracts the paper's qualitative evidence from the run logs across four dimensions
(communication content, social-module interpretability, failure modes, beliefs and hallucination)
plus the bonds-vs-behaviour correlations behind the appendix. Every stage writes under
`analysis/qualitative/out/`; nothing here writes into `paper_assets/`.

## Stages (one CLI)

```bash
python analysis/qualitative/run.py parse      # llm_logs + episodes -> per-run tables
python analysis/qualitative/run.py metrics    # 4 dimensions + flags + bonds
python analysis/qualitative/run.py sample     # stratified batches for annotation
#   ... an LLM annotator labels out/samples/<dim>/batch_*.jsonl in-session
#       into out/annotations/<dim>/batch_*.jsonl (rubrics in samples/RUBRICS.md)
python analysis/qualitative/run.py validate   # schema/coverage + kappa
python analysis/qualitative/run.py cases      # archetype shortlists + transcripts
python analysis/qualitative/run.py collab     # collaboration success/failure cases
python analysis/qualitative/run.py report     # out/report/qual_report.md
```

The annotations are the one input here that is not derived from the logs — they are model-produced
labels, so `out/annotations/` is version-controlled while the rest of `out/` is not.

`collab` (needs `parse` outputs only) mines per-interaction collaboration episodes — fulfilled and
ignored requests, real (non-timer) cooperative milestones, Ch3 mutual deadlocks, Ch2 anvil neglect
— and writes `tables/collab_{run_level,condition,matched_pairs}.csv`, `collab/events.jsonl.gz`,
exemplar transcripts under `cases/collab/`, and the baseline-vs-Hebbian matched-pair contrast in
`report/collab_report.md` (request outcomes conditioned on bond strength `W` for Hebbian arms).

Global flags: `--runs-root`, `--out`, `--only exp_dir/seed_N`, `--force`. `parse` is incremental,
keyed on file sizes/mtimes plus `PARSER_VERSION`, so re-running the chain only processes runs whose
logs changed.

## Mechanics that the numbers depend on

- **Step alignment.** LLM runs align by ordinal position, validated against `step_log` message text
  (≥98% exact required, ±2 repair). RL runs cannot be text-anchored — the driver retries, and the
  routed message differs from the `rl_thoughts` comm field — so a wall-clock → (episode, step)
  table is extracted from `log.txt` step markers (`qual_lib/stepclock.py`), robust to appended
  relaunch attempts and to a concurrent job interleaving markers.
- **Stale units.** Relaunches append to `llm_logs`, so units from aborted attempts are flagged
  `stale` in `parsed/*/llm_calls.jsonl.gz`. Always filter on it.
- **Cooperation metrics.** `final_metrics.json["coop_metrics"]` pair/dwell/damage totals are zero
  for real runs — `coop_eval` reads the episode summaries at the wrong nesting level. Everything
  here goes through `qual_lib/episode_io.read_cooperation_metrics`, which reads
  `episode_summary.json["cooperation_metrics"]` directly.
- **Quarantine.** `exp11_llm_9b_allied_none/seed_456` shows cross-job contamination (text
  alignment fails *and* message counts disagree with `comm_metrics` by more than 10%) and is
  excluded from step-keyed analyses.
- `log.txt`, `run.log`, `communication_log.json` and the recorded videos are denylisted except for
  the streaming step-marker scan.

## Outputs

| Path | Contents |
|---|---|
| `out/parsed/<exp>/<seed>/` | `llm_calls.jsonl.gz`, `timeline.jsonl.gz`, `alignment_report.json` |
| `out/tables/*.csv` | Run-level and condition-level metric tables |
| `out/flags/flags.jsonl.gz` | Failure-detector records with provenance |
| `out/samples/`, `out/annotations/` | The annotation workflow, plus `agreement.json` |
| `out/cases/` | `shortlist.csv` and `transcript.md` per archetype exemplar |
| `out/report/` | The compiled qualitative report and its staged LaTeX rows |

## Tests

```bash
python -m pytest tests/test_qual_log_parser.py tests/test_qual_detectors.py
```

Synthetic fixtures in `tests/qual_fixtures.py` emit the exact `llm_call.py` log grammar
(multi-line JSON, retries, U+FFFD) and the real `EpisodeLogger` layout, including the nested
`cooperation_metrics` regression case.
