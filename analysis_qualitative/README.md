# analysis_qualitative — qualitative analysis of medium-run logs

Extracts paper-ready qualitative evidence from `runs_from_daic/medium_runs/`
across four dimensions (communication content, social-module
interpretability, failure modes, beliefs/hallucination) plus the
bonds-vs-behaviour correlation the appendix promises. **Nothing here touches
the paper**; outputs live under `analysis_qualitative/out/`.

## Stages (one CLI)

```bash
python analysis_qualitative/run.py parse      # llm_logs+episodes -> per-run tables
python analysis_qualitative/run.py metrics    # 4 dimensions + flags + bonds
python analysis_qualitative/run.py sample     # stratified batches for annotation
#   ... Claude annotates out/samples/<dim>/batch_*.jsonl in-session
#       into out/annotations/<dim>/batch_*.jsonl (see samples/RUBRICS.md)
python analysis_qualitative/run.py validate   # schema/coverage + kappa
python analysis_qualitative/run.py cases      # archetype shortlists + transcripts
python analysis_qualitative/run.py collab     # collaboration success/failure cases
python analysis_qualitative/run.py report     # out/report/qual_report.md
```

`collab` (needs `parse` outputs only) mines per-interaction collaboration
episodes — fulfilled/ignored requests, real (non-timer) coop milestones,
Ch3 mutual deadlocks/stalls, Ch2 anvil neglect — writes
`tables/collab_{run_level,condition,matched_pairs}.csv`,
`collab/events.jsonl.gz`, exemplar transcripts under `cases/collab/`, and
`report/collab_report.md` with the baseline-vs-Hebbian matched-pair contrast
(request outcomes conditioned on bond strength W for Hebbian runs).

Global flags: `--runs-root`, `--out`, `--only exp_dir/seed_N`, `--force`.
`parse` is incremental (manifest keyed on file sizes/mtimes + PARSER_VERSION);
when the missing exp05/exp06 seeds land, re-running the chain only processes
the new runs.

## Key mechanics & gotchas

- **Step alignment**: LLM runs align by ordinal position validated against
  step_log message text (>=98% exact required; ±2 repair). RL runs cannot be
  text-anchored (driver retries + the routed message differs from the
  rl_thoughts comm field) -> a wall-clock->(ep,step) table is extracted from
  log.txt step markers (`qual_lib/stepclock.py`), robust to appended relaunch
  attempts and to a concurrent foreign-config job interleaving markers.
- **Stale units**: relaunches APPEND llm_logs; units from aborted attempts are
  flagged `stale` in `parsed/*/llm_calls.jsonl.gz` — always filter on it.
- **coop_eval nesting bug**: `final_metrics.json["coop_metrics"]`
  pair/dwell/damage totals are silently ZERO for real runs (coop_eval reads
  episode summaries at the wrong nesting level). All cooperation data here
  flows through `qual_lib/episode_io.read_cooperation_metrics`, which reads
  `episode_summary.json["cooperation_metrics"]` directly.
- **Quarantine**: a run whose artifacts show cross-job contamination (text
  alignment fails AND message counts disagree >10% with comm_metrics) is
  excluded from step-keyed analyses (currently:
  exp11_llm_9b_allied_none/seed_456).
- `log.txt`/`run.log`/`communication_log.json`/gifs are denylisted except the
  streaming step-marker scan.

## Outputs

- `out/parsed/<exp>/<seed>/` — llm_calls.jsonl.gz, timeline.jsonl.gz,
  alignment_report.json (per-run validation + cross-checks)
- `out/tables/*.csv` — run-level + condition-level metric tables
- `out/flags/flags.jsonl.gz` — failure-detector records with provenance
- `out/samples/`, `out/annotations/` — annotation workflow (+ agreement.json)
- `out/cases/` — shortlist.csv + transcript.md per archetype exemplar
- `out/report/qual_report.md` — the compiled qualitative report
- `out/report/qual_tables.tex` — staged LaTeX rows (do not paste into the
  paper until the suite completes)

## Tests

`python -m pytest tests/test_qual_log_parser.py tests/test_qual_detectors.py`
— synthetic fixtures in `tests/qual_fixtures.py` emit the exact llm_call.py
log grammar (multi-line JSON, retries, U+FFFD) and the real EpisodeLogger
layout, including the nested-cooperation_metrics regression case.
