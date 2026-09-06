# Run artifacts

Every number in the paper comes from a run directory. The collected runs are 46 GB, so they are
not in this repository — `runs_from_daic/` is git-ignored, and the released dataset is built from
it by `analysis/runs_dataset.py`.

## Layout

The cluster wrote one directory per submission wave. Those groups are filed under the question
they answer:

```
runs_from_daic/<question>/<group>/<arm>/seed_<N>/
```

| Question | Groups |
|---|---|
| `rq1_social_plasticity/` | `medium_runs` (Qwen 2B/9B, MAPPO/IPPO, ±Hebbian, ±social module), `new_exp_0_gemma`, `gemma4`, `orchestrator` |
| `rq2_cofiring/` | `cofiring_final` (the seven channel arms in the paper), `cofiring`, `cofiring_bidi`, `cofiring_noreward` |
| `rq3_topology_transfer/` | `medium_2k`, `pair_bonding`, `pair_bonding_qwen` |
| `compute/` | `pareto_social` (deliberation interval), `pareto_gemma4` (model size), `agent_scaling` (team size) |
| `social_replay/` | `social_replay_gemma4`, `social_replay_qwen_textonly` |
| `cluster_logs/` | SLURM stdout/stderr, kept as provenance |
| `smoke/` | Smoke tests; not part of the release |

Group names are unique across the dataset, so the analysis scripts ask for a group by name through
`paths.group("medium_runs")` and never encode the grouping — regrouping the tree does not touch
them.

Two things the table cannot show:

- `medium_runs` holds both the RQ1 baselines (`exp01`–`exp08`) and the RQ3 topology arms
  (`exp09`–`exp11`). They were one submission wave and `make_results.py` builds the main and
  topology tables from a single run root, so the group stays whole.
- `social_replay_qwen_textonly` ran text-only while the rest of the pooled Qwen suite ran with
  vision. It is not comparable to the other Qwen arms and is excluded from every pooled result.

## Layers

The layers stack and none repeats another's files, so nobody downloads more than they need.

| Layer | Holds | Raw | Needed for |
|---|---|---|---|
| `core` | metrics, events, per-episode tables, graph snapshots, configs | 2.6 GB | every table and quantitative figure |
| `logs` | `llm_logs/*.log`, `log.txt` | 26 GB | `analysis/qualitative/`, and step-clock alignment for the RL arms |
| `media` | the `.mp4` recordings | 7.5 GB | the story-timeline figures only |

`log.txt` compresses about 18×, so `core` + `logs` is roughly 2 GB of archives despite the raw
size. RL checkpoints (`checkpoints/`, 5.2 GB) are never bundled — nothing in `analysis/` reads
them.

## Tooling

```bash
python analysis/runs_dataset.py plan             # the grouping, and what each layer weighs
python analysis/runs_dataset.py regroup --apply   # file groups under their question
python analysis/runs_dataset.py bundle            # build dist/runs_dataset/
python analysis/runs_dataset.py verify dist/runs_dataset
```

`bundle` writes one `<question>__<layer>.tar.gz` per pair, plus `MANIFEST.json` (what each archive
holds, per-group descriptions and arm lists) and `SHA256SUMS`. `--layers core` builds just the
small tier; `--include-smoke` adds the smoke groups. Extract the archives into `runs_from_daic/`
and the analysis scripts find them.

## Excluded runs

`analysis/qualitative/` drops `exp11_llm_9b_allied_none/seed_456` from step-keyed analyses: its
artifacts show cross-job contamination — text alignment fails and message counts disagree with
`comm_metrics` by more than 10%.
