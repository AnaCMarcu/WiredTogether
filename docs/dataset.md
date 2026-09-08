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
| `rq3_topology_transfer/` | `medium_2k`, `pair_bonding` |
| `compute/` | `pareto_social` (deliberation interval), `pareto_gemma4` (model size), `agent_scaling` (team size) |
| `social_replay/` | `social_replay_gemma4`, `social_replay_qwen` |
| `cluster_logs/` | SLURM stdout/stderr, kept as provenance |
| `smoke/` | Smoke tests; not part of the release |

Group names are unique across the dataset, so the analysis scripts ask for a group by name through
`paths.group("medium_runs")` and never encode the grouping — regrouping the tree does not touch
them.

Two things the table cannot show:

- `medium_runs` holds both the RQ1 baselines (`exp01`–`exp08`) and the RQ3 topology arms
  (`exp09`–`exp11`). They were one submission wave and `make_results.py` builds the main and
  topology tables from a single run root, so the group stays whole.
- `social_replay_qwen` supplies the `MAPPO+Heb` and `IPPO+Heb` experience-sharing rows of the
  cross-model table. It ran with `LLM_VISION_MODE=text`; `exp05`/`exp06`, the Qwen arms it is
  tabled against, ran with vision. Each run's `log.txt` records which it was.

## Layers

The layers stack and none repeats another's files, so nobody downloads more than they need.

| Layer | Holds | Raw | Packed | Needed for |
|---|---|---|---|---|
| `core` | metrics, events, per-episode tables, graph snapshots, configs | 1.9 GB | 385 MB | every table and quantitative figure |
| `logs` | `llm_logs/*.log`, `log.txt`, SLURM `.out`/`.err` | 28 GB | 2.5 GB | `analysis/qualitative/`, and step-clock alignment for the RL arms |
| `media` | the `.mp4` recordings | 6.9 GB | ~6.9 GB | the story-timeline figures only |

`core` and `logs` together are 11 archives and 2.8 GB — `log.txt` compresses about 18×, so the
bulky layer packs down hardest. The recordings are already compressed and gain nothing, which is
why they are their own layer.

Model state is never bundled, because nothing in `analysis/` reads it: `checkpoints/`, and
`rl_live/` — the in-flight adapter and optimiser state (`rl_state.pt`, `value_head.pt`, a PEFT
adapter), which is a checkpoint under another name.

Per question, packed:

| Question | `core` | `logs` |
|---|---|---|
| `rq1_social_plasticity` | 143 MB | 946 MB |
| `rq2_cofiring` | 98 MB | 457 MB |
| `rq3_topology_transfer` | 35 MB | 414 MB |
| `compute` | 95 MB | 562 MB |
| `social_replay` | 14 MB | 119 MB |
| `cluster_logs` | — | 20 MB |

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
