# Analysis

Every table and figure in the paper is produced by a script here. The scripts read run
directories from `runs/<question>/<group>/<arm>/seed_<N>/` (see
[../docs/dataset.md](../docs/dataset.md)) and write to `paper_assets/`. Run them from anywhere:
`paths.py` anchors inputs and outputs to the repository root, puts `src/` and the qualitative
pipeline on `sys.path`, and resolves run groups by name through `paths.group()`.

`make_results.py` is the shared aggregation layer: condition registry, milestone accounting,
episode slicing. Every other script imports it, so a table and the figure next to it cannot
disagree about how a milestone counts.

## Paper tables and figures

| Paper | Script | Run groups it reads |
|---|---|---|
| Tables 1–3, 12 | `make_agent_completion_tables.py` | `medium_runs`, `new_exp_0_gemma`, `gemma4`, `orchestrator`, `social_replay_3f_{qwen,gemma4}`, `cofiring_bidi_3f`, `pair_bonding_3f`, `medium_2k` |
| Table 9 | `make_steps_table_pct.py` | as Table 1 |
| Table 10 | `make_bond_behaviour_rho.py` | `medium_runs` (exp34, exp35), `new_exp_0_gemma` (hebbian3f) |
| Table 11 | `make_transplant_tables.py` | `pair_bonding_3f` |
| Figures 4, 10, 11 | `make_counterfactual_compact_n6.py`, `make_counterfactual_n6.py` | `agent_scaling_orch`, `agent_scaling_3f` |
| Figure 5 | `make_chamber_gallery.py` | frames written by `make_final_figures.py` |
| Figure 6 | `make_agent_completion_figs.py` | `new_exp_0_gemma`, `pareto_social_3f` |
| Figure 7 | `make_agent_completion_figs.py` | `cofiring_bidi_3f` |
| Figure 8 | `make_agent_completion_figs.py` | `medium_runs`, `new_exp_0_gemma`, `pareto_gemma4`, `pareto_gemma4_3f`, and the perception table below |
| Figure 9 | `make_final_figures.py` (arm `gemma3f_seed123`) | `new_exp_0_gemma` (hebbian3f, seed 123) and its recordings |
| Figures 12, 13 | `make_counterfactual_story.py` | `orchestrator`, `pareto_social_3f` (si3f8), seed 42 |
| Figure 14 | `make_team_tenure.py` | `orchestrator`, `new_exp_0_gemma` (hebbian3f), seed 456 |

Figure 8's perception axes come from `paper_assets/perception_3f/beliefs_3f.csv`, which
`make_beliefs_3f_view.py` assembles from the qualitative pipeline's belief tables. The released
dataset includes that CSV, so the figure does not require re-running the pipeline.

## Shared modules

| Module | Role |
|---|---|
| `paths.py` | Repository paths and run-group lookup |
| `make_results.py` | Aggregation layer used by every script |
| `make_final_table.py`, `make_final_table_extended.py`, `make_final_table_latex.py` | Condition rows behind Tables 1 and 9 |
| `cofire_table.py` | Per-cue act use, attributed bond growth and ρ behind Table 2 |
| `compute_flops.py` | Inference FLOPs per run, for the deliberation-interval sweep |
| `replay_hebbian_terms.py`, `prototype_three_factor_rule.py` | Offline replay of the plasticity rule from logged inputs; reproduces the stored `W` |
| `make_directive_timelines.py`, `make_counterfactual_table.py` | Loaders and layout shared by the timeline figures |
| `runs_dataset.py` | Groups the run tree by research question and builds the released archives |

## Qualitative pipeline

`qualitative/` is a staged CLI (`parse`, `metrics`, `sample`, `validate`, `cases`, `collab`,
`report`) over the per-module LLM logs. It produces the belief and message tables used above. The
annotation labels under `qualitative/out/annotations/` are kept because they were produced by an
LLM annotator and cannot be regenerated deterministically. See `qualitative/README.md`.
