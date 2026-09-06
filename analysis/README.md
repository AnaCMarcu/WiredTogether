# Analysis

Every table and figure in the paper is produced by a script here. They read run directories
(`runs_from_daic/<group>/<arm>/seed_<N>/`, git-ignored) and write into `paper_assets/`.

Run them from anywhere — `paths.py` anchors the inputs and outputs to the repo root and puts the
sibling modules, `src/` and `qual_lib` on `sys.path`:

```bash
python analysis/make_final_table.py
python analysis/make_pareto_social_fig.py --out paper_assets/pareto_social
```

`make_results.py` is the shared aggregation layer, not just a script: the condition registry,
milestone accounting, episode slicing and the pooling convention live there and everything else
imports them, so the numbers cannot drift between a table and the figure next to it.

## Tables

| Script | Produces |
|---|---|
| `make_final_table.py` | The cross-model comparison: task return, milestone and cooperative completion, steps-to-first-completion, perception columns |
| `make_results.py` | Main results, steps-to-milestone, topology ablation and graph-statistics rows, plus the milestone progression and timeline figures |
| `cofire_table.py` | The co-firing channel table: per-cue act use, bond growth attribution, ρ(W, acts) |
| `make_bond_asymmetry_report.py` | `BOND_ASYMMETRY.md` — W_ij vs W_ji across every run carrying a graph |
| `wandb_compute_budget.py` | Wall-clock and GPU budget table (needs `wandb`, run where you are logged in) |

## Figures

| Script | Produces |
|---|---|
| `make_pareto_social_fig.py` | Performance vs social-module compute across deliberation intervals |
| `make_scaling_fig.py` | Performance vs whole-system compute across team sizes |
| `make_pareto_fig.py`, `make_pareto_grid.py` | Model-size sweep: compute vs performance, headline and grid |
| `make_pareto_perception_fig.py`, `make_pareto_delta.py` | Milestone completion vs perception grounding; Hebbian benefit vs partner localisation |
| `make_bond_asymmetry_fig.py` | Directedness of the learned graph |
| `make_bond_behavior_fig.py` | Correlation between bonds and messages / joint digging / proximity |
| `make_mechanistic_figure.py` | Long-horizon bond dynamics with magnified case studies |
| `make_final_figures.py`, `make_final_figures_callouts.py`, `make_iclr_figure.py` | Qualitative timelines per arm: frames, chat, bonds (two layout variants + the single-column cut) |
| `make_qualitative_figure.py`, `make_story_timelines_multi.py` | Earlier timeline variants kept for the appendix |
| `make_directive_timelines.py`, `make_coordination_timelines.py` | Team formation over time: base vs orchestrator vs Hebbian |
| `make_team_comparison.py`, `make_team_tenure.py`, `make_plan_vs_completion.py` | Planned vs realised teams, team tenure, when cooperation actually starts |
| `make_social_dynamics.py` | Standalone communication-rate and bond-strength figures |
| `make_qwen_hebbian_analysis.py` | Why the 9B backbone loses under the social module |
| `make_chamber_gallery.py` | Environment gallery, one row per chamber |
| `make_compose_assets.py` | Component export of the qualitative figures for hand layout |

## Appendix material

| Script | Produces |
|---|---|
| `make_cofire_excerpts.py` | Co-firing events joined with milestones, per run — candidates for the channel excerpts |
| `make_transplant_excerpts.py` | Phase-B interaction excerpts: reunions, re-pairing, partner preference |
| `assist_inference.py` | Assisted-kill classification behind the combat annotations |

## Mechanism replay

| Script | Produces |
|---|---|
| `replay_hebbian_terms.py` | Replays the deployed rule offline from logged positions, events and rewards, term by term — reproduces the stored `W` exactly |
| `prototype_three_factor_rule.py` | The same trajectories under an eligibility-trace variant (open-loop counterfactual) |
| `compute_flops.py` | Inference FLOPs per run from the logs; imported by the Pareto figures and usable on its own |

## Qualitative pipeline

`qualitative/` is a seven-stage CLI (`parse`, `metrics`, `sample`, `validate`, `cases`, `collab`,
`report`) over the same run logs, producing the communication, interpretability, failure and belief
tables plus the bonds-vs-behaviour correlations. See `qualitative/README.md`.
