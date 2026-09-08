# Figures

The figures used in the paper, so the repo carries them without the LaTeX sources.

| File | Shows | Paper |
|---|---|---|
| `overview_social_plasticity_loop.png` | The interaction–plasticity–behaviour loop: co-firing drives reward-modulated updates to `W(t)`, which couples back through reward diffusion and experience sharing (RL) or bond-conditioned partner selection (LLM) | Introduction |
| `wire_five_chambers.png` | WIRE: the five chambers and the coordination competency each targets | Methodology |
| `wire_first_person_views.png` | First-person agent views, one per chamber | Methodology |
| `coordination_timeline_gemma_e4b_hebbian_seed123.pdf` | Gemma-E4B+Heb2.0: mutual messages and bond strength across the five chambers, with interaction windows for successful, partial and failed coordination | Discussion |
| `coordination_timeline_qwen9b_hebbian_seed42.pdf` | Qwen3.5-9B+Heb, same view — a confirmed joint anvil action, an exhausted-anvil failure, and team combat | Appendix, coordination examples |
| `milestone_completion_timeline.pdf` | First-completion step per milestone, baseline vs +Hebbian, grouped by chamber | Appendix, additional results |
| `reward_modulation_replay_seed456_a_deployed_rule.pdf` | Open-loop replay of seed 456 under the rule used in the experiments | Appendix, additional results |
| `reward_modulation_replay_seed456_b_eligibility_trace.pdf` | The same replay with an eligibility trace over co-activity, homeostatic decay unchanged | Appendix, additional results |
| `reward_modulation_replay_seed456_c_trace_reduced_decay.pdf` | The same trace with `λ` reduced, so reward-gated credit persists instead of being erased | Appendix, additional results |

Regenerating: `wire_first_person_views.png` comes from `analysis/make_chamber_gallery.py`,
`milestone_completion_timeline.pdf` from `analysis/make_results.py`, and the
`reward_modulation_replay_*` set from `analysis/prototype_three_factor_rule.py`.
`analysis/README.md` maps every script to what it produces.
