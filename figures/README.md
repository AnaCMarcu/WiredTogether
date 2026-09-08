# Figures

The figures used in the paper, so the repo carries them without the LaTeX sources.

| File | Shows | Paper |
|---|---|---|
| `FIG_1.3.png` | The interaction–plasticity–behaviour loop: co-firing drives reward-modulated updates to `W(t)`, which couples back through reward diffusion and experience sharing (RL) or bond-conditioned partner selection (LLM) | Introduction |
| `env_minecraft_v5.png` | WIRE: the five chambers and the coordination competency each targets | Methodology |
| `chamber_gallery.png` | First-person agent views, one per chamber | Methodology |
| `wide_gemma_hebbian3f.pdf` | Gemma-E4B+Heb2.0 (seed 123): mutual messages and bond strength across the five chambers, with interaction windows for successful, partial and failed coordination | Discussion |
| `wide_qwen9b_hebbian.pdf` | Qwen3.5-9B+Heb (seed 42): the same view — a confirmed joint anvil action, an exhausted-anvil failure, and team combat | Appendix, coordination examples |
| `milestone_timeline.pdf` | First-completion step per milestone, baseline vs +Hebbian, grouped by chamber | Appendix, additional results |
| `three_factor_prototype_seed456_{A,B,C}.pdf` | Open-loop replay of seed 456 under the deployed rule (A) and longer-timescale reward-modulation variants (B, C) | Appendix, additional results |

Regenerating: `chamber_gallery.png` comes from `analysis/make_chamber_gallery.py`,
`milestone_timeline.pdf` from `analysis/make_results.py`, and the
`three_factor_prototype_*` set from `analysis/prototype_three_factor_rule.py`.
`analysis/README.md` maps every script to what it produces.
