## Table 1 — Experiment design

| Experiment | Base model | Learning | Social coupling | Seeds |
|---|---|---|---|---|
| LLM-2B | Qwen3.5-2B | none (frozen LLM) | — | 6 |
| LLM-2B+Heb | Qwen3.5-2B | none (frozen LLM) | Hebbian + prompt | 6 |
| LLM-9B | Qwen3.5-9B | none (frozen LLM) | — | 6 |
| LLM-9B+Heb | Qwen3.5-9B | none (frozen LLM) | Hebbian + prompt | 6 |
| Gemma-E4B | Gemma-4-E4B | none (frozen LLM) | — | 6 |
| Gemma-E4B+Heb | Gemma-4-E4B | none (frozen LLM) | Hebbian + prompt | 6 |
| IPPO | Qwen3.5-2B | IPPO (LoRA) | — | 4 |
| IPPO+Heb | Qwen3.5-2B | IPPO (LoRA) | Hebbian graph + replay | 3 |
| MAPPO | Qwen3.5-2B | MAPPO (shared critic) | — | 6 |
| MAPPO+Heb | Qwen3.5-2B | MAPPO (shared critic) | Hebbian graph + replay | 3 |

All conditions: 3 agents, 3 episodes × 1000 steps, communication enabled.

## Table 2 — Task performance and perception

Milestone figures are percentage completion per episode (team union; denominators 25 non-comm, 17 cooperative Ch2–Ch5). Steps-to-milestone is the median within-episode step of first completion, with completing episodes in parentheses.

| Experiment | Task return | Milestone % | Coop. % | Anvil A | Anvil B | Switch | Ch4 mob | Grounding | Partner loc. |
|---|---|---|---|---|---|---|---|---|---|
| LLM-2B | 358 ± 163 | 22.9 ± 4.6 | 9.2 ± 4.9 | — | — | 434 (14) | — | 0.417 | 0.594 |
| LLM-2B+Heb | 363 ± 196 | 20.4 ± 5.6 | 6.9 ± 5.6 | — | — | 421 (10) | 756 (1) | 0.393 | 0.592 |
| LLM-9B | 622 ± 150 | 31.1 ± 5.1 | 17.6 ± 6.2 | 201 (2) | 218 (2) | 415 (18) | 679 (10) | 0.822 | 0.721 |
| LLM-9B+Heb | 574 ± 123 | 27.3 ± 5.2 | 14.1 ± 5.6 | 301 (1) | — | 403 (17) | 696 (7) | 0.809 | 0.754 |
| Gemma-E4B | 521 ± 117 | 16.4 ± 4.8 | 8.5 ± 4.9 | — | 199 (1) | 406 (7) | 670 (10) | 0.845 | 0.799 |
| Gemma-E4B+Heb | 570 ± 125 | 22.0 ± 5.5 | 12.1 ± 5.0 | — | — | 407 (12) | 662 (13) | 0.843 | 0.790 |
| IPPO | 425 ± 99 | 18.3 ± 4.5 | 6.9 ± 5.8 | — | — | 488 (7) | — | 0.448 | 0.451 |
| IPPO+Heb | 427 ± 108 | 18.2 ± 4.7 | 5.2 ± 5.8 | — | — | 456 (4) | — | 0.133 | 0.638 |
| MAPPO | 390 ± 86 | 17.8 ± 4.5 | 4.6 ± 5.7 | — | — | 486 (7) | — | 0.392 | 0.565 |
| MAPPO+Heb | 470 ± 110 | 20.4 ± 4.4 | 7.8 ± 5.5 | — | — | 490 (6) | — | 0.164 | 0.636 |
