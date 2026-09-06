# RL layer

`src/rl_layer/` — action-level PPO on top of the same LLM the agents reason with. Off unless
`--rl`; when off the whole layer is a no-op and the run is the zero-shot LLM condition.

## Policy

One frozen base model shared by the whole team, plus one LoRA adapter per agent — an N-agent run
holds a single copy of the weights and N small trainable adapters.

The policy is discrete over the 14 unmasked primitives. Given the observation, each candidate
action string is scored by its summed token log-probability under the adapter, and the softmax over
those scores is `π(a|o)` (constrained decoding, `rl_layer.py::select_action`). Reasoning and the
message are generated first and appended to the context, so reasoning conditions the action while
PPO only ever optimises the discrete action distribution — neither the reasoning nor the message
receives a policy gradient.

`peft_compat.resolve_lora_targets` scopes the adapter to explicit text-tower paths. This matters
for multimodal checkpoints, where the naive target list also matches the vision/audio towers'
wrapper layers and every RL arm dies at model load.

## Critics

`--rl-critic-mode centralized` (MAPPO) trains one shared `V(s_t)` over an explicit joint-state
encoding, built once per step before anyone acts:

- compact stream, per agent: position, chamber one-hot, HP, inventory bag-of-words, milestone
  bitmap, raw step reward;
- semantic stream, per agent: a sentence-transformer embedding of its last action and last message.

Actors stay decentralised — they only ever see their own observation — so this is CTDE, not
centralised control. The critic's returns come from the team-mean diffused reward over living
agents, and it updates on its own cadence from its own buffer.

`--rl-critic-mode independent` (IPPO) instead gives each agent a value head over its own model
representation, with no cross-agent information. Reward normalisation is applied only in this mode:
under the centralised critic both streams stay on the raw milestone scale so the GAE deltas are
computed consistently.

## Update

Standard clipped PPO with GAE, an entropy bonus annealed from 0.05 to 0.001 over 500 updates, and
a value clip (`ξ`). Each actor updates after accumulating `update_interval` transitions; advantages
are standardised per rollout.

Rewards entering the buffer are already diffused along the social graph when `--hebbian` is on, so
the graph changes *what* an agent learns from. `--hebbian-rho` additionally changes *whose*
experience it learns from: `_collect_social_replay` draws a `ρ`-fraction of the batch from
neighbours in proportion to their normalised bond, recomputes GAE on copies under the source
agent's baseline, and lets PPO's clipped ratio `π_i/π_j` act as the importance correction. One
shared pre-update snapshot per cycle keeps every agent drawing from the same view.

`--rl-mode token` swaps the discrete policy for sequence-level PPO over the emitted tokens; it is
kept as an alternative and is not used in the paper's results.

## Practicalities

- `--rl-update-stagger` spreads the per-agent update rounds instead of running them back to back.
  Required for Gemma runs, where a long simultaneous update round leaves the environment bridge
  idle and eventually hangs it.
- Checkpoints (`--checkpoint-interval`) carry adapters, optimiser state, the critic, the Hebbian
  graph and each agent's cognitive state, so a job that hits its wall-clock limit resumes exactly
  where it stopped with `--resume`.
- All defaults are in `RLConfig` and pinned against the paper's table by
  `tests/test_paper_defaults.py`.
