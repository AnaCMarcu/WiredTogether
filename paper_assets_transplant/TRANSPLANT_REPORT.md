# Pair-Bonding Transplant Experiment — Results Report

*Generated 2026-08-12 23:06 by `src/mindforge/tools/transplant_report.py` — every number is computed from the run artifacts; regenerate rather than hand-edit.*

## Design

**Question:** do agent pairs that *co-fired* together in one context keep
preferring each other ("wire together") in a new context — and is that carried
by the relationship actually having happened, rather than by bond magnitude or
memory volume?

- **Phase A** — 9 independent 2-agent runs (Gemma 4 E4B, exp08 flags,
  `--max-chamber 2`, 3 episodes x 1000 steps): solo skills + anvil co-op in
  Chambers 1–2. Ground-truth co-firing = anvil-break milestones (impossible
  alone: solo digging is net-zero).
- **Selection** — 2 genuine co-firing pairs + 1 non-co-firing control pair
  (most joint digs among non-co-firers).
- **Merge** — bonds: block-diagonal 6x6 W, per-block mean-normalized so every
  dyad starts equal; cross-pair entries at init weight 0.1. Memories: full
  per-agent skills / 150 episodic memories / curriculum, partner references
  renamed to the new seats.
- **Phase B** — 6-agent runs starting directly in Chamber 3
  (`--start-chamber 3`, 3 episodes x 1000 steps), two arms with **identical
  W** and identical memory volume:
  - **T (transplant)**: real pairs seated together — seats (0,1) and (2,3)
    GENUINE, seats (4,5) CONTROL (shared history, never co-fired);
  - **S (shuffled)**: strangers seated together, each holding memories that
    claim a shared history with the seatmate which never happened.
- **Readout** — seatmate preference P(message target = seatmate), chance =
  0.20 at N=6; co-earned milestones; bond-matrix evolution.

## Phase A (Gemma) — pair bonding, Chambers 1–2

8 completed runs (of 9 seeds). Ranked by final within-pair bond; **COFIRED** = earned a milestone that is impossible alone (anvil break / gear equip — solo digging is net-zero by construction).

| rank | run | bond (mean W01,W10) | joint digs | co-actions | COFIRED | co-firing milestones |
|---|---|---|---|---|---|---|
| 0 | seed_1213 | 0.2962 | 1 | 1212 | no | — |
| 1 | seed_456 | 0.2922 | 0 | 1682 | no | — |
| 2 | seed_1011 | 0.2728 | 0 | 1326 | no | — |
| 3 | seed_42 | 0.2726 | 6 | 1526 | no | — |
| 4 | seed_1617 | 0.2692 | 1 | 1194 | no | — |
| 5 | seed_1415 | 0.2687 | 1 | 980 | **YES** | m15_chestplate_equipped, m9_anvil_B1 |
| 6 | seed_789 | 0.2651 | 0 | 1318 | no | — |
| 7 | seed_123 | 0.2542 | 3 | 851 | **YES** | m15_chestplate_equipped, m9_anvil_B1 |
| 8 | seed_1819 | — | — | — | — | *incomplete: no final graph* |

**Key numbers:** 2/8 pairs genuinely co-fired; by bond they ranked 6, 8 of 8. Final bonds span only 0.2542–0.2962 (spread 0.0420).

**The bond is not a usable selection signal.** Between-seed spread of final W is 0.0420, while the mean within-seed episode-to-episode wobble is 0.0246 (max 0.0468, seed_1213) — the differences between seeds are the same size as each seed's own noise. Meanwhile the co-firing ground truth picks out different runs entirely: the genuine co-firers ranked 6, 8 of 8 by bond — the bond is (weakly) *anti*-correlated with real cooperative achievement, because the engagement term rewards proximity + constant messaging.

## Phase A replication (Qwen3.5-9B)

9 completed runs (of 9 seeds). Ranked by final within-pair bond; **COFIRED** = earned a milestone that is impossible alone (anvil break / gear equip — solo digging is net-zero by construction).

| rank | run | bond (mean W01,W10) | joint digs | co-actions | COFIRED | co-firing milestones |
|---|---|---|---|---|---|---|
| 0 | seed_456 | 0.3275 | 63 | 805 | no | — |
| 1 | seed_1617 | 0.2763 | 188 | 936 | no | — |
| 2 | seed_1011 | 0.2741 | 165 | 844 | no | — |
| 3 | seed_42 | 0.2557 | 213 | 865 | no | — |
| 4 | seed_1819 | 0.2552 | 127 | 837 | no | — |
| 5 | seed_1213 | 0.2541 | 249 | 929 | no | — |
| 6 | seed_789 | 0.2356 | 117 | 859 | **YES** | m14_sword_equipped, m8_anvil_A1 |
| 7 | seed_1415 | 0.2249 | 451 | 1058 | **YES** | m14_sword_equipped, m8_anvil_A1 |
| 8 | seed_123 | 0.2052 | 312 | 977 | **YES** | m15_chestplate_equipped, m9_anvil_B1 |

**Key numbers:** 3/9 pairs genuinely co-fired; by bond they ranked 7, 8, 9 of 9. Final bonds span only 0.2052–0.3275 (spread 0.1222).

The bond↔co-firing inversion **replicates across models**: Qwen's 3 genuine co-firers ranked 7, 8, 9 of 9 by bond (Qwen Phase B pending).

## Merged Phase B inputs (Gemma)

- Within-seat bond after block-mean normalization: **0.265** (every dyad identical by construction); cross-seat: **0.100**.
- W identical between arms: **True** — the arms differ only in who occupies each seat.
- Seats [0, 1]: **GENUINE** ← seed_123
- Seats [2, 3]: **GENUINE** ← seed_1415
- Seats [4, 5]: **CONTROL** ← seed_42

## Phase B results (Gemma) — behavioral wiring

### Seatmate preference (chance = 0.20)

| seed | transplant | shuffled | T − S |
|---|---|---|---|
| 42 | 0.44 | 0.28 | 0.16 |
| 123 | 0.68 | 0.52 | 0.16 |
| 456 | 0.57 | 0.42 | 0.15 |
| **pooled** | **0.561** | **0.404** | **+0.157** |

Transplant beats shuffled in **3/3 paired seeds** (sign-consistent), and both arms sit above the 0.20 chance level.

### Seat-pair breakdown (pooled over seeds)

| arm | seats | label | mean pref | msgs within (per seed) | co-milestones (per seed) |
|---|---|---|---|---|---|
| transplant | [0, 1] | GENUINE | 0.612 | [1914, 3806, 3685] | [2, 2, 2] |
| transplant | [2, 3] | GENUINE | 0.565 | [2057, 3338, 3218] | [2, 3, 0] |
| transplant | [4, 5] | CONTROL | 0.505 | [1759, 3695, 2260] | [0, 3, 2] |
| shuffled | [0, 1] | strangers | 0.398 | [572, 2136, 2114] | [1, 2, 4] |
| shuffled | [2, 3] | strangers | 0.356 | [477, 2719, 1381] | [1, 1, 1] |
| shuffled | [4, 5] | strangers | 0.457 | [2646, 2878, 666] | [2, 0, 0] |

Ordering as predicted: GENUINE (0.612, 0.565) > CONTROL (0.505) > strangers (0.356–0.457). The control dyad (shared history, no anvil) sits closer to genuine than to strangers — most of the effect is carried by *real shared history*, with genuine co-firing adding a further increment.

### Per-episode trend and the re-pairing event

| arm | seed | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| transplant | seed_123 | 0.62 | 0.86 | 0.61 |
| transplant | seed_42 | 0.77 | 0.02 | 0.58 |
| transplant | seed_456 | 0.92 | 0.60 | 0.29 |
| shuffled | seed_123 | 0.54 | 0.72 | 0.37 |
| shuffled | seed_42 | 0.41 | 0.39 | 0.05 |
| shuffled | seed_456 | 0.54 | 0.23 | 0.26 |

Preference is strongest early and noisy across episodes. The transplant/seed_42 ep2 collapse is NOT messaging breakdown — the team *re-paired* into different reciprocal dyads for one episode and returned to the transplanted pairs in ep3 (dominant message target per agent: ep1 `0→1 1→0 2→3 3→2 4→5 5→4`, ep2 `0→4 1→2 2→1 3→5 4→0 5→3`, ep3 `0→1 1→0 2→3 3→2 4→5 5→4`). Dyadic organization itself is robust; the transplanted pairing acts as the attractor teams return to.

### Bond-matrix evolution (mean within-seat W / mean cross-seat W at episode end)

| arm | seed | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| transplant | seed_123 | 0.123 / 0.137 | 0.288 / 0.086 | 0.136 / 0.136 |
| transplant | seed_42 | 0.299 / 0.083 | 0.092 / 0.123 | 0.283 / 0.096 |
| transplant | seed_456 | 0.290 / 0.090 | 0.175 / 0.121 | 0.138 / 0.120 |
| shuffled | seed_123 | 0.275 / 0.102 | 0.148 / 0.121 | 0.214 / 0.129 |
| shuffled | seed_42 | 0.156 / 0.132 | 0.158 / 0.122 | 0.089 / 0.130 |
| shuffled | seed_456 | 0.162 / 0.136 | 0.171 / 0.155 | 0.147 / 0.128 |

Both arms start at the same 0.265. Episode-1 within-seat means: transplant 0.299, 0.123, 0.290 vs shuffled 0.156, 0.275, 0.162 — leaning toward survival for true pairs, but with one crossover each way at n=3: **W is a fast readout of the current episode's pairing behavior, not a persistent store** (the seed_42 dip to 0.092 lands exactly in its re-pairing episode and recovers to 0.283 when the pairs re-form). The persistent carrier of the relationship is the episodic memory.

## Phase B task performance

Milestone completion counts a milestone type as done if any agent earned it in any episode, out of the 13 task milestones achievable in Chambers 3–5 (Ch1/Ch2 milestones are unreachable in a `--start-chamber 3` run; per-chamber communication rewards `m_comm_*` are excluded).

| arm | seed | milestones completed (of 13) | reward events | mean cum. return | reached Ch5 | switch puzzle done | wall time (h) |
|---|---|---|---|---|---|---|---|
| transplant | seed_123 | 4/13 (**31%**) | 92 | 357 | yes | yes | 89.0 |
| transplant | seed_42 | 4/13 (**31%**) | 77 | 320 | yes | yes | 73.0 |
| transplant | seed_456 | 4/13 (**31%**) | 90 | 472 | yes | yes | 89.2 |
| shuffled | seed_123 | 3/13 (**23%**) | 76 | 347 | yes | NO | 81.8 |
| shuffled | seed_42 | 4/13 (**31%**) | 80 | 377 | yes | yes | 76.0 |
| shuffled | seed_456 | 4/13 (**31%**) | 81 | 314 | yes | yes | 55.6 |

- **transplant**: mean completion 31%, mean return 383.

- **shuffled**: mean completion 28%, mean return 346.

Transplant shows a small edge (31% vs 28% mean completion) — a trend, not a claim. Every run reached Ch5; wiring together is a social-structural effect, not a task-competence one, which also rules out 'transplant helps because it boosts performance' as a confound for the preference results.

Milestones completed somewhere in the suite: `m17_switch_pressed`, `m18_door_opened`, `m20_enter_ch4`, `m21_first_mob_kill`, `m24_enter_ch5`. Never completed by any run: `m16_enter_cell`, `m19_all_in_communal`, `m22_all_mobs_killed`, `m23_all_alive_ch4`, `m25_first_boss_dmg`, `m26_boss_half_hp`, `m27_boss_defeated`, `m28_all_alive_bonus` — the communal-regroup gate (m19) and the full combat/boss clears remain beyond 6-agent teams at these horizons.

*Known artifact: `m1_move_5` (a Ch1 milestone) leaked 18 reward events across the six runs despite Ch1 never being visited — the Ch1 movement tracker fires mid-episode under rare conditions. Excluded from the completion metric; worth a Lua fix before reusing the ch1_solo track in analyses.*

## Findings (summary)

1. **Pairs that co-fired together still wire together.** Transplanted real
   pairs message their former partner at 0.56 (chance 0.20) vs 0.40
   for strangers holding identical bonds and equally detailed fabricated
   memories — transplant > shuffled in 3/3 paired seeds.
2. **The effect is carried by the relationship being real.** Identical W,
   identical memory volume; only the truth of the pairing differs between
   arms. Within the transplant arm: GENUINE > CONTROL > strangers, with the
   control dyad (real history, no co-fired achievement) much closer to
   genuine — shared history is the main carrier, co-firing adds on top.
3. **Hebbian W is a live index, not the memory.** Bond values track the
   current episode's pairing behavior (dip during the re-pairing episode,
   recovery after) and sit at a fixed point that erases initialization over
   time; the transplanted episodic memories are what keep pulling agents
   back to their partners.
4. **The transplanted pairing is an attractor.** One team re-paired into
   different reciprocal dyads for a full episode and spontaneously returned
   to the transplanted pairs the next episode.
5. **Bond magnitude is anti-correlated with genuine co-firing at selection
   time** (Phase A, both models): the pairs that actually broke anvils ranked
   last by bond, because the engagement term rewards proximity + constant
   messaging. Any future use of W as a selection signal needs a co-firing
   ground truth next to it.
6. **No meaningful task-performance difference** between arms (small
   transplant-side trend); the wiring effect is not explained by competence.

## Caveats

- n = 3 seeds per arm, one model (Gemma 4 E4B); the Qwen3.5-9B
  Phase B replication is running and this report should be regenerated when
  it lands.
- Seatmate preference is message-based; proximity/joint-action based measures
  would strengthen the claim.
- The shuffled arm shows emergent *new* pairs (strangers bootstrapping real
  partnerships mid-run) — expected with plasticity on, and it keeps the
  baseline honest, but it means the S-arm number is not a floor.
- Phase A pair selection affects only the transplanted memories: after
  block-mean normalization the merged W is identical whichever pairs are
  chosen.
