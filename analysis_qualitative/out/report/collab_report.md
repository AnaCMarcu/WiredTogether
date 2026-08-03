# Collaboration case mining — success/failure episodes, Hebbian vs baseline

Detected **57289 events** across all runs (`collab/events.jsonl.gz`). A *request* is a valid directed message classified request/directive; it is **fulfilled** if the target performs the asked action within 8 steps or a real (non-timer) cooperative milestone fires within 30 steps, **answered** if it only gets a directed reply, **ignored** otherwise. Milestones within ±5 steps of chamber-timer boundaries and entry milestones are never counted as successes. RL rows use the rl_thoughts comm stream; treat their message text as generated commentary, not the policy input.

## Condition-level outcomes

| label | n_runs | n_requests | frac_fulfilled | frac_ignored | comply_rate | n_coop_success | frac_comm_before_coop | n_deadlocks | n_ch3_stalls | ch3_stall_latency_mean | n_ch2_neglect |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Allied-all | 6 | 2128 | 0.5042 | 0.007 | 0.7114 | 55 | 1.0 | 152 | 3 | 148.0 | 17 |
| Allied-pair | 6 | 2312 | 0.4485 | 0.0091 | 0.6664 | 37 | 1.0 | 160 | 6 |  | 18 |
| IPPO | 6 | 5232 | 0.5921 | 0.0155 | 0.646 | 16 | 1.0 | 60 | 10 |  | 17 |
| IPPO+Heb | 5 | 4260 | 0.5746 | 0.0113 | 0.6144 | 10 | 1.0 | 48 | 9 |  | 15 |
| LLM-2B | 6 | 9031 | 0.8581 | 0.0047 | 0.9198 | 33 | 1.0 | 94 | 7 |  | 18 |
| LLM-2B+Heb | 6 | 10156 | 0.799 | 0.0091 | 0.918 | 25 | 1.0 | 102 | 9 |  | 18 |
| LLM-9B | 6 | 3717 | 0.4587 | 0.0132 | 0.6639 | 59 | 1.0 | 151 | 5 |  | 16 |
| LLM-9B+Heb | 6 | 2270 | 0.4916 | 0.0097 | 0.7037 | 62 | 1.0 | 156 | 6 |  | 17 |
| MAPPO | 6 | 7614 | 0.5107 | 0.0206 | 0.5791 | 10 | 1.0 | 113 | 13 |  | 18 |
| MAPPO+Heb | 6 | 6730 | 0.6349 | 0.0141 | 0.678 | 10 | 1.0 | 88 | 14 |  | 18 |
| No-bonds | 5 | 1955 | 0.4404 | 0.0102 | 0.6997 | 42 | 1.0 | 127 | 7 | 86.6 | 13 |

## Matched pairs — baseline vs Hebbian

| pair | condition | n_requests | frac_fulfilled | frac_ignored | comply_rate | n_coop_success | frac_comm_before_coop | n_deadlocks | mean_W_fulfilled | mean_W_ignored | eps_W_fulfilled_higher | eps_W_fulfilled_lower | eps_W_tie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM-2B vs LLM-2B+Heb | LLM-2B | 9031 | 0.8581 | 0.0047 | 0.9198 | 33 | 1.0 | 94 |  |  | 0 | 0 | 0 |
| LLM-2B vs LLM-2B+Heb | LLM-2B+Heb | 10156 | 0.799 | 0.0091 | 0.918 | 25 | 1.0 | 102 | 0.2432 | 0.2478 | 8 | 9 | 0 |
| LLM-9B vs LLM-9B+Heb | LLM-9B | 3717 | 0.4587 | 0.0132 | 0.6639 | 59 | 1.0 | 151 |  |  | 0 | 0 | 0 |
| LLM-9B vs LLM-9B+Heb | LLM-9B+Heb | 2270 | 0.4916 | 0.0097 | 0.7037 | 62 | 1.0 | 156 | 0.2482 | 0.2205 | 8 | 4 | 0 |
| IPPO vs IPPO+Heb | IPPO | 5232 | 0.5921 | 0.0155 | 0.646 | 16 | 1.0 | 60 |  |  | 0 | 0 | 0 |
| IPPO vs IPPO+Heb | IPPO+Heb | 4260 | 0.5746 | 0.0113 | 0.6144 | 10 | 1.0 | 48 | 0.2796 | 0.1766 | 9 | 0 | 0 |
| MAPPO vs MAPPO+Heb | MAPPO | 7614 | 0.5107 | 0.0206 | 0.5791 | 10 | 1.0 | 113 |  |  | 0 | 0 | 0 |
| MAPPO vs MAPPO+Heb | MAPPO+Heb | 6730 | 0.6349 | 0.0141 | 0.678 | 10 | 1.0 | 88 | 0.2649 | 0.1821 | 14 | 2 | 0 |

`mean_W_fulfilled` vs `mean_W_ignored` (Hebbian rows only): mean end-of-episode bond strength on the sender->target edge of fulfilled vs ignored requests. A gap in favour of fulfilled means bonds track who actually responds. `eps_W_fulfilled_higher/lower` is the paired within-episode version (episodes containing both outcomes), immune to cross-episode W-scale differences; static-topology controls tie by construction, which doubles as a sanity check of the measure.

## Exemplar transcripts (`cases/collab/`)

- **collab_success_Allied-all_0** — `exp09_llm_9b_allied_all/seed_1011` ep1 t398-433: m18_door_opened (+60) at t=428, comm_before_coop=True
- **collab_success_Allied-all_1** — `exp09_llm_9b_allied_all/seed_1011` ep2 t442-477: m18_door_opened (+60) at t=472, comm_before_coop=True
- **collab_success_Allied-all_2** — `exp09_llm_9b_allied_all/seed_1011` ep2 t664-699: m21_first_mob_kill (+60) at t=694, comm_before_coop=True
- **collab_failure_Allied-all_0** — `exp09_llm_9b_allied_all/seed_42` ep1 t506-566: locked >? steps before NO real press (202 switch-talk msgs)
- **collab_failure_Allied-all_1** — `exp09_llm_9b_allied_all/seed_1213` ep2 t328-378: reached ch2, first anvil never broken (265 anvil msgs, 244 ch2 digs)
- **collab_failure_Allied-all_2** — `exp09_llm_9b_allied_all/seed_1011` ep1 t424-472: a1<->a0 both ask, nobody presses: "Agent_0, I am looking down to find the exit path. My door st" / "I am turning left to find the exit gap in my cell. Let me kn"
- **collab_success_Allied-pair_0** — `exp10_llm_9b_allied_pair/seed_1011` ep1 t474-509: m18_door_opened (+60) at t=504, comm_before_coop=True
- **collab_success_Allied-pair_1** — `exp10_llm_9b_allied_pair/seed_1011` ep1 t492-522: fulfilled request/status_report a0->a2 -> m17_switch_pressed,m18_door_opened: "I am moving forward to explore the cell. Please confirm if your door is open aft"
- **collab_success_Allied-pair_2** — `exp10_llm_9b_allied_pair/seed_1011` ep1 t733-768: m21_first_mob_kill (+60) at t=763, comm_before_coop=True
- **collab_failure_Allied-pair_0** — `exp10_llm_9b_allied_pair/seed_1213` ep3 t538-598: locked >? steps before NO real press (133 switch-talk msgs)
- **collab_failure_Allied-pair_1** — `exp10_llm_9b_allied_pair/seed_1011` ep2 t264-314: reached ch2, first anvil never broken (242 anvil msgs, 240 ch2 digs)
- **collab_failure_Allied-pair_2** — `exp10_llm_9b_allied_pair/seed_1011` ep1 t397-445: a0<->a1 both ask, nobody presses: "I am pressing the switch in my cell now to open your door. P" / "I am searching for the blue switch in my cell. Please check "
- **collab_success_IPPO_0** — `exp04_ippo/seed_123` ep2 t435-470: m18_door_opened (+60) at t=465, comm_before_coop=True
- **collab_success_IPPO_1** — `exp04_ippo/seed_123` ep3 t482-517: m18_door_opened (+60) at t=512, comm_before_coop=True
- **collab_success_IPPO_2** — `exp04_ippo/seed_42` ep2 t462-497: m18_door_opened (+60) at t=492, comm_before_coop=True
- **collab_failure_IPPO_0** — `exp04_ippo/seed_42` ep3 t715-745: ignored directive/commitment a0->a1 streak x7: "Move forward to collect the cobblestone and continue toward the teleporter in Ch"
- **collab_failure_IPPO_1** — `exp04_ippo/seed_456` ep3 t711-771: locked >? steps before NO real press (179 switch-talk msgs)
- **collab_failure_IPPO_2** — `exp04_ippo/seed_123` ep1 t700-750: reached ch2, first anvil never broken (332 anvil msgs, 271 ch2 digs)
- **collab_success_IPPOpHeb_0** — `exp06_ippo_hebbian/seed_123` ep3 t429-464: m18_door_opened (+60) at t=459, comm_before_coop=True
- **collab_success_IPPOpHeb_1** — `exp06_ippo_hebbian/seed_42` ep3 t496-531: m18_door_opened (+60) at t=526, comm_before_coop=True
- **collab_success_IPPOpHeb_2** — `exp06_ippo_hebbian/seed_456` ep1 t409-444: m18_door_opened (+60) at t=439, comm_before_coop=True
- **collab_failure_IPPOpHeb_0** — `exp06_ippo_hebbian/seed_42` ep1 t660-720: locked >? steps before NO real press (89 switch-talk msgs)
- **collab_failure_IPPOpHeb_1** — `exp06_ippo_hebbian/seed_42` ep2 t488-518: ignored request/status_report a0->a1 streak x4: "Switch is centered and ready for cooperative Dig. Please press it to open the do"
- **collab_failure_IPPOpHeb_2** — `exp06_ippo_hebbian/seed_42` ep1 t297-347: reached ch2, first anvil never broken (152 anvil msgs, 100 ch2 digs)
- **collab_success_LLM-2B_0** — `exp01_llm_2b/seed_1011` ep1 t414-449: m18_door_opened (+60) at t=444, comm_before_coop=True
- **collab_success_LLM-2B_1** — `exp01_llm_2b/seed_1011` ep2 t416-451: m18_door_opened (+60) at t=446, comm_before_coop=True
- **collab_success_LLM-2B_2** — `exp01_llm_2b/seed_1011` ep3 t402-437: m18_door_opened (+60) at t=432, comm_before_coop=True
- **collab_failure_LLM-2B_0** — `exp01_llm_2b/seed_1213` ep3 t428-458: ignored directive a2->a0 streak x6: "Press the switch to open my door."
- **collab_failure_LLM-2B_1** — `exp01_llm_2b/seed_123` ep1 t1623-1683: locked >? steps before NO real press (168 switch-talk msgs)
- **collab_failure_LLM-2B_2** — `exp01_llm_2b/seed_1011` ep1 t278-328: reached ch2, first anvil never broken (71 anvil msgs, 222 ch2 digs)
- **collab_success_LLM-2BpHeb_0** — `exp07_llm_2b_social_prompt/seed_1213` ep3 t728-763: m21_first_mob_kill (+60) at t=758, comm_before_coop=True
- **collab_success_LLM-2BpHeb_1** — `exp07_llm_2b_social_prompt/seed_123` ep1 t418-453: m18_door_opened (+60) at t=448, comm_before_coop=True
- **collab_success_LLM-2BpHeb_2** — `exp07_llm_2b_social_prompt/seed_123` ep3 t396-431: m18_door_opened (+60) at t=426, comm_before_coop=True
- **collab_failure_LLM-2BpHeb_0** — `exp07_llm_2b_social_prompt/seed_456` ep3 t557-587: ignored request a1->a2 streak x17: "Please press the switch to enter Ch4."
- **collab_failure_LLM-2BpHeb_1** — `exp07_llm_2b_social_prompt/seed_1011` ep3 t667-727: locked >? steps before NO real press (84 switch-talk msgs)
- **collab_failure_LLM-2BpHeb_2** — `exp07_llm_2b_social_prompt/seed_1011` ep1 t334-384: reached ch2, first anvil never broken (143 anvil msgs, 245 ch2 digs)
- **collab_success_LLM-9B_0** — `exp02_llm_9b/seed_1011` ep1 t472-507: m18_door_opened (+60) at t=502, comm_before_coop=True
- **collab_success_LLM-9B_1** — `exp02_llm_9b/seed_1011` ep1 t481-511: fulfilled request a0->a1 -> m18_door_opened: "I am turning left to find the blue switch. I cannot see it yet, please check you"
- **collab_success_LLM-9B_2** — `exp02_llm_9b/seed_1011` ep1 t608-643: m21_first_mob_kill (+60) at t=638, comm_before_coop=True
- **collab_failure_LLM-9B_0** — `exp02_llm_9b/seed_1011` ep1 t504-534: ignored request/status_report a2->a0 streak x7: "I am pressing my switch now, please press yours to open your door."
- **collab_failure_LLM-9B_1** — `exp02_llm_9b/seed_1011` ep3 t482-542: locked >? steps before NO real press (248 switch-talk msgs)
- **collab_failure_LLM-9B_2** — `exp02_llm_9b/seed_1011` ep2 t256-306: reached ch2, first anvil never broken (327 anvil msgs, 246 ch2 digs)
- **collab_success_LLM-9BpHeb_0** — `exp08_llm_9b_social_prompt/seed_1011` ep2 t381-416: m18_door_opened (+60) at t=411, comm_before_coop=True
- **collab_success_LLM-9BpHeb_1** — `exp08_llm_9b_social_prompt/seed_1011` ep2 t401-431: fulfilled request/status_report a0->a1 -> m18_door_opened,m17_switch_pressed,m18_door_opened: "I am pressing my switch now. Please move to your cell and press your switch next"
- **collab_success_LLM-9BpHeb_2** — `exp08_llm_9b_social_prompt/seed_1011` ep3 t419-454: m18_door_opened (+60) at t=449, comm_before_coop=True
- **collab_failure_LLM-9BpHeb_0** — `exp08_llm_9b_social_prompt/seed_1213` ep1 t538-598: locked >? steps before NO real press (90 switch-talk msgs)
- **collab_failure_LLM-9BpHeb_1** — `exp08_llm_9b_social_prompt/seed_1213` ep1 t200-250: reached ch2, first anvil never broken (250 anvil msgs, 257 ch2 digs)
- **collab_failure_LLM-9BpHeb_2** — `exp08_llm_9b_social_prompt/seed_1011` ep1 t395-443: a1<->a0 both ask, nobody presses: "I am pressing my switch now. Wait for me to free you." / "I am turning right to face South. Once I find the switch, I'"
- **collab_success_MAPPO_0** — `exp03_mappo/seed_123` ep2 t521-556: m18_door_opened (+60) at t=551, comm_before_coop=True
- **collab_success_MAPPO_1** — `exp03_mappo/seed_456` ep2 t457-492: m18_door_opened (+60) at t=487, comm_before_coop=True
- **collab_success_MAPPO_2** — `exp03_mappo/seed_456` ep3 t483-518: m18_door_opened (+60) at t=513, comm_before_coop=True
- **collab_failure_MAPPO_0** — `exp03_mappo/seed_1213` ep3 t194-224: ignored request/status_report a0->a1 streak x12: "I see the iron ore block centered in front of us. Please pair-dig with me to rai"
- **collab_failure_MAPPO_1** — `exp03_mappo/seed_1011` ep1 t744-799: locked >? steps before NO real press (68 switch-talk msgs)
- **collab_failure_MAPPO_2** — `exp03_mappo/seed_42` ep1 t268-318: reached ch2, first anvil never broken (375 anvil msgs, 129 ch2 digs)
- **collab_success_MAPPOpHeb_0** — `exp05_mappo_hebbian/seed_1213` ep1 t484-519: m18_door_opened (+60) at t=514, comm_before_coop=True
- **collab_success_MAPPOpHeb_1** — `exp05_mappo_hebbian/seed_123` ep1 t474-509: m18_door_opened (+60) at t=504, comm_before_coop=True
- **collab_success_MAPPOpHeb_2** — `exp05_mappo_hebbian/seed_42` ep2 t494-529: m18_door_opened (+60) at t=524, comm_before_coop=True
- **collab_failure_MAPPOpHeb_0** — `exp05_mappo_hebbian/seed_789` ep3 t259-289: ignored directive a0->a2 streak x10: "Turn left to center the tree trunk so we can start digging the anvil."
- **collab_failure_MAPPOpHeb_1** — `exp05_mappo_hebbian/seed_123` ep2 t664-724: locked >? steps before NO real press (72 switch-talk msgs)
- **collab_failure_MAPPOpHeb_2** — `exp05_mappo_hebbian/seed_789` ep1 t136-186: reached ch2, first anvil never broken (48 anvil msgs, 157 ch2 digs)
- **collab_success_No-bonds_0** — `exp11_llm_9b_allied_none/seed_1011` ep1 t735-770: m21_first_mob_kill (+60) at t=765, comm_before_coop=True
- **collab_success_No-bonds_1** — `exp11_llm_9b_allied_none/seed_1011` ep2 t388-418: fulfilled request/status_report a1->a2 -> m18_door_opened: "I am digging the anvil now, please dig it too to break it faster."
- **collab_success_No-bonds_2** — `exp11_llm_9b_allied_none/seed_1011` ep3 t378-413: m18_door_opened (+60) at t=408, comm_before_coop=True
- **collab_failure_No-bonds_0** — `exp11_llm_9b_allied_none/seed_1213` ep3 t527-587: locked >? steps before NO real press (220 switch-talk msgs)
- **collab_failure_No-bonds_1** — `exp11_llm_9b_allied_none/seed_1011` ep2 t252-302: reached ch2, first anvil never broken (289 anvil msgs, 205 ch2 digs)
- **collab_failure_No-bonds_2** — `exp11_llm_9b_allied_none/seed_42` ep1 t592-622: ignored request/status_report a2->a1 streak x4: "I am in Cell C and have pressed my switch. This should open your door in Cell A."
