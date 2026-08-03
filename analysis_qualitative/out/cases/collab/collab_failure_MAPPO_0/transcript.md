# Case: collab_failure_MAPPO_0

**Run:** `exp03_mappo/seed_1213` (MAPPO)  **Episode:** 3  **Steps:** 194-224
**Why shortlisted:** ignored request/status_report a0->a1 streak x12: "I see the iron ore block centered in front of us. Please pair-dig with me to rai"

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 194 | a0 | ch1 | MoveBackward | ->agent_1: I see the iron ore block centered in front of us. Please pair-dig with me to raise the anv | NEW TASK: Move forward to center the zombies; expect the 3 zombies to be centere |
| 194 | a1 | ch1 | Drop | ->agent_1 [random_fallback:self_target]: Agent_1: Pair-dig the iron ore to raise HP and drop a diamond sword. |  |
| 194 | a2 | ch1 | NoOp | ->agent_1: I see the iron ore centered left. Pair-dig with agent_1 to raise HP. |  |
| 195 | a0 | ch1 | NoOp | ->agent_1: Agent_1: Iron ore is centered. Dig to break and pair-dig. | NEW TASK: Coordinate with Agent_1 to pair-dig the iron ore block; expect the iro |
| 195 | a1 | ch1 | LookUp | ->agent_1 [random_fallback:self_target]: Agent_1, I see the iron ore centered left. Pair-dig with me to raise HP. |  |
| 195 | a2 | ch1 | LookUp | ->agent_1: I see the iron ore centered, ready to pair-dig with you and Agent_2 to raise HP. |  |
| 196 | a0 | ch1 | LookUp | ->agent_1: Dig the iron ore, move forward, and pair-dig with Agent_1 to raise the anvil HP. | NEW TASK: Center the iron ore block; expect the inventory to update with iron or |
| 196 | a1 | ch1 | Jump | ->agent_2: Agent_2, the iron ore is centered and ready for pair-digging. Dig to break it. |  |
| 196 | a2 | ch1 | Sneak | ->agent_1: agent_1: Iron ore centered, ready to pair-dig with you and agent_2 to raise HP. |  |
| 197 | a0 | ch1 | Jump | ->agent_1: Agent_1: Iron ore is perfectly centered. I am ready to break it. Pair-dig with me. | NEW TASK: Re-center the iron ore block; expect the iron ore to be perfectly cent |
| 197 | a1 | ch1 | MoveLeft | ->agent_1 [random_fallback:self_target]: agent_1: Iron ore centered, ready to pair-dig with you to raise HP. |  |
| 197 | a2 | ch1 | NoOp | ->agent_1: Turn right to center the sword on the iron ore, then dig to raise HP. I expect the sword t |  |
| 198 | a0 | ch1 | MoveLeft | ->agent_1: I broke the iron ore. Pair-dig with me to raise HP. | NEW TASK: Wait for the iron ore block to center and confirm HP increase; expect  |
| 198 | a1 | ch1 | MoveLeft | ->agent_2: Agent_2, please pair-dig with me to raise the iron ore's HP. |  |
| 198 | a2 | ch1 | MoveBackward | ->agent_1: Dig to break the iron ore and raise HP. |  |
| 199 | a0 | ch1 | TurnLeft | ->agent_1: I am digging the stone block in front of me. Please pair-dig with me to raise the Row A an | critic:FAIL |
| 199 | a1 | ch1 | Drop | ->agent_1 [random_fallback:self_target]: Move forward to the right to find the iron ore block. | critic:FAIL |
| 199 | a2 | ch1 | Sneak | ->agent_1: I need to pair-dig with agent_1 to raise the iron ore's HP. Please pair-dig with me to rai |  |
| 200 | a0 | ch2 | LookDown | ->agent_1: I see the iron ore centered and ready for break. Pair-dig with Agent_1 to raise HP. |  |
| 200 | a1 | ch2 | LookUp | ->agent_1 [random_fallback:self_target]: I see the iron ore block centered. Please pair-dig with me to raise its HP. |  |
| 200 | a2 | ch2 | Dig | ->agent_1: Agent_1: Iron ore centered, ready to pair-dig with you to raise HP. |  |
| 201 | a0 | ch2 | Drop | ->agent_1: Agent_1, we are centered on the iron ore and ready to pair-dig. Please pair-dig with me to | NEW TASK: Orient team to the iron ore block; expect to see the iron ore centered |
| 201 | a1 | ch2 | Jump | ->agent_1 [random_fallback:self_target]: Agent_1: Iron ore is ahead. Please pair-dig with me to raise HP. | NEW TASK: Pair-dig with agent_1 to raise the iron ore's HP and drop the diamond  |
| 201 | a2 | ch2 | Place | ->agent_1: agent_1: Iron ore centered. Please pair-dig with me to raise its HP. |  |
| 202 | a0 | ch2 | MoveBackward | ->agent_1: Agent_1: Iron ore is centered and ready for pair-dig. Please pair-dig with me to raise HP. |  |
| 202 | a1 | ch2 | Sneak | ->agent_1 [random_fallback:self_target]: agent_1: Iron ore centered. Please pair-dig with me to raise its HP. |  |
| 202 | a2 | ch2 | NoOp | ->agent_1: Agent_1, I see the iron ore block centered. Please pair-dig with me to raise its HP. I'm r |  |
| 203 | a0 | ch2 | MoveBackward | ->agent_1: Agent_1: Iron ore is centered. Please pair-dig with me to raise its HP. |  |
| 203 | a1 | ch2 | Place | ->agent_1 [random_fallback:self_target]: Agent_1, I am centered on the iron ore block and ready to pair-dig. Please pair-dig with m |  |
| 203 | a2 | ch2 | MoveLeft | ->agent_1: Agent_1, please pair-dig with me to raise the iron ore's HP. |  |
| 204 | a0 | ch2 | MoveRight | ->agent_1: Agent_1: Stone block is centered. Please pair-dig with me to raise HP. |  |
| 204 | a1 | ch2 | NoOp | ->agent_1 [random_fallback:self_target]: I see Agent_2 centered on the iron ore. I will move forward to approach it and pair-dig wi |  |
| 204 | a2 | ch2 | Place | ->agent_1: Please pair-dig with me to raise the iron ore block's HP. |  |
| 205 | a0 | ch2 | Jump | ->agent_1: I am centered on the iron ore, ready to pair-dig with Agent_1 to raise its HP. |  |
| 205 | a1 | ch2 | Drop | ->agent_1 [random_fallback:self_target]: Pair-dig with me to raise the iron ore block's HP. |  |
| 205 | a2 | ch2 | MoveBackward | ->agent_1: I see the iron ore block centered. Please pair-dig with me to raise its HP. |  |
| 206 | a0 | ch2 | Place | ->agent_1: I am aiming at the iron ore block and ready to pair-dig with Agent_1. | ** m_comm_ch2 (+10.0) **; ** m_comm_ch2 (+10.0) ** |
| 206 | a1 | ch2 | MoveRight | ->agent_2: I see the iron ore block centered and ready. Pair-digging with agent_2 to raise its HP. | ** m_comm_ch2 (+10.0) **; ** m_comm_ch2 (+10.0) ** |
| 206 | a2 | ch2 | LookDown | ->agent_1: I am centered on the iron ore block and ready to pair-dig. Please pair-dig with me to rais | ** m_comm_ch2 (+10.0) **; ** m_comm_ch2 (+10.0) ** |
| 207 | a0 | ch2 | Place | ->agent_1: Turn left to face the iron ore, move forward to approach it, then center it for digging. | ** m_comm_ch2 (+10.0) ** |
| 207 | a1 | ch2 | Place | ->agent_2: Agent_2, I am ready to pair-dig with you to raise the iron ore's HP. | ** m_comm_ch2 (+10.0) ** |
| 207 | a2 | ch2 | LookUp | ->agent_1: agent_1: Move to the exit, then pair-dig with agent_2 to raise the iron ore's HP and drop  | ** m_comm_ch2 (+10.0) ** |
| 208 | a0 | ch2 | Place | ->agent_1: I see Agent_1 centered on the iron ore. Please pair-dig with me to raise its HP. |  |
| 208 | a1 | ch2 | NoOp | ->agent_2: Iron ore centered in front; ready to pair-dig with Agent_2 to raise HP. |  |
| 208 | a2 | ch2 | LookUp | ->agent_1: Agent_1, I am ready to pair-dig with you to raise the iron ore's HP. |  |
| 209 | a0 | ch2 | MoveForward | ->agent_1: I see the iron ore on the left. Pair-dig with Agent_1 to raise its HP. |  |
| 209 | a1 | ch2 | Place | ->agent_1 [random_fallback:self_target]: I am centered on the iron ore and ready to pair-dig with Agent_1 to raise its HP. |  |
| 209 | a2 | ch2 | Drop | ->agent_1: Agent_1, Iron ore is centered and ready for pair-dig. |  |
| 210 | a0 | ch2 | Place | ->agent_1: Dig the iron ore, pair-dig with Agent_1 to raise HP. |  |
| 210 | a1 | ch2 | MoveLeft | ->agent_1 [random_fallback:self_target]: Agent_1, Iron ore is centered and ready for pair-dig. Pair-dig with you to raise its HP. |  |
| 210 | a2 | ch2 | Place | ->agent_1: Agent_1: Iron ore centered and ready to pair-dig with you to raise HP. |  |
| 211 | a0 | ch2 | Jump | ->agent_2: Agent_2, iron ore centered. I will pair-dig with you to raise its HP. |  |
| 211 | a1 | ch2 | MoveForward | ->agent_0: Agent_0: Iron ore centered and ready to pair-dig with Agent_1 to raise HP. |  |
| 211 | a2 | ch2 | Dig | ->agent_1: Agent_1, Iron ore is centered and ready for pair-dig. Pair-dig with you to raise its HP. |  |
| 212 | a0 | ch2 | LookUp | ->agent_1: I am centered on the iron ore, ready to pair-dig with Agent_1 to raise HP. |  |
| 212 | a1 | ch2 | LookUp | ->agent_2: Agent_2: Iron ore centered and ready for pair-dig. Pair-dig with me to raise its HP. |  |
| 212 | a2 | ch2 | MoveLeft | ->agent_1: Pair-dig with you to raise its HP. |  |
| 213 | a0 | ch2 | Jump | ->agent_1: I will pair-dig with Agent_1 to raise the iron ore's HP. |  |
| 213 | a1 | ch2 | MoveBackward | ->agent_2: Agent_2, I am centered on the iron ore, ready to pair-dig with you. |  |
| 213 | a2 | ch2 | LookUp | ->agent_1: Agent_1: Iron ore centered, ready for pair-dig. Please move north to the exit. |  |
| 214 | a0 | ch2 | Drop | ->agent_1: Agent_1, Iron ore centered and ready to pair-dig with you. |  |
| 214 | a1 | ch2 | NoOp | ->agent_1 [random_fallback:self_target]: Agent_1, I am alone in this chamber. Please pair-dig with me to raise the iron ore's HP. |  |
| 214 | a2 | ch2 | Dig | ->agent_1: I am centered on the iron ore, ready to pair-dig with you to raise its HP. |  |
| 215 | a0 | ch2 | MoveBackward | ->agent_1: I see Agent_2 centered on the iron ore. I will move forward to approach it and pair-dig wi |  |
| 215 | a1 | ch2 | MoveRight | ->agent_0: Agent_0, I am looking for the iron ore block to pair-dig with you. |  |
| 215 | a2 | ch2 | Place | ->agent_2 [random_fallback:self_target]: I am centered on the iron ore, ready to pair-dig with Agent_2 to raise its HP. I expect th |  |
| 216 | a0 | ch2 | MoveRight | ->agent_1: Agent_1, Iron ore is centered and ready to pair-dig with you to raise HP. |  |
| 216 | a1 | ch2 | Drop | ->agent_2: I am centered on the iron ore with Agent_2, ready to pair-dig and raise its HP. |  |
| 216 | a2 | ch2 | TurnRight | ->agent_1: Agent_1, I am centered on the iron ore, ready to pair-dig with you. |  |
| 217 | a0 | ch2 | MoveRight | ->agent_1: Agent_1, Iron ore centered and ready to pair-dig with you. |  |
| 217 | a1 | ch2 | NoOp | ->agent_1 [random_fallback:self_target]: Agent_1, I am centered on the iron ore, ready to pair-dig with you. |  |
| 217 | a2 | ch2 | MoveBackward | ->agent_1: Agent_1, I am centered on the iron ore with you, ready to pair-dig and raise its HP. |  |
| 218 | a0 | ch2 | MoveRight | ->agent_1: Agent_1, I am looking for the iron ore block to pair-dig with you. |  |
| 218 | a1 | ch2 | NoOp | ->agent_2: Agent_2, I am centered on the iron ore with you, ready to pair-dig and raise its HP. |  |
| 218 | a2 | ch2 | TurnRight | ->agent_1: agent_1: I am centered on the iron ore, ready to pair-dig. Please pair-dig with me to rais |  |
| 219 | a0 | ch2 | MoveBackward | ->agent_1: I am centered on the iron ore and ready to pair-dig with you. | critic:FAIL |
| 219 | a1 | ch2 | MoveForward | ->agent_2: Agent_2, I am centered on the iron ore, ready to pair-dig. Please pair-dig with me to rais | critic:FAIL |
| 219 | a2 | ch2 | Place | ->agent_1: I am centered on the iron ore with Agent_1, ready to pair-dig and raise its HP. |  |
| 220 | a0 | ch2 | Drop | ->agent_1: Agent_1, I am centered on the iron ore, ready to pair-dig with you. |  |
| 220 | a1 | ch2 | MoveLeft | ->agent_1 [random_fallback:self_target]: I am centered on the iron ore, ready to pair-dig with you. |  |
| 220 | a2 | ch2 | Drop | ->agent_1: Agent_1, I am ready to pair-dig with you to raise the iron ore's HP. |  |
| 221 | a0 | ch2 | MoveBackward | ->agent_1: I am centered on the iron ore and ready to pair-dig with Agent_1 to raise its HP. |  |
| 221 | a1 | ch2 | MoveLeft | ->agent_2: Agent_2, I am centered on the iron ore, ready to pair-dig with you. Please pair-dig with m |  |
| 221 | a2 | ch2 | LookUp | ->agent_1: agent_1: We are centered on the iron ore, ready to pair-dig. Please pair-dig with me to ra |  |
| 222 | a0 | ch2 | TurnRight | ->agent_1: I am centered on the iron ore and ready to pair-dig with Agent_1. |  |
| 222 | a1 | ? | Jump | ->agent_2: I am centered on the iron ore and ready to pair-dig with Agent_2 to raise its HP. |  |
| 222 | a2 | ch2 | MoveLeft | ->agent_1: agent_1: [in ch2] I am centered on the iron ore, ready to pair-dig with you. Please pair-d |  |
| 223 | a0 | ch2 | Sneak | ->agent_1: I am centered on the iron ore, ready to pair-dig with you. |  |
| 223 | a1 | ? | Sneak | ->agent_2: I am facing the anvil, ready to pair-dig with Agent_2. Please pair-dig with me to raise th |  |
| 223 | a2 | ch2 | Drop | ->agent_1: I am centered on the iron ore, ready to pair-dig with agent_1 to raise its HP. I expect th |  |
| 224 | a0 | ch2 | Drop | ->agent_1: Agent_1, I am digging the iron ore block. Please pair-dig with me to break it. |  |
| 224 | a1 | ? | Place | ->agent_1 [random_fallback:self_target]: I am centered on the iron ore, ready to pair-dig. Please pair-dig with me to raise the iro |  |
| 224 | a2 | ch2 | Sneak | ->agent_1: Dig on iron ore to raise HP and drop sword, then move north. |  |