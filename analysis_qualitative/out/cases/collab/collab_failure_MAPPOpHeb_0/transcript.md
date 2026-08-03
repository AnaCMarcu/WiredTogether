# Case: collab_failure_MAPPOpHeb_0

**Run:** `exp05_mappo_hebbian/seed_789` (MAPPO+Heb)  **Episode:** 3  **Steps:** 259-289
**Why shortlisted:** ignored directive a0->a2 streak x10: "Turn left to center the tree trunk so we can start digging the anvil."

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 259 | a0 | ch2 | MoveRight | ->agent_2: Turn left to center the tree trunk so we can start digging the anvil. | critic:FAIL |
| 259 | a1 | ? | MoveLeft | ->agent_2: I am centered on the cobblestone block with the diamond sword. I am ready to initiate the  | critic:FAIL |
| 259 | a2 | ? | Dig | ->agent_1: I am centered and ready to initiate the cooperative Dig scheme. |  |
| 260 | a0 | ch2 | Dig | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 260 | a1 | ? | LookDown | ->agent_2: Press block to drop diamond sword and initiate cooperative Dig scheme with Agent_2. I am c |  |
| 260 | a2 | ch1 | Jump | ->agent_1: Initiate the cooperative Dig scheme with Agent_1. |  |
| 261 | a0 | ch2 | TurnLeft | ->agent_2: Turn left to center the sword so I can initiate the cooperative Dig scheme. |  |
| 261 | a1 | ? | MoveLeft | ->agent_2: I will move forward to center the camera on the cobblestone block, then turn left to face  |  |
| 261 | a2 | ch1 | Drop | ->agent_1: I am centered on the diamond sword block and ready to initiate the cooperative Dig scheme  |  |
| 262 | a0 | ch2 | Sneak | ->agent_1: Turn left to center the tree trunk, then move forward to align the sword. |  |
| 262 | a1 | ? | Jump | ->agent_2: I will TurnLeft to center the sword on the block, then MoveForward to approach it for coop |  |
| 262 | a2 | ch1 | MoveForward | ->agent_1: agent_1: Initiating cooperative Dig scheme. |  |
| 263 | a0 | ch2 | MoveLeft | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 263 | a1 | ch2 | MoveRight | ->agent_2: Dig on cobblestone block. |  |
| 263 | a2 | ch1 | Dig | ->agent_1: Agent_1, I moved forward to get close to the cobblestone block where the diamond sword is. |  |
| 264 | a0 | ch2 | MoveLeft | ->agent_2: Turn left to center the chickens so I can attack them. |  |
| 264 | a1 | ch2 | Jump | ->agent_2: I will turn left to face the cobblestone block and initiate the cooperative Dig scheme wit |  |
| 264 | a2 | ch1 | Dig | ->agent_1: I am centered on the cobblestone block and ready to initiate the cooperative Dig scheme wi |  |
| 265 | a0 | ch2 | Place | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 265 | a1 | ch2 | TurnRight | ->agent_2: Hold diamond sword and turn left to face the block. |  |
| 265 | a2 | ch1 | MoveLeft | ->agent_1: I am about to initiate the cooperative Dig scheme on the diamond sword. I expect to drop t |  |
| 266 | a0 | ch2 | Dig | ->agent_2: Agent 2: Turn left to center the chicken, then MoveForward 3 blocks to reach me. |  |
| 266 | a1 | ch2 | TurnLeft | ->agent_2: I am holding the diamond sword on the cobblestone block, ready to initiate the cooperative |  |
| 266 | a2 | ch1 | MoveLeft | ->agent_1: Initiate the cooperative Dig scheme. |  |
| 267 | a0 | ch2 | MoveRight | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 267 | a1 | ch2 | Jump | ->agent_2: I am aiming for the cobblestone block and ready to initiate cooperative Dig with Agent_2. |  |
| 267 | a2 | ch1 | TurnRight | ->agent_1: Agent_1: The diamond sword is centered on the cobblestone block. I am ready to initiate th |  |
| 268 | a0 | ch2 | Sneak | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 268 | a1 | ch2 | TurnRight | ->agent_2: Agent_2, the diamond sword is centered on the block. I am ready to initiate the cooperativ |  |
| 268 | a2 | ch1 | LookUp | ->agent_2 [hebbian_fallback:self_target]: Agent_2: Initiate cooperative Dig scheme with the diamond sword on the cobblestone. |  |
| 269 | a0 | ch2 | Jump | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 269 | a1 | ch2 | TurnRight | ->agent_2: I am ready to initiate the cooperative Dig scheme with the diamond sword on the cobbleston |  |
| 269 | a2 | ch1 | MoveRight | ->agent_1: Initiate the cooperative Dig scheme on the diamond sword by pressing the block to drop it. |  |
| 270 | a0 | ch2 | Jump | ->agent_2: Move forward 3 blocks to reach Agent 1, then turn left to center the sword. |  |
| 270 | a1 | ch2 | Place | ->agent_2: HoldDiamondSword: I am holding the diamond sword and ready to initiate the cooperative Dig |  |
| 270 | a2 | ch1 | NoOp | ->agent_1: Agent_1, the diamond sword is centered on the block and ready to initiate the cooperative  |  |
| 271 | a0 | ch2 | Sneak | ->agent_2: Agent 2: Move forward 3 blocks to reach the center of the room. |  |
| 271 | a1 | ch2 | Dig | ->agent_2: Agent_2: The diamond sword is centered on the cobblestone block and ready to initiate the  |  |
| 271 | a2 | ch1 | Drop | ->agent_1: I have the diamond sword centered and ready to press the block to initiate the cooperative |  |
| 272 | a0 | ch2 | Dig | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 272 | a1 | ch2 | Sneak | ->agent_2: I am ready to initiate the cooperative Dig scheme with Agent_2. |  |
| 272 | a2 | ch1 | MoveRight | ->agent_1: Agent_1, the diamond sword is centered on the cobblestone block and ready to initiate the  |  |
| 273 | a0 | ch2 | Sneak | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 273 | a1 | ch2 | MoveLeft | ->agent_2: Press the block to initiate the cooperative Dig scheme with Agent_2. |  |
| 273 | a2 | ch1 | Sneak | ->agent_2 [hebbian_fallback:self_target]: Agent_2: You are centered on the sword. Ready to initiate the cooperative Dig scheme. |  |
| 274 | a0 | ch2 | LookDown | ->agent_1: Turn left to center the tree, then move forward 3 blocks to reach Agent 1, then turn left  |  |
| 274 | a1 | ch2 | TurnLeft | ->agent_2: I am centered on the diamond sword and ready to initiate the cooperative Dig scheme with A |  |
| 274 | a2 | ch1 | MoveLeft | ->agent_1: Agent_1, I see the stone block in the center. I am moving forward to get closer to it. |  |
| 275 | a0 | ch2 | Sneak | ->agent_2: Turn left to center the diamond sword. |  |
| 275 | a1 | ch2 | Dig | ->agent_2: I am centering the diamond sword on the cobblestone block and ready to initiate the cooper |  |
| 275 | a2 | ch1 | MoveLeft | ->agent_1: Agent_1: I am centered on the diamond sword and ready to initiate the cooperative Dig sche |  |
| 276 | a0 | ch2 | Jump | ->agent_1: Move forward 3 blocks to reach Agent 1, then turn left to center the diamond sword. |  |
| 276 | a1 | ch2 | Dig | ->agent_2: I am centered on the diamond sword and ready to initiate the cooperative Dig scheme with A |  |
| 276 | a2 | ch1 | MoveLeft | ->agent_1: Initiate the cooperative Dig scheme with Agent_1. |  |
| 277 | a0 | ch2 | MoveBackward | ->agent_1: Turn left to center the tree trunk. |  |
| 277 | a1 | ch2 | Jump | ->agent_1 [hebbian_fallback:self_target]: Initiate the cooperative Dig scheme with Agent_1. |  |
| 277 | a2 | ch1 | Drop | ->agent_1: I am centered on the diamond sword and ready to initiate the cooperative Dig scheme with A |  |
| 278 | a0 | ch2 | TurnLeft | ->agent_2: Agent 2: Turn left to center the sword so I can initiate the cooperative Dig scheme. |  |
| 278 | a1 | ch2 | Dig | ->agent_2: I am centered on the diamond sword and ready to initiate the cooperative Dig scheme with A |  |
| 278 | a2 | ch1 | Dig | ->agent_1: Agent_1, initiate the cooperative Dig scheme by pressing the block. |  |
| 279 | a0 | ch2 | TurnRight | ->agent_1: Turn left to center the tree trunk, then move forward 3 blocks to reach me, then turn left | critic:FAIL |
| 279 | a1 | ch2 | MoveLeft | ->agent_2: Agent_2, the diamond sword is centered on the cobblestone block, ready for cooperative Dig | critic:FAIL |
| 279 | a2 | ch1 | Place | ->agent_1: I am centered on the diamond sword and ready to initiate the cooperative Dig scheme. |  |
| 280 | a0 | ch2 | MoveBackward | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 280 | a1 | ch2 | Sneak | ->agent_2: Turn left to center the cobblestone block, then move forward to initiate the cooperative D |  |
| 280 | a2 | ch1 | Drop | ->agent_1: Agent 1, the left chicken is slightly left of center. Turn left to center it and prepare f |  |
| 281 | a0 | ch2 | MoveBackward | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 281 | a1 | ch2 | MoveBackward | ->agent_2: I am centered on the cobblestone block and ready to initiate the cooperative Dig scheme wi |  |
| 281 | a2 | ch1 | LookDown | ->agent_1: Agent_1: I turned left to center the pig and move forward to dig. Please confirm the pig i |  |
| 282 | a0 | ch2 | TurnLeft | ->agent_1: Turn left to center the tree trunk, then move forward 2 blocks to reach Agent 1, then turn |  |
| 282 | a1 | ch2 | Dig | ->agent_2: Turn left to center the pig so I can dig with Agent_2. |  |
| 282 | a2 | ch1 | Drop | ->agent_1: I am centered on the cobblestone block with the diamond sword. I will initiate the coopera |  |
| 283 | a0 | ch2 | Dig | ->agent_2: Agent_2: Turn right to center the right anvil so I can initiate the cooperative Dig scheme |  |
| 283 | a1 | ch2 | MoveForward | ->agent_1 [hebbian_fallback:self_target]: I am centered on the cobblestone block with the diamond sword. I will turn left to center  |  |
| 283 | a2 | ch1 | Jump | ->agent_1: Agent_1, I am centered on the diamond sword, ready to initiate the cooperative Dig scheme  |  |
| 284 | a0 | ch2 | MoveLeft | ->agent_1: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 284 | a1 | ch2 | TurnLeft | ->agent_2: I am centered on the diamond sword and ready to initiate the cooperative Dig scheme with A |  |
| 284 | a2 | ch1 | MoveForward | ->agent_1: I am centered on the cobblestone block with the diamond sword. I will turn left to center  |  |
| 285 | a0 | ch2 | MoveLeft | ->agent_2: Agent_2: Turn left to center the sword, then move forward. |  |
| 285 | a1 | ch2 | Dig | ->agent_2: Agent 2 is centered on the cobblestone block with the diamond sword. I will initiate the c |  |
| 285 | a2 | ch1 | LookUp | ->agent_1: I am centered on the diamond sword. I will turn left to center the sword and then initiate |  |
| 286 | a0 | ch2 | Dig | ->agent_2: Move forward 3 blocks to reach me, then turn left to center the sword. |  |
| 286 | a1 | ch2 | Jump | ->agent_2: I am centered on the cobblestone block with the diamond sword, ready to initiate the coope |  |
| 286 | a2 | ch1 | Drop | ->agent_1: Agent_1, I am centered on the cobblestone with the diamond sword. Initiate the cooperative |  |
| 287 | a0 | ch2 | Dig | ->agent_1: Turn left to center the tree, then move forward 3 blocks to reach me, then turn left to ce |  |
| 287 | a1 | ch2 | MoveBackward | ->agent_2: Agent_2, I am centered on the cobblestone with the diamond sword. Initiate the cooperative |  |
| 287 | a2 | ch1 | MoveLeft | ->agent_2 [hebbian_fallback:self_target]: I am centered on the diamond sword and ready to initiate the cooperative Dig scheme. |  |
| 288 | a0 | ch2 | Jump | ->agent_2: Move forward to break the anvil, then turn left to center the sword and initiate the coope |  |
| 288 | a1 | ch2 | Dig | ->agent_2: Move forward to center the camera on the cobblestone block. |  |
| 288 | a2 | ch1 | Place | ->agent_1: Initiate the cooperative Dig scheme by pressing the cobblestone block. |  |
| 289 | a0 | ch2 | Sneak | ->agent_2: Move forward 5 blocks to reach the north wall, turn left to center the tree trunk, then Di |  |
| 289 | a1 | ch2 | TurnRight | ->agent_2: agent_2: You are centered on the cobblestone block. I am ready to initiate the cooperative |  |
| 289 | a2 | ch1 | MoveBackward | ->agent_1: Agent 1: The diamond sword is centered on the cobblestone block. I am ready to initiate th |  |