# Case: collab_success_MAPPOpHeb_1

**Run:** `exp05_mappo_hebbian/seed_123` (MAPPO+Heb)  **Episode:** 1  **Steps:** 474-509
**Why shortlisted:** m18_door_opened (+60) at t=504, comm_before_coop=True

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 474 | a0 | ch3 | Sneak | ->agent_2: I am aiming at the switch. I will initiate cooperative Dig with agent_2. |  |
| 474 | a1 | ch3 | TurnLeft | ->agent_1 [hebbian_fallback:self_target]: Turn left to center the red glow block, then move forward to dig. I need Agent_1 to enter  |  |
| 474 | a2 | ch3 | Sneak | ->agent_0: I've centered the switch cube and am ready to Dig with you to open the door. |  |
| 475 | a0 | ch3 | Place | ->agent_2: Agent 2, I've turned left and moved forward to center the red sandstone block. I am ready  |  |
| 475 | a1 | ch3 | MoveRight | ->agent_2: I am centered on the red sandstone block, ready to Dig. Please turn right to center the re |  |
| 475 | a2 | ch3 | TurnLeft | ->agent_1: I am aiming at the red glow block on the left. I will turn left to center it, then move fo |  |
| 476 | a0 | ch3 | MoveRight | ->agent_2: Agent 2, I've centered the red sandstone block and am ready to initiate cooperative Dig wi |  |
| 476 | a1 | ch3 | MoveRight | ->agent_2: I am aiming at the red glow block. I will turn left to center it, then move forward to rea |  |
| 476 | a2 | ch3 | Dig | ->agent_1: Agent 1: I am centered on the red sandstone block, ready to Dig. |  |
| 477 | a0 | ch3 | Sneak | ->agent_2: I am aiming at the red sandstone block and ready to initiate Dig with you. Please center i |  |
| 477 | a1 | ch3 | MoveLeft | ->agent_2: I am aiming at the red glow block on the left, ready to Dig. |  |
| 477 | a2 | ch3 | MoveBackward | ->agent_1: I will turn right to center the anvil, then move forward to reach it, then turn left to ce |  |
| 478 | a0 | ch3 | Place | ->agent_2: agent_2: I am centered on the switch block and ready to press it with you. I will initiate |  |
| 478 | a1 | ch3 | TurnRight | ->agent_1 [hebbian_fallback:self_target]: I will turn right to center the red glow block, then dig to break it and receive cobblesto |  |
| 478 | a2 | ch3 | Sneak | ->agent_1: I see the stone tile on the ceiling. I am centered on it, and I expect to be aimed at it n | critic:FAIL |
| 479 | a0 | ch3 | TurnLeft | ->agent_2: Agent 2, I'm centered and ready for cooperative Dig. Please ensure you are also centered a | critic:FAIL |
| 479 | a1 | ch3 | Dig | ->agent_0: I will dig the tree trunk to drop wood. |  |
| 479 | a2 | ch3 | Dig | ->agent_1: Agent 1, I have reached the floor level and the anvil is centered ahead. I will initiate D | critic:FAIL |
| 480 | a0 | ch3 | Jump | ->agent_2: Agent 2: I have centered the switch cube and am ready to Dig with you to open the door. |  |
| 480 | a1 | ch3 | MoveBackward | ->agent_2: agent_2: I see the ceiling stone tile centered. I am aiming at it. Please turn left to cen |  |
| 480 | a2 | ch3 | MoveForward | ->agent_1: Turn right to center the agent on the stone block, move forward to reach it, then turn lef |  |
| 481 | a0 | ch3 | Sneak | ->agent_2: Switch cube centered by Agent 2. Ready to initiate cooperative Dig sequence. |  |
| 481 | a1 | ch3 | MoveForward | ->agent_2: I am looking at the stone tile on the floor. I will turn left to face it, then dig. |  |
| 481 | a2 | ch3 | MoveLeft | ->agent_0: I see the purple anvil centered left. I will turn right to center it, then move forward 3  |  |
| 482 | a0 | ch3 | Place | ->agent_2: Agent_2, the block is centered and ready for cooperative Dig. Please initiate the Dig acti |  |
| 482 | a1 | ch3 | LookDown | ->agent_2: I've moved forward and will turn left to center the blue switch. I expect to be aimed at i |  |
| 482 | a2 | ch3 | LookUp | ->agent_1: Turn left to center the anvil on the left, then move forward 3 blocks to reach it, then di |  |
| 483 | a0 | ch3 | TurnLeft | ->agent_2: Switch cube centered, ready for cooperative Dig. |  |
| 483 | a1 | ch3 | TurnLeft | ->agent_2: agent_2: [in ch3] Center the ceiling tile, look up, then dig. I expect to break the red gl |  |
| 483 | a2 | ch3 | MoveForward | ->agent_1: Turn left to center the anvil. Then move forward to reach it and dig. |  |
| 484 | a0 | ch3 | MoveBackward | ->agent_2: Agent 2, you're centering the anvil, move forward to reach the switch, then turn left to c |  |
| 484 | a1 | ch3 | Dig | ->agent_2: agent_2: [in ch3] Dig the stone pillar to drop cobblestone. |  |
| 484 | a2 | ch3 | LookUp | ->agent_1: agent_1: I am centered on the anvil; turn left to center it, then move forward to reach it |  |
| 485 | a0 | ch3 | MoveForward | ->agent_2: I am centering the red sandstone block for cooperative Dig with Agent_2. I expect the bloc |  |
| 485 | a1 | ch3 | TurnRight | ->agent_2: Switch 1 is pressed. Move forward to check the new position. |  |
| 485 | a2 | ch3 | LookDown | ->agent_1: I am centering the red glow block and moving forward to reach the purple anvil. |  |
| 486 | a0 | ch3 | Sneak | ->agent_2: Dig on red sandstone block. |  |
| 486 | a1 | ch3 | LookDown | ->agent_2: agent_2: I am centered on the stone pillar; turn left to center it, then move forward to r |  |
| 486 | a2 | ch3 | TurnLeft | ->agent_0: I am centering the red sandstone block for cooperative Dig with Agent_0. I expect the bloc |  |
| 487 | a0 | ch3 | LookUp | ->agent_2: I see the red sandstone block centered. I will initiate cooperative Dig with Agent_2. I ex |  |
| 487 | a1 | ch3 | LookDown | ->agent_2: I am centering the stone tile on the ceiling and moving forward to reach it. Please wait f |  |
| 487 | a2 | ch3 | Sneak | ->agent_1: agent_1: I am centered on the stone pillar; turn left to center it, then move forward to r |  |
| 488 | a0 | ch3 | Jump | ->agent_2: Agent_2: I will center the red sandstone block for cooperative Dig. I expect it to drop co |  |
| 488 | a1 | ch3 | TurnRight | ->agent_2: You are centering the red glow block. I am looking down and will turn to face the ceiling  |  |
| 488 | a2 | ch3 | MoveLeft | ->agent_1: Agent_1, please wait for me to approach the red sandstone block. I will initiate cooperati |  |
| 489 | a0 | ch3 | MoveBackward | ->agent_2: Agent 2, I see the red sandstone block centered. Dig to break it and drop cobblestone. |  |
| 489 | a1 | ch3 | MoveBackward | ->agent_2: Agent_2, I am centered on the stone tile; turn left to center it, then move forward to rea |  |
| 489 | a2 | ch3 | Drop | ->agent_0: I am centering the red sandstone block for cooperative Dig. I expect it to drop cobbleston |  |
| 490 | a0 | ch3 | TurnRight | ->agent_2: I am centering the red sandstone block and ready to Dig for cooperative action with Agent_ |  |
| 490 | a1 | ch3 | Place | ->agent_2: I am centered on the ceiling stone tile. Please turn left to center the red sandstone bloc |  |
| 490 | a2 | ch3 | Dig | ->agent_1: I see the red sandstone block centered. Dig to break it and drop cobblestone. |  |
| 491 | a0 | ch3 | Jump | ->agent_2: I am centering the red sandstone block for cooperative Dig with Agent_2. I expect the bloc |  |
| 491 | a1 | ch3 | TurnLeft | ->agent_2: Agent_2: Dig on the red sandstone block to drop cobblestone. |  |
| 491 | a2 | ch3 | Jump | ->agent_1: I am aiming the red glow block, then moving toward the anvil. I will turn left to center t |  |
| 492 | a0 | ch3 | NoOp | ->agent_2: Agent 2, I am centering the red sandstone block. Please move forward to reach the blue swi |  |
| 492 | a1 | ch3 | Sneak | ->agent_2: I see the ceiling stone tile centered. I will Dig to break it and drop cobblestone. |  |
| 492 | a2 | ch3 | TurnLeft | ->agent_0: I am centering the red sandstone block for cooperative Dig with Agent_2. I expect the bloc |  |
| 493 | a0 | ch3 | Dig | ->agent_2: Agent 2, I am centering the red sandstone block for cooperative Dig. I expect it to drop c |  |
| 493 | a1 | ch3 | MoveRight | ->agent_2: Switch is centered. I will dig it to open the door. |  |
| 493 | a2 | ch3 | Dig | ->agent_1: agent_1: I am centering the stone block to reach the switch. Please move forward to the bl |  |
| 494 | a0 | ch3 | LookUp | ->agent_2: Move forward to reach the blue switch cube on the south wall. |  |
| 494 | a1 | ch3 | TurnRight | ->agent_2: agent_2: I am aiming the red glow block, then moving toward the anvil. I will turn left to |  |
| 494 | a2 | ch3 | TurnRight | ->agent_1: Switch is centered. I will dig it to open the door. |  |
| 495 | a0 | ch3 | NoOp | ->agent_2: Turn left to face the anvil, then move forward to center it for cooperative Dig. |  |
| 495 | a1 | ch3 | TurnRight | ->agent_2: Switch is centered. I will dig it to break the switch. |  |
| 495 | a2 | ch3 | Place | ->agent_1: agent_1: I am aiming the red glow block, then moving toward the anvil. I will turn left to |  |
| 496 | a0 | ch3 | LookUp | ->agent_2: Turn left to center the anvil, then move forward to align it. I expect to be centered on t |  |
| 496 | a1 | ch3 | TurnLeft | ->agent_2: Switch is centered, ready to press. Expect to open the door next step. |  |
| 496 | a2 | ch3 | Sneak | ->agent_0: Agent_0, I moved back to step down the anvil pedestal. Please turn left to center it for t |  |
| 497 | a0 | ch3 | TurnRight | ->agent_2: Agent 2: Dig the red sandstone block to drop cobblestone and open the door. |  |
| 497 | a1 | ch3 | Jump | ->agent_2: agent_2: I will center the anvil and Dig to break it. |  |
| 497 | a2 | ch3 | Sneak | ->agent_1: I am centered on the red sandstone block in Chamber 3 and awaiting confirmation to proceed |  |
| 498 | a0 | ch3 | Place | ->agent_2: I am centering the red sandstone block for cooperative Dig with Agent_2. |  |
| 498 | a1 | ch3 | TurnRight | ->agent_2: I am centered on the red glow block, ready to Dig to break it and drop a stone tile. |  |
| 498 | a2 | ch3 | MoveBackward | ->agent_0: agent_0: Dig the red sandstone block to drop cobblestone and open the door. | critic:FAIL |
| 499 | a0 | ch3 | Place | ->agent_2: Please turn left to center the red sandstone block for cooperative Dig. |  |
| 499 | a1 | ch3 | TurnLeft | ->agent_1 [hebbian_fallback:self_target]: Turn left to center the anvil setup and prepare for the next Dig attempt. | critic:FAIL |
| 499 | a2 | ch3 | LookDown | ->agent_1: agent_1: I am centering the red glow block, ready to Dig to break it and drop a stone tile | critic:FAIL |
| 500 | a0 | ch3 | Drop | ->agent_2: agent_2: Dig the red sandstone block to drop cobblestone and open the door. |  |
| 500 | a1 | ch3 | Dig | ->agent_0: I am centered on the red glow block, ready to Dig. |  |
| 500 | a2 | ch3 | TurnLeft | ->agent_1: Turn left to center the red glow block and prepare for the cooperative Dig sequence. |  |
| 501 | a0 | ch3 | Jump | ->agent_2: I am centered on the red glow block, ready to Dig. Please turn left to face the block and  |  |
| 501 | a1 | ch3 | TurnLeft | ->agent_1 [hebbian_fallback:self_target]: I am centering the red glow block to unlock the door, then I will move forward to exit the |  |
| 501 | a2 | ch3 | Jump | ->agent_0: agent_0: I am centered on the red glow block, ready for the next step. |  |
| 502 | a0 | ch3 | TurnRight | ->agent_2: agent_2: I am centered on the red glow block, ready to Dig. |  |
| 502 | a1 | ch3 | MoveForward | ->agent_2: I am ready to Dig on the red glow block. |  |
| 502 | a2 | ch3 | LookDown | ->agent_0: Turn left to center the red glow block, then turn right to center the blue switch. I need  |  |
| 503 | a0 | ch3 | Dig | ->agent_2: I am centered on the red glow block and ready for the next step. |  |
| 503 | a1 | ch3 | TurnRight | ->agent_1 [hebbian_fallback:self_target]: Agent_1, I am centering the red glow block, ready to wait for the cooperative Dig sequence |  |
| 503 | a2 | ch3 | MoveLeft | ->agent_0: Switch state is OPEN; wait for door open. |  |
| 504 | a0 | ch3 | Dig | ->agent_2: I am centered on the red glow block. agent_2, I will turn left to center it and initiate c | ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 504 | a1 | ch3 | NoOp | ->agent_1 [hebbian_fallback:self_target]: Turn left to center the red glow block and prepare for the cooperative Dig sequence. | ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 504 | a2 | ch3 | Dig | ->agent_0: Turn left to center agent_0, then move forward 5 blocks to reach the switch. | ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 505 | a0 | ch3 | TurnLeft | ->agent_2: Turn left to center the red glow block; expect cooperative Dig with Agent_2. | NEW TASK: Turn left to center the blue switch cube; expect to hear the switch ac |
| 505 | a1 | ch3 | Jump | ->agent_2: I am centering the red glow block, ready to Dig to break it and drop a stone tile. | NEW TASK: Move forward 5 blocks to reach the communal room door and announce you |
| 505 | a2 | ch3 | Sneak | ->agent_0: agent_0: I will break the blue switch cube with the diamond sword to open your door. |  |
| 506 | a0 | ch3 | Sneak | ->agent_2: agent_2: I am centered on the red sandstone block and ready to initiate cooperative Dig. |  |
| 506 | a1 | ch3 | Dig | ->agent_2: I am centering the red glow block, ready to Dig to break the anvil and drop a stone tile. |  |
| 506 | a2 | ch3 | Sneak | ->agent_0: Turn left to center the switch cube; expect cooperative Dig with Agent_0. |  |
| 507 | a0 | ch3 | TurnRight | ->agent_2: I am centered on the red sandstone block and ready for cooperative Dig with agent_2. |  |
| 507 | a1 | ch3 | Jump | ->agent_2: Turn left to center the red glow block and prepare for the cooperative Dig sequence. |  |
| 507 | a2 | ch3 | Dig | ->agent_0: Switch cube is centered, ready to break. |  |
| 508 | a0 | ch3 | Sneak | ->agent_2: Turn right to center the red glow block. Then move forward to align it for cooperative Dig |  |
| 508 | a1 | ch3 | MoveBackward | ->agent_2: agent_2: Please turn left to center the red glow block so we can Dig on it to break the do |  |
| 508 | a2 | ch3 | Drop | ->agent_1: I just broke the switch cube. |  |
| 509 | a0 | ch3 | Drop | ->agent_2: Turn left to center the red sandstone block; expect to see it centered and ready for coope |  |
| 509 | a1 | ch3 | TurnLeft | ->agent_2: Turn left to center the stone pillar and prepare for the cooperative Dig sequence. |  |
| 509 | a2 | ch3 | Place | ->agent_0: agent_0: Please turn right to center the red glow block so we can Dig on it to break the d |  |