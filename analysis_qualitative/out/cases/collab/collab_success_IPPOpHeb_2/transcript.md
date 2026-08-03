# Case: collab_success_IPPOpHeb_2

**Run:** `exp06_ippo_hebbian/seed_456` (IPPO+Heb)  **Episode:** 1  **Steps:** 409-444
**Why shortlisted:** m18_door_opened (+60) at t=439, comm_before_coop=True

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 409 | a0 | ch3 | MoveForward | ->agent_1: Align the sword to center the anvil, then pair-dig with Agent 1. |  |
| 409 | a1 | ch3 | Jump | ->agent_0: Turn right to center the sword. I need to break the communal anvil to equip the sword. |  |
| 409 | a2 | ch3 | Dig | ->agent_1: Agent 1 is centered on the anvil, ready to pair-dig with Agent 2. Please align the sword t |  |
| 410 | a0 | ch3 | Jump | ->agent_1: Turn left to center the sword on the anvil. I need to break the communal anvil to equip th |  |
| 410 | a1 | ch3 | TurnRight | ->agent_1 [hebbian_fallback:self_target]: Align the sword to center the anvil, then pair-dig with Agent 1. |  |
| 410 | a2 | ch3 | MoveForward | ->agent_1: Turn right to align the sword handle with the block. |  |
| 411 | a0 | ch3 | TurnLeft | ->agent_1: I am centered on the anvil, ready to pair-dig. Please turn left to center the diamond swor |  |
| 411 | a1 | ch3 | Drop | ->agent_0: Turn right to align the sword with the switch cube. I need to break the switch to open my  |  |
| 411 | a2 | ch3 | Place | ->agent_1: Turn your sword up to face the block, then pair-dig with me. |  |
| 412 | a0 | ch3 | Jump | ->agent_1: Agent 1: Move forward to get closer to the anvil. |  |
| 412 | a1 | ch3 | Drop | ->agent_0: Pick up the diamond sword from inventory. |  |
| 412 | a2 | ch3 | Jump | ->agent_1: Agent 1, turn left to center the anvil and prepare for pair-digging. |  |
| 413 | a0 | ch3 | Dig | ->agent_1: Agent 1, your switch should be open now. We need to pair-dig on the anvil together. |  |
| 413 | a1 | ch3 | Drop | ->agent_0: I am centering the anvil with the sword. Please align the sword for me. |  |
| 413 | a2 | ch3 | MoveForward | ->agent_1: Align the sword to center the anvil, then pair-dig with Agent 1. |  |
| 414 | a0 | ch3 | Dig | ->agent_1: Agent 1: Align the sword with the anvil for me. |  |
| 414 | a1 | ch3 | TurnRight | ->agent_2: Agent_2, Turn right to center the anvil, then Dig to break it. I expect the anvil to drop  |  |
| 414 | a2 | ch3 | MoveForward | ->agent_1: Turn the sword upward to face the anvil, then pair-dig with Agent 1. |  |
| 415 | a0 | ch3 | Dig | ->agent_1: Turn right to see the right side of the anvil. |  |
| 415 | a1 | ch3 | Drop | ->agent_0: Align the sword with the anvil for me. |  |
| 415 | a2 | ch3 | Place | ->agent_1: Turn left to center the diamond sword, then dig. I expect the anvil to drop the diamond sw |  |
| 416 | a0 | ch3 | Place | ->agent_1: I am centering the anvil with the sword. Please align the sword for me. |  |
| 416 | a1 | ch3 | Dig | ->agent_2: Agent_2, align the handle with the anvil center, then dig to break the anvil. |  |
| 416 | a2 | ch3 | LookDown | ->agent_1: I am centering the sword on the anvil and moving forward to reach it. |  |
| 417 | a0 | ch3 | Place | ->agent_1: Align the sword with the anvil for me. |  |
| 417 | a1 | ch3 | MoveLeft | ->agent_2: Align the sword for the anvil, then Dig. |  |
| 417 | a2 | ch3 | MoveForward | ->agent_1: Agent_1, I am aligning the diamond sword upward to break the communal anvil. I expect the  |  |
| 418 | a0 | ch3 | Dig | ->agent_1: Agent 1, the anvil is broken. I am ready to pair-dig with you to break the next block. Ple |  |
| 418 | a1 | ch3 | TurnRight | ->agent_2: I need to select the diamond sword from my inventory, then I will break the communal anvil |  |
| 418 | a2 | ch3 | MoveRight | ->agent_1: Agent 1, please align the handle with the anvil center so we can dig. |  |
| 419 | a0 | ch3 | MoveForward | ->agent_1: I am centered on the stone block, moving forward to survey the open red area on the right. | critic:FAIL |
| 419 | a1 | ch3 | TurnRight | ->agent_2: I am centering the anvil with the sword, ready to break it with you. | critic:FAIL |
| 419 | a2 | ch3 | MoveLeft | ->agent_1: Are you holding the diamond sword? I need to center the anvil and break it with you. |  |
| 420 | a0 | ch3 | Jump | ->agent_1: Align the sword with the anvil for me. |  |
| 420 | a1 | ch3 | Dig | ->agent_2: Your handle is centered; turn left to align the anvil and prepare for the pair-dig sequenc |  |
| 420 | a2 | ch3 | Drop | ->agent_1: Turn left to center the anvil, then dig. |  |
| 421 | a0 | ch3 | Dig | ->agent_1: I am centering the stone block with the anvil handle to break it. |  |
| 421 | a1 | ch3 | Place | ->agent_2: Turn right to center the anvil, then dig. |  |
| 421 | a2 | ch3 | Drop | ->agent_1: Your handle is centered; turn left to align the anvil and prepare for the pair-dig sequenc |  |
| 422 | a0 | ch3 | MoveRight | ->agent_1: I am breaking the stone block at the bottom of the incline. Please check the red area on t |  |
| 422 | a1 | ch3 | Place | ->agent_2: Agent_2, the anvil is centered and ready for the pair-dig sequence. |  |
| 422 | a2 | ch3 | TurnLeft | ->agent_1: Turn left to align the anvil, then dig. |  |
| 423 | a0 | ch3 | TurnLeft | ->agent_1: Agent_1, the anvil is aligned for me to break. |  |
| 423 | a1 | ch3 | Drop | ->agent_2: Turn right to center the anvil and left to align the sword, then dig. |  |
| 423 | a2 | ch3 | LookUp | ->agent_1: Turn up the sword to align it with the anvil. |  |
| 424 | a0 | ch3 | MoveRight | ->agent_1: Align the sword with the anvil for me. |  |
| 424 | a1 | ch3 | MoveForward | ->agent_1 [hebbian_fallback:self_target]: I have centered the anvil for you to break. |  |
| 424 | a2 | ch3 | Jump | ->agent_1: Turn left to center the block and align the sword, then dig. |  |
| 425 | a0 | ch3 | Dig | ->agent_1: I am aligned with the anvil. Please align the sword for me. |  |
| 425 | a1 | ch3 | MoveLeft | ->agent_2: Align the stone block with the center, then dig. |  |
| 425 | a2 | ch3 | Place | ->agent_1: Turn left to align the diamond sword upward for the anvil, then Dig. |  |
| 426 | a0 | ch3 | MoveLeft | ->agent_1: Agent_1, I am centering the anvil with the sword. Please align the sword for me. |  |
| 426 | a1 | ch3 | TurnRight | ->agent_2: Align the diamond sword upward for the anvil, then Dig. |  |
| 426 | a2 | ch3 | Place | ->agent_1: Agent 1, the anvil is 3 blocks away to the right. I am heading there to break it. |  |
| 427 | a0 | ch3 | MoveLeft | ->agent_1: agent_1, I am centered and ready to break the anvil with the diamond sword. Please align t |  |
| 427 | a1 | ch3 | Jump | ->agent_1 [hebbian_fallback:self_target]: Agent_1, I am centering the anvil with the sword. Please align the sword for me before I b |  |
| 427 | a2 | ch3 | LookUp | ->agent_1: Agent_1, I am centering the anvil with my sword, ready to break it. |  |
| 428 | a0 | ch3 | Dig | ->agent_1: agent_1, I am aligned with the anvil, ready to break. Please align the sword for me before |  |
| 428 | a1 | ch3 | TurnLeft | ->agent_2: I am centered on the communal anvil. I will turn right to align the anvil with my center,  |  |
| 428 | a2 | ch3 | LookUp | ->agent_1: Align the anvil with the center, then dig. |  |
| 429 | a0 | ch3 | Dig | ->agent_1: Agent 1, I am centering the anvil with the sword. Please align the sword for me before I b |  |
| 429 | a1 | ch3 | Drop | ->agent_2: Agent 2, align the anvil with the center, then dig. |  |
| 429 | a2 | ch3 | MoveRight | ->agent_1: Turn left to center the anvil, then dig. |  |
| 430 | a0 | ch3 | Dig | ->agent_1: Agent_1, I am centering the anvil with the sword. Please align the sword for me before I b |  |
| 430 | a1 | ch3 | Place | ->agent_2: Turn right to center the anvil, then DigForward 3 times. |  |
| 430 | a2 | ch3 | Dig | ->agent_1: Agent 1, are you ready to align the anvil with the center and break it with the diamond sw |  |
| 431 | a0 | ch3 | TurnRight | ->agent_1: I am centering the anvil with the sword. Please align the sword for me before I break it. |  |
| 431 | a1 | ch3 | TurnLeft | ->agent_2: Agent 2, the sword is centered and ready. Please align it with the anvil's center so I can |  |
| 431 | a2 | ch3 | LookUp | ->agent_1: Agent 1, turn right to align the anvil with the center, then dig. |  |
| 432 | a0 | ch3 | Jump | ->agent_1: I am adjusting the sword's handle to eye level, ready to break the stone pillar. |  |
| 432 | a1 | ch3 | MoveLeft | ->agent_2: Turn right to align the anvil with the center, then dig. |  |
| 432 | a2 | ch3 | LookDown | ->agent_1: I am ready to break the communal anvil with the diamond sword to start the pair-dig sequen |  |
| 433 | a0 | ch3 | MoveRight | ->agent_1: Agent_1, the anvil is aligned and ready for me to break it. |  |
| 433 | a1 | ch3 | Place | ->agent_2: Agent 2, I am centered on the communal anvil, ready to break it. I expect the diamond swor |  |
| 433 | a2 | ch3 | Place | ->agent_1: Agent 1: Align the diamond sword with the anvil's center so I can dig. |  |
| 434 | a0 | ch3 | TurnRight | ->agent_1: Agent 1, I am ready to pair-dig on the anvil. Please align the sword for me before I break |  |
| 434 | a1 | ch3 | TurnLeft | ->agent_2: Agent 2, the anvil is aligned and ready for me to break it. |  |
| 434 | a2 | ch3 | Jump | ->agent_1: Turn right to align the anvil with the center, then dig. |  |
| 435 | a0 | ch3 | Dig | ->agent_1: Agent_1, I am centering the anvil with the sword. Please align the sword for me before I b |  |
| 435 | a1 | ch3 | Drop | ->agent_1 [hebbian_fallback:self_target]: Align the anvil with the center, then dig. |  |
| 435 | a2 | ch3 | Place | ->agent_2 [hebbian_fallback:self_target]: Agent 2, I am aligned with the anvil and ready to dig, please confirm I am centered. |  |
| 436 | a0 | ch3 | MoveRight | ->agent_1: Turn the sword down to eye level, then move forward and dig the switch. |  |
| 436 | a1 | ch3 | TurnLeft | ->agent_2: Agent 2, I am aligned with the anvil and ready to dig, please confirm I am centered. |  |
| 436 | a2 | ch3 | Place | ->agent_1: Agent 1, I am centered on the communal anvil and ready to break it. I expect the diamond s |  |
| 437 | a0 | ch3 | Dig | ->agent_1: Agent 1, I am aligned with the anvil. Please wait for my command to break it. |  |
| 437 | a1 | ch3 | MoveLeft | ->agent_2: Agent 2, I am centered on the communal anvil and ready to break it. I expect the diamond s |  |
| 437 | a2 | ch3 | Place | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the diamond sword with the anvil. Agent 2 is ready to press the switch |  |
| 438 | a0 | ch3 | Dig | ->agent_1: Agent_1, I am centering the diamond sword on the anvil. Please align the sword for me befo |  |
| 438 | a1 | ch3 | Drop | ->agent_1 [hebbian_fallback:self_target]: The communal anvil is perfectly centered. Please wait for my command to break it. |  |
| 438 | a2 | ch3 | Place | ->agent_1: Turn left to center the anvil and align the diamond sword so it points up, then I will dig |  |
| 439 | a0 | ch3 | TurnRight | ->agent_1: Agent_1, the communal anvil is centered and ready for the break. | critic:FAIL; ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 439 | a1 | ch3 | Dig | ->agent_2: Agent_2, I am centering the anvil. Please align the sword for me before I break it. | critic:FAIL; ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 439 | a2 | ch3 | MoveLeft | ->agent_1: Agent 1, I am centered on the communal anvil and ready to break it. I expect the diamond s | critic:FAIL; ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 440 | a0 | ch3 | TurnLeft | ->agent_1: agent_1: Dig on stone, I am aiming. |  |
| 440 | a1 | ch3 | Jump | ->agent_0: Turn left to center the diamond sword on the communal anvil, then Dig. |  |
| 440 | a2 | ch3 | MoveRight | ->agent_1: Agent 1, I am ready to break the stone block with my diamond sword. Please align the sword |  |
| 441 | a0 | ch3 | MoveRight | ->agent_1: I am looking down at the diamond sword to center it, then I will move forward to align the |  |
| 441 | a1 | ch3 | TurnRight | ->agent_1 [hebbian_fallback:self_target]: Agent 1, I am aiming at the communal anvil with my diamond sword. Please align the sword s |  |
| 441 | a2 | ch3 | LookDown | ->agent_1: Agent 1, I am centering the anvil. Please align the sword for me before I break it. |  |
| 442 | a0 | ch3 | MoveRight | ->agent_1: Wait for my command to break the anvil. |  |
| 442 | a1 | ch3 | Place | ->agent_0: I am centering the anvil. Please align the sword for me before I break it. |  |
| 442 | a2 | ch3 | Jump | ->agent_1: Agent 1, align the sword up so the blade points straight at the switch so I can break it. |  |
| 443 | a0 | ch3 | MoveLeft | ->agent_1: I am centering the anvil. Please align the sword for me before I break it. |  |
| 443 | a1 | ch3 | Place | ->agent_2: Agent 2, align the sword up so the blade points straight at the switch so I can break it. |  |
| 443 | a2 | ch3 | MoveRight | ->agent_1: Turn left to align the sword, then Dig. |  |
| 444 | a0 | ch3 | TurnLeft | ->agent_1: agent_1: [in ch3] Dig the zombie for health. I am ready to pair-dig. |  |
| 444 | a1 | ch3 | MoveLeft | ->agent_0: Agent_0, Turn right to center the anvil, then Dig. |  |
| 444 | a2 | ch3 | Jump | ->agent_1: Agent 1, the sword is pointing right; turn left to align it up so I can break the switch. |  |