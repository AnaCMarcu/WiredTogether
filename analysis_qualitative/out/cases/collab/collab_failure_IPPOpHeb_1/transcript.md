# Case: collab_failure_IPPOpHeb_1

**Run:** `exp06_ippo_hebbian/seed_42` (IPPO+Heb)  **Episode:** 2  **Steps:** 488-518
**Why shortlisted:** ignored request/status_report a0->a1 streak x4: "Switch is centered and ready for cooperative Dig. Please press it to open the do"

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 488 | a0 | ch3 | Sneak | ->agent_1: Switch is centered and ready for cooperative Dig. Please press it to open the door. |  |
| 488 | a1 | ch3 | Jump | ->agent_2: Switch is centered and facing me. Please press it to open the door. |  |
| 488 | a2 | ch3 | MoveForward | ->agent_1: Agent_1, I am centered on the diamond block, but it is slightly right of center. Please tu |  |
| 489 | a0 | ch3 | Sneak | ->agent_1: Switch is centered and ready for cooperative Dig. Please press it to open the door. |  |
| 489 | a1 | ch3 | MoveBackward | ->agent_0: I have seen the switch on the south wall. Please press it to open the door. |  |
| 489 | a2 | ch3 | NoOp | ->agent_1: Switch is centered and facing me. Please press it to open the door. |  |
| 490 | a0 | ch3 | MoveLeft | ->agent_1: Agent_1, switch is centered and ready for cooperative Dig. Please press it to open the doo | critic:FAIL |
| 490 | a1 | ch3 | LookUp | ->agent_2: Switch is pressed. Ready for door opening. |  |
| 490 | a2 | ch3 | MoveRight | ->agent_1: Switch is centered and facing me. Please press it to open the door. | critic:FAIL |
| 491 | a0 | ch3 | MoveLeft | ->agent_1: Agent_1: Please press the switch to open the door. |  |
| 491 | a1 | ch3 | Drop | ->agent_0: Switch is centered and ready for cooperative Dig. Please press it to open the door. |  |
| 491 | a2 | ch3 | MoveBackward | ->agent_1: Turn left to center the diamond block, then move forward to break it. |  |
| 492 | a0 | ch3 | NoOp | ->agent_2: Switch is centered and ready. Please press it to open the door. |  |
| 492 | a1 | ch3 | Jump | ->agent_2: Agent 2: The switch is active! Please press the switch to open the door. |  |
| 492 | a2 | ch3 | MoveForward | ->agent_1: Agent 1, the switch is centered and facing me. Please press it to open the door. |  |
| 493 | a0 | ch3 | MoveLeft | ->agent_1: Agent_1, please press the switch to open the door. |  |
| 493 | a1 | ch3 | Drop | ->agent_2: Agent 2, I am facing the switch and need to press it to open the door. |  |
| 493 | a2 | ch3 | MoveRight | ->agent_1: The switch is centered and ready. Please press it to open the door. |  |
| 494 | a0 | ch3 | MoveRight | ->agent_2: Please press the switch to open the door. |  |
| 494 | a1 | ch3 | TurnLeft | ->agent_2: Press the switch to open the door. |  |
| 494 | a2 | ch3 | TurnLeft | ->agent_1: agent_1, I am facing the switch and need to press it to open the door. |  |
| 495 | a0 | ch3 | TurnRight | ->agent_2: agent_2: I'm breaking the diamond block next! Please help me coordinate the switch press.  |  |
| 495 | a1 | ch3 | TurnRight | ->agent_2: agent_2: Press the switch to open the door. |  |
| 495 | a2 | ch3 | Place | ->agent_0: I'm centered on the diamond block. Please press the switch to open the door. |  |
| 496 | a0 | ch3 | TurnLeft | ->agent_2: We are ready to break the diamond block together. Please press the switch to open the door |  |
| 496 | a1 | ch3 | MoveForward | ->agent_2: Please press the switch to open the door. |  |
| 496 | a2 | ch3 | MoveLeft | ->agent_1: Turn left to center the diamond block, then dig to break it. |  |
| 497 | a0 | ch3 | TurnRight | ->agent_2: agent_2: Turn left to center the diamond block, then dig. I expect you to be nearby. |  |
| 497 | a1 | ch3 | LookDown | ->agent_2: Turn left to center the diamond block, then dig to break it. |  |
| 497 | a2 | ch3 | MoveLeft | ->agent_0: Please press the switch to open the door. |  |
| 498 | a0 | ch3 | LookDown | ->agent_1: I'm centered on the diamond block. Please press the switch to open the door. |  |
| 498 | a1 | ch3 | MoveForward | ->agent_2: agent_2: Dig the switch to open the door. |  |
| 498 | a2 | ch3 | NoOp | ->agent_1: Turn left to center the diamond block, then dig to break it. |  |
| 499 | a0 | ch3 | LookUp | ->agent_2: Switch is open and diamond centered. Please press the switch to open the door. |  |
| 499 | a1 | ch3 | Jump | ->agent_0: The switch is centered on the diamond block. I will press the switch to open the door. |  |
| 499 | a2 | ch3 | MoveRight | ->agent_1: agent_1: Turn left to center the diamond block, then move forward to break it. |  |
| 500 | a0 | ch3 | Jump | ->agent_1: I will turn left to center the diamond sword and move forward to aim at the zombie. I expe |  |
| 500 | a1 | ch3 | LookDown | ->agent_2: agent_2: [in ch3] Turn left to center the diamond block, then dig to break it. |  |
| 500 | a2 | ch3 | MoveLeft | ->agent_1: Turn left to center the diamond block, then move forward to break it. |  |
| 501 | a0 | ch3 | Place | ->agent_1: Agent_1, I am ready to break the switch and collect the wood drop. Please press the switch |  |
| 501 | a1 | ch3 | MoveBackward | ->agent_2: Turn left to center the diamond block, then move forward to break it. |  |
| 501 | a2 | ch3 | MoveLeft | ->agent_1: agent_1: I am centered on the diamond block, ready to Dig. Please press the switch to open |  |
| 502 | a0 | ch3 | LookDown | ->agent_2: Please press the switch to open the door. |  |
| 502 | a1 | ch3 | LookUp | ->agent_2: Turn left to face the movement direction, then MoveForward 3 blocks to survey the area and |  |
| 502 | a2 | ch3 | MoveLeft | ->agent_1: I am aiming at the diamond block. Turn left to center it and dig to break it. |  |
| 503 | a0 | ch3 | Drop | ->agent_1: Switch is centered. Please press it to open the door. |  |
| 503 | a1 | ch3 | Dig | ->agent_2: Switch activated. Cell door should open. |  |
| 503 | a2 | ch3 | Jump | ->agent_1: Turn left to center the diamond block. Press the switch to open the door. |  |
| 504 | a0 | ch3 | MoveRight | ->agent_2: I am digging the diamond block to collect wood and break the switch. Please press the swit |  |
| 504 | a1 | ch3 | TurnLeft | ->agent_2: Switch is centered, ready for you to press. I am ready to cooperate. |  |
| 504 | a2 | ch3 | MoveForward | ->agent_1: Turn left to center the sword, then move forward 3 blocks to survey the area and locate th |  |
| 505 | a0 | ch3 | MoveBackward | ->agent_2: Switch is centered on the diamond block. I am ready to Dig. Agent_2, please press the swit |  |
| 505 | a1 | ch3 | MoveRight | ->agent_2: Turn left to center the diamond block. Press the switch to open the door. |  |
| 505 | a2 | ch3 | NoOp | ->agent_1: Switch is centered, ready for you to press. I am ready to cooperate. |  |
| 506 | a0 | ch3 | MoveLeft | ->agent_1: agent_1: The switch is centered on the diamond block. I will press the switch to open the  |  |
| 506 | a1 | ch3 | LookDown | ->agent_2: Turn left to center the sword, then move forward 3 blocks to survey the area and locate th |  |
| 506 | a2 | ch3 | Sneak | ->agent_1: Agent_1, the diamond block is centered. I need you to press the switch so I can Dig on it. |  |
| 507 | a0 | ch3 | TurnLeft | ->agent_2: Switch is centered on the diamond block. I will press the switch to open the door. |  |
| 507 | a1 | ch3 | LookUp | ->agent_2: Agent_2, the diamond block is centered. I need you to press the switch so I can Dig on it. |  |
| 507 | a2 | ch3 | MoveLeft | ->agent_1: Turn left to center the diamond sword. Press the switch to open the door. |  |
| 508 | a0 | ch3 | TurnRight | ->agent_2: agent_2: Please press the switch to open the door. |  |
| 508 | a1 | ch3 | MoveForward | ->agent_2: Turn left to center the sword. Press the switch to open the door. |  |
| 508 | a2 | ch3 | Dig | ->agent_0: The diamond block is centered. I will press the switch to open the door. |  |
| 509 | a0 | ch3 | LookUp | ->agent_2: agent_2: Switch is centered on the diamond block. I will press it to open the door. |  |
| 509 | a1 | ch3 | MoveBackward | ->agent_2: Press the switch so I can Dig on it. |  |
| 509 | a2 | ch3 | MoveRight | ->agent_1: Switch is centered and ready for cooperative Dig. | critic:FAIL |
| 510 | a0 | ch3 | Sneak | ->agent_1: I'll press the switch to open the door and bring you to the communal room. |  |
| 510 | a1 | ch3 | Drop | ->agent_2: Agent 2, I will turn left to center the diamond sword, then dig to break the stone block. | critic:FAIL |
| 510 | a2 | ch3 | NoOp | ->agent_1: Switch is centered on the diamond block. I will turn left to center it and then press the  |  |
| 511 | a0 | ch3 | Sneak | ->agent_1: Agent_1, the diamond block is centered and ready for the team. I'll press the switch to op |  |
| 511 | a1 | ch3 | MoveBackward | ->agent_2: Turn left to face the switch and press it to open the door. |  |
| 511 | a2 | ch3 | LookUp | ->agent_1: Agent_1, I will turn left to center the diamond sword, then dig to break the stone block. |  |
| 512 | a0 | ch3 | Sneak | ->agent_1: Switch is centered on the diamond block. I will press the switch to open the door. |  |
| 512 | a1 | ch3 | MoveForward | ->agent_2: Agent 2, the diamond sword is centered and ready for the team. |  |
| 512 | a2 | ch3 | Place | ->agent_1: Agent 1, I will turn left to center the diamond sword, then move forward to break the ston |  |
| 513 | a0 | ch3 | LookUp | ->agent_1: The diamond block is centered and accessible. I will turn left to center it and then dig t |  |
| 513 | a1 | ch3 | Sneak | ->agent_2: Switch is centered on the diamond block on the right wall. I will turn right to face it an |  |
| 513 | a2 | ch3 | Place | ->agent_1: Agent 1, the diamond block is centered and ready for the team to Dig. |  |
| 514 | a0 | ch3 | LookUp | ->agent_2: I will move forward to center the diamond sword, then dig to collect the wood drop and act |  |
| 514 | a1 | ch3 | LookUp | ->agent_2: Agent 2, the switch is centered and ready for the team to Dig. |  |
| 514 | a2 | ch3 | Jump | ->agent_1: Agent 1, the diamond block is centered in front of you. Turn right to center it, then move |  |
| 515 | a0 | ch3 | Sneak | ->agent_1: Switch is centered. I will press it to open the door. You should press the switch in the c |  |
| 515 | a1 | ch3 | Sneak | ->agent_2: Agent 2, the diamond block is centered and ready for the team to Dig. |  |
| 515 | a2 | ch3 | Drop | ->agent_0: I am aiming to center the diamond sword. Agent 0 is ready to press the switch. Please coor |  |
| 516 | a0 | ch3 | MoveLeft | ->agent_1: Agent_1: Press the switch to open the door. |  |
| 516 | a1 | ch3 | Place | ->agent_2: I will press the switch to open the door. |  |
| 516 | a2 | ch3 | MoveRight | ->agent_1: I see the diamond block to the right and forward, ready for cooperative Dig. |  |
| 517 | a0 | ch3 | Sneak | ->agent_0 [hebbian_fallback:self_target]: I am aiming to center the diamond sword. Agent 0 is ready to press the switch. Please coor |  |
| 517 | a1 | ch3 | LookUp | ->agent_1 [hebbian_fallback:self_target]: Dig the switch. |  |
| 517 | a2 | ch3 | MoveLeft | ->agent_1: Agent 1, the diamond block is left of center. Please turn left to center it so we can Dig  |  |
| 518 | a0 | ch3 | Drop | ->agent_2: I am ready to Dig with the team. Please coordinate so I can Dig. |  |
| 518 | a1 | ch3 | MoveBackward | ->agent_2: Agent 2, the diamond block is centered. Please turn left to center it so we can Dig togeth |  |
| 518 | a2 | ch3 | TurnLeft | ->agent_1: agent_1: I see the diamond block on the left. Turn left to center it, then dig. I need you |  |