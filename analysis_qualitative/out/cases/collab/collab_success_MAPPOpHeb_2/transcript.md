# Case: collab_success_MAPPOpHeb_2

**Run:** `exp05_mappo_hebbian/seed_42` (MAPPO+Heb)  **Episode:** 2  **Steps:** 494-529
**Why shortlisted:** m18_door_opened (+60) at t=524, comm_before_coop=True

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 494 | a0 | ch3 | MoveRight | ->agent_1: I will turn left to center the lever and move forward to eye level, then jump to break it. |  |
| 494 | a1 | ch3 | MoveForward | ->agent_0: I have reached the lever and turned right to center it. Please press the lever to drop the |  |
| 494 | a2 | ch3 | TurnRight | ->agent_1: I am aiming at the lever on the left wall. I expect it to be centered next step. I will br |  |
| 495 | a0 | ch3 | TurnRight | ->agent_1: I have centered the lever and am ready to break it. Please press the lever to drop the dia |  |
| 495 | a1 | ch3 | MoveRight | ->agent_2: I am aiming at the lever on the left wall. I expect it to be centered next step. I will br |  |
| 495 | a2 | ch3 | MoveForward | ->agent_1: TurnLeft to center the switch, then Dig to press it. I expect the switch to break. |  |
| 496 | a0 | ch3 | Jump | ->agent_1: I am aiming at the lever and ready to break it. I expect the diamond sword to drop after b |  |
| 496 | a1 | ch3 | Sneak | ->agent_2: Agent 2 has confirmed the lever break scheme to initiate the sequence. Turn left to center |  |
| 496 | a2 | ch3 | MoveForward | ->agent_1: Press the lever to drop the diamond sword and heal HP. |  |
| 497 | a0 | ch3 | MoveRight | ->agent_1: agent_1: Please press the lever to drop the diamond sword and open the door. |  |
| 497 | a1 | ch3 | TurnLeft | ->agent_0: I am aiming at the lever and ready to break it. I expect the diamond sword to drop after b |  |
| 497 | a2 | ch3 | Place | ->agent_1: I am aiming at the lever. I expect breaking it will drop the diamond sword and open Door 1 |  |
| 498 | a0 | ch3 | MoveForward | ->agent_1: agent_1: Ready to press lever to drop sword and open door. |  |
| 498 | a1 | ch3 | MoveLeft | ->agent_2: Agent 2, I am aiming at the lever. I expect breaking it will drop the diamond sword and op |  |
| 498 | a2 | ch3 | LookUp | ->agent_1: agent_1: The lever is centered and broken. I need to press it again to drop the diamond sw |  |
| 499 | a0 | ch3 | Jump | ->agent_1: I have looked up and centered the lever. I am aiming at it and ready to jump and break it  |  |
| 499 | a1 | ch3 | Drop | ->agent_0: agent_0: The lever is centered and broken. I need to press it again to drop the diamond sw |  |
| 499 | a2 | ch3 | MoveLeft | ->agent_1: Agent 1, breaking the lever drops the diamond sword and heals me. Please press the lever t |  |
| 500 | a0 | ch3 | MoveRight | ->agent_1: agent_1: The lever is centered; I am facing it. I will TurnRight to align it with the cros |  |
| 500 | a1 | ch3 | Drop | ->agent_0: The lever is centered directly in front of the crosshair. I should move forward to reach t |  |
| 500 | a2 | ch3 | Place | ->agent_1: I am aiming at the lever. I expect breaking it will drop the diamond sword and open Door 1 |  |
| 501 | a0 | ch3 | Drop | ->agent_1: Jump to break the lever. I will drop the diamond sword. |  |
| 501 | a1 | ch3 | LookUp | ->agent_1 [hebbian_fallback:self_target]: I am ready to break the lever and expect the diamond sword to drop. |  |
| 501 | a2 | ch3 | MoveForward | ->agent_1: agent_1: Confirm lever break scheme. Ask to help break the anvil together. |  |
| 502 | a0 | ch3 | NoOp | ->agent_1: The lever is centered in front of me. I should move forward to reach it, then jump to brea |  |
| 502 | a1 | ch3 | LookDown | ->agent_1 [hebbian_fallback:self_target]: I will move forward to reach the lever surface, then turn right to face it so the lever is |  |
| 502 | a2 | ch3 | MoveLeft | ->agent_1: I am aiming at the lever, expecting to break it to drop the diamond sword and heal HP. |  |
| 503 | a0 | ch3 | Jump | ->agent_1: Move forward to reach the lever surface. |  |
| 503 | a1 | ch3 | MoveBackward | ->agent_2: I am aiming at the lever, expecting to break it to drop the diamond sword and heal HP. |  |
| 503 | a2 | ch3 | LookUp | ->agent_1: I am ready to break the lever. Breaking this lever will drop the diamond sword, heal HP to |  |
| 504 | a0 | ch3 | NoOp | ->agent_1: agent_1: The lever is centered in front of you. I will press the lever to drop the diamond | critic:FAIL |
| 504 | a1 | ch3 | Place | ->agent_2: Move forward to reach the lever. | critic:FAIL |
| 504 | a2 | ch3 | Dig | ->agent_1: I am breaking the lever to drop the diamond sword and heal HP. |  |
| 505 | a0 | ch3 | MoveLeft | ->agent_1: I am breaking the lever; expect to see it drop the diamond sword and equip me. |  |
| 505 | a1 | ch3 | MoveForward | ->agent_0: I am breaking the lever. Please press the lever switch to drop the diamond sword and heal  |  |
| 505 | a2 | ch3 | Jump | ->agent_1: agent_1: I am aiming at the lever. I need you to help break it by pressing the break butto |  |
| 506 | a0 | ch3 | Place | ->agent_1: Agent 1, the lever is centered and ready. Please press the lever switch to drop the diamon |  |
| 506 | a1 | ch3 | Dig | ->agent_2: agent_2: I am aiming at the lever. I need you to help break it by pressing the break butto |  |
| 506 | a2 | ch3 | MoveBackward | ->agent_1: agent_1: I am aiming at the lever, expecting to break it to drop the diamond sword and hea |  |
| 507 | a0 | ch3 | LookUp | ->agent_1: The lever is centered and at eye level, but slightly to the left. I should turn left to ce |  |
| 507 | a1 | ch3 | TurnLeft | ->agent_2: agent_2: The lever is centered and ready for the break. I am positioning myself to break i |  |
| 507 | a2 | ch3 | Place | ->agent_1: Dig the stone block to drop the diamond sword. |  |
| 508 | a0 | ch3 | Place | ->agent_1: agent_1: The lever is centered in front of me. I should move forward to reach the lever su |  |
| 508 | a1 | ch3 | LookDown | ->agent_1 [hebbian_fallback:self_target]: Turn left to center the lever. Then move forward to reach it, and dig to break the lever. |  |
| 508 | a2 | ch3 | MoveRight | ->agent_1: Break the lever to drop the diamond sword and heal HP to 20/20. |  |
| 509 | a0 | ch3 | Sneak | ->agent_1: Press the lever switch to drop the diamond sword. |  |
| 509 | a1 | ch3 | Dig | ->agent_2: Agent 2, the lever is centered in front of me. I should move forward to reach the lever su |  |
| 509 | a2 | ch3 | MoveLeft | ->agent_1: agent_1: I am ready to break the lever again. I will press the break button to drop the di |  |
| 510 | a0 | ch3 | MoveRight | ->agent_1: agent_1: The lever is centered in front of me. I am jumping to reach the lever surface to  |  |
| 510 | a1 | ch3 | Place | ->agent_2: Turn right to align the lever for the next break. |  |
| 510 | a2 | ch3 | MoveRight | ->agent_1: Agent 1, I am centered on the lever. I expect breaking it will drop the diamond sword and  |  |
| 511 | a0 | ch3 | TurnRight | ->agent_1: Turn left to center the lever. I am ready to dig. |  |
| 511 | a1 | ch3 | Drop | ->agent_1 [hebbian_fallback:self_target]: I am centered on the lever. Please confirm the scheme. |  |
| 511 | a2 | ch3 | Drop | ->agent_1: agent_1: Break the lever to drop the diamond sword and heal HP. |  |
| 512 | a0 | ch3 | Sneak | ->agent_1: I will drop the diamond sword. Please confirm you are ready to dig with the sword. |  |
| 512 | a1 | ch3 | MoveForward | ->agent_2: agent_2: I am ready to dig the lever. |  |
| 512 | a2 | ch3 | MoveLeft | ->agent_1: agent_1: Please confirm the scheme that breaking this lever drops the diamond sword and he |  |
| 513 | a0 | ch3 | Place | ->agent_1: The lever is centered in the view, but slightly to the right. I should turn right to cente |  |
| 513 | a1 | ch3 | Jump | ->agent_2: I am aiming the lever. Please confirm the scheme that breaking this lever drops the diamon |  |
| 513 | a2 | ch3 | MoveBackward | ->agent_1: agent_1: Confirm the scheme that breaking the lever drops the sword and heals HP. |  |
| 514 | a0 | ch3 | LookDown | ->agent_1: Agent 1, lever is centered in front of me, ready to break. I will turn forward to reach it |  |
| 514 | a1 | ch3 | MoveRight | ->agent_2: I am breaking the lever to drop the diamond sword and open the Ch5 door. |  |
| 514 | a2 | ch3 | MoveForward | ->agent_1: I am aiming the lever. Please confirm the scheme that breaking this lever drops the diamon |  |
| 515 | a0 | ch3 | Place | ->agent_1: I am moving forward to reach the lever surface, then turning right to face it so the lever |  |
| 515 | a1 | ch3 | MoveRight | ->agent_2: Agent 2, I am aiming the lever. Please confirm the scheme that breaking this lever drops t |  |
| 515 | a2 | ch3 | NoOp | ->agent_1: I am breaking the lever to drop the diamond sword and open the Ch5 door. Please confirm th |  |
| 516 | a0 | ch3 | Place | ->agent_1: Agent 1 is breaking the lever. I will jump to break the lever and drop the diamond sword. |  |
| 516 | a1 | ch3 | LookDown | ->agent_2: Agent 2: I am breaking the lever to drop the diamond sword and open the Ch5 door. Please c |  |
| 516 | a2 | ch3 | MoveRight | ->agent_1: Agent 1, please confirm the scheme that breaking the lever drops the diamond sword and hea |  |
| 517 | a0 | ch3 | Place | ->agent_1: I am breaking the lever. Please press the lever switch to drop the diamond sword and heal  |  |
| 517 | a1 | ch3 | MoveLeft | ->agent_2: TurnLeft to aim at the lever, then Dig. |  |
| 517 | a2 | ch3 | LookUp | ->agent_1: I am turning right to face the lever. I am breaking the lever to drop the diamond sword an |  |
| 518 | a0 | ch3 | Sneak | ->agent_1: Agent 1 is breaking the lever. Please press the lever switch to drop the diamond sword and |  |
| 518 | a1 | ch3 | MoveForward | ->agent_1 [hebbian_fallback:self_target]: I am breaking the lever to drop the diamond sword and heal HP. |  |
| 518 | a2 | ch3 | MoveBackward | ->agent_1: I am breaking the lever again to drop the diamond sword and heal my HP. Please confirm the |  |
| 519 | a0 | ch3 | Sneak | ->agent_1: I am breaking the lever. Please press the lever switch to drop the diamond sword and heal  |  |
| 519 | a1 | ch3 | LookDown | ->agent_2: Agent 2, confirm the scheme that breaking the lever drops the sword and heals HP. |  |
| 519 | a2 | ch3 | MoveBackward | ->agent_1: Confirming the scheme: breaking the lever drops the sword and heals HP. |  |
| 520 | a0 | ch3 | MoveBackward | ->agent_1: I am breaking the lever to open the door. Please press the lever switch to drop the diamon |  |
| 520 | a1 | ch3 | MoveLeft | ->agent_0: I am breaking the lever to drop the diamond sword. Please press the lever switch to drop t |  |
| 520 | a2 | ch3 | Dig | ->agent_1: Agent 1, I am breaking the lever again to drop the diamond sword. Please confirm the schem |  |
| 521 | a0 | ch3 | TurnLeft | ->agent_1: Jump to break the lever and equip the sword. agent_1 is ready to help. |  |
| 521 | a1 | ch3 | Sneak | ->agent_1 [hebbian_fallback:self_target]: Agent 1, I am breaking the lever to drop the diamond sword. Please press the lever switch  |  |
| 521 | a2 | ch3 | Place | ->agent_1: agent_1: Confirm the scheme that breaking the lever drops the sword and heals HP. |  |
| 522 | a0 | ch3 | MoveLeft | ->agent_1: I am breaking the lever. Please press the lever switch to drop the diamond sword and heal  |  |
| 522 | a1 | ch3 | Jump | ->agent_1 [hebbian_fallback:self_target]: I've moved forward and centered the lever. Dig to break it. |  |
| 522 | a2 | ch3 | MoveForward | ->agent_1: Breaking the lever drops the sword and heals HP. Please press the switch to confirm. |  |
| 523 | a0 | ch3 | MoveRight | ->agent_1: I am centering the lever. Please confirm your position and readiness to help break an anvi |  |
| 523 | a1 | ch3 | Place | ->agent_2: Agent 2: Please press the lever switch to drop the diamond sword. |  |
| 523 | a2 | ch3 | MoveRight | ->agent_1: I've centered the lever. Dig to break it and drop the sword. |  |
| 524 | a0 | ch3 | Dig | ->agent_1: Turn left to center the lever. Ready to dig. | critic:FAIL; ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 524 | a1 | ch3 | MoveLeft | ->agent_2: Dig to break the lever. | critic:FAIL; ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 524 | a2 | ch3 | Jump | ->agent_1: Dig to break the lever. | ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 525 | a0 | ch3 | TurnRight | ->agent_1: Press the lever switch to drop the diamond sword and heal HP. | NEW TASK: Turn to face the lever; expect the lever to appear centered in the vie |
| 525 | a1 | ch3 | MoveForward | ->agent_2: Move forward 3 blocks to reach the lever, then turn right to face it. Ready to Dig. | NEW TASK: Turn until the lever is perfectly centered in the screen; expect the l |
| 525 | a2 | ch3 | MoveForward | ->agent_1: Please press the lever switch to drop the diamond sword. |  |
| 526 | a0 | ch3 | Place | ->agent_1: I am breaking the lever with the diamond sword to open Ch5 door. |  |
| 526 | a1 | ch3 | Jump | ->agent_2: Press the lever switch to drop the diamond sword. |  |
| 526 | a2 | ch3 | Drop | ->agent_1: I will dig again to break the lever and collect the diamond sword, then press it to drop i |  |
| 527 | a0 | ch3 | NoOp | ->agent_1: Turn left to center the lever, then press to break it and drop the diamond sword. |  |
| 527 | a1 | ch3 | MoveBackward | ->agent_1 [hebbian_fallback:self_target]: I am breaking the lever to open the Ch5 door. |  |
| 527 | a2 | ch3 | TurnRight | ->agent_1: Break the lever to drop the diamond sword. We need to press the switch again to trigger th |  |
| 528 | a0 | ch3 | Dig | ->agent_1: Press the lever switch to drop the diamond sword and heal HP. |  |
| 528 | a1 | ch3 | MoveRight | ->agent_1 [hebbian_fallback:self_target]: Press the lever to break it and drop the diamond sword. |  |
| 528 | a2 | ch3 | MoveBackward | ->agent_1: I see the lever is high up. I need to look down to center it, then move forward to break i |  |
| 529 | a0 | ch3 | MoveRight | ->agent_1: Turn left to center the lever, then dig. |  |
| 529 | a1 | ch3 | MoveLeft | ->agent_2: Agent 2, I see the lever is centered directly in front of me. I should move forward to rea |  |
| 529 | a2 | ch3 | Jump | ->agent_1: I am breaking the lever again to drop the diamond sword and heal my HP. |  |