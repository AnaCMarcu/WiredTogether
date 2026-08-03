# Case: collab_success_IPPOpHeb_0

**Run:** `exp06_ippo_hebbian/seed_123` (IPPO+Heb)  **Episode:** 3  **Steps:** 429-464
**Why shortlisted:** m18_door_opened (+60) at t=459, comm_before_coop=True

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 429 | a0 | ch3 | MoveForward | ->agent_1: Turn left to center the lever head against the wall; expect to hear the success sound and  |  |
| 429 | a1 | ch3 | Drop | ->agent_2: I am aiming the lever head against the wall, waiting for it to center. |  |
| 429 | a2 | ch3 | Sneak | ->agent_1: agent_1: I am aiming the lever head against the wall, waiting for it to center. |  |
| 430 | a0 | ch3 | Drop | ->agent_1: Turn left to align the lever head against the wall. Move forward to break it. |  |
| 430 | a1 | ch3 | Drop | ->agent_2: I am aiming the lever head against the wall, waiting for it to center. |  |
| 430 | a2 | ch3 | NoOp | ->agent_1: agent_1: Move forward to center the lever base, then turn left to face it and dig. |  |
| 431 | a0 | ch3 | MoveRight | ->agent_1: I will turn left to align the lever head against the wall, then move forward to approach i |  |
| 431 | a1 | ch3 | Place | ->agent_1 [hebbian_fallback:self_target]: agent_1: I will turn left to center the lever base against the wall, then move forward to  |  |
| 431 | a2 | ch3 | MoveRight | ->agent_1: I am aiming the lever base against the wall, waiting for it to center. Agent_1 is also aim |  |
| 432 | a0 | ch3 | TurnRight | ->agent_1: I will turn left to center the lever head against the wall, then move forward to approach  |  |
| 432 | a1 | ch3 | MoveBackward | ->agent_2: Turn left to center the lever base against the wall, then move forward to approach it, and |  |
| 432 | a2 | ch3 | TurnRight | ->agent_1: Agent_1: You are aligned with the lever head; turn left to face the base and approach. |  |
| 433 | a0 | ch3 | LookUp | ->agent_1: Turn left to center the lever head against the wall, then move forward to approach it. Bre |  |
| 433 | a1 | ch3 | Dig | ->agent_2: I will turn left to center the lever base against the wall, then move forward to approach  |  |
| 433 | a2 | ch3 | Place | ->agent_1: agent_1: I am aiming the lever head against the wall. Please wait for it to center before  |  |
| 434 | a0 | ch3 | Drop | ->agent_1: I am aiming the lever head against the wall, waiting for it to center before breaking. |  |
| 434 | a1 | ch3 | Jump | ->agent_1 [hebbian_fallback:self_target]: Break the lever head against the wall. |  |
| 434 | a2 | ch3 | TurnRight | ->agent_1: agent_1: Dig on the lever base! |  |
| 435 | a0 | ch3 | LookDown | ->agent_1: Turn left to align the lever head against the wall, then move forward to approach it. |  |
| 435 | a1 | ch3 | MoveBackward | ->agent_1 [hebbian_fallback:self_target]: agent_1: Dig on the lever base! |  |
| 435 | a2 | ch3 | MoveRight | ->agent_1: agent_1: Turn left to center the lever base against the wall, then move forward to approac |  |
| 436 | a0 | ch3 | Jump | ->agent_1: Turn left to center the lever head against the wall; expect to hear the success sound and  |  |
| 436 | a1 | ch3 | Place | ->agent_2: agent_2: Turn left to center the lever base against the wall, then move forward to approac |  |
| 436 | a2 | ch3 | Jump | ->agent_1: agent_1: I am aligned with the lever head; turn left to face the base and dig. |  |
| 437 | a0 | ch3 | Dig | ->agent_1: Turn left to center the lever head against the wall; expect to hear the success sound and  |  |
| 437 | a1 | ch3 | TurnLeft | ->agent_0: Agent_0, I am centering the lever base. I expect the lever head to align with the wall and |  |
| 437 | a2 | ch3 | Jump | ->agent_1: agent_1: I am centering the lever base, then I will move forward to approach it. Please wa |  |
| 438 | a0 | ch3 | TurnLeft | ->agent_1: Turn left to center the lever head against the wall; expect to hear the success sound and  |  |
| 438 | a1 | ch3 | MoveRight | ->agent_0: Turn left to center the lever base against the wall; expect to hear the success sound and  |  |
| 438 | a2 | ch3 | LookDown | ->agent_1: agent_1, I have turned left to face the lever base, moving forward to approach it. I am re | critic:FAIL |
| 439 | a0 | ch3 | Jump | ->agent_1: Dig the lever head against the wall; expect to hear the success sound and receive the bonu |  |
| 439 | a1 | ch3 | Jump | ->agent_2: Turn left to center the lever head against the wall; expect to hear the success sound and  | critic:FAIL |
| 439 | a2 | ch3 | MoveForward | ->agent_1: agent_1: I will turn left to center the lever base against the wall, then move forward to  |  |
| 440 | a0 | ch3 | MoveLeft | ->agent_1: I am centering the lever head against the wall. I expect to hear the success sound and rec |  |
| 440 | a1 | ch3 | MoveForward | ->agent_1 [hebbian_fallback:self_target]: I am centering the lever head and base. I will dig the lever base against the wall, then m |  |
| 440 | a2 | ch3 | MoveForward | ->agent_1: agent_1: Turn left to center the lever base against the wall, then move forward to approac |  |
| 441 | a0 | ch3 | LookDown | ->agent_1: Turn left to center the lever head against the wall; expect to hear the success sound and  |  |
| 441 | a1 | ch3 | MoveBackward | ->agent_2: I will turn left to center the lever base against the wall, then move forward to approach  |  |
| 441 | a2 | ch3 | Place | ->agent_1: agent_1: Turn left to center the lever base against the wall, then move forward to approac |  |
| 442 | a0 | ch3 | Place | ->agent_1: Turn left to align the lever base against the wall, then move forward to approach it. I wi |  |
| 442 | a1 | ch3 | Drop | ->agent_2: Turn left to center the lever base against the wall, then move forward to approach it, and |  |
| 442 | a2 | ch3 | Jump | ->agent_1: I am ready to break the lever base. Please confirm the alignment and ask if you need help. |  |
| 443 | a0 | ch3 | MoveLeft | ->agent_1: I am centering the lever head and base. I will dig the lever base against the wall, then m |  |
| 443 | a1 | ch3 | MoveLeft | ->agent_2: I will turn left to center the lever base against the wall, then move forward to approach  |  |
| 443 | a2 | ch3 | LookDown | ->agent_1: Turn left to align the lever base precisely before breaking. |  |
| 444 | a0 | ch3 | LookDown | ->agent_1: I will dig the lever base against the wall, then move forward to approach it. Teammate age |  |
| 444 | a1 | ch3 | TurnRight | ->agent_0: Turn left to center the lever base against the wall, then break it. |  |
| 444 | a2 | ch3 | Drop | ->agent_1: agent_1: I have turned left to align the lever base precisely. Please confirm the lever is |  |
| 445 | a0 | ch3 | Place | ->agent_1: agent_1: I will turn left to center the lever head against the wall, then move forward to  |  |
| 445 | a1 | ch3 | LookUp | ->agent_0: Turn left to center the lever head, then move forward to approach the lever base. |  |
| 445 | a2 | ch3 | Jump | ->agent_1: Turn left to center the lever base against the wall, then move forward to approach it, and |  |
| 446 | a0 | ch3 | Place | ->agent_1: Agent_1, I'll turn left to center the lever head against the wall. Once centered, move for |  |
| 446 | a1 | ch3 | Jump | ->agent_0: Turn left to center the lever base against the wall, then move forward to approach it, and |  |
| 446 | a2 | ch3 | MoveLeft | ->agent_1: agent_1: I am aligned with the lever base and ready to break it. |  |
| 447 | a0 | ch3 | MoveForward | ->agent_1: agent_1: Dig the lever head now; expect success sound and stone block drop. |  |
| 447 | a1 | ch3 | NoOp | ->agent_1 [hebbian_fallback:self_target]: Agent_1, I'll turn left to center the lever base against the wall. Once centered, move for |  |
| 447 | a2 | ch3 | Place | ->agent_1: agent_1: I will turn left to align with the lever base, then move forward to approach it,  |  |
| 448 | a0 | ch3 | Place | ->agent_1: I'll turn left to center the lever head against the wall, then move forward to approach it |  |
| 448 | a1 | ch3 | Sneak | ->agent_1 [hebbian_fallback:self_target]: Agent_1, I will turn left to center the lever base against the wall, then move forward to  |  |
| 448 | a2 | ch3 | Jump | ->agent_1: Turn left to center the lever base, then move forward to approach it. I expect the lever b |  |
| 449 | a0 | ch3 | Place | ->agent_1: Agent_1: I will turn left to center the lever base against the wall, then move forward to  |  |
| 449 | a1 | ch3 | Sneak | ->agent_2: Turn left to center the lever base against the wall, then move forward to approach it. |  |
| 449 | a2 | ch3 | MoveLeft | ->agent_1: agent_1: [in ch3] I am centering the lever head. I need to turn left to face the lever bas |  |
| 450 | a0 | ch3 | Place | ->agent_1: Agent_1, I will turn left to center the lever base against the wall, then move forward to  |  |
| 450 | a1 | ch3 | NoOp | ->agent_1 [hebbian_fallback:self_target]: Agent 1 is breaking the lever base. I will turn left to center the lever base against the  |  |
| 450 | a2 | ch3 | Place | ->agent_1: I am aligning the lever base against the wall to trigger rotation. Agent_1 is waiting for  |  |
| 451 | a0 | ch3 | MoveRight | ->agent_1: Agent_1, I will turn left to center the lever base against the wall, then move forward to  |  |
| 451 | a1 | ch3 | Dig | ->agent_2: I am aligning the lever base against the wall to trigger rotation. |  |
| 451 | a2 | ch3 | Sneak | ->agent_1: Turn left to center the lever base against the wall, then move forward to approach it. I e |  |
| 452 | a0 | ch3 | MoveBackward | ->agent_1: I am centering the lever base against the wall, then moving forward to approach it. Breaki |  |
| 452 | a1 | ch3 | Jump | ->agent_2: Turn left to center the lever head against the wall, then move forward to approach it, and |  |
| 452 | a2 | ch3 | TurnLeft | ->agent_1: Turn left to center the lever head and approach it. I am ready to break it. |  |
| 453 | a0 | ch3 | Jump | ->agent_1: I will turn left to center the lever head against the wall, then move forward to approach  |  |
| 453 | a1 | ch3 | Drop | ->agent_0: I am centering the lever head against the wall, then moving forward to approach it. Breaki |  |
| 453 | a2 | ch3 | Jump | ->agent_1: Turn left to align with the lever base, then move forward to approach it, and finally turn |  |
| 454 | a0 | ch3 | TurnLeft | ->agent_1: Agent_1, I am centering the lever head against the wall. I will turn left to align the lev |  |
| 454 | a1 | ch3 | Sneak | ->agent_1 [hebbian_fallback:self_target]: Turn left to align the lever head against the wall, then move forward to approach it. |  |
| 454 | a2 | ch3 | MoveLeft | ->agent_1: agent_1: Please open the door in Cell 1 so I can trigger the rotation sequence. |  |
| 455 | a0 | ch3 | Dig | ->agent_1: agent_1: I am centering the lever base against the wall, then moving forward to approach i |  |
| 455 | a1 | ch3 | Sneak | ->agent_1 [hebbian_fallback:self_target]: I will turn left to center the lever base against the wall, then move forward to approach  |  |
| 455 | a2 | ch3 | MoveBackward | ->agent_1: Turn left to center the lever base, then move forward to approach it, and finally turn lef |  |
| 456 | a0 | ch3 | LookDown | ->agent_1: Turn left to center the lever head against the wall, then move forward to approach it. |  |
| 456 | a1 | ch3 | TurnLeft | ->agent_2: agent_2: Turn left to center the lever base, then move forward to approach it, and finally |  |
| 456 | a2 | ch3 | Place | ->agent_1: Turn left to center the lever head, then move forward to approach it, and finally turn lef |  |
| 457 | a0 | ch3 | Dig | ->agent_1: Ready to break the lever head with agent_1. |  |
| 457 | a1 | ch3 | Jump | ->agent_1 [hebbian_fallback:self_target]: Agent_1, the lever head is centered on the wall and the base is on the ground. I am ready  |  |
| 457 | a2 | ch3 | Place | ->agent_1: Turn left to center the lever base, then move forward to approach it, and finally turn lef |  |
| 458 | a0 | ch3 | Jump | ->agent_1: Agent_1, the lever head is centered against the wall and the base is aligned. I am ready t |  |
| 458 | a1 | ch3 | Jump | ->agent_2: Turn left to center the lever base against the wall, then move forward to approach it, and |  |
| 458 | a2 | ch3 | Sneak | ->agent_1: agent_1: You are aiming right at the switch. Turn left to face it, then move forward to ce |  |
| 459 | a0 | ch3 | Dig | ->agent_1: Agent_1, I will turn left to center the switch against the wall, then move forward to appr | critic:FAIL; ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 459 | a1 | ch3 | MoveForward | ->agent_1 [hebbian_fallback:self_target]: I am aiming right at the switch. Turn left to face it, then move forward to center the wea | critic:FAIL; ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 459 | a2 | ch3 | Dig | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the lever base, then move forward to approach it, and finally turn lef | ** m17_switch_pressed (+40) **; ** m18_door_opened (+60) ** |
| 460 | a0 | ch3 | Dig | ->agent_1: Turn left to center the lever base against the wall. Break it! | NEW TASK: Announce your cell (A), turn left to align the switch against the wall |
| 460 | a1 | ch3 | MoveForward | ->agent_1 [hebbian_fallback:self_target]: Turn left to center the lever head against the wall, then move forward to approach it, and |  |
| 460 | a2 | ch3 | Drop | ->agent_1: Agent_1, the lever head is centered. I need you to turn left to center the lever base agai | NEW TASK: Turn left to center the lever head in the view; expect to see it cente |
| 461 | a0 | ch3 | Dig | ->agent_1: agent_1: I am aiming right at the lever head. Turn left to face it, then move forward to c |  |
| 461 | a1 | ch3 | MoveLeft | ->agent_0: I will turn left to center the lever base against the wall, then move forward to approach  |  |
| 461 | a2 | ch3 | LookUp | ->agent_1: Switch pressed! Cell A door should now open for teleportation. |  |
| 462 | a0 | ch3 | Place | ->agent_1: Agent_1: You turned left to center the lever base, but I am not sure you are aligned preci |  |
| 462 | a1 | ch3 | Place | ->agent_2: I will turn left to center the lever base against the wall, then move forward to approach  |  |
| 462 | a2 | ch3 | Dig | ->agent_1: agent_1, I am turning left to center the lever head against the wall, then moving forward  |  |
| 463 | a0 | ch3 | Jump | ->agent_1: Turn left to align the lever base against the wall, then move forward to approach it. |  |
| 463 | a1 | ch3 | MoveRight | ->agent_1 [hebbian_fallback:self_target]: Turn left to center the lever base against the wall, then move forward to approach it, and |  |
| 463 | a2 | ch3 | MoveRight | ->agent_1: I am centering the lever base against the wall to prepare for breaking. |  |
| 464 | a0 | ch3 | Dig | ->agent_1: I am turning left to center the lever base against the wall. I will turn left again to ali |  |
| 464 | a1 | ch3 | NoOp | ->agent_2: I am centering the lever base against the wall to prepare for breaking. |  |
| 464 | a2 | ch3 | Place | ->agent_1: Agent_1, I am aligned with the lever base. Please turn left to center the lever base again |  |