# Case: hebbian_fallback_heavy_0

**Run:** `exp05_mappo_hebbian/seed_123` (MAPPO+Heb)  **Episode:** 3  **Steps:** 675-705
**Why shortlisted:** 588 hebbian_fallback-routed messages in run

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 675 | a0 | ch4 | Place | ->agent_1: Turn left to center the sword on the block. |  |
| 675 | a1 | ch4 | MoveForward | ->agent_2: Turn left to center the diamond sword on the block so I can initiate cooperative Dig with  |  |
| 675 | a2 | ch4 | TurnRight | ->agent_2 [hebbian_fallback:self_target]: Agent_2, I am centered on the sword. Turn left to center it, then move forward 3 blocks to |  |
| 676 | a0 | ch4 | Jump | ->agent_1: Turn right to center the sword on the ground, then move forward to reach the block. Agent_ |  |
| 676 | a1 | ch4 | MoveLeft | ->agent_2: Agent_2, I am centered on the cooperative dirt block and ready to initiate the Dig with yo |  |
| 676 | a2 | ch4 | MoveLeft | ->agent_1: I am turning left to center the diamond sword on the block so I can initiate cooperative D |  |
| 677 | a0 | ch4 | Sneak | ->agent_1: Agent_1, I'll turn right to center the diamond sword on the block, then move forward to re |  |
| 677 | a1 | ch4 | Jump | ->agent_2: Agent_2, I am centered on the block. Please turn left to center the sword on it so we can  |  |
| 677 | a2 | ch4 | MoveLeft | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the diamond sword on the block so I can initiate cooperative Dig with  |  |
| 678 | a0 | ch4 | Dig | ->agent_1: Turn right to center the diamond block on the ground, then move forward to reach it. |  |
| 678 | a1 | ch4 | Place | ->agent_2: Agent_2: Ready to pair-dig the cooperative dirt block. |  |
| 678 | a2 | ch4 | TurnRight | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the zombie, then move forward 3 blocks to reach the cooperative spot. |  |
| 679 | a0 | ch4 | TurnRight | ->agent_1: Turn right to center the diamond sword on the zombie. I will Dig to kill it. |  |
| 679 | a1 | ch4 | Dig | ->agent_2: Agent_2, I am centered on the block, ready to Dig. You are already in position to pair-dig |  |
| 679 | a2 | ch4 | MoveLeft | ->agent_1: I am centered on the cooperative spot with my diamond sword. Please turn left to center th |  |
| 680 | a0 | ch4 | MoveForward | ->agent_1: Agent_1, center the red sandstone block on the ground. |  |
| 680 | a1 | ch4 | MoveLeft | ->agent_2: Agent_2 is centered on the cooperative spot with my diamond sword. Please turn left to cen |  |
| 680 | a2 | ch4 | TurnLeft | ->agent_1: Look down at the switch, turn left to center the diamond sword, then move forward 3 blocks |  |
| 681 | a0 | ch4 | MoveBackward | ->agent_1: Agent_1, you turned left but now face away from the block. Turn left to center the diamond | critic:FAIL |
| 681 | a1 | ch4 | Jump | ->agent_2: Agent_2, I am centered on the cooperative dirt block, ready to initiate cooperative Dig wi | critic:FAIL |
| 681 | a2 | ch4 | MoveRight | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the sword on the cooperative dirt block so we can dig together. | critic:FAIL |
| 682 | a0 | ch4 | Sneak | ->agent_1: Turn left to center the red sandstone block on the ground. I am ready to initiate cooperat |  |
| 682 | a1 | ch4 | Dig | ->agent_2: I am centered and ready to initiate cooperative Dig with Agent_2. |  |
| 682 | a2 | ch4 | Place | ->agent_1: Agent_1, I am centered on the cooperative dirt block, ready to initiate cooperative Dig wi |  |
| 683 | a0 | ch4 | MoveRight | ->agent_1: Turn left to center the diamond sword. Turn left to center the diamond sword. |  |
| 683 | a1 | ch4 | Place | ->agent_2: Agent_2, I am centered on the cooperative dirt block, ready to initiate cooperative Dig wi |  |
| 683 | a2 | ch4 | TurnLeft | ->agent_1: Agent_1, I am centered on the cooperative dirt block, ready to initiate cooperative Dig wi |  |
| 684 | a0 | ch4 | TurnRight | ->agent_1: Turn left to center the zombie. Turn left to center the zombie. |  |
| 684 | a1 | ch4 | TurnLeft | ->agent_1 [hebbian_fallback:self_target]: I am centered on the cooperative dirt block, ready to initiate cooperative Dig with Agent_ |  |
| 684 | a2 | ch4 | TurnLeft | ->agent_1: Agent_1, I am centered on the cooperative spot with my diamond sword. I will turn left to  |  |
| 685 | a0 | ch4 | Jump | ->agent_1: I am ready to dig. Please initiate cooperative Dig with me. |  |
| 685 | a1 | ch4 | TurnLeft | ->agent_2: Agent_2, I am centered on the cooperative spot with my diamond sword. I will turn left to  |  |
| 685 | a2 | ch4 | LookUp | ->agent_2 [hebbian_fallback:self_target]: Agent_2, I am centered on the dirt block and ready to pair-dig. Please initiate cooperativ |  |
| 686 | a0 | ch4 | MoveLeft | ->agent_1: Turn left to center the diamond sword. |  |
| 686 | a1 | ch4 | MoveRight | ->agent_2: Agent_2, I am centered on the dirt block and ready to pair-dig. Please initiate cooperativ |  |
| 686 | a2 | ch4 | NoOp | ->agent_2 [hebbian_fallback:self_target]: I'll turn left to center the diamond sword on the dirt block, then move forward to reach i |  |
| 687 | a0 | ch4 | MoveRight | ->agent_1: I will turn left to center the diamond sword on the dirt block. Turn left to center the di |  |
| 687 | a1 | ch4 | Drop | ->agent_0: I'm centered on the diamond sword. Ready to initiate cooperative Dig with Agent_0. |  |
| 687 | a2 | ? | Place | ->agent_2 [hebbian_fallback:self_target]: Agent_2, I am centered on the dirt block and ready to pair-dig. Please initiate cooperativ |  |
| 688 | a0 | ch4 | MoveBackward | ->agent_1: Turn left to center the zombie, then move forward to reach it. |  |
| 688 | a1 | ch4 | MoveLeft | ->agent_2: Agent_2, I will pair-dig the cooperative dirt block to raise the anvil's HP. |  |
| 688 | a2 | ? | TurnLeft | ->agent_2 [hebbian_fallback:self_target]: Agent_2, I am centered on the dirt block and ready to pair-dig. Please initiate cooperativ |  |
| 689 | a0 | ch4 | TurnRight | ->agent_1: Turn left to center the diamond sword on the dirt block, then move forward to reach Agent_ |  |
| 689 | a1 | ch4 | MoveLeft | ->agent_2: Agent_2, I am centered on the dirt block and ready to pair-dig. Please initiate cooperativ |  |
| 689 | a2 | ch4 | MoveRight | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the diamond sword, then move forward to reach the dirt block, then ini |  |
| 690 | a0 | ch4 | MoveForward | ->agent_1: Turn left to center the diamond sword on the red sandstone block. Initiate cooperative Dig |  |
| 690 | a1 | ch4 | TurnRight | ->agent_2: Agent_2, I am aiming the sword for the zombies on the right. You are positioned to center  |  |
| 690 | a2 | ch4 | Drop | ->agent_1: Agent_1, I am centered on the dirt block and ready to initiate cooperative Dig. Please tur |  |
| 691 | a0 | ch4 | MoveLeft | ->agent_1: Turn right to center the diamond sword on the red sandstone block, then initiate cooperati |  |
| 691 | a1 | ch4 | NoOp | ->agent_0: I am centered on the dirt block. I will turn left to center the diamond sword on the red s |  |
| 691 | a2 | ch4 | TurnLeft | ->agent_1: I am centered on the dirt block. I will move forward 3 blocks to reach the dirt block, the |  |
| 692 | a0 | ch4 | MoveBackward | ->agent_1: Agent_1: I am centered on the red sandstone block. I will turn left to center the diamond  |  |
| 692 | a1 | ch4 | MoveLeft | ->agent_1 [hebbian_fallback:self_target]: Agent_2: I am centered on the diamond sword. I will move forward 3 blocks to reach the red |  |
| 692 | a2 | ch4 | MoveForward | ->agent_2 [hebbian_fallback:self_target]: I will turn left to center the sword on the dirt block, then move forward to reach it, and |  |
| 693 | a0 | ch4 | MoveLeft | ->agent_1: TurnLeft to center the diamond sword, then MoveForward to reach Agent_1's position. |  |
| 693 | a1 | ch4 | Place | ->agent_0: I am centered on the red sandstone block. I will turn left to center the diamond sword on  |  |
| 693 | a2 | ch4 | Place | ->agent_1: Turn left to center the diamond sword on the dirt block, then move forward 3 blocks to rea |  |
| 694 | a0 | ch4 | Sneak | ->agent_1: Turn right to center the red sandstone block on your diamond sword, then move forward to r |  |
| 694 | a1 | ch4 | NoOp | ->agent_2: TurnLeft to center the diamond sword on the cooperative dirt block, then MoveForward to re |  |
| 694 | a2 | ch4 | MoveLeft | ->agent_2 [hebbian_fallback:self_target]: I am centered on the diamond sword, ready to initiate cooperative Dig with Agent_2 on the  |  |
| 695 | a0 | ch4 | Jump | ->agent_1: Agent_1, I am centered on the zombie. I will turn left to center the diamond sword on it,  |  |
| 695 | a1 | ch4 | TurnLeft | ->agent_0: Agent_0, please turn left to center the diamond sword on the cooperative dirt block. Once  |  |
| 695 | a2 | ch4 | Dig | ->agent_2 [hebbian_fallback:self_target]: Move forward 3 blocks to reach the dirt block, then turn left to center the sword. We are  |  |
| 696 | a0 | ch4 | Place | ->agent_1: Agent_1, I am centered on the red sandstone block with my sword. Turn left to center the d |  |
| 696 | a1 | ch4 | Dig | ->agent_2: We are ready to initiate cooperative Dig with Agent_2. Please move forward 3 blocks to rea |  |
| 696 | a2 | ch4 | TurnRight | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the sword. Move forward 3 blocks. Dig with Agent_2. |  |
| 697 | a0 | ch4 | MoveLeft | ->agent_1: Turn right to approach the red sandstone block. Once there, center it and initiate coopera |  |
| 697 | a1 | ch4 | Drop | ->agent_2: I am centered on the diamond sword and ready to initiate the cooperative Dig with Agent_2. |  |
| 697 | a2 | ch4 | MoveLeft | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the diamond sword on the dirt block, then move forward 3 blocks to rea |  |
| 698 | a0 | ch4 | MoveBackward | ->agent_1: Agent_1, please turn right to center the diamond sword on the dirt block. |  |
| 698 | a1 | ch4 | Dig | ->agent_1 [hebbian_fallback:self_target]: I have centered the cooperative dirt block. I will initiate cooperative Dig with Agent_1 t |  |
| 698 | a2 | ch4 | TurnLeft | ->agent_2 [hebbian_fallback:self_target]: Turn left to center the sword on the dirt block. Move forward 3 blocks to reach the dirt b |  |
| 699 | a0 | ch4 | Dig | ->agent_1: Agent_1, please turn right to center the zombie on the block so we can initiate cooperativ |  |
| 699 | a1 | ch4 | Place | ->agent_2: Agent_2, please move forward 3 blocks to center the block and turn left to align the sword |  |
| 699 | a2 | ch4 | TurnRight | ->agent_1: I am centered on the sword and ready to turn left and move forward to reach the dirt block |  |
| 700 | a0 | ch4 | Dig | ->agent_1: Agent_1, please turn left to center the diamond sword on the zombie. Once centered, move f |  |
| 700 | a1 | ch4 | MoveLeft | ->agent_2: I am centered on the sword and ready to move forward to reach the block. Please turn left  |  |
| 700 | a2 | ch4 | MoveLeft | ->agent_2 [hebbian_fallback:self_target]: Agent_2, I am centered on the cooperative dirt block. Please move forward 3 blocks to reac |  |
| 701 | a0 | ch4 | LookUp | ->agent_1: Agent_1, please turn left to center the diamond sword on the cooperative dirt block. Once  | critic:OK |
| 701 | a1 | ch4 | MoveLeft | ->agent_2: Agent_2, I am centered on the cooperative dirt block. Please move forward to reach the blo | NEW TASK: Initiate cooperative Dig on the cooperative dirt block; expect the blo |
| 701 | a2 | ch4 | MoveLeft | ->agent_2 [hebbian_fallback:self_target]: Agent_2, please move forward 3 blocks to center the block and turn left to align the sword | critic:FAIL |
| 702 | a0 | ch4 | Place | ->agent_1: Agent_1, please turn left to center the diamond sword on the cooperative dirt block. Once  | NEW TASK: Move forward and turn left to center the cooperative dirt block; ask A |
| 702 | a1 | ch4 | Sneak | ->agent_0: Agent_0, the diamond sword is currently to the right, not centered. Please turn left to ce |  |
| 702 | a2 | ch4 | MoveRight | ->agent_2 [hebbian_fallback:self_target]: Turn right to center the zombie, then move forward to reach the dirt block. We will initia |  |
| 703 | a0 | ch4 | MoveRight | ->agent_1: Agent_1, the diamond sword is centered on the zombie, please turn left to center the diamo |  |
| 703 | a1 | ch4 | MoveBackward | ->agent_1 [hebbian_fallback:self_target]: Agent_1, the diamond sword is now centered on the stone block. I will move forward to reac | NEW TASK: Turn left to center the diamond sword; expect to see the diamond sword |
| 703 | a2 | ch4 | MoveLeft | ->agent_1: Turn left to center the sword and move forward to reach the block, then pair-dig with Agen |  |
| 704 | a0 | ch4 | MoveBackward | ->agent_1: Agent_1, the diamond sword is to the right of the center. Please turn left to center it on | NEW TASK: Align directly over the cooperative dirt block; expect to hear a coope |
| 704 | a1 | ch4 | Jump | ->agent_2: I have aligned the sword on the zombie, I will move forward to reach the block, then pair- |  |
| 704 | a2 | ch4 | Sneak | ->agent_2 [hebbian_fallback:self_target]: Agent_2, I am centered on the sword. Please move forward to reach the dirt block and turn  |  |
| 705 | a0 | ch4 | MoveForward | ->agent_1: Agent_1, I've turned left to center the diamond sword on the block. Move forward to reach  | NEW TASK: Move forward; expect to see the zombie centered in the view. Turn left |
| 705 | a1 | ch4 | TurnRight | ->agent_1 [hebbian_fallback:self_target]: I am centered on the diamond sword. Please turn left to center it on the zombie, then move |  |
| 705 | a2 | ch4 | TurnRight | ->agent_2 [hebbian_fallback:self_target]: Agent_2, I am looking at a stone block. I will turn right to see the dirt block you mentio |  |