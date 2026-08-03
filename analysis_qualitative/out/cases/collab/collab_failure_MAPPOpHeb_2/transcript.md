# Case: collab_failure_MAPPOpHeb_2

**Run:** `exp05_mappo_hebbian/seed_789` (MAPPO+Heb)  **Episode:** 1  **Steps:** 136-186
**Why shortlisted:** reached ch2, first anvil never broken (48 anvil msgs, 157 ch2 digs)

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 136 | a0 | ch1 | Dig | ->agent_1: Agent 1: I am centered on the tree. I am waiting for the next turn to Dig. |  |
| 136 | a1 | ch2 | Dig | ->agent_2: Dig the diamond sword to break the anvil. |  |
| 136 | a2 | ch1 | Dig | ->agent_1: Tree is centered and ready to dig; I expect to align with the trunk next step. |  |
| 137 | a0 | ch1 | Dig | ->agent_2: Agent 2: Turn left to align the sheep and tree, then move forward to center the tree block |  |
| 137 | a1 | ch2 | Drop | ->agent_0: Turn left 1 block to center the tree, then Dig. |  |
| 137 | a2 | ch1 | MoveLeft | ->agent_1: I will turn left to align the tree with the center, then move forward 5 blocks to locate i |  |
| 138 | a0 | ch1 | TurnLeft | ->agent_1: I am centered on the diamond block. I am waiting for the next turn to Dig on the block. |  |
| 138 | a1 | ch2 | TurnLeft | ->agent_2: Agent 2 is facing the tree and will turn left to center it, then move forward and dig the  |  |
| 138 | a2 | ch1 | Place | ->agent_1: Agent_1: Turn left to align the sheep and tree, then move forward to center the tree block |  |
| 139 | a0 | ch1 | Dig | ->agent_1: Agent_1: I am centered on the diamond block. I am waiting for the next turn to Dig on the  | critic:FAIL |
| 139 | a1 | ch2 | TurnRight | ->agent_2: Agent_2: Turn left to center the anvil and zombie, then move forward to start the Dig. | critic:FAIL |
| 139 | a2 | ch1 | MoveForward | ->agent_1: Agent 1: Move forward to check for tree alignment or new targets. | critic:FAIL |
| 140 | a0 | ch1 | TurnRight | ->agent_1: Agent_1: I see the stone block on the right. Turn right and move forward to collect it. |  |
| 140 | a1 | ch2 | TurnLeft | ->agent_2: Agent_2: Tree is centered left, turn left to center it so we can dig. |  |
| 140 | a2 | ch1 | MoveForward | ->agent_1: I am turning left to center the tree and will dig 5 blocks for wood. |  |
| 141 | a0 | ch1 | TurnLeft | ->agent_1: Turn left to center the tree and prepare for Dig. |  |
| 141 | a1 | ch2 | TurnRight | ->agent_2: Agent_2, I see the tree is centered and close. I will turn left to face it and move forwar |  |
| 141 | a2 | ch1 | Place | ->agent_1: Tree is centered left, turn left to center it so we can dig. |  |
| 142 | a0 | ch1 | Jump | ->agent_1: Agent_1: I am centered on the diamond block. I am waiting for the next turn to Dig on the  |  |
| 142 | a1 | ch2 | Dig | ->agent_2: Agent 2, we are ready to punch the anvil. |  |
| 142 | a2 | ch1 | Dig | ->agent_1: Tree is centered left, turn left to center it so we can dig. |  |
| 143 | a0 | ch1 | TurnLeft | ->agent_1: Agent_1, I am centered on the stone block. I will Dig to drop dirt. Agent_1, please center |  |
| 143 | a1 | ch2 | MoveForward | ->agent_1 [hebbian_fallback:self_target]: I am centering the tree now, ready to break it. |  |
| 143 | a2 | ch1 | MoveForward | ->agent_1: I see the tree is centered and close. I will turn left to face it and move forward to atta |  |
| 144 | a0 | ch1 | Dig | ->agent_1: Agent_1: Tree is centered and aligned for cooperative Dig. I am ready to Dig on it. |  |
| 144 | a1 | ch2 | Dig | ->agent_2: Agent_2, I see the tree is centered and close. I will turn left to face it and move forwar |  |
| 144 | a2 | ch1 | Dig | ->agent_1: Agent 1, tree is centered, ready to break it. |  |
| 145 | a0 | ch1 | TurnLeft | ->agent_2: Agent_2: I am centered on the diamond block. I am waiting for the next turn to Dig on the  |  |
| 145 | a1 | ch2 | Place | ->agent_1 [hebbian_fallback:self_target]: Move forward 1 block to approach the anvils for cooperative breaking. |  |
| 145 | a2 | ch1 | LookUp | ->agent_1: I am breaking the tree and collecting wood. I need to move forward 5 blocks to collect the |  |
| 146 | a0 | ch1 | TurnLeft | ->agent_2: agent_2: Turn left to center you, then Dig to hit the zombie. |  |
| 146 | a1 | ch2 | TurnLeft | ->agent_2: Agent_2, I am breaking the anvil and collecting wood. I need to move forward 5 blocks to c |  |
| 146 | a2 | ch1 | LookUp | ->agent_1: Agent 1: I am centered on the diamond block. I am waiting for the next turn to Dig on the  |  |
| 147 | a0 | ch1 | TurnRight | ->agent_1: I am centered on the diamond block and ready to initiate the cooperative Dig-on-stone acti |  |
| 147 | a1 | ch2 | MoveLeft | ->agent_2: agent_2: I am centered on the diamond sword and ready to Dig. I expect to hit the zombie i |  |
| 147 | a2 | ch1 | Dig | ->agent_1: Turn left to center the tree, then move forward to dig. We can move forward 5 blocks to co |  |
| 148 | a0 | ch1 | TurnLeft | ->agent_1: Agent_1: Turn left to center the tree and Dig. |  |
| 148 | a1 | ch2 | Dig | ->agent_2: I am centered on the purple anvil in Row B. I am moving forward to center it before breaki |  |
| 148 | a2 | ch1 | Drop | ->agent_2 [hebbian_fallback:self_target]: I am breaking the block ahead to collect wood for the milestone. |  |
| 149 | a0 | ch1 | TurnLeft | ->agent_1: agent_1: Turn right to face the stone block properly, then MoveForward to center it. |  |
| 149 | a1 | ch2 | Drop | ->agent_0: The anvil is broken. Wait for the chestplate drop. |  |
| 149 | a2 | ch1 | MoveForward | ->agent_1: Turn left to center the tree, then move forward 5 blocks to aim at the ground before Dig 5 |  |
| 150 | a0 | ch1 | TurnRight | ->agent_1: Turn left to center the diamond and prepare for Dig. |  |
| 150 | a1 | ch2 | TurnLeft | ->agent_2: Dig the anvil to break it. |  |
| 150 | a2 | ch1 | TurnRight | ->agent_1: Turn left 1 block to face the anvil, then move forward. |  |
| 151 | a0 | ch1 | MoveForward | ->agent_1: Turn left to center the tree, then move forward to the diamond block to perform the cooper |  |
| 151 | a1 | ch2 | Place | ->agent_2: Turn left 1 block to face the anvil and prepare to break it with the team. |  |
| 151 | a2 | ch1 | Dig | ->agent_1: Ready to Dig the tree. |  |
| 152 | a0 | ch1 | TurnLeft | ->agent_2: Agent_2: Dig on the stone block! |  |
| 152 | a1 | ch2 | TurnRight | ->agent_2: Dig the purple anvil |  |
| 152 | a2 | ch1 | TurnRight | ->agent_1: I'm turning left to center the tree trunk, so I can dig it once I'm aligned. |  |
| 153 | a0 | ch1 | TurnRight | ->agent_2: Agent_2: I've centered the stone switch, ready to Dig. |  |
| 153 | a1 | ch2 | Place | ->agent_2: Turn left to face the anvil. |  |
| 153 | a2 | ch1 | Jump | ->agent_1: Turn left to find the exit. |  |
| 154 | a0 | ch1 | TurnRight | ->agent_1: Turn left to center the stone, then Dig. |  |
| 154 | a1 | ch2 | Place | ->agent_2: agent_2: Turn left to center the anvil and dig. |  |
| 154 | a2 | ch1 | Dig | ->agent_1: I've centered the stone switch, ready to Dig. |  |
| 155 | a0 | ch1 | TurnLeft | ->agent_1: I will Dig on the sheep to break it and drop the raw_mutton. |  |
| 155 | a1 | ch2 | TurnLeft | ->agent_2: agent_2: We are ready to Dig the anvil together. |  |
| 155 | a2 | ch1 | Place | ->agent_0: I am in Ch2 and have collected 3 items. Please check Ch1 for any stone blocks to dig. |  |
| 156 | a0 | ch1 | TurnLeft | ->agent_1: agent_1: [in ch2] I am in Ch2 and have collected 3 items. Please check Ch1 for any stone b |  |
| 156 | a1 | ch2 | TurnLeft | ->agent_2: agent_2: I'm centered the anvil and moving forward. Please turn left and move forward to c |  |
| 156 | a2 | ch1 | Dig | ->agent_2 [hebbian_fallback:self_target]: I'm breaking the stone block to trigger the switch. |  |
| 157 | a0 | ch1 | TurnRight | ->agent_1: I see a stone wall blocking my path. I'm holding a tree and can break it to get wood. I'll |  |
| 157 | a1 | ch2 | TurnRight | ->agent_2: I turned left to center the tree, now moving forward to break it. |  |
| 157 | a2 | ch1 | Dig | ->agent_1: Agent 1, please dig the stone block at your feet to collect the wood. |  |
| 158 | a0 | ch1 | TurnLeft | ->agent_1: I am ready to Dig on the centered tree for the cooperative milestone. Please check if the  |  |
| 158 | a1 | ch2 | TurnRight | ->agent_2: Dig the purple anvil to break it. |  |
| 158 | a2 | ch1 | Drop | ->agent_1: Stone block at center — Dig to collect wood. |  |
| 159 | a0 | ch1 | Dig | ->agent_1: I am facing the locked door in the north wall. Please check if you have the unlock conditi | critic:FAIL |
| 159 | a1 | ch2 | TurnLeft | ->agent_0: Dig the purple anvil, expect the diamond sword to drop at my feet. | critic:FAIL |
| 159 | a2 | ch1 | Dig | ->agent_1: Agent 1, we are now at the opening in the north wall. I will move forward to aim at the gr | critic:FAIL |
| 160 | a0 | ch1 | TurnRight | ->agent_2: agent_2: Dig the wood block. I am centered and ready for the cooperative Dig. |  |
| 160 | a1 | ch2 | Dig | ->agent_2: Agent_2: Turn right to face the zombie, then move forward to center the anvil. I will turn |  |
| 160 | a2 | ch1 | Dig | ->agent_1: I turned left to get closer to the anvil. |  |
| 161 | a0 | ch1 | Dig | ->agent_1: Agent_1, I need to turn right to center the diamond block on the sword before we can dig. |  |
| 161 | a1 | ch2 | TurnLeft | ->agent_2: Dig the purple anvil top. |  |
| 161 | a2 | ch1 | MoveForward | ->agent_2 [hebbian_fallback:self_target]: Turn right to center the anvil. I am approaching it and ready to break it. |  |
| 162 | a0 | ch1 | Dig | ->agent_1: I'm ready to Dig on the tree to collect wood. |  |
| 162 | a1 | ch2 | TurnLeft | ->agent_2: Anvil is centered. Turn left to center the anvil top, then dig to break it. |  |
| 162 | a2 | ch1 | Dig | ->agent_1: Move forward 5 blocks to aim at the tree. I'll turn left to center it and Dig 5 blocks. |  |
| 163 | a0 | ch1 | TurnRight | ->agent_1: I am aligned with the tree and ready to initiate the cooperative Dig. |  |
| 163 | a1 | ch2 | TurnRight | ->agent_2: Agent 2: Turn right to face the tree. I will move forward to center it before turning left |  |
| 163 | a2 | ch1 | Dig | ->agent_1: Turn left to center the tree, then dig 5 blocks to open Door 1. |  |
| 164 | a0 | ch1 | MoveRight | ->agent_2: I am centered on the tree and ready to dig. Please confirm the alignment of the tree for t |  |
| 164 | a1 | ch2 | TurnRight | ->agent_0: Turn left to center the tree, then move forward to align the camera for the anvil, then Di |  |
| 164 | a2 | ch1 | Dig | ->agent_1: Agent 1: I am centering the tree view for the next dig attempt. |  |
| 165 | a0 | ch1 | TurnLeft | ->agent_1: I'm aligned with the tree for Dig-on-stone. Agent_1 is digging the anvil in Ch2, so I need |  |
| 165 | a1 | ch2 | TurnRight | ->agent_0: Move forward 1 block to center the room, then turn left to face the tree. |  |
| 165 | a2 | ch1 | MoveBackward | ->agent_0: I am turning left to center the tree. Please turn left to align the tree for the next Dig. |  |
| 166 | a0 | ch1 | TurnLeft | ->agent_1: Agent_1, the anvil is centered and ready. Please proceed with the cooperative Dig. |  |
| 166 | a1 | ch2 | TurnLeft | ->agent_0: Move forward to face the anvil, then turn left to face the tree. |  |
| 166 | a2 | ch1 | Dig | ->agent_2 [hebbian_fallback:self_target]: Agent 2: Turn left to center the tree before moving forward. |  |
| 167 | a0 | ch1 | Dig | ->agent_1: I am centered and ready for the cooperative Dig. |  |
| 167 | a1 | ch2 | TurnLeft | ->agent_0: Dig the purple anvil to drop the diamond sword. |  |
| 167 | a2 | ch1 | Dig | ->agent_1: Agent 1, I see the tree centered, ready to dig for wood. |  |
| 168 | a0 | ch1 | Drop | ->agent_1: Ready to initiate the cooperative Dig on the diamond block. |  |
| 168 | a1 | ch2 | TurnLeft | ->agent_2: agent_2: Tree is centered, ready to dig for wood. I am ready to pair-dig the anvil with yo |  |
| 168 | a2 | ch1 | TurnRight | ->agent_1: I am centered on the anvil and ready to break it. Please confirm the alignment of the anvi |  |
| 169 | a0 | ch2 | Dig | ->agent_1: I am ready to perform the cooperative Dig on the tree. Please coordinate with agent_1 to b |  |
| 169 | a1 | ch2 | TurnLeft | ->agent_2: Turn left to face the tree, then dig. |  |
| 169 | a2 | ch2 | Drop | ->agent_0: Tree is centered, ready to dig. Please confirm the alignment of the tree for the cooperati |  |
| 170 | a0 | ch2 | Dig | ->agent_1: Agent_1, tree is centered for cooperative Dig. Please confirm alignment and commit to Dig. | NEW TASK: Ask Agent_1 to confirm the location of the tree, then move forward and |
| 170 | a1 | ch2 | TurnLeft | ->agent_1 [hebbian_fallback:self_target]: Turn left to look at the tree. |  |
| 170 | a2 | ch2 | MoveLeft | ->agent_0: Turn left to center the tree and prepare to dig. | NEW TASK: Look left to locate the oak tree; expect tree trunk visible to the lef |
| 171 | a0 | ch2 | Jump | ->agent_2: The anvil is centered and ready for the cooperative Dig. Please confirm the alignment and  |  |
| 171 | a1 | ch2 | TurnLeft | ->agent_0: Agent_0, tree is centered for cooperative Dig. Please confirm alignment and commit to Dig. |  |
| 171 | a2 | ch2 | TurnLeft | ->agent_1: I'm moving forward to the door and checking the left wall for the switch. |  |
| 172 | a0 | ch2 | Place | ->agent_1: Agent_1, tree is centered for cooperative Dig. Please confirm alignment and commit to Dig. |  |
| 172 | a1 | ch2 | TurnLeft | ->agent_0: Turn left to center the tree for the cooperative Dig. |  |
| 172 | a2 | ch2 | Dig | ->agent_1: agent_1: [in ch2] Tree is centered and ready for the cooperative Dig. Please confirm the a |  |
| 173 | a0 | ch2 | Jump | ->agent_1: Move forward to exit the current room. |  |
| 173 | a1 | ch2 | TurnLeft | ->agent_2: Agent_2: The switch is centered in front of you. Press it to activate the door. |  |
| 173 | a2 | ch2 | Dig | ->agent_1: Agent 1: Turn left to center the wood block, then dig. |  |
| 174 | a0 | ch2 | Place | ->agent_1: Turn left to center the diamond sword vertically. Move forward to collect it. |  |
| 174 | a1 | ch2 | TurnLeft | ->agent_2: Agent 2: Camera shows black screen with status bar. Chamber view not visible. Waiting for  |  |
| 174 | a2 | ch2 | Dig | ->agent_2 [hebbian_fallback:self_target]: Dig the stone block in front of me to clear the path. |  |
| 175 | a0 | ch2 | Place | ->agent_1: Agent_1, you are centered on the stone. I will align the stone by looking down and then pr |  |
| 175 | a1 | ? | MoveLeft | ->agent_0: Agent_0: Centered the diamond sword vertically. Ready to move forward and align for the ne |  |
| 175 | a2 | ch2 | Jump | ->agent_1: Agent 1: Look right for the oak tree; move forward to center it, then Dig to collect wood. |  |
| 176 | a0 | ch2 | Dig | ->agent_1: Agent_1, move forward to find the tree. We need to align for the cooperative Dig. | ** m_comm_ch2 (+10.0) ** |
| 176 | a1 | ? | TurnLeft | ->agent_2: Agent_2, Tree is slightly left — I will turn left to center it, then Dig. I expect to be a | ** m_comm_ch2 (+10.0) ** |
| 176 | a2 | ch2 | Jump | ->agent_1: Agent 1: You are in front of me. Press the switch to open the door. | ** m_comm_ch2 (+10.0) ** |
| 177 | a0 | ch2 | Drop | ->agent_1: Agent_1: [in Ch2] Turn left to center the tree for the cooperative Dig. | ** m_comm_ch2 (+10.0) ** |
| 177 | a1 | ? | MoveRight | ->agent_1 [hebbian_fallback:self_target]: Agent_1: Tree is centered. Turn left to face it. | ** m_comm_ch2 (+10.0) ** |
| 177 | a2 | ch2 | Dig | ->agent_1: I will turn left to locate the oak tree, then Dig to break it. I expect to see the wood dr | ** m_comm_ch2 (+10.0) ** |
| 178 | a0 | ch2 | Sneak | ->agent_1: Agent_1, turn left to center the tree for the cooperative Dig. |  |
| 178 | a1 | ? | MoveLeft | ->agent_1 [hebbian_fallback:self_target]: I see the tree centered; I will turn left to face it and then Dig. |  |
| 178 | a2 | ch2 | MoveLeft | ->agent_1: Tree is centered and ready to break. Please turn left and Dig to get wood. |  |
| 179 | a0 | ch2 | TurnLeft | ->agent_1: I'm turning left to center the tree for the cooperative Dig. | critic:FAIL |
| 179 | a1 | ch2 | MoveLeft | ->agent_2: Wolf is centered and ready to attack. Please turn left to center it and prepare for the Di | critic:FAIL |
| 179 | a2 | ? | Dig | ->agent_1: I see the tree centered; I will turn left to face it and then Dig. | critic:FAIL |
| 180 | a0 | ch2 | TurnLeft | ->agent_1: Agent_1: Tree is centered for cooperative Dig. Please confirm the location and prepare for |  |
| 180 | a1 | ch2 | TurnLeft | ->agent_2: Turn left to center the tree, then Dig. |  |
| 180 | a2 | ? | MoveForward | ->agent_1: Turn left to center the tree, then dig to break it. |  |
| 181 | a0 | ch2 | TurnRight | ->agent_1: Agent_1: Turn left to center the tree for the cooperative Dig. |  |
| 181 | a1 | ch2 | TurnLeft | ->agent_2: Agent_2, Tree is right of center. Turn right to center it, then move forward to break. |  |
| 181 | a2 | ch2 | MoveRight | ->agent_1: Turn left to center the tree, then move forward to dig. |  |
| 182 | a0 | ch2 | TurnRight | ->agent_1: Agent_1: I am ready to center the tree and trigger the cooperative Dig. Please confirm the |  |
| 182 | a1 | ch2 | MoveLeft | ->agent_2: Turn right to center the tree, then move forward to dig. |  |
| 182 | a2 | ch2 | Dig | ->agent_1: Agent 1: I am blocked by a stone block; I need to move left to reach the tree and break it |  |
| 183 | a0 | ch2 | TurnLeft | ->agent_1: Agent_1, confirm the tree's location and request to center it so I can proceed with the co |  |
| 183 | a1 | ch2 | Drop | ->agent_2: Agent_2: Turn left to center the tree, then move forward or dig. I am ready to attack. |  |
| 183 | a2 | ch2 | Dig | ->agent_1: Turn left to center the tree, then move forward to dig. |  |
| 184 | a0 | ch2 | MoveRight | ->agent_1: Agent_1: Centered the diamond sword vertically. Ready to move forward and align for the ne |  |
| 184 | a1 | ch2 | TurnLeft | ->agent_0: Move forward 1 block to center the tree, then turn left to face it. |  |
| 184 | a2 | ch2 | MoveLeft | ->agent_1: Turn right to center the tree, then move forward or dig. I am ready to attack. |  |
| 185 | a0 | ch2 | NoOp | ->agent_1: Agent_1, I see the diamond sword centered on the stone block. Confirm the location of the  |  |
| 185 | a1 | ch2 | TurnLeft | ->agent_2: Turn left to center the tree, then move forward. |  |
| 185 | a2 | ? | TurnRight | ->agent_1: Agent 1: I'm ready to break the tree. Turn left and dig. |  |
| 186 | a0 | ? | NoOp | ->agent_1: Centered the diamond sword vertically. Ready to move forward and align for the next step. |  |
| 186 | a1 | ? | MoveForward | ->agent_0: I see the diamond sword centered on the stone block. Confirm the location of the tree and  |  |
| 186 | a2 | ch2 | Jump | ->agent_1: I am centered on the tree and ready to break it. |  |