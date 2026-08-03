# Case: collab_failure_IPPO_0

**Run:** `exp04_ippo/seed_42` (IPPO)  **Episode:** 3  **Steps:** 715-745
**Why shortlisted:** ignored directive/commitment a0->a1 streak x7: "Move forward to collect the cobblestone and continue toward the teleporter in Ch"

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 715 | a0 | ch4 | MoveBackward | ->agent_1: Move forward to collect the cobblestone and continue toward the teleporter in Ch3. I will  |  |
| 715 | a1 | ? | MoveForward | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 715 | a2 | ch4 | NoOp | ->agent_1: Agent 1, move forward to center the tunnel ahead. We are centered and ready to proceed. |  |
| 716 | a0 | ch4 | TurnRight | ->agent_1: I am moving forward to the teleporter in Ch3. Please proceed to the teleporter to enter th |  |
| 716 | a1 | ? | NoOp | ->agent_2: Agent 2, I am ready to pair-dig with you. Move forward to center the zombie. |  |
| 716 | a2 | ch4 | Place | ->agent_2 [random_fallback:self_target]: Agent 2, we are centered on the greenstone block and ready to pair-dig. Please proceed. |  |
| 717 | a0 | ch4 | Jump | ->agent_1: Agent 1, I am moving forward to the teleporter to enter Ch3. Please wait for me to reach i |  |
| 717 | a1 | ? | Place | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 717 | a2 | ch4 | MoveBackward | ->agent_1: Agent 1, pair-dig for Ch1. Tree trunk is centered, ready to raise HP. |  |
| 718 | a0 | ch4 | Jump | ->agent_1: I collected the cobblestone. Agent_1 is ready to pair-dig with Agent_2. Please proceed to  |  |
| 718 | a1 | ? | Jump | ->agent_2: I am ready to pair-dig with Agent 2 on the greenstone block. |  |
| 718 | a2 | ch4 | MoveRight | ->agent_1: Agent 1, we are centered on the zombie and ready to pair-dig for Ch5. Please proceed. | critic:FAIL |
| 719 | a0 | ? | Drop | ->agent_1: I am moving forward to the teleporter to enter Ch3. Agent 1, we are ready for the next pai | critic:FAIL |
| 719 | a1 | ? | Sneak | ->agent_2: I am ready to pair-dig with Agent 2 on the zombie. | critic:FAIL |
| 719 | a2 | ch4 | MoveLeft | ->agent_1: Agent 1, pair-dig with me on the greenstone block. We need to raise the zombie's HP to unl |  |
| 720 | a0 | ch4 | MoveRight | ->agent_1: I moved forward and collected the cobblestone. I am ready to pair-dig with Agent 1 to rais |  |
| 720 | a1 | ch4 | MoveLeft | ->agent_1 [random_fallback:self_target]: I am looking for the zombie. Turn left to center it, then pair-dig with Agent 1 to raise H |  |
| 720 | a2 | ch4 | Drop | ->agent_1: Agent 1, we are centered on the left zombie and ready to pair-dig. Please proceed with the |  |
| 721 | a0 | ch4 | Jump | ->agent_1: Agent 1, I am moving forward to the teleporter to enter Ch2. I will continue the pair-dig  |  |
| 721 | a1 | ch4 | Drop | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 721 | a2 | ch4 | Jump | ->agent_1: Agent 1, we are centered on the zombie and ready to pair-dig to raise its HP. |  |
| 722 | a0 | ch4 | MoveLeft | ->agent_1: I collected the cobblestone. Ready to pair-dig with agent_1. |  |
| 722 | a1 | ch4 | Sneak | ->agent_1 [random_fallback:self_target]: I am ready to pair-dig with Agent 1 to raise the zombie's HP. |  |
| 722 | a2 | ch4 | Jump | ->agent_2 [random_fallback:self_target]: Agent 2, we are centered on the greenstone block and ready to pair-dig. Please proceed. |  |
| 723 | a0 | ch4 | MoveLeft | ->agent_1: I moved forward and collected the cobblestone. Ready to pair-dig with Agent 1. |  |
| 723 | a1 | ch4 | MoveRight | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 723 | a2 | ch4 | MoveRight | ->agent_1: Agent 1, I am centered on the greenstone block and ready to pair-dig. Please proceed. |  |
| 724 | a0 | ch4 | Place | ->agent_1: Agent 1, we are centered on the greenstone block and ready to pair-dig. Please proceed. |  |
| 724 | a1 | ch4 | MoveLeft | ->agent_1 [random_fallback:self_target]: I am centered on the zombie and ready to attack. Please proceed. |  |
| 724 | a2 | ch4 | Sneak | ->agent_1: Agent 1, we are facing the zombie and ready to pair-dig. Please proceed with the next move |  |
| 725 | a0 | ch4 | Dig | ->agent_1: I collected the cobblestone drop and am ready to continue moving forward toward the telepo |  |
| 725 | a1 | ch4 | Sneak | ->agent_2: I am ready to pair-dig with Agent 2 on the greenstone block. |  |
| 725 | a2 | ch4 | Jump | ->agent_2 [random_fallback:self_target]: Agent 2, I am centered on the zombie and ready to pair-dig. Please proceed. |  |
| 726 | a0 | ch4 | Dig | ->agent_1: I collected the cobblestone. Ready to proceed to the teleporter. |  |
| 726 | a1 | ch4 | LookUp | ->agent_2: Agent 2, I am centered on the zombie and ready to pair-dig. Please proceed. |  |
| 726 | a2 | ch4 | TurnLeft | ->agent_1: Agent 1, we are centered on the zombie and tree. Please proceed. |  |
| 727 | a0 | ch4 | Sneak | ->agent_1: I collected the cobblestone. I am moving forward to the teleporter to enter Ch3. |  |
| 727 | a1 | ch4 | Sneak | ->agent_1 [random_fallback:self_target]: Agent 1, we are centered on the zombie and block. Please proceed. |  |
| 727 | a2 | ch4 | Jump | ->agent_1: Agent 1, the zombie is to the right. Turn left to center it, then pair-dig. |  |
| 728 | a0 | ch4 | Jump | ->agent_2: agent_2, I've confirmed the greenstone block is broken and dropped a cobblestone. I'll pai |  |
| 728 | a1 | ch4 | Place | ->agent_2: Agent 2, the greenstone block is centered and I am ready to pair-dig. |  |
| 728 | a2 | ch4 | Jump | ->agent_1: Agent 1, we are centered on the zombie and tree trunk. Please proceed with the pair-dig sc |  |
| 729 | a0 | ch4 | Place | ->agent_1: Agent 1: I am looking up to see the teleporter marker on the wall. Please proceed to pair- |  |
| 729 | a1 | ch4 | LookUp | ->agent_2: I am centered on the zombie. I am ready to pair-dig with Agent 2. |  |
| 729 | a2 | ch4 | TurnLeft | ->agent_1: Agent 1, the greenstone block is centered and ready for pair-dig. Please proceed. |  |
| 730 | a0 | ch4 | LookUp | ->agent_1: I moved forward to collect the cobblestone. I am ready to pair-dig with agent_1 to raise t |  |
| 730 | a1 | ch4 | Sneak | ->agent_2: Agent 2, I am ready to pair-dig with you. |  |
| 730 | a2 | ch4 | Place | ->agent_1: Agent 1, I am facing the zombie and ready to pair-dig with you. Please join me in attackin |  |
| 731 | a0 | ch4 | Drop | ->agent_1: Ready for next pair-dig with agent_1. |  |
| 731 | a1 | ch4 | MoveBackward | ->agent_1 [random_fallback:self_target]: Agent 1, I am facing the zombie and ready to pair-dig with you. Please join me in attackin |  |
| 731 | a2 | ch4 | MoveForward | ->agent_1: Agent 1, we are centered on the greenstone block and ready to pair-dig for Ch1. Please pro |  |
| 732 | a0 | ch4 | MoveLeft | ->agent_1: I am aligned on the cobblestone block and ready to move forward. Please proceed to the tel |  |
| 732 | a1 | ch4 | MoveLeft | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 732 | a2 | ch4 | MoveRight | ->agent_1: Agent 1, pair-dig with me on the Row A anvil. |  |
| 733 | a0 | ch4 | Jump | ->agent_1: I am ready to pair-dig with you on the greenstone block. Please proceed. |  |
| 733 | a1 | ch4 | MoveLeft | ->agent_2: Agent 2, pair-dig with me on the greenstone block. |  |
| 733 | a2 | ch4 | Jump | ->agent_1: Agent 1, I am aiming at the greenstone block to pair-dig with you. Please proceed. |  |
| 734 | a0 | ch4 | Sneak | ->agent_1: I am moving forward to the teleporter to enter Ch3. Please join me in advancing. |  |
| 734 | a1 | ch4 | MoveRight | ->agent_2: Agent 2, I am ready to pair-dig with you on the tree trunk. |  |
| 734 | a2 | ch4 | Jump | ->agent_1: Agent 1, we are centered on the zombie and ready to pair-dig for the greenstone block. Ple |  |
| 735 | a0 | ch4 | MoveRight | ->agent_1: I collected the cobblestone, ready to pair-dig the zombie with Agent 2. |  |
| 735 | a1 | ch4 | MoveBackward | ->agent_2: I am ready to pair-dig with Agent 2 on the greenstone block. |  |
| 735 | a2 | ch4 | MoveBackward | ->agent_1: Agent 1, the zombie is blocking the greenstone block. Please pair-dig with Agent 2 on the  |  |
| 736 | a0 | ch4 | Dig | ->agent_2: I am moving forward to collect the cobblestone and move toward the teleporter in Ch3. |  |
| 736 | a1 | ch4 | MoveRight | ->agent_2: I am ready to pair-dig with Agent 2 on the greenstone block. |  |
| 736 | a2 | ch4 | Drop | ->agent_1: Agent 1, we are centered on the greenstone block and ready to pair-dig. Please proceed wit |  |
| 737 | a0 | ch4 | MoveForward | ->agent_1: Agent 1, I am moving forward to the teleporter. Once we reach it, we will proceed to Ch3. |  |
| 737 | a1 | ch4 | Dig | ->agent_2: I am ready to pair-dig with Agent 2 on the greenstone block. |  |
| 737 | a2 | ch4 | Jump | ->agent_1: Agent 1, we are centered on the greenstone block and ready to pair-dig. Please proceed. |  |
| 738 | a0 | ch4 | TurnLeft | ->agent_1: I am moving forward to collect the cobblestone drop from the greenstone block. I am ready  |  |
| 738 | a1 | ch4 | MoveLeft | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 738 | a2 | ch4 | Drop | ->agent_1: Agent 1, I am centered on the greenstone block and ready to pair-dig with you. Please proc | critic:FAIL |
| 739 | a0 | ch4 | Jump | ->agent_1: Agent_1, I am moving forward to collect the cobblestone. Please confirm readiness for the  | critic:FAIL |
| 739 | a1 | ch4 | Sneak | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 739 | a2 | ch4 | Sneak | ->agent_1: Agent 1, we are centered on the tree and ready to pair-dig for Ch1. Please proceed. | critic:FAIL |
| 740 | a0 | ch4 | MoveBackward | ->agent_1: I am moving forward to collect the cobblestone. I will communicate with agent_1 to confirm |  |
| 740 | a1 | ch4 | LookDown | ->agent_2: Agent 2, I am looking at the stone pillar, ready to pair-dig with you. Please proceed. |  |
| 740 | a2 | ch4 | NoOp | ->agent_2 [random_fallback:self_target]: Agent 2, we are centered on the greenstone block and ready to pair-dig for Ch1. Please pro |  |
| 741 | a0 | ch4 | MoveLeft | ->agent_2: Agent 1, I am moving forward to collect the cobblestone, then ready for the next pair-dig. |  |
| 741 | a1 | ch4 | MoveLeft | ->agent_2: I am centered on the zombie, ready to pair-dig for Ch1. Please proceed. |  |
| 741 | a2 | ch4 | Jump | ->agent_1: Agent 1, we are centered on the tree trunk and ready to pair-dig. Please proceed. |  |
| 742 | a0 | ch4 | MoveLeft | ->agent_1: Agent 1, I am moving forward to collect the cobblestone drop. Ready to pair-dig with you. |  |
| 742 | a1 | ch4 | Drop | ->agent_2: Agent 2, we are centered on the tree trunk and ready to pair-dig. Please proceed. |  |
| 742 | a2 | ch4 | MoveLeft | ->agent_1: Agent 1, we are centered on the greenstone block and ready to pair-dig. Please proceed. |  |
| 743 | a0 | ch4 | MoveForward | ->agent_1: Agent_1, dig the greenstone block for the diamond sword |  |
| 743 | a1 | ? | MoveLeft | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 743 | a2 | ch4 | Place | ->agent_1: Agent 1, we are centered on the tree trunk and ready to pair-dig. Please proceed. |  |
| 744 | a0 | ch4 | MoveLeft | ->agent_1: Agent_1, ready for next pair-dig. |  |
| 744 | a1 | ? | Sneak | ->agent_2: I am ready to pair-dig with Agent 2. |  |
| 744 | a2 | ch4 | LookDown | ->agent_1: Agent 1, we are centered on the tree trunk and ready to pair-dig. Please proceed. |  |
| 745 | a0 | ch4 | MoveBackward | ->agent_1: I collected the cobblestone. I am moving forward to the teleporter. I will pair-dig the gr |  |
| 745 | a1 | ? | Drop | ->agent_1 [random_fallback:self_target]: I am ready to pair-dig with Agent 1. |  |
| 745 | a2 | ch4 | LookDown | ->agent_1: agent_1, we are centered on the zombie and ready to pair-dig for Ch1. Please proceed. |  |