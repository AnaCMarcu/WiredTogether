# Case: rl_action_collapse_1

**Run:** `exp04_ippo/seed_1011` (IPPO)  **Episode:** 2  **Steps:** 0-40
**Why shortlisted:** RL emits NoOp 1.0 of window

| t | agent | chamber | action | message | events/notes |
|---|---|---|---|---|---|
| 0 | a0 | ch1 | NoOp |  | NEW TASK: Move forward 1 block to a teammate; expect to see a second agent stand; ** m1_move_5 (+10) ** |
| 0 | a1 | ch1 | NoOp |  | NEW TASK: Attack the Boss; expect to take 1 dmg/hit and see its HP decrease.; ** m1_move_5 (+10) ** |
| 0 | a2 | ch1 | Dig | ->agent_0: agent_0: Check on boss health, we need to keep hitting it. | NEW TASK: Turn until the purple anvil is centered in the view, then move forward; ** m1_move_5 (+10) ** |
| 1 | a0 | ch1 | NoOp |  | NEW TASK: Move forward 10 blocks to reach the center of the room; expect to see  |
| 1 | a1 | ch1 | NoOp |  | NEW TASK: Move left to find a wood source and dig 5 wood; expect to see logs dro |
| 1 | a2 | ch1 | MoveForward | ->agent_0: Dig the red block on the door to unlock Door 1. | NEW TASK: Turn to face the purple anvil, then move forward and hover over it; ex |
| 2 | a0 | ch1 | NoOp |  |  |
| 2 | a1 | ch1 | NoOp |  |  |
| 2 | a2 | ch1 | NoOp |  |  |
| 3 | a0 | ch1 | NoOp |  |  |
| 3 | a1 | ch1 | NoOp |  |  |
| 3 | a2 | ch1 | NoOp |  |  |
| 4 | a0 | ch1 | NoOp |  |  |
| 4 | a1 | ch1 | NoOp |  |  |
| 4 | a2 | ch1 | NoOp |  |  |
| 5 | a0 | ch1 | NoOp |  |  |
| 5 | a1 | ch1 | NoOp |  |  |
| 5 | a2 | ch1 | NoOp |  |  |
| 6 | a0 | ch1 | NoOp |  |  |
| 6 | a1 | ch1 | NoOp |  |  |
| 6 | a2 | ch1 | NoOp |  |  |
| 7 | a0 | ch1 | NoOp |  |  |
| 7 | a1 | ch1 | NoOp |  |  |
| 7 | a2 | ch1 | NoOp |  |  |
| 8 | a0 | ch1 | NoOp |  |  |
| 8 | a1 | ch1 | NoOp |  |  |
| 8 | a2 | ch1 | NoOp |  |  |
| 9 | a0 | ch1 | NoOp |  |  |
| 9 | a1 | ch1 | NoOp |  |  |
| 9 | a2 | ch1 | NoOp |  |  |
| 10 | a0 | ch1 | NoOp |  |  |
| 10 | a1 | ch1 | NoOp |  |  |
| 10 | a2 | ch1 | NoOp |  |  |
| 11 | a0 | ch1 | NoOp |  |  |
| 11 | a1 | ch1 | NoOp |  |  |
| 11 | a2 | ch1 | NoOp |  |  |
| 12 | a0 | ch1 | NoOp |  |  |
| 12 | a1 | ch1 | NoOp |  |  |
| 12 | a2 | ch1 | NoOp |  |  |
| 13 | a0 | ch1 | NoOp |  |  |
| 13 | a1 | ch1 | NoOp |  |  |
| 13 | a2 | ch1 | NoOp |  |  |
| 14 | a0 | ch1 | NoOp |  |  |
| 14 | a1 | ch1 | NoOp |  |  |
| 14 | a2 | ch1 | NoOp |  |  |
| 15 | a0 | ch1 | NoOp |  |  |
| 15 | a1 | ch1 | NoOp |  |  |
| 15 | a2 | ch1 | NoOp |  |  |
| 16 | a0 | ch1 | NoOp |  | ** m1_move_5 (+10) ** |
| 16 | a1 | ch1 | NoOp |  | ** m1_move_5 (+10) ** |
| 16 | a2 | ch1 | NoOp |  | ** m1_move_5 (+10) ** |
| 17 | a0 | ch1 | NoOp |  | critic:FAIL |
| 17 | a1 | ch1 | NoOp |  | critic:FAIL |
| 17 | a2 | ch1 | NoOp |  | NEW TASK: Move forward and turn right; expect to enter the chamber containing th; critic:OK |
| 18 | a0 | ch1 | NoOp |  |  |
| 18 | a1 | ch1 | NoOp |  |  |
| 18 | a2 | ch1 | NoOp |  | NEW TASK: Dig 5 wood in the center of the current room to find potential gear or |
| 19 | a0 | ch1 | NoOp |  |  |
| 19 | a1 | ch1 | NoOp |  |  |
| 19 | a2 | ch1 | NoOp |  | NEW TASK: Move forward and turn to face the red door; expect to see a red locked |
| 20 | a0 | ch1 | NoOp |  |  |
| 20 | a1 | ch1 | NoOp |  |  |
| 20 | a2 | ch1 | NoOp |  | NEW TASK: Move forward and turn to face the red door; expect to enter the next c |
| 21 | a0 | ch1 | NoOp |  |  |
| 21 | a1 | ch1 | NoOp |  |  |
| 21 | a2 | ch1 | NoOp |  | NEW TASK: Press interaction key (e.g., 'I' or 'L') on the red locked door to bre |
| 22 | a0 | ch1 | NoOp |  |  |
| 22 | a1 | ch1 | NoOp |  |  |
| 22 | a2 | ch1 | NoOp |  | NEW TASK: Press interaction key (e.g., 'I' or 'L') on the red locked door to ope |
| 23 | a0 | ch1 | NoOp |  |  |
| 23 | a1 | ch1 | NoOp |  |  |
| 23 | a2 | ch1 | NoOp |  | NEW TASK: Move forward into the next chamber to reach the communal room; expect  |
| 24 | a0 | ch1 | NoOp |  |  |
| 24 | a1 | ch1 | NoOp |  |  |
| 24 | a2 | ch1 | NoOp |  | NEW TASK: Move forward and turn to survey the area ahead; expect to see the entr |
| 25 | a0 | ch1 | NoOp |  |  |
| 25 | a1 | ch1 | NoOp |  |  |
| 25 | a2 | ch1 | NoOp |  | NEW TASK: Move forward to enter Ch2; expect to see the purple anvils and the ent |
| 26 | a0 | ch1 | NoOp |  |  |
| 26 | a1 | ch1 | NoOp |  |  |
| 26 | a2 | ch1 | NoOp |  | NEW TASK: Press interaction key on the red locked door; expect the red door to o |
| 27 | a0 | ch1 | NoOp |  |  |
| 27 | a1 | ch1 | NoOp |  |  |
| 27 | a2 | ch1 | NoOp |  | NEW TASK: Survey the area ahead to locate the two purple anvils and the entrance |
| 28 | a0 | ch1 | NoOp |  |  |
| 28 | a1 | ch1 | NoOp |  |  |
| 28 | a2 | ch1 | NoOp |  | NEW TASK: Move forward and turn to face the purple anvil; expect another agent ( |
| 29 | a0 | ch1 | NoOp |  |  |
| 29 | a1 | ch1 | NoOp |  |  |
| 29 | a2 | ch1 | NoOp |  | NEW TASK: Dig the wood block ahead of the red door to continue progress toward M |
| 30 | a0 | ch1 | NoOp |  |  |
| 30 | a1 | ch1 | NoOp |  |  |
| 30 | a2 | ch1 | NoOp |  | NEW TASK: Dig on the red locked door ahead to break the lock and enter Ch2; expe |
| 31 | a0 | ch1 | NoOp |  |  |
| 31 | a1 | ch1 | NoOp |  |  |
| 31 | a2 | ch1 | NoOp |  | NEW TASK: Dig the red door block ahead; expect the red door to be broken, the lo |
| 32 | a0 | ch1 | NoOp |  |  |
| 32 | a1 | ch1 | NoOp |  |  |
| 32 | a2 | ch1 | NoOp |  | NEW TASK: Move forward to find the next wood block and check for an animal to ki |
| 33 | a0 | ch1 | NoOp |  |  |
| 33 | a1 | ch1 | NoOp |  |  |
| 33 | a2 | ch1 | NoOp |  | NEW TASK: Move forward to the area with the two purple anvils; expect to see two |
| 34 | a0 | ch1 | NoOp |  |  |
| 34 | a1 | ch1 | NoOp |  |  |
| 34 | a2 | ch1 | NoOp |  | NEW TASK: Press interaction key (e.g., 'I' or 'L') on the red locked door; expec |
| 35 | a0 | ch1 | NoOp |  |  |
| 35 | a1 | ch1 | NoOp |  |  |
| 35 | a2 | ch1 | NoOp |  | NEW TASK: Move forward to enter Ch3; expect to see the entrance to Ch3 with the  |
| 36 | a0 | ch1 | NoOp |  |  |
| 36 | a1 | ch1 | NoOp |  |  |
| 36 | a2 | ch1 | NoOp |  | NEW TASK: Dig on the red locked door block in Ch1 to break the lock and enter Ch |
| 37 | a0 | ch1 | NoOp |  | critic:FAIL |
| 37 | a1 | ch1 | NoOp |  | critic:FAIL |
| 37 | a2 | ch1 | NoOp |  | critic:FAIL |
| 38 | a0 | ch1 | NoOp |  |  |
| 38 | a1 | ch1 | NoOp |  |  |
| 38 | a2 | ch1 | NoOp |  |  |
| 39 | a0 | ch1 | NoOp |  |  |
| 39 | a1 | ch1 | NoOp |  |  |
| 39 | a2 | ch1 | NoOp |  |  |
| 40 | a0 | ch1 | NoOp |  |  |
| 40 | a1 | ch1 | NoOp |  |  |
| 40 | a2 | ch1 | NoOp |  |  |