# WiredTogether System Architecture

```
                            SLURM / CLI Entry Point
                        scripts/mindforge_slurm.sh
                                    |
                        LLM_MODEL_PATH or LLM_BASE_URL
                                    |
                                    v
    ================================================================
    |           MAIN LOOP  (multi_agent_craftium.py)                |
    |                                                              |
    |  parse_args() --> load_prompts() --> build_agents()           |
    |                                                              |
    |  for episode in episodes:                                    |
    |    environment.reset()                                       |
    |    for step in max_steps:                                    |
    |      for agent_id in agents:                                 |
    |        agent_do_action()  ----+                              |
    |      metric.store_timestep() |                               |
    |      [social_graph.update()] | <-- PROPOSED (Phase 1)        |
    |    metric.save_run_metrics() |                               |
    ================================================================
                                   |
            +----------------------+----------------------+
            |                      |                      |
            v                      v                      v
    +--------------+    +------------------+    +------------------+
    | ENVIRONMENT  |    |  AGENT (x N)     |    |    METRICS       |
    +--------------+    |  custom_agent.py  |    |  craftium_metric |
    |              |    +------------------+    +------------------+
    | Craftium     |    |                  |    | - rewards/agent  |
    | Environment  |    |  Decision Flow:  |    | - milestones     |
    | Interface    |    |                  |    | - comm counts    |
    |              |    |  1. Critic       |    | - RL loss/entropy|
    | Wraps:       |    |  2. Curriculum   |    | - [social graph] |
    | OpenWorld    |    |  3. Memory       |    |     snapshots    |
    | MultiAgent   |    |  4. Beliefs      |    |                  |
    | Env          |    |  5. Action       |    | Outputs:         |
    | (PettingZoo) |    |     Selection    |    |  JSON + plots    |
    |              |    |                  |    |  + summary.txt   |
    | Wraps:       |    +--------+---------+    +------------------+
    | _Patched     |             |
    | MarlCraftium |             |
    | Env          |    +--------+---------+
    |              |    |                  |
    | Methods:     |    v                  v
    | reset()      |  LLM Path         RL Path (optional)
    | step()       |  (default)        (--rl flag)
    | get_image()  |    |                  |
    | get_reward() |    v                  v
    | get_positions|  +--------+    +------------------+
    +--------------+  | LLM    |    |   RL LAYER       |
                      | Call   |    |   rl_layer/      |
                      +--------+    +------------------+
                      |        |    |                  |
                      v        v    | LoRA-adapted LLM |
              +---------+ +------+  | + ActionHead(17) |
              | Local   | | HTTP |  | + ValueHead(1)   |
              | Model   | | API  |  |                  |
              | Client  | |(vLLM/|  | MAPPO Optimizer  |
              | (trans-  | |Open- |  | (PPO updates)   |
              | formers)| |Router|  |                  |
              +---------+ +------+  | TrajectoryBuffer |
                                    | (GAE computation)|
                                    +------------------+


    AGENT MODULES DETAIL (agent_modules/)
    ======================================

    +-------------------+     +------------------+     +------------------+
    |  AutoCurriculum   |     |   SkillManager   |     | EpisodicMemory   |
    |  (task proposer)  |     |  (skill library)  |     |   Manager        |
    +-------------------+     +------------------+     +------------------+
    | - completed_tasks |     | - ChromaDB store |     | - ChromaDB store |
    | - failed_tasks    |     | - sentence-      |     | - episode history|
    | - generates new   |     |   transformers   |     | - semantic search|
    |   tasks via LLM   |     |   embeddings     |     | - LLM summary   |
    +-------------------+     | - semantic search|     +------------------+
                              +------------------+

    +-------------------+     +------------------+     +------------------+
    |     Critic        |     |   BeliefSystem   |     | ActionSelection  |
    |  (task evaluator) |     |  (4 belief types)|     |  (LLM action)    |
    +-------------------+     +------------------+     +------------------+
    | - checks success/ |     | - perception     |     | - builds prompt  |
    |   failure via LLM |     | - partner        |     |   from beliefs,  |
    | - returns bool +  |     | - interaction    |     |   memories, task |
    |   critique text   |     | - task           |     | - LLM call ->    |
    +-------------------+     | - updated via LLM|     |   {action,       |
                              +------------------+     |    thoughts,     |
                                                       |    communication}|
                                                       +------------------+


    PROPOSED: SOCIAL PLASTICITY MODULE
    ====================================

    Placement: AFTER metric.store_timestep(), OUTSIDE per-agent loop.
    Reads data only (Phase 1: log_only mode).

    +-------------------------------------------------------------+
    |                                                             |
    |  social_plasticity/                                         |
    |                                                             |
    |  +------------------+  +----------------+  +--------------+ |
    |  | SocialPlasticity |  |  SocialGraph   |  | social_      | |
    |  | Config           |  |                |  | metrics.py   | |
    |  +------------------+  +----------------+  +--------------+ |
    |  | eta    = 0.01    |  | W(t) [N x N]   |  | mean_weight  | |
    |  | decay  = 0.001   |  |                |  | sparsity     | |
    |  | beta   = 1.0     |  | update():      |  | reciprocity  | |
    |  | prox_d = 10.0    |  |  1. m_i = tanh |  | intra/inter  | |
    |  | w_init = 0.1     |  |     (beta*r_i) |  |   role mean  | |
    |  | mode   = log_only|  |  2. g_i = has  |  | degree       | |
    |  +------------------+  |     comm/reward |  | asymmetry    | |
    |                        |  3. c_ij =     |  +--------------+ |
    |                        |     proximity  |                   |
    |                        |     * g_i * g_j|                   |
    |                        |  4. dw_ij =    |                   |
    |                        |     eta*m*c*   |                   |
    |                        |     (1-w) -    |                   |
    |                        |     decay*w    |                   |
    |                        |  5. clip [0,1] |                   |
    |                        |                |                   |
    |                        | reset()        |                   |
    |                        | snapshot()     |                   |
    |                        +----------------+                   |
    |                                                             |
    +-------------------------------------------------------------+
                    |                           |
        Reads from:                   Writes to:
        - positions (env)              - metric.record_social_graph()
        - rewards (env)                - JSON snapshots
        - communications (agents)      - bond_strength.png
        - advantages (RL, optional)    - summary.txt stats


    DATA FLOW (per step)
    =====================

    Environment                  Agent i                    Metrics
    ----------                  -------                    -------
        |                          |                          |
        |--- frame (PIL image) --->|                          |
        |--- reward summary ------>|                          |
        |                          |                          |
        |                    1. Critic: success?              |
        |                    2. Curriculum: new task?          |
        |                    3. Memory: query skills/episodes  |
        |                    4. Beliefs: update all 4 types    |
        |                    5. Action: LLM or RL path         |
        |                          |                          |
        |<-- action (discrete) ----|                          |
        |                          |--- communication ------->|
        |--- step reward --------->|                          |
        |                          |--- reward ------------->>|
        |                          |                          |
        |                    [If RL: store in buffer]          |
        |                    [If RL & buffer full: MAPPO update]
        |                          |                          |
        +--- positions ---------->>+--- [social_graph.update()]
                                                              |
                                                    store_timestep()
                                                    [record_social()]
```
