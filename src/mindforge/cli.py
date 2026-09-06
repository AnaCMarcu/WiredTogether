"""Command-line interface for a WIRE run.

Every knob of the system is set here: environment and episode shape, the
LLM backbone, the RL layer, the Hebbian graph and its couplings, the
orchestrator baselines, and the logging/checkpointing behaviour. Defaults
reproduce the paper's zero-shot LLM configuration; the launchers under
``hpc/`` override them per experiment.
"""

from __future__ import annotations

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mindforge agents in Craftium OpenWorld")
    parser.add_argument("--num-agents", type=int, default=3,
                        help="Number of agents in WIRE (all share the agent role)")
    parser.add_argument("--team-scaling", action="store_true",
                        help="Master switch for the agent-count scaling suite. "
                             "ON: prompt text is rendered truthfully for the "
                             "actual --num-agents (team size, cell letters, "
                             "switch ring) and Lua uses the collision-free "
                             "generic Ch1 spawn row (WT_TEAM_SCALING=1 is "
                             "exported for the Lua side). OFF (default): every "
                             "prompt renders the historical 3-agent wording "
                             "byte-identically and the env behaves exactly as "
                             "all pre-scaling suites — leave this off for "
                             "legacy/medium/cofiring/transplant runs.")
    parser.add_argument("--ch4-mob-count", type=int, default=None,
                        help="Pin the Ch4 zombie count to this value regardless of "
                             "--num-agents (exported as FC_CH4_MOB_COUNT to the Lua "
                             "server AND used for the prompt text, so agents are "
                             "told the true count). Default: unset — legacy "
                             "one-zombie-per-agent, min(num_agents, 6). The "
                             "agent-count scaling suite pins 3 so the environment "
                             "is identical for every team size.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="Maximum steps per episode (default 1500 — fits the "
                             "DAIC 36h SLURM budget). Each chamber timeout fires at "
                             "20%% of this budget (Ch1->Ch2 at step ~300 of 1500), "
                             "so the five chambers get a 20%% window apiece. "
                             "Override with a larger value (e.g. 2500) when "
                             "running on qos=long / --time=72:00:00 to give "
                             "agents more headroom for organic Ch2-Ch3 "
                             "coordination.")
    parser.add_argument("--obs-width", type=int, default=320,
                        help="Observation width in pixels")
    parser.add_argument("--obs-height", type=int, default=180,
                        help="Observation height in pixels")
    parser.add_argument("--no-communication", action="store_true",
                        help="Disable inter-agent communication entirely.")
    parser.add_argument("--simultaneous", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Simultaneous-move stepping (DEFAULT ON): all agents "
                             "choose actions concurrently on the shared state s_t "
                             "and the env advances once via step_all(). Pass "
                             "--no-simultaneous for the legacy turn-based "
                             "round-robin (e.g. parity testing). Works with both "
                             "LLM and --rl agents (macro actions were removed).")
    parser.add_argument("--sleep-time", type=float, default=0.0,
                        help="Seconds to sleep between LLM calls (rate-limit protection)")
    parser.add_argument("--belief-interval", type=int, default=5,
                        help="Refresh beliefs every N steps (default 5). Between refreshes "
                             "cached beliefs are reused, saving 4 LLM calls per skipped step.")
    parser.add_argument("--critic-interval", type=int, default=20,
                        help="Run critic every N steps (default 20). Between evaluations "
                             "cached success/critique are reused, saving 1 LLM call per skipped step.")
    parser.add_argument("--no-gif", action="store_true",
                        help="Disable GIF saving")
    parser.add_argument("--gif-dir", type=str, default="auto",
                        help="Directory to save GIFs. Default 'auto' resolves "
                             "to <run_dir>/gifs/ so each run's media stays "
                             "bundled with its other artifacts. Pass an "
                             "explicit path (e.g. /scratch/$USER/gifs) to "
                             "override.")
    parser.add_argument("--gif-interval", type=int, default=300,
                        help="Save a checkpoint GIF every N steps (default 300). 0 = only save at episode end. "
                             "Raised from 100 after exp3_mappo crashed mid-ep3 (job 12616286): the GIF+MP4 dump "
                             "every 100 steps × 3 agents × 320×180 frames spiked memory enough to OOM-kill one of "
                             "the luanti client processes via SLURM cgroup. 300 cuts the dump rate by 3× and "
                             "leaves more headroom in the per-job memory cap.")
    parser.add_argument("--warmup-time", type=int, default=60,
                        help="Minimum seconds before checking if media loaded (default 60). "
                             "Smart detection exits early once all clients show game world.")
    parser.add_argument("--ch1-timeout-steps", type=int, default=400,
                        help="Lua-side Ch1-timeout fallback budget, in env steps. "
                             "The Python primary now fires unconditionally at "
                             "20%% of --max-steps (each chamber gets 20%% of the "
                             "episode; Ch1→Ch2, Ch2→Ch3, Ch3→Ch4, Ch4→Ch5 are "
                             "all on the same 20%% timer). "
                             "This flag only sizes the Lua-side backstop in case "
                             "Python's force-flag never reaches the world (mod "
                             "I/O error, etc.). Default 400 → 60000 Lua ticks.")
    # ── Weights & Biases ──
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging. Requires WANDB_API_KEY "
                             "in the environment. Failures during init/log are "
                             "tolerated and do not kill training.")
    parser.add_argument("--wandb-project", type=str, default="wired-together",
                        help="W&B project name (default 'wired-together').")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team or user). Defaults to your "
                             "wandb-configured default entity.")
    parser.add_argument("--wandb-tags", type=str, default="",
                        help="Comma-separated list of tags applied to the W&B "
                             "run (e.g. 'llm,hebbian,seed_42').")
    parser.add_argument("--wandb-id", type=str, default=None,
                        help="Explicit W&B run id. Defaults to the sanitised "
                             "run_id, which makes chunked SLURM jobs resume "
                             "into the same W&B run (resume='allow').")
    parser.add_argument("--wandb-upload-artifacts", action="store_true",
                        help="Also upload final_metrics.json (and summary.txt) "
                             "as W&B artifacts at run end. Off by default to "
                             "save bandwidth on chunked runs.")
    # ── Reproducibility ──
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. Seeds torch, numpy, random, "
                             "and the Minetest world. LLM sampling remains stochastic — "
                             "run multiple trials and report mean/std.")
    # ── RL layer ──
    parser.add_argument("--rl", action="store_true",
                        help="Enable the modular RL layer (action-level MAPPO)")
    parser.add_argument("--rl-model-path", type=str, default=None,
                        help="Path to base model for RL (e.g. /scratch/.../Qwen3.5-2B)")
    parser.add_argument("--rl-lora-rank", type=int, default=8,
                        help="LoRA rank for RL adapter")
    parser.add_argument("--rl-update-interval", type=int, default=256,
                        help="Steps between MAPPO updates")
    parser.add_argument("--rl-update-stagger", action="store_true",
                        default=os.environ.get("RL_UPDATE_STAGGER", "0") == "1",
                        help="Stagger per-agent PPO updates by agent_id steps "
                             "so the env steps between them instead of idling "
                             "for the whole update round (default off; also "
                             "via RL_UPDATE_STAGGER=1). Needed for Gemma E4B, "
                             "whose ~40-min update rounds hang the Minetest "
                             "bridge; Qwen's ~20-min rounds are safe without.")
    parser.add_argument("--rl-lr", type=float, default=1e-4,
                        help="Learning rate for RL optimiser")
    parser.add_argument("--rl-auto-token-opt", action="store_true",
                        help="Let agents self-trigger token-level optimisation")
    parser.add_argument("--rl-mode", type=str, default="action",
                        choices=["action", "token"],
                        help="RL mode: 'action' = MAPPO action head, "
                             "'token' = token-opt only (LLM picks actions)")
    parser.add_argument("--rl-critic-mode", type=str, default="centralized",
                        choices=["centralized", "independent"],
                        help="Critic architecture for action-mode RL. "
                             "'centralized' (default) = shared V(joint_state) critic across "
                             "all agents (true MAPPO). "
                             "'independent' = legacy per-agent value head on per-agent LLM "
                             "hidden state (IPPO).")
    parser.add_argument("--rl-prompt-max-tokens", type=int, default=512,
                        help="Max tokens for RL prompt encoding. Capping this is critical "
                             "for VRAM: at model_max_length=32768 a mini-batch of 8 prompts "
                             "needs ~21 GB just for hidden states. 512 is sufficient for "
                             "discrete action policy learning.")
    # ── Hebbian social plasticity ──
    parser.add_argument("--hebbian", action="store_true",
                        help="Enable Hebbian social plasticity graph")
    parser.add_argument("--hebbian-mode", type=str, default="reward_modulated",
                        choices=["legacy", "coactivity", "reward_modulated",
                                 "three_factor"],
                        help="Graph-update rule. 'reward_modulated' (default, "
                             "Variant B): growth (η0 + η+·|r_bond|/R)·c·(1−W). "
                             "'coactivity' (Variant A): flat η+·c·(1−W). "
                             "'three_factor': eligibility trace e←ρe+c with "
                             "growth η0·c·(1−W) + η+·(|r_bond|/R)·e·(1−W) and "
                             "monotone co-activity — reward credits recent "
                             "joint work and persists (pair with a lower "
                             "--hebbian-decay). 'legacy': old advantage-"
                             "modulator + failure-window rule.")
    # ── Gated-variant knobs (mode = coactivity | reward_modulated) ──
    parser.add_argument("--hebbian-eta-plus", type=float, default=0.05,
                        help="η+ growth rate (Variant A flat rate / Variant B "
                             "salience scale)")
    parser.add_argument("--hebbian-eta-0", type=float, default=0.01,
                        help="η0 association floor (Variant B only)")
    parser.add_argument("--hebbian-eta-minus", type=float, default=0.025,
                        help="η- failure-gated decay rate")
    parser.add_argument("--hebbian-coop-eps", type=float, default=0.05,
                        help="ε 'no co-activity' / activity-floor threshold")
    parser.add_argument("--hebbian-coop-window", type=int, default=50,
                        help="n rolling-window length (steps) for coop/neg")
    parser.add_argument("--hebbian-neg-theta", type=float, default=5.0,
                        help="θ negative-reward threshold (between |futile|=1 "
                             "and the death-class penalties |would-die|=10 / |death|=50)")
    parser.add_argument("--hebbian-eligibility-rho", type=float, default=0.9,
                        help="three_factor mode: eligibility-trace decay ρ_e "
                             "(e ← ρ_e·e + c; memory ≈ 1/(1−ρ_e) steps)")
    parser.add_argument("--hebbian-coact-floor", type=float, default=0.25,
                        help="three_factor mode: co-location counts at least "
                             "this much co-activity even for a silent pair; "
                             "0 restores the engagement-gated spatial term")
    parser.add_argument("--hebbian-death-ltd", type=float, default=0.0,
                        help="three_factor mode: η₋ᵈ signed death LTD rate — "
                             "a drained death/would-die penalty converts the "
                             "eligibility trace into bond WEAKENING "
                             "(ΔW⁻ = η₋ᵈ·(min(|death|,cap)/R)·e·W) on the "
                             "dying agent's outgoing row. 0 (default) = off, "
                             "byte-identical to the audited three_factor rule")
    parser.add_argument("--hebbian-death-cap", type=float, default=10.0,
                        help="cap on |death signal| before /R in the death-LTD "
                             "term: would-die (−10) and real death (−50) "
                             "blame equally")
    parser.add_argument("--hebbian-reward-norm", type=float, default=300.0,
                        help="R fixed bondable-reward normalizer (Variant B); "
                             "default = largest milestone reward (m27=300)")
    parser.add_argument("--hebbian-alpha", type=float, default=0.5,
                        help="α engagement reward/comm mix in g_i")
    parser.add_argument("--hebbian-radius", type=float, default=5.0,
                        help="Interaction radius d (Minetest world units)")
    parser.add_argument("--hebbian-ltp", type=float, default=0.01,
                        help="η_+ LTP learning rate")
    parser.add_argument("--hebbian-ltd", type=float, default=0.005,
                        help="η_- LTD learning rate")
    parser.add_argument("--hebbian-decay", type=float, default=0.005,
                        help="λ passive decay rate")
    parser.add_argument("--hebbian-beta", type=float, default=1.0,
                        help="β modulation sensitivity")
    parser.add_argument("--hebbian-rho", type=float, default=0.0,
                        help="ρ social replay blend factor (Eq. 7 weight-gated "
                             "experience sharing). 0 = off (paper default; "
                             "matches HebbianConfig). Requires --rl and "
                             "--hebbian; e.g. 0.3 makes ~30%% of each PPO "
                             "pool bond-weighted neighbour transitions.")
    parser.add_argument("--hebbian-gamma", type=float, default=0.2,
                        help="γ reward diffusion strength")
    parser.add_argument("--hebbian-init-weight", type=float, default=0.1,
                        help="Initial bond weight W_0 (default 0.1 = warm start)")
    parser.add_argument("--hebbian-no-comm-bond", action="store_true",
                        help="Set δ_comm=0 (spatial-only, for RQ4 ablation)")
    # ── Hardcoded / frozen graph (LLM-only social-bias ablation) ──
    parser.add_argument("--hebbian-freeze", action="store_true",
                        help="Freeze W for the whole run (no plasticity). Use "
                             "with --hebbian-preset + --social-module bias to "
                             "test an IMPOSED social topology. Pair with "
                             "--hebbian-gamma 0.")
    parser.add_argument("--hebbian-preset", type=str, default="none",
                        choices=["none", "uniform", "star", "ring", "pair"],
                        help="Hardcoded starting topology for W. 'uniform' = "
                             "flat control; 'star' = all bond to the hub; "
                             "'ring' = directed help chain; 'pair' = 0↔1 dyad "
                             "+ loner.")
    parser.add_argument("--hebbian-bond-strong", type=float, default=0.8,
                        help="Value of a 'strong' hardcoded bond (preset)")
    parser.add_argument("--hebbian-bond-weak", type=float, default=0.1,
                        help="Value of a 'weak' hardcoded bond (preset)")
    parser.add_argument("--hebbian-hub", type=int, default=0,
                        help="Hub agent index for the 'star' preset")
    # ── Pair-bonding transplant experiment (all default None = no-op) ──
    parser.add_argument("--max-chamber", type=int, default=None,
                        choices=[1, 2, 3, 4],
                        help="Highest chamber the Python force-teleport timers "
                             "will push agents into. E.g. 2 = the Ch1 timer "
                             "still fires but agents are never force-moved "
                             "past Ch2 (organic progression stays possible). "
                             "Default: no cap (current behavior).")
    parser.add_argument("--start-chamber", type=int, default=None,
                        choices=[2, 3, 4, 5],
                        help="Force-teleport all agents into this chamber at "
                             "the start of every episode (after warmup) and "
                             "suppress the timers for earlier chambers; the "
                             "remaining chambers split the episode evenly. "
                             "3 = start in the Ch3 cells. Default: normal "
                             "Ch1 start.")
    parser.add_argument("--hebbian-init-file", type=str, default=None,
                        help="JSON file holding a full N×N starting W matrix "
                             "(either {\"W\": [[...]]} or a raw nested list), "
                             "e.g. merged_W.json from "
                             "mindforge/tools/merge_pair_runs.py. Requires "
                             "--hebbian; mutually exclusive with "
                             "--hebbian-preset and --resume.")
    parser.add_argument("--agent-state-init", type=str, default=None,
                        help="Merged agent-state manifest JSON (skills, "
                             "episodic memory, curriculum per agent slot) "
                             "produced by merge_pair_runs.py. Imported into "
                             "the fresh per-agent vector DBs after agent "
                             "construction. Mutually exclusive with --resume.")
    # ── Phase B+ thesis comparison: interpretability sidecar ──
    # (`--reward-propagation` was removed alongside the deleted rlvr module
    #  that provided per_teammate_contributions / attribute_source_events /
    #  format_propagation_prompt. Reintroduce here if a local replacement
    #  for those helpers is added.)
    parser.add_argument("--interpretability", action="store_true",
                        help="Emit interpretability.jsonl with per-step "
                             "(agent, bond_row, action, comm_target, "
                             "propagated_deltas) records. Auto-enabled when "
                             "--hebbian is on; off otherwise.")
    # ── Social module (Hebbian-driven social-reasoning layer) ──
    parser.add_argument("--social-module", type=str, default="none",
                        choices=["none", "prompt", "bias"],
                        help="Social-reasoning module coupling: 'none' = "
                             "disabled (legacy raw bond text in action "
                             "prompt), 'prompt' = deliberation rendered as "
                             "directive text in the action prompt, 'bias' = "
                             "directive's ask_target also overwrites the "
                             "agent's communication_target at the routing "
                             "site. Requires --hebbian.")
    parser.add_argument("--social-interval", type=int, default=8,
                        help="Run the social-module deliberation every N "
                             "steps (cached in between). 1 = every step. "
                             "Default 8: bonds/directives change slowly, so "
                             "deliberating every step burned ~200 LLM calls/"
                             "room/agent for no behavioral gain.")
    # ── Choice-mode social acts (Experiment 2) ──
    parser.add_argument("--social-act-mode", type=str, default="legacy",
                        choices=["legacy", "choice"],
                        help="'legacy' (default): communication is a mandatory "
                             "per-step field, exactly the historical behavior. "
                             "'choice': each step the agent picks AT MOST ONE "
                             "social act from --social-acts (communicate / "
                             "observe / imitate / none); co-firing credits the "
                             "channels in --cofiring-channels. LLM-only "
                             "(incompatible with --rl).")
    parser.add_argument("--social-acts", type=str, default="comm,obs,imit",
                        help="Choice mode's affordance MENU: comma-separated "
                             "subset of comm,obs,imit — or 'none' for a mute "
                             "arm (proximity+reward floor). Ignored in legacy "
                             "mode.")
    parser.add_argument("--cofiring-channels", type=str, default=None,
                        help="Choice mode's co-firing CREDIT mask: subset of "
                             "comm,obs,imit or 'none'. Defaults to the value "
                             "of --social-acts (credit what is afforded). "
                             "Ignored in legacy mode (legacy credits comm).")
    parser.add_argument("--social-bidirectional", action="store_true",
                        help="Delivery-symmetric obs/imit ('agents that "
                             "co-fire wire together'): one observation/"
                             "imitation event credits BOTH directions of the "
                             "pair (as comm already does), and the target is "
                             "notified next step who observed/imitated it. "
                             "Choice mode only. Default off: directed "
                             "obs/imit terms, no notice — byte-identical to "
                             "the historical behavior.")
    parser.add_argument("--comm-distance-free", action="store_true",
                        help="Drop the (1 - spatial) factor from the comm "
                             "co-firing term: a message co-fires at ANY "
                             "distance, unifying comm with obs/imit "
                             "(c_k = delta*1[event]; the c_ij clip bounds "
                             "stacking with the spatial term). Default off: "
                             "legacy long-range-only comm.")
    parser.add_argument("--social-act-rewards", action="store_true",
                        help="Pay observation and imitation acts EXACTLY "
                             "like communication (same 0.5 base reward, cap, "
                             "rate limit, and per-chamber act milestones "
                             "m_obs_chN/m_imit_chN at the comm-track "
                             "values) — the act-reward symmetry suite. "
                             "Choice mode only. Default off: reward streams "
                             "byte-identical to the historical behavior.")
    parser.add_argument("--comm-reward-scale", type=float, default=1.0,
                        help="Scale on every communication PAYOUT (base msg "
                             "reward + chamber comm milestones). 0.0 = the "
                             "Experiment-2 noreward suite: messages still "
                             "route and comm milestones still fire as "
                             "events, but talking pays nothing — so it can "
                             "neither manufacture bondable reward nor trip "
                             "the milestone-success banner. Default 1.0 = "
                             "historical behavior.")
    parser.add_argument("--hebbian-delta", type=float, default=None,
                        help="δ: the co-activity value of ONE social act "
                             "(comm/obs/imit channel terms alike). Default "
                             "None keeps the historical 0.5. Set 1.0 with "
                             "--comm-reward-scale 0 so act-driven bonds "
                             "(growing at the η0 floor, without comm-reward "
                             "salience) still equilibrate in the analyzable "
                             "band against the homeostatic decay.")
    # ── Centralized task-ledger orchestrator (O2 baseline) ──────────────
    # Mutually exclusive with the Hebbian condition (validated in __main__).
    # All flags default to the disabled/no-op values so legacy runs are
    # byte-identical.
    parser.add_argument("--orchestrator", action="store_true",
                        help="Enable the O2 centralized orchestrator: a "
                             "non-embodied coordinator called every "
                             "--orchestrator-cadence steps (and on events) "
                             "that keeps a within-episode task ledger and "
                             "issues per-agent comm_target/help directives. "
                             "Runs INSTEAD of the Hebbian coupling.")
    parser.add_argument("--orchestrator-variant", type=str, default="task",
                        choices=["task", "social", "plan", "villager"],
                        help="'task' (default) = the O2 task-ledger "
                             "orchestrator (map + event digest in, "
                             "comm_target/help out, relational content "
                             "filtered, ledger reset per episode). 'social' "
                             "= centralized social deliberation, information-"
                             "matched to the Hebbian rule: pair co-presence/"
                             "message-count/co-reward digest in, a per-agent "
                             "SocialThought (ask_target/ask_message/"
                             "respond_to) out, rendered in the SocialModule's "
                             "exact directive format; relational notes "
                             "allowed; ledger persists across episodes like "
                             "W(t). 'plan' = social + each agent's auto-"
                             "curriculum task in view + a per-agent plan_note "
                             "delivered to that agent's curriculum at its "
                             "next task generation (upper baseline). "
                             "'villager' = VillagerAgent-style centralized "
                             "DAG orchestration: a decomposer LLM proposes "
                             "milestone-verified subtasks into a dependency "
                             "graph, an allocator LLM HARD-assigns ready "
                             "tasks to free agents (curriculum constrained "
                             "to the objective; replans on reassignment); "
                             "no communication routing.")
    parser.add_argument("--orchestrator-node-timeout-steps", type=int,
                        default=60,
                        help="Villager only: a running DAG task fails after "
                             "this many steps without one of its milestones "
                             "firing (default 60)")
    parser.add_argument("--orchestrator-max-open-tasks", type=int, default=0,
                        help="Villager only: cap on open+running DAG tasks; "
                             "0 = auto (2 x num agents)")
    parser.add_argument("--orchestrator-decompose-min-interval", type=int,
                        default=8,
                        help="Villager only: minimum steps between "
                             "decomposer calls, and the cooldown after a "
                             "failed allocator call (default 8)")
    parser.add_argument("--orchestrator-mode", type=str, default="advisory",
                        choices=["advisory", "bias"],
                        help="'advisory' = directives rendered into the same "
                             "{social_directive} action-prompt slot the "
                             "social module uses; 'bias' = additionally "
                             "override the emitted communication_target at "
                             "the routing site (mirrors --social-module "
                             "bias exactly)")
    parser.add_argument("--orchestrator-cadence", type=int, default=8,
                        help="Steps between scheduled orchestrator calls "
                             "(default 8 = the social module's T_soc "
                             "default, --social-interval)")
    parser.add_argument("--orchestrator-event-triggers",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Also call the orchestrator when a milestone / "
                             "chamber change / death occurred since its "
                             "last call (default on)")
    parser.add_argument("--orchestrator-stall-threshold", type=int, default=2,
                        help="The orchestrator is told to replan when its "
                             "ledger stall_counter exceeds this (default 2)")
    parser.add_argument("--orchestrator-max-task-facts", type=int, default=15,
                        help="Ledger task-facts cap; FIFO eviction keeps the "
                             "most recent (default 15)")
    parser.add_argument("--orchestrator-max-digest-events", type=int,
                        default=30,
                        help="Events included in the since-last-call digest "
                             "(default 30; older events are dropped with a "
                             "'(showing last K of M)' banner)")
    parser.add_argument("--orchestrator-use-map-image",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Attach the schematic top-down map PNG to the "
                             "orchestrator call (default on; falls back to "
                             "a text world-state block when off or when the "
                             "client lacks vision)")
    parser.add_argument("--orchestrator-model", type=str, default=None,
                        help="LLM for the orchestrator (default None = reuse "
                             "the agents' backbone/client). Only supported "
                             "on the HTTP-client path — rejected when "
                             "LLM_MODEL_PATH pins a local in-process model.")
    parser.add_argument("--orchestrator-log-dir-name", type=str,
                        default="orchestrator",
                        help="Subdirectory of the run dir for orchestrator "
                             "calls.jsonl / compliance.jsonl / maps/")
    # ── Experiment tracking ──
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Experiment identifier (e.g. E1a, E5) — saved in metrics for traceability")
    parser.add_argument("--run-group", type=str, default=None,
                        help="Suite subtree for --tag runs: output lands at "
                             "runs/<group>/<tag>/seed_<seed>/ and the W&B run "
                             "id is namespaced by <group> so re-running an "
                             "exp+seed under a new group starts a new W&B run "
                             "instead of resuming the old suite's. Defaults to "
                             "$WIREDTOGETHER_RUN_GROUP, then 'legacy' (which "
                             "keeps the pre-grouping paths and ids).")
    parser.add_argument("--tag", type=str, default=None,
                        help="Phase B++ tagged-run layout: when set, output "
                             "lands at runs/<group>/<tag>/seed_<seed>/ instead "
                             "of the default runs/<timestamp>_<experiment_id>/. "
                             "Lines up with the GRPO runs/grpo/<tag>/seed_<N>/ "
                             "pattern so build_results.py and the legacy "
                             "schema bridge discover both stacks uniformly.")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print a reward/metric summary every N steps (default 10)")
    # ── Team composition ─────────────────────────────────────────────
    # homogeneous-agent : all agents share --homogeneous-role (default).
    # heterogeneous     : agents take distinct roles from --roles
    #                     (comma-separated list, len must == --num-agents).
    parser.add_argument(
        "--team-mode",
        type=str,
        default="homogeneous-agent",
        choices=["homogeneous-agent", "heterogeneous"],
        help="homogeneous-agent: all agents share --homogeneous-role. "
             "heterogeneous: each agent gets a distinct role from --roles.",
    )
    parser.add_argument(
        "--homogeneous-role",
        type=str,
        default="agent",
        choices=["agent", "hunter", "harvester", "scouter"],
        help="Role for all agents in homogeneous-agent mode (default: agent).",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help="Comma-separated role list for heterogeneous mode "
             "(e.g. 'hunter,harvester,scouter'). Length must equal --num-agents. "
             "Each role must be one of: agent, hunter, harvester, scouter. "
             "Order maps to agent_0, agent_1, ... so changing the order changes "
             "which physical spawn gets which role — keep it stable across runs.",
    )
    # (Survival-mode CLI was removed — the env is permanently in
    # exploration mode: mobs passive in Ch1, hunger drain disabled. The
    # five-chamber curriculum supplied its own difficulty progression via
    # chamber-entry milestones, so the phased difficulty layer was redundant.)
    # ── Checkpoint / resume ──
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to write checkpoints into. "
                             "Default: ./checkpoints/<run_id>")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                        help="Save a checkpoint every N steps within an episode (default 500)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint directory from a previous job. "
                             "Restores cognitive/RL/Hebbian state and continues from saved ep/step.")
    parser.add_argument("--resume-skip-warmup", action="store_true",
                        help="Skip the media-load warmup detection on resume "
                             "(use when VoxeLibre media cache is already populated)")
    parser.add_argument("--checkpoint-frames", action="store_true",
                        help="Include raw frames in the checkpoint for GIF continuity. "
                             "Off by default as frame arrays can be large.")
    parser.add_argument("--voxel-obs", action="store_true",
                        help="Enable Craftium's per-agent voxel observations "
                             "(node-id + light + param2 grid around each agent). "
                             "When set, the per-step prompt gains a "
                             "'Nearby voxels: ...' line summarising the most "
                             "common blocks within ~10 blocks. Intended as a "
                             "hallucination-resistant grounding signal: the LLM "
                             "cannot perceive a zombie that isn't in the voxel "
                             "readout. Adds ~50 KB per agent per env step to the "
                             "TCP payload; OFF by default.")
    return parser.parse_args()


def validate_args(args) -> None:
    """Reject flag combinations that would silently produce a useless run.

    Each check corresponds to a coupling that only exists when another
    flag is set, or to two conditions that would confound each other.
    Raises :class:`SystemExit` with the reason rather than degrading to a
    no-op that costs a day of cluster time.
    """
    if args.social_module != "none" and not args.hebbian:
        # The social module reads from bond_weights / bond_deltas, which
        # are only populated when the Hebbian graph is enabled. Without
        # --hebbian, deliberation never runs and the directive falls back
        # to "Social bonds: N/A" every step — a silent no-op. Fail loudly
        # rather than letting an experiment run for 24h producing useless
        # output.
        raise SystemExit(
            "--social-module requires --hebbian to be set (the social "
            "module needs bond weights to reason over)"
        )
    if args.social_act_mode == "choice" and args.rl:
        # Choice mode is LLM-only by design: the social-act choice, the
        # observe/imitate payloads and the guided-imitation instructions
        # all live in the LLM prompt/schema layer, which the RL policy's
        # constrained decoding never sees. Fail loudly rather than running
        # an arm whose manipulation the policy cannot perceive.
        raise SystemExit(
            "--social-act-mode choice is LLM-only and incompatible with "
            "--rl (the social-act choice lives in the LLM prompt/schema "
            "layer, which the RL policy never sees)"
        )
    if args.social_act_mode == "choice" and not args.hebbian:
        raise SystemExit(
            "--social-act-mode choice requires --hebbian (the co-firing "
            "credit mask has no graph to act on otherwise)"
        )
    if args.social_act_rewards and args.social_act_mode != "choice":
        raise SystemExit(
            "--social-act-rewards requires --social-act-mode choice (there "
            "are no observation/imitation acts to pay in legacy mode)"
        )
    if args.hebbian_init_file and not args.hebbian:
        raise SystemExit(
            "--hebbian-init-file requires --hebbian (there is no graph to "
            "initialize otherwise)"
        )
    if args.hebbian_init_file and args.hebbian_preset != "none":
        raise SystemExit(
            "--hebbian-init-file and --hebbian-preset are mutually exclusive "
            "(init_matrix would silently override the preset)"
        )
    if args.resume and (args.hebbian_init_file or args.agent_state_init):
        raise SystemExit(
            "--hebbian-init-file/--agent-state-init cannot be combined with "
            "--resume: the checkpoint restores its own Hebbian graph and "
            "curriculum state and would clobber/duplicate the transplant. "
            "(Resuming a transplant run WITHOUT these flags is fine — the "
            "checkpoint already carries the transplanted state forward.)"
        )
    if args.start_chamber and args.max_chamber:
        raise SystemExit(
            "--start-chamber and --max-chamber are mutually exclusive"
        )
    if args.orchestrator and args.hebbian:
        # The orchestrator (O2) is a BASELINE against the Hebbian condition;
        # enabling both would confound the comparison. Reward diffusion
        # (--hebbian-gamma) belongs to the Hebbian condition and is only
        # active under --hebbian, so this one check also excludes it. Fail
        # loudly rather than silently disabling either.
        raise SystemExit(
            "orchestrator and Hebbian coupling are mutually exclusive "
            "conditions; disable one (--orchestrator vs --hebbian; reward "
            "diffusion is part of the Hebbian condition)"
        )
    if args.orchestrator and args.social_module != "none":
        raise SystemExit(
            "--orchestrator and --social-module both write the "
            "{social_directive} action-prompt slot; disable one"
        )
    if (args.orchestrator and args.orchestrator_variant == "villager"
            and args.orchestrator_mode == "bias"):
        raise SystemExit(
            "the villager variant issues task assignments, not comm "
            "directives — there is no comm_target to bias; use "
            "--orchestrator-mode advisory"
        )
