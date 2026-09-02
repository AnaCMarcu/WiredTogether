import argparse
import asyncio
import os
import random
import sys
import time
import logging
from pathlib import Path

sys.setrecursionlimit(10000)
from datetime import datetime

import numpy as np
import PIL
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import CancellationToken, Image
from autogen_core import EVENT_LOGGER_NAME

from custom_agent import CustomAgent
from custom_environment_craftium import CraftiumEnvironmentInterface, VALID_ACTIONS
from agent_modules.action_selection import ActionSelection
from agent_modules.auto_curriculum import AutoCurriculum
from agent_modules.belief_system import BeliefSystem
from agent_modules.critic import Critic
from agent_modules.skill_manager import SkillManager
from agent_modules.episodic_memory_manager import EpisodicMemoryManager
from agent_modules.craftium_metric import CraftiumMetric, format_milestone_progress
from agent_modules.social_module import SocialModule
from mindforge.env.communication_rewards import CommunicationTracker
from mindforge.env.cooperation_metric import CooperationMetric
from mindforge.env.episode_logger import EpisodeLogger
from mindforge.run_layout import RunPaths
import json as _json

from rl_layer import RLConfig, RLLayer, HebbianConfig, HebbianSocialGraph

from mindforge.chamber_schedule import compute_chamber_schedule

ROLE_NAMES = ["agent", "hunter", "harvester", "scouter"]

# Macro actions removed — agents use only primitives. The macro
# reward-deferral / macro-skip scaffolding (kept around as a no-op for a
# while after the macro removal) was deleted in the T1.6 cleanup.


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mindforge agents in Craftium OpenWorld")
    parser.add_argument("--num-agents", type=int, default=3,
                        help="Number of agents in five-chambers (all share the agent role)")
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


def load_prompts():
    """Load all prompt files and return them as a dict."""
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    belief_dir = os.path.join(prompt_dir, "belief_system")

    def _read(path):
        with open(path, "r") as f:
            return f.read()

    prompts = {
        "environment": _read(os.path.join(prompt_dir, "environment_prompt.txt")),
        "system_template": _read(os.path.join(prompt_dir, "system_prompt.txt")),
        "critic": _read(os.path.join(prompt_dir, "critic_prompt.txt")),
        "curriculum_questions": _read(os.path.join(prompt_dir, "curriculum_questions.txt")),
        "skill_description": _read(os.path.join(prompt_dir, "skill_description_prompt.txt")),
        "skill_info": _read(os.path.join(prompt_dir, "skill_description_info.txt")),
        "perception": _read(os.path.join(belief_dir, "perception_beliefs.txt")),
        "partner": _read(os.path.join(belief_dir, "partner_beliefs.txt")),
        "interaction": _read(os.path.join(belief_dir, "interaction_belief.txt")),
        "context": _read(os.path.join(belief_dir, "update_context.txt")),
    }

    # Role prompts
    prompts["roles"] = {}
    for role in ROLE_NAMES:
        prompts["roles"][role] = _read(os.path.join(prompt_dir, f"role_{role}.txt"))

    return prompts


def build_role_configs(
    num_agents,
    role_prompts,
    team_mode="homogeneous-agent",
    homogeneous_role="agent",
    roles=None,
):
    """Build ROLE_CONFIGS for num_agents.

    Two modes:
      * homogeneous-agent: every agent gets ``homogeneous_role`` (default
        "agent" — matches all prior runs).
      * heterogeneous: ``roles`` is a list of role names of length
        ``num_agents``; agent_i is assigned roles[i]. This is what makes
        the Hebbian bonds meaningful — symmetric teams give symmetric W.

    ``roles`` may also be a comma-separated string for CLI convenience.
    """
    if team_mode == "heterogeneous":
        if roles is None:
            raise ValueError(
                "team_mode='heterogeneous' requires --roles "
                "(comma-separated list, length == num_agents)."
            )
        if isinstance(roles, str):
            roles = [r.strip() for r in roles.split(",") if r.strip()]
        if len(roles) != num_agents:
            raise ValueError(
                f"--roles has {len(roles)} entries but num_agents is "
                f"{num_agents}. They must match."
            )
        for r in roles:
            if r not in role_prompts:
                raise ValueError(
                    f"Unknown role: {r!r}. Available roles: "
                    f"{sorted(role_prompts.keys())}."
                )
        assigned_roles = list(roles)
    else:
        if homogeneous_role not in role_prompts:
            raise ValueError(
                f"Unknown homogeneous_role: {homogeneous_role!r}. "
                f"Available roles: {sorted(role_prompts.keys())}."
            )
        assigned_roles = [homogeneous_role] * num_agents

    return [
        {
            "name": assigned_roles[i],
            "agent_name": f"agent_{i}",
            "curriculum_prompt": role_prompts[assigned_roles[i]].format(
                num_agents=num_agents
            ),
        }
        for i in range(num_agents)
    ]


def build_agents(role_configs, system_prompt, prompts, num_agents, communication, metric,
                 rl_config=None, belief_interval=5, critic_interval=20,
                 centralized_critic=None, is_resume: bool = False,
                 social_module_mode: str = "none", social_interval: int = 8,
                 social_act_mode: str = "legacy",
                 social_act_channels: tuple = (),
                 orchestrator_plan: bool = False,
                 orchestrator_villager: bool = False):
    """Initialize all Mindforge agents.

    ``centralized_critic`` (when not None) is shared by all agents' RLLayers
    and turns the value-loss off in their PPO updates.

    ``is_resume`` controls whether per-agent persistent stores (skill DB) are
    wiped at construction. Fresh runs reset; chained-checkpoint resumes
    preserve previously-learned skills.

    ``social_act_mode`` / ``social_act_channels`` (Experiment 2): in
    "choice" mode agents are built with the PARALLEL choice-mode prompt
    templates + the SocialAgentResponse schema; "legacy" (default) keeps the
    original templates and AgentResponse byte-for-byte.
    """
    # O-plan orchestrator variant: curriculum USER template with the
    # {team_plan_note} placeholder appended. None in every other
    # configuration → AutoCurriculum falls back to its module-level default,
    # byte-identical to the historical prompt.
    _task_info_override = None
    if orchestrator_plan:
        from agent_modules.auto_curriculum import curriculum_info as _cur_info
        from orchestrator.curriculum_hook import apply_plan_suffix
        _task_info_override = apply_plan_suffix(_cur_info, True)
    elif orchestrator_villager:
        # Villager: HARD assignment block ({assigned_objective}) instead of
        # the advisory plan-note block.
        from agent_modules.auto_curriculum import curriculum_info as _cur_info
        from orchestrator.curriculum_hook import apply_villager_suffix
        _task_info_override = apply_villager_suffix(_cur_info, True)

    # Choice-mode template/client setup — built once, shared by all agents.
    _choice_action_kwargs = {}
    _choice_sm_prompt = None
    if social_act_mode == "choice":
        from agent_modules import social_acts as _sacts
        from agent_modules.util import (
            SocialAgentResponse, SocialThoughtChoice, create_model_client,
            safe_format as _safe_format,
        )
        _choice_system_txt, _choice_instruction = _sacts.load_choice_templates(
            social_act_channels
        )
        _choice_action_kwargs = {
            "system_prompt": _safe_format(
                _choice_system_txt, environment_prompt=prompts["environment"]
            ),
            "user_prompt_template": _choice_instruction,
        }
        if social_module_mode != "none":
            _choice_sm_prompt = _sacts.load_social_module_choice_prompt(
                social_act_channels
            )

    agents = []
    for i, role_cfg in enumerate(role_configs):
        # Build per-agent RL layer (no-op when rl_config.enabled is False)
        rl_layer = None
        if rl_config and rl_config.enabled:
            rl_layer = RLLayer(
                config=rl_config, role=role_cfg["name"], agent_id=i,
                centralized_critic=centralized_critic,
            )

        # Targeted communication policy lives in the static prompts. The LLM
        # uses its own agent name (passed in via the user message) to exclude
        # itself from the recipient list.
        agent_system_prompt = system_prompt

        # Optional Hebbian-driven social-reasoning module. Stays None when
        # --social-module=none so the agent loop falls back to the legacy
        # raw bond text in the action prompt.
        agent_social_module = None
        if social_module_mode != "none":
            if social_act_mode == "choice":
                from agent_modules.util import (
                    SocialThoughtChoice as _STC,
                    create_model_client as _cmc,
                )
                agent_social_module = SocialModule(
                    agent_name=role_cfg["agent_name"],
                    num_agents=num_agents,
                    social_interval=social_interval,
                    social_model_client=_cmc(response_format=_STC),
                    override_prompt=_choice_sm_prompt,
                )
            else:
                agent_social_module = SocialModule(
                    agent_name=role_cfg["agent_name"],
                    num_agents=num_agents,
                    social_interval=social_interval,
                )

        if social_act_mode == "choice":
            from agent_modules.util import (
                SocialAgentResponse as _SAR,
                create_model_client as _cmc2,
            )
            _action_selection = ActionSelection(
                action_model_client=_cmc2(response_format=_SAR),
                **_choice_action_kwargs,
            )
        else:
            _action_selection = ActionSelection(system_prompt=agent_system_prompt)

        agent = CustomAgent(
            name=role_cfg["agent_name"],
            description=f"{role_cfg['name']} agent in Craftium open world",
            action_selection=_action_selection,
            auto_curriculum=AutoCurriculum(
                override_curriculum_prompt=role_cfg["curriculum_prompt"],
                override_questions_prompt=prompts["curriculum_questions"],
                override_task_info_prompt=_task_info_override,
                agent_name=role_cfg["agent_name"],
            ),
            critic=Critic(override_critic_prompt=prompts["critic"]),
            skill_manager=SkillManager(
                override_skill_prompt=prompts["skill_description"],
                override_skill_info_prompt=prompts["skill_info"],
                agent_name=role_cfg["agent_name"],
                # On resume from a checkpoint, preserve the per-agent
                # skill DB so skills learned in earlier chained runs
                # remain available. Fresh runs wipe (default).
                reset=not is_resume,
            ),
            episode_manager=EpisodicMemoryManager(
                agent_name=role_cfg["agent_name"],
            ),
            belief_system=BeliefSystem(
                number_of_agents=num_agents,
                override_perception_prompt=prompts["perception"],
                override_partner_prompt=prompts["partner"],
                override_interaction_prompt=prompts["interaction"],
                override_context_prompt=prompts["context"],
            ),
            number_of_agents=num_agents,
            metric=metric,
            voyager=False,
            rl_layer=rl_layer,
            belief_interval=belief_interval,
            critic_interval=critic_interval,
            num_agents=num_agents,
            social_module=agent_social_module,
        )
        agents.append(agent)
        rl_status = " [RL enabled]" if rl_layer else ""
        print(f"Initialized agent {i}: {role_cfg['agent_name']} ({role_cfg['name']}){rl_status}")
    return agents


def _resume_run_paths(run_id: str, group: str | None = None) -> RunPaths:
    """Reconstruct a ``RunPaths`` from a saved ``run_id`` regardless of layout.

    Two cases:
      * ``"<tag>/seed_<N>"`` (Phase B++ tagged layout) → lives under
        ``runs/<group>/<tag>/seed_<N>/``.
      * Anything else (legacy timestamp-based id) → lives under
        ``runs/<run_id>/``.

    Detected by the presence of ``/seed_`` in the run_id, which can only
    come from ``RunPaths.create_tagged``. The dataclass stores both
    forms identically — only the on-disk root differs.

    ``group`` must match the run being resumed (the run_id does not carry
    it); passing ``None`` resolves it from the environment exactly as the
    original run did.
    """
    if "/seed_" in run_id and run_id.count("/") == 1:
        tag, seed_part = run_id.split("/", 1)
        try:
            seed = int(seed_part.removeprefix("seed_"))
        except ValueError:
            # Malformed id — fall back to the untagged factory.
            return RunPaths.create(run_id=run_id, root="runs")
        return RunPaths.create_tagged(tag=tag, seed=seed, group=group)
    return RunPaths.create(run_id=run_id, root="runs")


# ===========================
# Agent action loop
# ===========================
async def agent_do_action(
    agent,
    agent_id: int,
    frame_image,
    communications: list,
    reward_text: str,
    environment,
    error=None,
    error_count=0,
    social_bonds=None,
    propagation_summary=None,
    position_text=None,
    player_status_text=None,
    current_chamber=None,
    visited_chambers=None,
    completed_milestones=None,
    milestone_progress=None,
    chamber_state=None,
    bond_weights=None,
    bond_deltas=None,
    social_returns=None,
    orchestrator_directive=None,
    orchestrator_plan_note=None,
    orchestrator_assigned_objective=None,
):
    """Have one agent observe and choose an action.

    Returns:
        (content_dict, last_action_str, error_count)
    """
    formatted_communication = [
        f"{msg.source}: {msg.content}"
        for msg in communications
        if msg.source != agent.name
    ]

    # Don't pre-fill the instruction template here — action_selection.select_action()
    # will fill it once with real cognitive data (beliefs, skills, episodes) via llm_call.
    # Only pass communication context as the message content.
    comm_text = f"Communications from other agents: {formatted_communication}.\n"

    multi_modal_message = MultiModalMessage(
        content=[comm_text, Image.from_pil(frame_image)],
        source="user",
    )

    content, error_count = await agent.on_messages(
        [multi_modal_message],
        CancellationToken(),
        communication=formatted_communication,
        error=error,
        error_count=error_count,
        picked_object=environment.pickedup_object(agentId=agent_id),
        reward_text=reward_text,
        social_bonds=social_bonds,
        propagation_summary=propagation_summary,
        position_text=position_text,
        player_status_text=player_status_text,
        current_chamber=current_chamber,
        visited_chambers=visited_chambers,
        completed_milestones=completed_milestones,
        milestone_progress=milestone_progress,
        chamber_state=chamber_state,
        bond_weights=bond_weights,
        bond_deltas=bond_deltas,
        social_returns=social_returns,
        orchestrator_directive=orchestrator_directive,
        orchestrator_plan_note=orchestrator_plan_note,
        orchestrator_assigned_objective=orchestrator_assigned_objective,
    )

    last_action = "NoOp"
    try:
        action = content.get("action", "NoOp") if content else "NoOp"
        _, last_action = environment.step(action, agentId=agent_id)
    except Exception as e:
        logging.error(f"Error in environment step for agent {agent_id}: {e}")
        if error_count < 5:
            content, last_action, error_count = await agent_do_action(
                agent, agent_id, frame_image, communications, reward_text,
                environment,
                error=str(e), error_count=error_count + 1,
                social_bonds=social_bonds,
                propagation_summary=propagation_summary,
                position_text=position_text,
                player_status_text=player_status_text,
                current_chamber=current_chamber,
                visited_chambers=visited_chambers,
                completed_milestones=completed_milestones,
                milestone_progress=milestone_progress,
                chamber_state=chamber_state,
                bond_weights=bond_weights,
                bond_deltas=bond_deltas,
                social_returns=social_returns,
                orchestrator_directive=orchestrator_directive,
                orchestrator_plan_note=orchestrator_plan_note,
                orchestrator_assigned_objective=orchestrator_assigned_objective,
            )
        else:
            logging.error(f"Agent {agent_id} exceeded retry limit, using NoOp")
            environment.step("NoOp", agentId=agent_id)
            last_action = "NoOp"

    return content, last_action, error_count


def save_checkpoint(
    checkpoint_dir: str,
    episode: int,
    step: int,
    run_id: str,
    args,
    metric: "CraftiumMetric",
    agents,
    hebbian_graph: "HebbianSocialGraph",
    frames_list=None,
    save_frames: bool = False,
    global_step: int = 0,
) -> None:
    """Serialize full run state to *checkpoint_dir* so a new SLURM job can resume.

    Files written:
      run_state.json         — scalar counters, CLI args, metric dicts
      hebbian_graph.json     — Hebbian weight matrix + config
      rl_agent_{i}/          — RL LoRA weights + optimizer (via rl_layer.save())
      agent_{i}_curriculum.json — curriculum task lists + current task/context
      frames_{i}.npy         — raw observation arrays (optional, --checkpoint-frames)

    The function is wrapped in try/except so a serialization error never kills the run.
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # --- run_state.json -------------------------------------------------
        metric_dict = {
            "num_agents": metric.num_agents,
            "communication": metric.communication,
            "run_id": metric.run_id,
            "timestep": metric.timestep,
            "cumulative_returns": [float(x) for x in metric.cumulative_returns],
            "episode_returns": [float(x) for x in metric.episode_returns],
            "per_episode_returns": [
                [float(x) for x in ep_list]
                for ep_list in metric.per_episode_returns
            ],
            "track_rewards_episode": {
                str(i): dict(metric.track_rewards_episode[i])
                for i in range(metric.num_agents)
            },
            "track_rewards_per_episode": [
                [dict(d) for d in agent_eps]
                for agent_eps in metric.track_rewards_per_episode
            ],
            "agent_milestones_episode": {
                str(i): sorted(metric._agent_milestones_episode[i])
                for i in range(metric.num_agents)
            },
            "milestones_per_episode": [
                [list(ms) for ms in agent_eps]
                for agent_eps in metric.milestones_per_episode
            ],
            "comm_count_episode": list(metric.comm_count_episode),
            "comm_count_per_episode": [
                list(c) for c in metric.comm_count_per_episode
            ],
            "episode_lengths": list(metric.episode_lengths),
            "comm_counts_per_step": list(metric.comm_counts_per_step),
            "communication_log": metric.communication_log,
            "rl_updates": metric.rl_updates,
            "rl_token_opts": metric.rl_token_opts,
            "milestones_per_agent": {name: sorted(ms) for name, ms in metric._agent_milestones.items()},
            "track_rewards": metric.track_rewards,
            "_graph_snapshots": metric._graph_snapshots,
            "ts_data": metric.ts_data,
            "phase_transitions": getattr(metric, "phase_transitions", []),
            "team_mode": getattr(metric, "team_mode", "heterogeneous"),
            "homogeneous_role": getattr(metric, "homogeneous_role", "agent"),
        }
        run_state = {
            "episode": episode,
            "step": step,
            "run_id": run_id,
            "metric": metric_dict,
            "cli_args": vars(args),
            "global_step": global_step,
            # Team composition
            "team_mode": getattr(metric, "team_mode", "heterogeneous"),
            "homogeneous_role": getattr(metric, "homogeneous_role", "agent"),
        }
        with open(os.path.join(checkpoint_dir, "run_state.json"), "w") as f:
            _json.dump(run_state, f, indent=2, default=str)

        # --- hebbian_graph.json ---------------------------------------------
        with open(os.path.join(checkpoint_dir, "hebbian_graph.json"), "w") as f:
            _json.dump(hebbian_graph.to_dict(), f, indent=2)

        # --- per-agent RL weights + optimizer --------------------------------
        for i, agent in enumerate(agents):
            if agent.rl_layer and agent.rl_layer.enabled:
                rl_save_dir = os.path.join(checkpoint_dir, f"rl_agent_{i}")
                os.makedirs(rl_save_dir, exist_ok=True)
                agent.rl_layer.save(path=rl_save_dir)

        # --- per-agent curriculum state -------------------------------------
        for i, agent in enumerate(agents):
            cur = agent.auto_curriculum
            curriculum_state = {
                "current_task": cur.current_task,
                "current_context": getattr(cur, "current_context", ""),
                "completed_tasks": list(cur.completed_tasks),
                "failed_tasks": list(cur.failed_tasks),
            }
            with open(os.path.join(checkpoint_dir, f"agent_{i}_curriculum.json"), "w") as f:
                _json.dump(curriculum_state, f, indent=2)

        # --- per-agent cognitive state (skills / episodic memory) -----------
        # The vector DBs live on job-local /tmp under SLURM; this JSON copy in
        # the checkpoint dir is the durable form (used by merge_pair_runs.py).
        from agent_modules.agent_state_io import export_agent_state
        export_agent_state(
            agents, os.path.join(checkpoint_dir, "agent_state"),
            episode=episode,
        )

        # --- optional frames ------------------------------------------------
        if save_frames and frames_list:
            for i in range(len(agents)):
                agent_frames = [f[i] for f in frames_list if f[i] is not None]
                if agent_frames:
                    frames_path = os.path.join(checkpoint_dir, f"frames_{i}.npy")
                    np.save(frames_path, np.stack(agent_frames, axis=0))

        print(f"[CKPT] Saved checkpoint ep={episode} step={step} → {checkpoint_dir}")

    except Exception as exc:
        logging.warning(f"[CKPT] save_checkpoint failed (ep={episode} step={step}): {exc}")


def load_checkpoint(
    checkpoint_dir: str,
    agents,
    hebbian_graph: "HebbianSocialGraph",
    metric_path: str = "./run_metrics",
    run_paths=None,
) -> dict:
    """Restore run state from *checkpoint_dir*.

    Returns a dict with keys:
      episode  — last fully-checkpointed episode index
      step     — last checkpointed step within that episode
      run_id   — original run ID
      metric   — restored CraftiumMetric instance

    RL weights, optimizer, and Hebbian graph are restored in-place.
    Curriculum state is restored into each agent's auto_curriculum.

    When ``run_paths`` is supplied, the restored metric writes to that
    consolidated tree (``runs/<run_id>/``). Otherwise ``metric_path`` is
    used and the legacy ``./run_metrics/<run_id>/`` folder is created.
    """
    run_state_path = os.path.join(checkpoint_dir, "run_state.json")
    if not os.path.exists(run_state_path):
        raise FileNotFoundError(f"[CKPT] No run_state.json in {checkpoint_dir}")

    with open(run_state_path, "r") as f:
        run_state = _json.load(f)

    episode = run_state["episode"]
    step = run_state["step"]
    run_id = run_state.get("run_id", "resumed")

    # Restore metric — use run_paths when provided so resumed runs keep
    # writing into the same `runs/<run_id>/` tree as the live process.
    metric = CraftiumMetric.restore_from_dict(
        run_state["metric"], path=metric_path, run_paths=run_paths,
    )

    # Restore Hebbian graph
    hebbian_path = os.path.join(checkpoint_dir, "hebbian_graph.json")
    if os.path.exists(hebbian_path):
        with open(hebbian_path, "r") as f:
            hebbian_dict = _json.load(f)
        hebbian_graph.from_dict(hebbian_dict)
        print(f"[CKPT] Restored Hebbian graph from {hebbian_path}")
    else:
        logging.warning(f"[CKPT] No hebbian_graph.json in {checkpoint_dir}, graph untouched")

    # Restore per-agent RL state
    for i, agent in enumerate(agents):
        rl_save_dir = os.path.join(checkpoint_dir, f"rl_agent_{i}")
        if agent.rl_layer and agent.rl_layer.enabled and os.path.isdir(rl_save_dir):
            agent.rl_layer.load(path=rl_save_dir)
            print(f"[CKPT] Restored RL state for agent_{i} from {rl_save_dir}")

    # Restore per-agent curriculum state
    for i, agent in enumerate(agents):
        cur_path = os.path.join(checkpoint_dir, f"agent_{i}_curriculum.json")
        if os.path.exists(cur_path):
            with open(cur_path, "r") as f:
                cur_state = _json.load(f)
            cur = agent.auto_curriculum
            cur.current_task = cur_state.get("current_task")
            cur.current_context = cur_state.get("current_context", "")
            cur.completed_tasks = list(cur_state.get("completed_tasks", []))
            cur.failed_tasks = list(cur_state.get("failed_tasks", []))
            print(f"[CKPT] Restored curriculum for agent_{i}: task={cur.current_task!r}")

    metric._global_step_ckpt = run_state.get("global_step", 0)

    print(f"[CKPT] Loaded checkpoint: ep={episode} step={step} run_id={run_id}")
    return {"episode": episode, "step": step, "run_id": run_id, "metric": metric}


def _frames_to_mp4(pil_frames: list, mp4_path: str, fps: int = 2) -> None:
    """Write PIL frames directly to MP4 using imageio[ffmpeg] (bundled binary, no system ffmpeg)."""
    try:
        import imageio
        with imageio.get_writer(mp4_path, fps=fps, macro_block_size=1) as writer:
            for frame in pil_frames:
                writer.append_data(np.array(frame))
        print(f"  Saved MP4: {mp4_path}")
    except Exception as exc:
        logging.warning("MP4 save failed (%s): %s", mp4_path, exc)


# ===========================
# Main episode loop
# ===========================
async def run(args):
    num_agents = args.num_agents
    num_episodes = args.episodes
    max_steps = args.max_steps
    obs_width = args.obs_width
    obs_height = args.obs_height
    communication = not args.no_communication

    # ── Choice-mode social acts (Experiment 2) ─────────────────────────
    # legacy (default): everything below stays inert and the run is
    # bit-for-bit the historical behavior. choice: the agent picks at most
    # one social act per step from the MENU; co-firing credits the CREDIT
    # mask channels.
    social_act_mode = getattr(args, "social_act_mode", "legacy")
    _social_menu_channels: tuple = ()
    _cofire_channels: tuple = ("comm",)   # legacy credit = messages only
    if social_act_mode == "choice":
        from agent_modules import social_acts as _sacts
        _social_menu_channels = _sacts.parse_channels_csv(args.social_acts)
        _cofire_channels = (
            _sacts.parse_channels_csv(args.cofiring_channels)
            if args.cofiring_channels is not None
            else _social_menu_channels
        )
        print(f"[FEATURES] Social acts:      CHOICE mode  "
              f"menu={list(_social_menu_channels) or 'none'}  "
              f"credit={list(_cofire_channels) or 'none'}")
    sleep_time = args.sleep_time
    save_gif = not args.no_gif
    gif_dir: str | None = None
    gif_interval = args.gif_interval

    seed = args.seed
    if seed is not None:
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seeded RNG: seed={seed}")

    if args.tag is not None:
        seed_for_path = args.seed if args.seed is not None else 0
        run_paths = RunPaths.create_tagged(
            tag=args.tag, seed=seed_for_path, group=args.run_group,
        )
        run_id = run_paths.run_id
        print(f"[RUN ID] {run_id} (tagged, group={run_paths.group})")
    else:
        from uuid import uuid4
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _exp = args.experiment_id or ""
        run_id = f"{_exp+'_' if _exp else ''}{_ts}_{uuid4().hex[:6]}"
        print(f"[RUN ID] {run_id}")
        run_paths = RunPaths.create(run_id=run_id, root="runs")

    # ── Weights & Biases init ─────────────────────────────────────────────
    import wandb_logger as _wb
    _wb_tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    _wb.init(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_id=run_id,
        tags=_wb_tags,
        config=vars(args),
        explicit_id=args.wandb_id,
        group=run_paths.group,
    )

    if args.gif_dir == "auto":
        gif_dir = str(run_paths.root / "gifs")
    else:
        gif_dir = args.gif_dir
    if save_gif:
        os.makedirs(gif_dir, exist_ok=True)

    intermediate_gif_dir = (
        os.environ.get("WIREDTOGETHER_INTERMEDIATE_GIF_DIR")
        or os.path.join(os.getcwd(), "intermediate_gifs")
    )
    _abs_gif_dir = os.path.abspath(gif_dir)
    _abs_intermediate = os.path.abspath(intermediate_gif_dir)
    if (_abs_intermediate == _abs_gif_dir
            or _abs_intermediate.startswith(_abs_gif_dir + os.sep)):
        _redirected = os.path.join(os.getcwd(), "intermediate_gifs")
        if os.path.abspath(_redirected).startswith(_abs_gif_dir + os.sep):
            _redirected = str(run_paths.root.parent / "intermediate_gifs")
        print(f"[GIF] WARNING: intermediate_gif_dir resolved inside the run "
              f"({intermediate_gif_dir}); redirecting checkpoint media to "
              f"{_redirected} to keep runs/ small.")
        intermediate_gif_dir = _redirected
    if save_gif:
        os.makedirs(intermediate_gif_dir, exist_ok=True)

    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    event_logger.disabled = True
    logging.basicConfig(
        level=logging.INFO,
        filename=str(run_paths.log_txt),
        filemode="a",
        format=f"[{run_id}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from agent_modules.llm_call import setup_llm_logging as _setup_llm_logging
    _setup_llm_logging(run_paths.root / "llm_logs")

    log_interval = args.log_interval

    # FC_CH4_MOB_COUNT / WT_TEAM_SCALING must be set BEFORE the prompt
    # substitution below (which reads them) and before the env/server
    # construction (Lua reads both in config.lua via the same os.environ
    # merge that delivers FC_NUM_AGENTS).
    if args.ch4_mob_count is not None:
        os.environ["FC_CH4_MOB_COUNT"] = str(args.ch4_mob_count)
    team_scaling = bool(args.team_scaling) \
        or os.environ.get("WT_TEAM_SCALING") == "1"
    if team_scaling:
        os.environ["WT_TEAM_SCALING"] = "1"

    # Resolve the prompt placeholders ({num_agents_word}, {cell_letters},
    # {switch_rotation}, {ch4_zombies}, ...). Gated: with --team-scaling the
    # text is rendered truthfully for THIS run's team size; without it (all
    # legacy suites) the frozen 3-agent wording is rendered byte-identically
    # regardless of N. Must happen before build_role_configs (whose .format
    # would otherwise trip on the placeholders) and before any template
    # reaches llm_call.
    from agent_modules.team_scaling import apply_team_scaling_to_prompts
    prompts = apply_team_scaling_to_prompts(
        load_prompts(), num_agents, enabled=team_scaling)
    environment_prompt = prompts["environment"]

    from agent_modules.util import safe_format
    system_prompt_template = prompts["system_template"]
    system_prompt = safe_format(system_prompt_template, environment_prompt=environment_prompt)

    role_configs = build_role_configs(
        num_agents,
        prompts["roles"],
        team_mode=args.team_mode,
        homogeneous_role=args.homogeneous_role,
        roles=args.roles,
    )

    metric = CraftiumMetric(
        num_agents=num_agents,
        communication=communication,
        run_id=run_id,
        run_paths=run_paths,
    )
    metric.team_mode = args.team_mode
    metric.homogeneous_role = args.homogeneous_role
    metric.roles = [rc["name"] for rc in role_configs]

    _LUA_TICKS_PER_ENV_STEP = num_agents * 3
    _LUA_SAFETY_FACTOR = 50   
    _ch1_lua_ticks = (args.ch1_timeout_steps
                      * _LUA_TICKS_PER_ENV_STEP * _LUA_SAFETY_FACTOR)
    if args.start_chamber:
        # Skipping Ch1 entirely: disarm the Lua-side Ch1 fallback timer so it
        # can never fire a Ch1→Ch2 rescue in a run that starts past Ch1.
        _ch1_lua_ticks = 10 ** 9
    os.environ["CH1_TIMEOUT_TICKS"] = str(_ch1_lua_ticks)
    # Keep Lua's five_chambers.NUM_AGENTS in lockstep with --num-agents.
    # Without this, any N != 3 desyncs agent_index()/geometry on the Lua side
    # (agents with idx >= NUM_AGENTS get no cell, no switch, no milestones).
    os.environ["FC_NUM_AGENTS"] = str(num_agents)
    _chamber_schedule = compute_chamber_schedule(
        args.max_steps, args.start_chamber, args.max_chamber
    )
    _sched_str = ", ".join(
        f"Ch{n}→Ch{n+1}@" + (f"step {s}" if s is not None else "off")
        for n, s in sorted(_chamber_schedule.items())
    )
    print(
        f"[FEATURES] Chamber timer:      {_sched_str} of --max-steps="
        f"{args.max_steps}"
        + (f" (start_chamber={args.start_chamber})" if args.start_chamber else "")
        + (f" (max_chamber={args.max_chamber})" if args.max_chamber else "")
        + ". (Python primary, unconditional — fires even if the door opened "
        f"organically.) Lua fallback at {_ch1_lua_ticks} ticks "
        f"({_LUA_TICKS_PER_ENV_STEP}×{_LUA_SAFETY_FACTOR}/step safety margin)."
    )

    environment = CraftiumEnvironmentInterface(
        num_agents=num_agents,
        obs_width=obs_width,
        obs_height=obs_height,
        max_steps=max_steps,
        seed=seed,
        voxel_obs=args.voxel_obs,
    )

    # ── RL layer config ──
    _rl_run_id = run_id
    _rl_run_paths = run_paths
    if args.resume:
        _ckpt_state_path = os.path.join(args.resume, "run_state.json")
        if os.path.exists(_ckpt_state_path):
            with open(_ckpt_state_path) as _f:
                _rl_run_id = _json.load(_f).get("run_id", run_id)
            print(f"[RL] Resume: using adapter dir from original run_id={_rl_run_id!r}")
            _rl_run_paths = RunPaths(
                root=run_paths.root.parent / _rl_run_id,
                run_id=_rl_run_id,
            )
    rl_save_dir = str(_rl_run_paths.root / "rl_live")
    rl_config = RLConfig(
        enabled=args.rl,
        mode=args.rl_mode,
        model_path=args.rl_model_path,
        lora_rank=args.rl_lora_rank,
        update_interval=args.rl_update_interval,
        update_stagger=args.rl_update_stagger,
        lr=args.rl_lr,
        auto_token_opt=args.rl_auto_token_opt,
        rl_prompt_max_tokens=args.rl_prompt_max_tokens,
        lora_save_dir=rl_save_dir,
        critic_mode=args.rl_critic_mode,
    )
    if rl_config.enabled:
        print(f"RL layer ENABLED: model={rl_config.model_path}, "
              f"lora_rank={rl_config.lora_rank}, update_interval={rl_config.update_interval}, "
              f"critic_mode={rl_config.critic_mode}")

    # ── Centralised MAPPO critic (one shared V across all agents) ─────────
    centralized_critic = None
    if rl_config.enabled and rl_config.mode == "action" \
            and rl_config.critic_mode == "centralized":
        from rl_layer.centralized_critic import CentralizedCritic
        from agent_modules.craftium_metric import MILESTONE_TRACK
        milestone_ids = list(MILESTONE_TRACK.keys())
        centralized_critic = CentralizedCritic(
            num_agents=num_agents,
            config=rl_config,
            milestone_ids=milestone_ids,
        )
        print(f"[MAPPO] Centralised critic ENABLED: joint_dim={centralized_critic.joint_dim}, "
              f"milestones tracked={len(milestone_ids)}")

    agents = build_agents(role_configs, system_prompt, prompts, num_agents, communication, metric,
                         rl_config=rl_config,
                         belief_interval=args.belief_interval,
                         critic_interval=args.critic_interval,
                         centralized_critic=centralized_critic,
                         is_resume=bool(args.resume),
                         social_module_mode=args.social_module,
                         social_interval=args.social_interval,
                         social_act_mode=social_act_mode,
                         social_act_channels=_social_menu_channels,
                         orchestrator_plan=(
                             args.orchestrator
                             and args.orchestrator_variant == "plan"),
                         orchestrator_villager=(
                             args.orchestrator
                             and args.orchestrator_variant == "villager"))

    if args.agent_state_init:
        # Transplant memories into the freshly-constructed agents. Must run
        # after build_agents (construction wipes/recreates the per-agent
        # vector DBs) and before the episode loop (first on_messages call).
        # Raises on structural problems — fail at startup, not 24h in.
        from agent_modules.agent_state_io import import_agent_state
        print(f"[TRANSPLANT] Importing agent state from {args.agent_state_init}")
        import_agent_state(agents, args.agent_state_init)

    # ── Hebbian social plasticity ──
    _hebbian_init_matrix = None
    if args.hebbian_init_file:
        with open(args.hebbian_init_file) as _f:
            _init_payload = _json.load(_f)
        _hebbian_init_matrix = (
            _init_payload["W"] if isinstance(_init_payload, dict)
            else _init_payload
        )
        print(f"[FEATURES] Hebbian init matrix: loaded "
              f"{len(_hebbian_init_matrix)}×{len(_hebbian_init_matrix[0])} W "
              f"from {args.hebbian_init_file}")
        for _row in _hebbian_init_matrix:
            print("           " + " ".join(f"{float(_w):.3f}" for _w in _row))

    hebbian_config = HebbianConfig(
        enabled=args.hebbian,
        mode=args.hebbian_mode,
        num_agents=num_agents,
        init_matrix=_hebbian_init_matrix,
        interaction_radius=args.hebbian_radius,
        ltp_lr=args.hebbian_ltp,
        ltd_lr=args.hebbian_ltd,
        decay=args.hebbian_decay,
        modulation_beta=args.hebbian_beta,
        social_replay_rho=args.hebbian_rho,
        reward_diffusion_gamma=args.hebbian_gamma,
        communication_coactivity_bonus=(
            0.0 if args.hebbian_no_comm_bond
            else (args.hebbian_delta if args.hebbian_delta is not None else 0.5)
        ),
        # One δ for ALL channel terms (obs/imit follow comm via the None
        # alias unless --hebbian-delta sets them explicitly).
        social_coactivity_bonus=args.hebbian_delta,
        # Experiment 2 credit mask; ("comm",) in legacy mode = historical rule.
        social_act_channels=_cofire_channels,
        init_weight=args.hebbian_init_weight,
        # Gated-variant (coactivity / reward_modulated) knobs
        engagement_reward_weight=args.hebbian_alpha,
        eta_plus=args.hebbian_eta_plus,
        eta_0=args.hebbian_eta_0,
        eta_minus=args.hebbian_eta_minus,
        coop_eps=args.hebbian_coop_eps,
        coop_window=args.hebbian_coop_window,
        neg_theta=args.hebbian_neg_theta,
        reward_norm_R=args.hebbian_reward_norm,
        # Three-factor variant knobs (mode = three_factor)
        eligibility_rho=args.hebbian_eligibility_rho,
        coact_floor=args.hebbian_coact_floor,
        # Hardcoded / frozen graph (LLM-only social-bias ablation)
        freeze_weights=args.hebbian_freeze,
        social_bidirectional=args.social_bidirectional,
        comm_distance_free=args.comm_distance_free,
        init_preset=args.hebbian_preset,
        preset_bond_strong=args.hebbian_bond_strong,
        preset_bond_weak=args.hebbian_bond_weak,
        preset_hub=args.hebbian_hub,
    )
    agent_roles = [ROLE_NAMES.index(rc["name"]) for rc in role_configs]
    hebbian_graph = HebbianSocialGraph(hebbian_config, agent_roles=agent_roles)
    if hebbian_config.enabled and hebbian_config.freeze_weights:
        import numpy as _np
        print(f"[FEATURES] Hebbian graph FROZEN (no plasticity): "
              f"preset={hebbian_config.init_preset} "
              f"strong={hebbian_config.preset_bond_strong} "
              f"weak={hebbian_config.preset_bond_weak}"
              + (f" hub={hebbian_config.preset_hub}"
                 if hebbian_config.init_preset == "star" else ""))
        print("[FEATURES] Hardcoded W (row i = agent i's bonds toward j):")
        print(_np.array2string(hebbian_graph.get_all_weights(),
                               precision=2, suppress_small=True))

    # ── Centralized task-ledger orchestrator (O2 baseline) ────────────────
    # Everything below is None when --orchestrator is off, and every hook in
    # the loop is guarded on that — legacy runs are untouched.
    orchestrator_state = None
    orchestrator_config = None
    orchestrator_client = None
    orch_logger = None
    _orch_core = None
    _orch_events = None
    _orch_pair_acc = None
    villager_controller = None
    if args.orchestrator:
        from orchestrator import core as _orch_core
        from orchestrator import events as _orch_events
        from orchestrator.config import OrchestratorConfig
        from orchestrator.logging import OrchestratorLogger
        from orchestrator.state import OrchestratorState

        orchestrator_config = OrchestratorConfig(
            enabled=True,
            variant=args.orchestrator_variant,
            mode=args.orchestrator_mode,
            cadence=args.orchestrator_cadence,
            event_triggers=args.orchestrator_event_triggers,
            stall_threshold=args.orchestrator_stall_threshold,
            max_task_facts=args.orchestrator_max_task_facts,
            max_digest_events=args.orchestrator_max_digest_events,
            use_map_image=args.orchestrator_use_map_image,
            model=args.orchestrator_model,
            log_dir_name=args.orchestrator_log_dir_name,
            node_timeout_steps=args.orchestrator_node_timeout_steps,
            max_open_tasks=args.orchestrator_max_open_tasks,
            decompose_min_interval=args.orchestrator_decompose_min_interval,
        )
        orchestrator_config.validate()
        orchestrator_state = OrchestratorState()
        orch_logger = OrchestratorLogger(
            run_dir=str(run_paths.root),
            dir_name=orchestrator_config.log_dir_name,
        )
        if orchestrator_config.variant == "villager":
            # Villager: an event-driven controller with two dedicated
            # clients (distinct response schemas on the shared backbone —
            # same pattern as the curriculum/critic/social clients). The
            # generic orchestrate() client is never used for this variant.
            from orchestrator import villager as _orch_villager
            villager_controller = _orch_villager.VillagerController(
                orchestrator_config, num_agents,
                decompose_client=_orch_core.create_orchestrator_client(
                    orchestrator_config,
                    response_format=_orch_villager.VillagerDecomposeResponse,
                ),
                allocate_client=_orch_core.create_orchestrator_client(
                    orchestrator_config,
                    response_format=_orch_villager.VillagerAllocateResponse,
                ),
                orch_logger=orch_logger,
            )
        else:
            orchestrator_client = _orch_core.create_orchestrator_client(
                orchestrator_config
            )
        # social/plan variants: the Hebbian-rule-matched pair-activity
        # accumulator (fed each step from Phase-2 scope). Radius from
        # --hebbian-radius so a radius ablation stays matched.
        _orch_pair_acc = None
        if orchestrator_config.variant in ("social", "plan"):
            _orch_pair_acc = _orch_events.PairAccumulator(
                num_agents, radius=args.hebbian_radius,
            )
        _villager_info = (
            f"node_timeout={orchestrator_config.node_timeout_steps}  "
            f"max_open={orchestrator_config.max_open_tasks or 2 * num_agents}  "
            f"decompose_min_interval="
            f"{orchestrator_config.decompose_min_interval}  "
            if orchestrator_config.variant == "villager" else ""
        )
        print(f"[FEATURES] Orchestrator:     ENABLED "
              f"[{orchestrator_config.variant}/{orchestrator_config.mode}]  "
              f"cadence={orchestrator_config.cadence}  "
              f"event_triggers={orchestrator_config.event_triggers}  "
              f"stall_threshold={orchestrator_config.stall_threshold}  "
              f"max_task_facts={orchestrator_config.max_task_facts}  "
              f"map_image={orchestrator_config.use_map_image}  "
              f"{_villager_info}"
              f"model={orchestrator_config.model or 'backbone'}  "
              f"log_dir={orch_logger.dir}")

    comm_mode = "off" if not communication else "targeted"
    print(f"\nConfig: {num_agents} agents, {num_episodes} episodes, "
          f"{max_steps} max steps, comm={comm_mode}, "
          f"seed={seed}")

    # ── Feature activation summary ─────────────────────────────────────────
    _feat_sep = "─" * 60
    print(f"\n{_feat_sep}")
    print(f"[FEATURES] Team mode:        {args.team_mode}  ({num_agents} agents)")
    _roles_str = ", ".join(f"agent_{i}={rc['name']}" for i, rc in enumerate(role_configs))
    print(f"[FEATURES] Role assignment:  {_roles_str}")
    if hebbian_config.enabled:
        if hebbian_config.mode == "legacy":
            print(f"[FEATURES] Hebbian:          ENABLED [legacy]  "
                  f"ltp={hebbian_config.ltp_lr}  ltd={hebbian_config.ltd_lr}  "
                  f"decay={hebbian_config.decay}  "
                  f"radius={hebbian_config.interaction_radius}  "
                  f"gamma={hebbian_config.reward_diffusion_gamma}")
        else:
            print(f"[FEATURES] Hebbian:          ENABLED [{hebbian_config.mode}]  "
                  f"eta+={hebbian_config.eta_plus}  eta0={hebbian_config.eta_0}  "
                  f"eta-={hebbian_config.eta_minus}  eps={hebbian_config.coop_eps}  "
                  f"n={hebbian_config.coop_window}  theta={hebbian_config.neg_theta}  "
                  f"R={hebbian_config.reward_norm_R}  decay={hebbian_config.decay}  "
                  f"radius={hebbian_config.interaction_radius}  "
                  f"gamma={hebbian_config.reward_diffusion_gamma}")
    else:
        print(f"[FEATURES] Hebbian:          OFF")
    print(f"{_feat_sep}\n")
    # ─────────────────────────────────────────────────────────────────────────

    # ── Checkpoint directory: lives under runs/<run_id>/checkpoints/ ──
    checkpoint_dir = args.checkpoint_dir or str(run_paths.checkpoints_dir)
    checkpoint_interval = args.checkpoint_interval
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[CKPT] Checkpoint directory: {checkpoint_dir}")

    # ── config.json snapshot: durable record of how this run was launched ──
    try:
        import subprocess as _sp
        _git = _sp.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        _git = None
    with open(run_paths.config_json, "w") as _f:
        _json.dump({
            "run_id": run_id,
            "start_ts": datetime.now().isoformat(),
            "git_commit": _git,
            "cli_args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                         for k, v in vars(args).items()},
            "num_agents": num_agents,
            "communication_mode": communication,
        }, _f, indent=2)

    # ── Signal handler: gracefully save on SIGTERM / SIGINT ──
    import signal as _signal
    _shutdown_requested = False
    _shutdown_episode = 0
    _shutdown_step = 0

    def _handle_shutdown(signum, frame):
        nonlocal _shutdown_requested
        sig_name = "SIGTERM" if signum == _signal.SIGTERM else "SIGINT"
        print(f"\n[CKPT] {sig_name} received — will checkpoint at end of current step.")
        _shutdown_requested = True

    _signal.signal(_signal.SIGTERM, _handle_shutdown)
    _signal.signal(_signal.SIGINT, _handle_shutdown)

    # ── Resume: restore state from a previous checkpoint ──
    resume_episode = 0   # first episode to run (0-indexed)
    resume_step = 0      # unused currently — always restart episode from step 0
    if args.resume:
        print(f"[CKPT] Resuming from {args.resume}")
        restored = load_checkpoint(
            checkpoint_dir=args.resume,
            agents=agents,
            hebbian_graph=hebbian_graph,
            run_paths=run_paths,
        )
        resume_episode = restored["episode"]
        resume_step = restored["step"]
        run_id = restored["run_id"]
        metric = restored["metric"]
        rl_config.lora_save_dir = str(
            _resume_run_paths(run_id, group=args.run_group).root / "rl_live"
        )
        print(f"[CKPT] Resuming from episode {resume_episode} step {resume_step}")

    global_step = 0
    if args.resume:
        global_step = getattr(metric, "_global_step_ckpt", 0)

    # ── Choice-mode social-act counters (whole run; → final_metrics) ──
    _sa_metrics = {
        "act_counts": {},             # act -> n (incl. "none" and replay steps)
        "act_counts_by_agent": {},    # agent_N -> {act: n}
        "imitation_horizon_hist": {}, # str(h) -> n
        "imitation_requests": 0,
        "imitation_delivered_actions": 0,   # sum of delivered sequence lengths
        "imitation_adopted_steps": 0,       # steps where the next element was re-enacted
        "imitation_completed": 0,           # sequences adopted to the end
        "adopted_reward_sum": 0.0,
        "nonadopted_steps": 0,
        "nonadopted_reward_sum": 0.0,
        "cofire_pair_events": 0,
        "cofire_events_by_channel": {"spat": 0, "comm": 0, "obs": 0, "imit": 0},
    }

    def _sa_count(agent_name: str, act: str) -> None:
        _sa_metrics["act_counts"][act] = _sa_metrics["act_counts"].get(act, 0) + 1
        _by = _sa_metrics["act_counts_by_agent"].setdefault(agent_name, {})
        _by[act] = _by.get(act, 0) + 1

    for episode in range(resume_episode, num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        environment.reset()
        environment.reset_milestone_offset()
        environment.reset_anvil_coop_offset()
        environment.reset_death_offset()

        if episode > resume_episode:
            for _ag in agents:
                await _ag.on_reset(CancellationToken())

        # ── Orchestrator: memory horizon at the episode boundary ──
        # task variant: ledger/directives wiped every episode (the
        # experimental contrast with W(t)). social/plan variants: ledger AND
        # directives survive, matching W(t)'s cross-episode horizon; only
        # the episode-clock state resets. _orch_prev_chambers feeds the
        # chamber_change events; _orch_prev_tasks feeds the plan variant's
        # task-compliance log.
        _orch_prev_chambers: dict = {}
        _orch_prev_tasks: dict = {}
        if orchestrator_state is not None:
            # Only social/plan keep memory across episodes (W(t)'s horizon);
            # task AND villager start fresh — explicit variant set, not
            # `!= "task"`, so a new variant never inherits the wrong horizon.
            orchestrator_state.reset(
                keep_ledger=(orchestrator_config.variant
                             in ("social", "plan"))
            )
            if _orch_pair_acc is not None:
                _orch_pair_acc.clear()
            if villager_controller is not None:
                villager_controller.reset()

        import time as _time
        skip_warmup = (
            (args.resume is not None and args.resume_skip_warmup)
            or episode > 0
        )
        warmup_secs = 0 if skip_warmup else args.warmup_time
        max_warmup = 900  # hard cap: 15 min
        print(f"  * Waiting for media to load (min {warmup_secs}s, max {max_warmup}s)...")
        warmup_start = _time.time()
        all_loaded = False
        last_log_time = 0.0
        consecutive_loaded = 0  # require sustained signal across multiple checks
        while _time.time() - warmup_start < max_warmup:
            observations = environment.warmup_noop()
            elapsed = _time.time() - warmup_start

            # Compute per-client std-dev
            stds = []
            if observations:
                for obs in observations:
                    if obs is not None:
                        stds.append(np.std(obs.astype(np.float32)))
                    else:
                        stds.append(0.0)

            # Log progress every 30s
            if elapsed - last_log_time >= 30.0 and stds:
                std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds))
                print(f"    [{elapsed:.0f}s] std-dev: {std_str}  (>25 = loaded)")
                last_log_time = elapsed

            if elapsed >= warmup_secs and stds:
                if all(s > 25.0 for s in stds):
                    consecutive_loaded += 1
                    if consecutive_loaded >= 3:
                        all_loaded = True
                        break
                else:
                    consecutive_loaded = 0
            _time.sleep(2)
        elapsed = _time.time() - warmup_start
        if all_loaded:
            std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds))
            print(f"  * All clients loaded ({elapsed:.0f}s). std-dev: {std_str}")
        else:
            std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds)) if stds else "N/A"
            print(f"  * Warm-up timeout ({elapsed:.0f}s). std-dev: {std_str}. Starting anyway.")

        try:
            environment.signal_warmup_complete()
        except AttributeError:
            print("  * WARNING: env has no signal_warmup_complete; "
                  "Ch1 timeout may fire prematurely.")

        if args.start_chamber:
            # Start-chamber teleport: write the force-teleport flag for the
            # chamber BEFORE the start chamber (S=3 → ch2_force_teleport.txt →
            # Lua relocates everyone to their Ch3 cells). Runs every episode
            # because the Lua reset re-teleports all agents back to Ch1.
            # Placed after the warmup block (≥6s after reset) so the
            # post-reset retry in init.lua has already fired harmlessly.
            _start_force_fn = {
                2: environment.force_ch1_teleport,
                3: environment.force_ch2_teleport,
                4: environment.force_ch3_teleport,
                5: environment.force_ch4_teleport,
            }[args.start_chamber]
            if _start_force_fn():
                print(f"[START_CHAMBER] ep={episode+1}: wrote force-teleport "
                      f"flag -> agents start in ch{args.start_chamber}")
            else:
                print(f"[START_CHAMBER] ep={episode+1}: WARNING — flag write "
                      f"failed; agents will start in Ch1 this episode")

        # All communication is targeted: each agent has its own inbox.
        agent_communications = {i: [] for i in range(num_agents)}
        # Choice-mode observe/imitate payloads, delivered to the INITIATOR's
        # prompt next step (the observed/imitated party is never notified).
        agent_social_returns = {i: [] for i in range(num_agents)}
        # Guided imitation: one PendingImitation per agent, tracking the
        # delivered sequence for ADOPTION crediting (imit co-firing fires
        # when the agent re-enacts the next element, never automatically).
        _pending_imitations: dict = {}
        agents_error_count = [0] * num_agents
        comm_tracker = CommunicationTracker(
            agent_ids=list(range(num_agents)),
            reward_scale=args.comm_reward_scale,
        )
        # Act-reward symmetry suite: obs/imit acts paid like messages.
        act_reward_tracker = None
        if social_act_mode == "choice" and args.social_act_rewards:
            from env.social_act_rewards import SocialActRewardTracker
            act_reward_tracker = SocialActRewardTracker(
                agent_ids=list(range(num_agents)),
            )
        coop_metric = CooperationMetric(agent_ids=list(range(num_agents)))
        ep_logger = EpisodeLogger(run_dir=metric.target_folder, episode=episode + 1)
        _last_propagation_contribs: dict[int, dict[int, float]] = {
            i: {} for i in range(num_agents)
        }
        _last_milestone_sources: dict[int, str] = {}
        _interpretability_enabled = bool(
            args.interpretability or args.hebbian
        )
        _interp_path = (
            Path(metric.target_folder) / "interpretability.jsonl"
            if _interpretability_enabled else None
        )
        # Experiment 2 sidecars (choice mode only, so legacy run dirs are
        # byte-identical): per-act log + per-pair co-firing event log.
        _social_acts_path = (
            Path(metric.target_folder) / "social_acts.jsonl"
            if social_act_mode == "choice" else None
        )
        _cofiring_path = (
            Path(metric.target_folder) / "cofiring_events.jsonl"
            if (social_act_mode == "choice" and hebbian_config.enabled
                and hebbian_config.mode != "legacy")
            else None
        )

        def _append_jsonl(path, obj):
            try:
                with open(path, "a", encoding="utf-8") as _jf:
                    _jf.write(_json.dumps(obj) + "\n")
            except OSError as _jexc:
                logging.warning("sidecar write failed (%s): %s", path, _jexc)
        _visited_chambers = [set() for _ in range(num_agents)]
        _prev_step_actions = {i: "NoOp" for i in range(num_agents)}
        _prev_step_comms   = {i: "" for i in range(num_agents)}

        _bond_cache_version = -1
        _bond_strings_cached: dict = {}
        _bond_weights_cached: dict = {}
        _bond_deltas_cached: dict = {}

        import collections as _collections
        _mp4_writers: list = [None] * num_agents
        if save_gif:
            try:
                import imageio as _imageio_ep
                for _i in range(num_agents):
                    _ep_mp4_path = (
                        f"{gif_dir}/{run_id}_{role_configs[_i]['agent_name']}"
                        f"_ep{episode+1}.mp4"
                    )
                    os.makedirs(os.path.dirname(_ep_mp4_path), exist_ok=True)
                    _mp4_writers[_i] = _imageio_ep.get_writer(
                        _ep_mp4_path, fps=2, macro_block_size=1,
                    )
            except Exception as _exc:
                logging.warning(
                    "[GIF] streaming MP4 writer setup failed (%s); falling "
                    "back to deque-only capture (final MP4 will be from "
                    "the last gif_interval frames only).", _exc)
                _mp4_writers = [None] * num_agents

        # Rolling window — used for both intermediate GIF/MP4 checkpoints
        # and (as a fallback) the end-of-episode GIF. maxlen bounds RSS to
        # ~gif_interval × num_agents × frame-size (≈ 165 MB for 300×3 at
        # 320×180×3). `frames_list` name kept for save_checkpoint compat.
        _recent_window = max(gif_interval if gif_interval > 0 else 300, 1)
        frames_list: "_collections.deque" = _collections.deque(maxlen=_recent_window)

        def _save_gif_checkpoint(step_num):
            """Write a checkpoint GIF + MP4 per agent from the rolling window.

            Intermediate (mid-episode) checkpoint media go to
            intermediate_gif_dir — kept off the PRB share so the runs/
            tree stays small (the final per-episode MP4 lives in
            <run_dir>/gifs/ via the streaming writer above).

            The rolling window only carries the LAST gif_interval frames,
            so each checkpoint shows the most recent slice of the episode
            — perfect for "what was happening when this checkpoint fired?"
            and bounded in RAM.
            """
            _window_snapshot = list(frames_list)
            for i in range(num_agents):
                agent_frames = [
                    PIL.Image.fromarray(f[i])
                    for f in _window_snapshot if f[i] is not None
                ]
                if agent_frames:
                    gif_path = (
                        f"{intermediate_gif_dir}/{run_id}_"
                        f"{role_configs[i]['agent_name']}_ep{episode+1}"
                        f"_step{step_num}.gif"
                    )
                    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                    agent_frames[0].save(
                        gif_path,
                        format="GIF",
                        append_images=agent_frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )
                    print(f"  Saved GIF checkpoint: {gif_path}")
                    _frames_to_mp4(agent_frames, gif_path.replace(".gif", ".mp4"))

        _force_teleport_fired = {1: False, 2: False, 3: False, 4: False}

        _chamber_trigger_steps = compute_chamber_schedule(
            max_steps, args.start_chamber, args.max_chamber
        )
        _expected_chamber_after = {1: "ch1", 2: "ch2", 3: "ch3_cell", 4: "ch4"}
        _force_fn = {
            1: environment.force_ch1_teleport,
            2: environment.force_ch2_teleport,
            3: environment.force_ch3_teleport,
            4: environment.force_ch4_teleport,
        }

        # Social-replay snapshot state, scoped to one UPDATE CYCLE (not one
        # step): with --rl-update-stagger the agents' updates land on
        # consecutive steps, and all of them must sample the same pre-round
        # view. Taken lazily by the first updater of a cycle; cleared once
        # every agent that was expected to update has consumed it.
        _rl_replay_snapshot = None
        _rl_replay_pending = set()

        for step in range(max_steps):
            global_step += 1
            logging.info(f"ep={episode+1} step={step+1}/{max_steps} global_step={global_step}")

            for _from_ch in (1, 2, 3, 4):
                _trigger_step = _chamber_trigger_steps[_from_ch]
                if (_trigger_step is None
                        or _force_teleport_fired[_from_ch]
                        or step + 1 < _trigger_step):
                    continue
                try:
                    _trigger_chambers = [
                        environment.get_chamber(_i) for _i in range(num_agents)
                    ]
                except Exception:
                    _trigger_chambers = []
                _src_label = _expected_chamber_after[_from_ch]
                _n_advanced = sum(
                    1 for _c in _trigger_chambers
                    if _c is not None and _c != _src_label and (
                        # Ch3 has two sub-labels (cell/communal) — both count as still-in-Ch3
                        _from_ch != 3 or not str(_c).startswith("ch3")
                    )
                )
                if _force_fn[_from_ch]():
                    _force_teleport_fired[_from_ch] = True
                    if _n_advanced == 0:
                        _verb = "RESCUE"
                    elif _n_advanced >= num_agents:
                        _verb = "NUDGE"
                    else:
                        _verb = "REGROUP"
                    print(f"[CH{_from_ch}_TIMEOUT] {_verb} at ep={episode+1} "
                          f"step={step+1} "
                          f"(threshold={_trigger_step}"
                          f"≈{int(100 * _trigger_step / max_steps)}%, "
                          f"n_advanced={_n_advanced}, "
                          f"chambers={_trigger_chambers})")

            if step % log_interval == 0:
                _wb.log({
                    "step/ep": episode + 1,
                    "step/step_in_ep": step + 1,
                    **{
                        f"step/episode_return/agent_{i}": float(metric.episode_returns[i])
                        for i in range(num_agents)
                    },
                }, step=global_step)
                returns_str = "  ".join(
                    f"agent_{i}={metric.episode_returns[i]:.1f}"
                    for i in range(num_agents)
                )
                tasks_str = "  ".join(
                    f"agent_{i}={agents[i].auto_curriculum.current_task or 'None'!r}"
                    for i in range(num_agents)
                )
                # Per-agent chamber so the SLURM .out shows where everyone is.
                chambers_str = "  ".join(
                    f"agent_{i}={environment.get_chamber(i) or '?'}"
                    for i in range(num_agents)
                )
                print(
                    f"[{run_id}] ep={episode+1} step={step+1}/{max_steps} | "
                    f"chambers: {chambers_str} | "
                    f"returns: {returns_str} | "
                    f"tasks: {tasks_str}"
                )

            if environment.all_done():
                print(f"  All agents done at step {step+1}")
                break

            if environment.episode_over():
                print(f"  Team wipe (all agents dead in Ch5) at step {step+1} — ending episode")
                break

            current_frames = []
            for i in range(num_agents):
                frame = environment.get_agent_frame(i)
                if frame is None:
                    frame = np.zeros((obs_height, obs_width, 3), dtype=np.uint8)
                current_frames.append(frame)
                if save_gif and _mp4_writers[i] is not None:
                    try:
                        _mp4_writers[i].append_data(frame)
                    except Exception as _exc:
                        logging.warning(
                            "[GIF] streaming MP4 append failed for agent %d "
                            "step %d: %s", i, step + 1, _exc)
            frames_list.append(current_frames)

            if save_gif and gif_interval > 0 and (step + 1) % gif_interval == 0:
                _save_gif_checkpoint(step + 1)

            step_comm_count = 0
            step_rewards_raw = [0.0] * num_agents
            _step_envstep_reward = [0.0] * num_agents   # environment.get_step_reward
            _step_pitch_penalty  = [0.0] * num_agents   # negative; pitch-cap futile
            _step_milestone_drain = [0.0] * num_agents  # poll_milestone_events drain
            _step_death_drain    = [0.0] * num_agents   # negative; poll_death_events drain
            step_contents = [None] * num_agents
            comm_events = []
            # Choice-mode channel-tagged social events (initiator, target,
            # channel) — obs/imit only; comm rides comm_events as before.
            social_events = []
            _adopted_agents_this_step: set = set()
            # (agent_id, "obs"|"imit", rescued) — for SocialActRewardTracker.
            _act_events_this_step: list = []
            _messages_this_step = []
            _milestone_events_this_step: list = []
            _bond_strings = {}
            _bond_weights: dict[int, dict[str, float]] = {}
            _bond_deltas: dict[int, dict[str, float]] = {}
            if hebbian_config.enabled:
                _current_version = int(getattr(hebbian_graph, "_step_count", 0))
                if _current_version != _bond_cache_version:
                    for i in range(num_agents):
                        parts = []
                        for j in range(num_agents):
                            if j == i:
                                continue
                            raw_w = hebbian_graph.get_weight(i, j)
                            role_j = ROLE_NAMES[j % len(ROLE_NAMES)]
                            parts.append(f"agent_{j} ({role_j}): {raw_w:.2f}")
                        _bond_strings[i] = "Social bonds: " + ", ".join(parts)
                        # Also build the structured (weight, delta) dicts for the
                        # SocialModule. Keys are "agent_N" strings so they survive
                        # JSON round-trips and match the comm-target format the
                        # LLM emits.
                        delta_row = hebbian_graph.bond_delta_row(i)
                        _bond_weights[i] = {
                            f"agent_{j}": float(hebbian_graph.get_weight(i, j))
                            for j in range(num_agents) if j != i
                        }
                        _bond_deltas[i] = {
                            f"agent_{j}": float(delta_row.get(j, 0.0))
                            for j in range(num_agents) if j != i
                        }
                    _bond_cache_version = _current_version
                    _bond_strings_cached = dict(_bond_strings)
                    _bond_weights_cached = {k: dict(v) for k, v in _bond_weights.items()}
                    _bond_deltas_cached  = {k: dict(v) for k, v in _bond_deltas.items()}
                else:
                    # Bonds unchanged → reuse the cached dicts.
                    _bond_strings = dict(_bond_strings_cached)
                    _bond_weights = {k: dict(v) for k, v in _bond_weights_cached.items()}
                    _bond_deltas  = {k: dict(v) for k, v in _bond_deltas_cached.items()}

            # ── Phase B+ §1: per-teammate reward-propagation prompt lines ──
            # Disabled — the rlvr.reward_propagation module was removed.
            # Keep an empty dict so callers still find a value to look up;
            # they pass it through to the LLM prompt as an empty string.
            _propagation_strings: dict[int, str] = {}

            # Update each agent's visited-chambers set BEFORE the action loop,
            # so the prompt the LLM sees this step reflects its full history
            # (current chamber included).
            for _i in range(num_agents):
                _ch = environment.get_chamber(_i)
                if _ch:
                    _visited_chambers[_i].add(_ch)

            # ── Orchestrator (O2): chamber events, scheduled call, and
            # per-agent directive text for this step's action prompts ──
            _orch_directives_text: dict = {}
            _orch_plan_notes: dict = {}
            _orch_assigned_objectives: dict = {}
            if orchestrator_state is not None:
                _orch_new_chambers = set()
                for _i in range(num_agents):
                    _ch = environment.get_chamber(_i)
                    if _ch and _ch != _orch_prev_chambers.get(_i):
                        if (_orch_prev_chambers.get(_i) is not None
                                and _ch not in _orch_new_chambers):
                            _orch_new_chambers.add(_ch)
                            orchestrator_state.add_event(
                                _orch_events.chamber_change_event(step, _ch)
                            )
                        _orch_prev_chambers[_i] = _ch
                _orch_living = [
                    f"agent_{_i}" for _i in range(num_agents)
                    if not environment._terminations.get(f"agent_{_i}", False)
                ]
                if orchestrator_config.variant == "villager":
                    # ── Villager: event-driven controller tick ──
                    # Deterministic scheduling every step (event drain,
                    # timeouts, cascades); LLM decompose/allocate calls only
                    # when due. Short-circuits the cadence-based should_call
                    # path entirely. tick() also runs on a team wipe (fails
                    # running nodes, no LLM calls).
                    try:
                        _v_tick = await villager_controller.tick(
                            state=orchestrator_state,
                            living_agents=_orch_living,
                            episode=episode + 1, t=step,
                            environment=environment, agents=agents,
                            metric=metric,
                        )
                        # HARD enforcement: a reassigned agent's curriculum
                        # replans immediately under the new objective (the
                        # _initialized guard makes clearing current_task a
                        # pure replan, never a DB wipe).
                        for _nm in _v_tick.reassigned:
                            try:
                                _ri = int(str(_nm).rsplit("_", 1)[-1])
                            except (ValueError, IndexError):
                                continue
                            if 0 <= _ri < num_agents:
                                agents[_ri].auto_curriculum.current_task = None
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as _orch_exc:
                        logging.error(
                            "Villager tick crashed at ep=%d step=%d: %s "
                            "— keeping previous assignments",
                            episode + 1, step, _orch_exc,
                        )
                    for _i in range(num_agents):
                        _orch_directives_text[_i] = (
                            villager_controller.directive_text(f"agent_{_i}")
                        )
                        _orch_assigned_objectives[_i] = (
                            villager_controller.assigned_objective(
                                f"agent_{_i}")
                        )
                elif _orch_living and _orch_core.should_call(
                        orchestrator_state, step, orchestrator_config):
                    if orchestrator_config.variant == "task":
                        _orch_recent_msgs = [
                            (_ev.get("sender"), _ev.get("target"))
                            for _ev in orchestrator_state.event_buffer
                            if _ev.get("type") == "message"
                        ]
                        _orch_env_state = _orch_core.collect_env_state(
                            environment, num_agents, step,
                            recent_messages=_orch_recent_msgs,
                        )
                    else:
                        # social/plan: the Hebbian-rule-matched inputs only.
                        _orch_env_state = {
                            "pair_digest": _orch_pair_acc.render(),
                            "task_table": (
                                _orch_core.collect_task_table(
                                    agents, num_agents)
                                if orchestrator_config.variant == "plan"
                                else None
                            ),
                        }
                    _orch_fails_before = orchestrator_state.failed_calls
                    try:
                        await _orch_core.orchestrate(
                            orchestrator_state, _orch_env_state,
                            orchestrator_client, orchestrator_config,
                            living_agents=_orch_living,
                            episode=episode + 1, t=step,
                            orch_logger=orch_logger,
                            num_agents=num_agents,
                        )
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as _orch_exc:
                        logging.error(
                            "Orchestrator call crashed at ep=%d step=%d: %s "
                            "— keeping previous directives",
                            episode + 1, step, _orch_exc,
                        )
                    # Mirror the event-buffer policy: a successful call
                    # consumed the pair window; a failed one keeps it so the
                    # next call still sees those signals.
                    if (_orch_pair_acc is not None
                            and orchestrator_state.failed_calls
                            == _orch_fails_before):
                        _orch_pair_acc.clear()
                if orchestrator_config.variant != "villager":
                    # (villager filled its directives inside its own branch)
                    for _i in range(num_agents):
                        if orchestrator_config.variant == "task":
                            _orch_directives_text[_i] = (
                                _orch_core.render_agent_directive(
                                    f"agent_{_i}", orchestrator_state
                                )
                            )
                        else:
                            _orch_directives_text[_i] = (
                                _orch_core.render_social_directive(
                                    f"agent_{_i}", orchestrator_state
                                )
                            )
                            if orchestrator_config.variant == "plan":
                                _orch_plan_notes[_i] = (
                                    _orch_core.plan_note_for(
                                        orchestrator_state, f"agent_{_i}"
                                    )
                                )

            # ── Phase 0: encode the PRE-step joint state (s_t) ──
            # Computed BEFORE any agent acts so V_global represents V(s_t),
            # not V(s_{t+1}). Each agent's pending transition gets this V_global
            # attached right after its select_action runs (inside the per-agent
            # loop below). The centralised critic's store_step at the end of
            # the step also references these pre-step values, so its training
            # buffer is aligned on s_t too.
            joint_state_t = None
            v_global_t = 0.0
            if centralized_critic is not None:
                _pre_positions = {}
                _pre_chambers  = {}
                _pre_hps       = {}
                _pre_inv       = {}
                for _i in range(num_agents):
                    _pre_positions[_i] = environment.get_agent_position(_i)
                    _pre_chambers[_i]  = environment.get_chamber(_i)
                    _status = environment.get_player_status_text(_i) or ""
                    _hp = 20.0
                    if "Health:" in _status:
                        try:
                            _hp = float(_status.split("Health:")[1].split("/")[0].strip())
                        except (ValueError, IndexError):
                            _hp = 20.0
                    _pre_hps[_i] = _hp
                    _pre_inv[_i] = environment.pickedup_object(agentId=_i) or ""
                # raw_rewards at s_t = 0 by definition (rewards are earned by
                # the transition out of s_t, not in s_t itself). Pass an empty
                # dict so the encoder fills zeros.
                joint_state_t = centralized_critic.encode_joint(
                    positions=_pre_positions,
                    chambers=_pre_chambers,
                    hps=_pre_hps,
                    inventories=_pre_inv,
                    milestones_by_agent=metric._agent_milestones,
                    raw_rewards={i: 0.0 for i in range(num_agents)},
                    last_actions=_prev_step_actions,
                    last_comms=_prev_step_comms,
                )
                v_global_t = centralized_critic.evaluate(joint_state_t)

            # ── Simultaneous-move selection (Stage 1: LLM-only, no macros) ──
            # All living agents choose actions CONCURRENTLY on the shared
            # pre-step state s_t, then a single step_all() advances the env
            # once for everyone. The turn-based loop below selects+steps one
            # agent at a time instead. Comm routing, Hebbian and metric
            # bookkeeping run in the shared loop body either way — only
            # selection+stepping moves up here.
            _sim_contents: dict = {}
            if args.simultaneous:
                _sim_alive = [
                    _i for _i in range(num_agents)
                    if not environment._terminations.get(f"agent_{_i}", False)
                ]

                async def _sim_select(_i):
                    _ag = agents[_i]
                    try:
                        _frame = environment.get_pil_image(_i)
                        _formatted = [
                            f"{m.source}: {m.content}"
                            for m in agent_communications[_i]
                            if m.source != _ag.name
                        ]
                        _mm = MultiModalMessage(
                            content=[
                                f"Communications from other agents: {_formatted}.\n",
                                Image.from_pil(_frame),
                            ],
                            source="user",
                        )
                        _ag_done = metric._agent_milestones.get(f"agent_{_i}", set())
                        _tm_done = (
                            set().union(*metric._agent_milestones.values())
                            if metric._agent_milestones else set()
                        )
                        _mp = format_milestone_progress(
                            environment.get_chamber(_i), _ag_done, _tm_done
                        )
                        _content, _ = await _ag.on_messages(
                            [_mm], CancellationToken(),
                            communication=_formatted,
                            error=None, error_count=agents_error_count[_i],
                            picked_object=environment.pickedup_object(agentId=_i),
                            reward_text=environment.get_reward_summary(_i),
                            social_bonds=_bond_strings.get(_i),
                            propagation_summary=_propagation_strings.get(_i, ""),
                            position_text=environment.get_position_text(_i),
                            player_status_text=environment.get_player_status_text(_i),
                            current_chamber=environment.get_chamber(_i),
                            visited_chambers=sorted(_visited_chambers[_i]),
                            completed_milestones=_ag_done,
                            milestone_progress=_mp,
                            chamber_state=(
                                environment.get_chamber_state(_i)
                                + (
                                    "\n" + environment.get_voxel_summary(_i)
                                    if args.voxel_obs
                                    and environment.get_voxel_summary(_i)
                                    else ""
                                )
                            ),
                            bond_weights=_bond_weights.get(_i),
                            bond_deltas=_bond_deltas.get(_i),
                            social_returns=(
                                "\n".join(agent_social_returns[_i])
                                if agent_social_returns.get(_i) else ""
                            ),
                            orchestrator_directive=(
                                _orch_directives_text.get(_i)
                            ),
                            orchestrator_plan_note=(
                                _orch_plan_notes.get(_i)
                            ),
                            orchestrator_assigned_objective=(
                                _orch_assigned_objectives.get(_i)
                            ),
                        )
                        return _i, _content
                    except Exception as _exc:
                        logging.error(
                            "simultaneous select failed for agent %d: %s", _i, _exc
                        )
                        return _i, {"action": "NoOp", "thoughts": "", "communication": ""}

                # SEQUENTIAL, not asyncio.gather: the agents still all decide
                # on the shared pre-step state s_t before step_all() runs, so
                # simultaneous-move semantics hold — but interleaving their
                # on_messages coroutines corrupts the shared in-process model/
                # tokenizer state and yields NaN logits (CUDA assert). The
                # in-process LLM serializes on one GPU regardless, so this
                # costs no speed and matches the turn-based order that works.
                _sim_results = []
                for _i in _sim_alive:
                    _sim_results.append(await _sim_select(_i))
                _sim_actions = {}
                for _i, _content in _sim_results:
                    _sim_contents[_i] = _content
                    _sim_actions[_i] = (
                        _content.get("action", "NoOp") if _content else "NoOp"
                    )
                environment.step_all(_sim_actions)
                # step_all produced ONE _step_rewards set — drain it once
                # (turn-based drains once per agent inside its loop instead).
                for _i in range(num_agents):
                    _r_env = environment.get_step_reward(_i)
                    step_rewards_raw[_i] += _r_env
                    _step_envstep_reward[_i] += _r_env

            for agent_id, agent in enumerate(agents):
                agent_name = f"agent_{agent_id}"

                if environment._terminations.get(agent_name, False):
                    continue

                error_count = agents_error_count[agent_id]
                frame_image = environment.get_pil_image(agent_id)
                reward_text = environment.get_reward_summary(agent_id)

                comms_for_agent = agent_communications[agent_id]
                # Milestone-progress block: per-agent done + team done +
                # still-open per chamber. Plumbed to both the curriculum LLM
                # (task selection) and the action LLM (action selection)
                # via beliefs so neither has to re-derive what's still left.
                _agent_done = metric._agent_milestones.get(
                    f"agent_{agent_id}", set()
                )
                _team_done = set().union(
                    *metric._agent_milestones.values()
                ) if metric._agent_milestones else set()
                _milestone_progress = format_milestone_progress(
                    environment.get_chamber(agent_id),
                    _agent_done,
                    _team_done,
                )
                if args.simultaneous:
                    # Action was chosen concurrently and applied via step_all()
                    # in the pre-loop block; rewards were drained once there.
                    # Just consume the pre-computed content for this agent.
                    content = _sim_contents.get(agent_id)
                    last_action = environment._last_actions.get(
                        f"agent_{agent_id}", "NoOp"
                    )
                    step_contents[agent_id] = content
                else:
                    content, last_action, error_count = await agent_do_action(
                        agent, agent_id, frame_image, comms_for_agent, reward_text,
                        environment,
                        error_count=error_count,
                        social_bonds=_bond_strings.get(agent_id),
                        propagation_summary=_propagation_strings.get(agent_id, ""),
                        position_text=environment.get_position_text(agent_id),
                        player_status_text=environment.get_player_status_text(agent_id),
                        current_chamber=environment.get_chamber(agent_id),
                        visited_chambers=sorted(_visited_chambers[agent_id]),
                        completed_milestones=_agent_done,
                        milestone_progress=_milestone_progress,
                        chamber_state=(
                            environment.get_chamber_state(agent_id)
                            + (
                                "\n" + environment.get_voxel_summary(agent_id)
                                if args.voxel_obs
                                and environment.get_voxel_summary(agent_id)
                                else ""
                            )
                        ),
                        bond_weights=_bond_weights.get(agent_id),
                        bond_deltas=_bond_deltas.get(agent_id),
                        social_returns=(
                            "\n".join(agent_social_returns[agent_id])
                            if agent_social_returns.get(agent_id) else ""
                        ),
                        orchestrator_directive=(
                            _orch_directives_text.get(agent_id)
                        ),
                        orchestrator_plan_note=(
                            _orch_plan_notes.get(agent_id)
                        ),
                        orchestrator_assigned_objective=(
                            _orch_assigned_objectives.get(agent_id)
                        ),
                    )
                    agents_error_count[agent_id] = error_count
                    for _i in range(num_agents):
                        _r_env = environment.get_step_reward(_i)
                        step_rewards_raw[_i] += _r_env
                        _step_envstep_reward[_i] += _r_env
                    step_contents[agent_id] = content

                # Attach the PRE-step V_global(s_t) to the transition that
                # select_action just opened, so old_value_global is V(s_t) and
                # not V(s_{t+1}) (correct credit assignment). Safe no-op if no
                # transition is pending (e.g. agent terminated before its
                # action). For macro-continuation ticks this branch does not run
                # because the macro-skip path early-continues before agent_do_action.
                if (
                    centralized_critic is not None
                    and joint_state_t is not None
                    and agent.rl_layer is not None
                    and agent.rl_layer.enabled
                ):
                    agent.rl_layer.set_pending_value_global(v_global_t, joint_state_t)

                # ── Choice-mode act normalization + mutual exclusivity ──
                # Normalize the LLM's act against the MENU (the affordance
                # ablation is enforced in code, not trusted to the prompt)
                # and blank the comm fields unless the act IS communicate —
                # at most one social act per step.
                if social_act_mode == "choice" and content:
                    _act_norm = _sacts.normalize_social_act(
                        content.get("social_act"), _social_menu_channels
                    )
                    content["social_act"] = _act_norm
                    if _act_norm != "communicate":
                        content["communication"] = ""
                        content["communication_target"] = ""

                # Handle communication (collect comm_events for Hebbian)
                if (
                    content
                    and content.get("communication")
                    and content["communication"] not in ("", "None")
                    and communication
                ):
                    msg_text = content["communication"]
                    comm_target = content.get("communication_target") or "all"
                    # Annotate the wire message with the sender's chamber at
                    # send time. The receiver sees "[in ch3] <text>" so it can
                    # ground claims like "I'm pressing switch B" against the
                    # actual chamber, instead of inferring location from
                    # screenshot artifacts (a frequent hallucination source).
                    _sender_chamber = environment.get_chamber(agent_id) or "?"
                    _wire_content = f"[in {_sender_chamber}] {msg_text}"
                    message = TextMessage(content=_wire_content, source=agent.name)
                    metric.record_communication(agent.name, msg_text, target=comm_target)
                    step_comm_count += 1

                    try:
                        sender_idx = int(agent.name.split("_")[1])
                    except (IndexError, ValueError):
                        sender_idx = -1

                    if sender_idx < 0:
                        continue  # malformed agent name; drop the message

                    # Resolve recipient from communication_target. Tolerant
                    # parser: accept "agent_N", "agentN", "Agent_N" — the LLM
                    # sometimes drops the underscore or capitalises. We
                    # categorise the failure mode (self / out-of-range /
                    # malformed) so the routing_source field tells us *why*
                    # the model output had to be rescued, not just that it did.
                    import re as _re
                    recv_idx = -1
                    routing_source = "model"
                    target_failure = None  # set when the model output is rejected
                    canonical_target = None  # normalised "agent_N" we parsed out
                    if not comm_target or comm_target == "all":
                        target_failure = "missing_or_all"
                    else:
                        _m = _re.match(r"^\s*agent_?(\d+)\s*$",
                                       str(comm_target).lower())
                        if not _m:
                            target_failure = "unparseable"
                        else:
                            try:
                                cand = int(_m.group(1))
                                canonical_target = f"agent_{cand}"
                                if cand == sender_idx:
                                    target_failure = "self_target"
                                elif not (0 <= cand < num_agents):
                                    target_failure = "out_of_range"
                                else:
                                    recv_idx = cand
                            except (IndexError, ValueError):
                                target_failure = "unparseable"

                    if recv_idx < 0:
                        # Model returned a bad target despite the prompt
                        # naming its valid teammates. Reroute via Hebbian
                        # bond / random and tag routing_source with the
                        # *reason* so we can track whether prompt fixes
                        # actually reduce self-targeting over time.
                        if hebbian_graph.config.enabled:
                            candidates = [
                                (j, float(hebbian_graph.W[sender_idx, j]))
                                for j in range(num_agents) if j != sender_idx
                            ]
                            recv_idx = max(candidates, key=lambda x: x[1])[0]
                            routing_source = f"hebbian_fallback:{target_failure}"
                        else:
                            others = [j for j in range(num_agents) if j != sender_idx]
                            recv_idx = random.choice(others)
                            routing_source = f"random_fallback:{target_failure}"

                    # ── Social-module bias coupling ──
                    # When --social-module=bias, the sender's SocialModule's
                    # ask_target wins over whatever the action LLM emitted.
                    # This makes the social directive a hard guarantee on
                    # routing: the bond-weighted pick the social step made
                    # IS where the message goes. Skipped in 'prompt' mode
                    # (directive stays as prompt-text only) and 'none'.
                    if args.social_module == "bias":
                        _sender_sm = getattr(agent, "social_module", None)
                        _thought = (
                            _sender_sm.last_thought if _sender_sm else None
                        )
                        _sm_target = _thought.get("ask_target") if _thought else None
                        if _sm_target:
                            _m2 = _re.match(
                                r"^\s*agent_?(\d+)\s*$", str(_sm_target).lower()
                            )
                            if _m2:
                                try:
                                    _sm_idx = int(_m2.group(1))
                                    if (
                                        _sm_idx != sender_idx
                                        and 0 <= _sm_idx < num_agents
                                    ):
                                        recv_idx = _sm_idx
                                        routing_source = "social_bias"
                                except (IndexError, ValueError):
                                    pass

                    # ── Orchestrator bias coupling (O2) ──
                    # Mirrors the social-module bias exactly: in "bias" mode
                    # the orchestrator's standing comm_target for the sender
                    # overrides whatever the action LLM emitted, making the
                    # directive a hard guarantee on routing. Skipped in
                    # "advisory" mode (directive stays prompt-text only).
                    if (orchestrator_state is not None
                            and args.orchestrator_mode == "bias"):
                        _o_target = _orch_core.directive_comm_target(
                            orchestrator_state, agent.name
                        )
                        if _o_target:
                            _m3 = _re.match(
                                r"^\s*agent_?(\d+)\s*$", str(_o_target).lower()
                            )
                            if _m3:
                                try:
                                    _o_idx = int(_m3.group(1))
                                    if (
                                        _o_idx != sender_idx
                                        and 0 <= _o_idx < num_agents
                                    ):
                                        recv_idx = _o_idx
                                        routing_source = "orchestrator_bias"
                                except (IndexError, ValueError):
                                    pass

                    agent_communications[recv_idx].append(message)
                    if len(agent_communications[recv_idx]) > num_agents - 1:
                        agent_communications[recv_idx].pop(0)
                    comm_events.append((sender_idx, recv_idx))
                    # Stash per-message metadata; rewards are stamped in Phase 1b
                    # once CommunicationTracker has processed the step.
                    _messages_this_step.append({
                        "t": step,
                        "sender": f"agent_{sender_idx}",
                        "receiver": f"agent_{recv_idx}",
                        "text": msg_text,
                        "tokens": len(msg_text.split()),
                        "routing": routing_source,
                        "model_target": comm_target,
                        "model_target_canonical": canonical_target,
                    })

                # ── Choice-mode social-act router (Experiment 2) ──────────
                # observe → deliver the target's state+beliefs to the
                # INITIATOR next step (silent, directed). imitate (GUIDED,
                # 2026-08-07) → deliver the target's last-h actions + state
                # with judge-then-re-enact instructions; imit co-firing is
                # credited on ADOPTION — each step the agent's own chosen
                # action matches the next element of the delivered sequence.
                if social_act_mode == "choice":
                    import re as _re2
                    # This agent's select consumed its inbox this step.
                    agent_social_returns[agent_id].clear()

                    try:
                        _init_idx = int(agent.name.split("_")[1])
                    except (IndexError, ValueError):
                        _init_idx = -1

                    # Adoption check runs EVERY step, on the sequence
                    # delivered earlier, before any fresh act (a fresh
                    # imitate below replaces the pending one afterwards).
                    _pend = _pending_imitations.get(agent_id)
                    if _pend is not None and content is not None:
                        if _pend.expired(step):
                            if _social_acts_path:
                                _append_jsonl(_social_acts_path, {
                                    "ep": episode + 1, "step": step,
                                    "agent": agent.name, "act": "imitate",
                                    "gate": "expired",
                                    "adopted": _pend.ptr,
                                    "of": len(_pend.sequence),
                                })
                            _pending_imitations.pop(agent_id, None)
                        elif _pend.note_action(content.get("action")):
                            social_events.append(
                                (_init_idx, _pend.target, "imit")
                            )
                            if args.social_bidirectional:
                                # Delivery-symmetric channel: tell the target
                                # its behavior was adopted (mirrors a message
                                # landing in the recipient's inbox).
                                agent_social_returns[_pend.target].append(
                                    _sacts.render_imitated_notice(
                                        agent.name, content.get("action")
                                    )
                                )
                            _adopted_agents_this_step.add(_init_idx)
                            _sa_metrics["imitation_adopted_steps"] += 1
                            if _social_acts_path:
                                _append_jsonl(_social_acts_path, {
                                    "ep": episode + 1, "step": step,
                                    "agent": agent.name, "act": "imitate",
                                    "gate": "adopted",
                                    "target": f"agent_{_pend.target}",
                                    "adopted_step": _pend.ptr,
                                    "of": len(_pend.sequence),
                                })
                            if _pend.done():
                                _sa_metrics["imitation_completed"] += 1
                                _pending_imitations.pop(agent_id, None)

                    if content and _init_idx >= 0:
                            _act = content.get("social_act") or "none"
                            _sa_count(agent.name, _act)
                            _raw_target = content.get("social_target") or ""
                            _t_idx = -1
                            _sa_routing = "model"
                            if _act in ("observe", "imitate"):
                                _m4 = _re2.match(r"^\s*agent_?(\d+)\s*$",
                                                 str(_raw_target).lower())
                                if _m4:
                                    _cand = int(_m4.group(1))
                                    if (_cand != _init_idx
                                            and 0 <= _cand < num_agents):
                                        _t_idx = _cand
                                if _t_idx < 0:
                                    # Same rescue as comm routing: highest
                                    # Hebbian bond, else random.
                                    if hebbian_graph.config.enabled:
                                        _cands = [
                                            (j, float(hebbian_graph.W[_init_idx, j]))
                                            for j in range(num_agents)
                                            if j != _init_idx
                                        ]
                                        _t_idx = max(_cands, key=lambda x: x[1])[0]
                                        _sa_routing = "hebbian_fallback"
                                    else:
                                        _t_idx = random.choice([
                                            j for j in range(num_agents)
                                            if j != _init_idx
                                        ])
                                        _sa_routing = "random_fallback"

                            if _act == "observe":
                                # Target's ground state + its OWN beliefs.
                                # partner_beliefs is deliberately never read
                                # (conversation-index keyed — wrong teammate).
                                _tb = agents[_t_idx].belief_system
                                agent_social_returns[_init_idx].append(
                                    f"[observation of agent_{_t_idx}] "
                                    f"Position: {environment.get_position_text(_t_idx) or 'Unknown'} | "
                                    f"Chamber: {environment.get_chamber(_t_idx) or '?'} | "
                                    f"Status: {environment.get_player_status_text(_t_idx) or '?'} | "
                                    f"Inventory: {environment.pickedup_object(agentId=_t_idx) or '(none)'}\n"
                                    f"  Their beliefs — See now: {_tb.perception_beliefs or '(none)'} | "
                                    f"Task: {_tb.task_beliefs or '(none)'} | "
                                    f"Interactions: {_tb.interaction_beliefs or '(none)'}"
                                )
                                social_events.append((_init_idx, _t_idx, "obs"))
                                if args.social_bidirectional:
                                    # Delivery-symmetric channel: tell the
                                    # target who observed it.
                                    agent_social_returns[_t_idx].append(
                                        _sacts.render_observed_notice(agent.name)
                                    )
                                if act_reward_tracker is not None:
                                    _act_events_this_step.append(
                                        (_init_idx, "obs", _sa_routing != "model")
                                    )
                                if _social_acts_path:
                                    _append_jsonl(_social_acts_path, {
                                        "ep": episode + 1, "step": step,
                                        "agent": agent.name, "act": "observe",
                                        "target": f"agent_{_t_idx}",
                                        "routing_source": _sa_routing,
                                    })
                            elif _act == "imitate":
                                _h = _sacts.clamp_horizon(
                                    content.get("imitate_horizon")
                                )
                                _sa_metrics["imitation_requests"] += 1
                                _hh = str(_h)
                                _sa_metrics["imitation_horizon_hist"][_hh] = (
                                    _sa_metrics["imitation_horizon_hist"].get(_hh, 0) + 1
                                )
                                # GUIDED imitation: deliver the target's
                                # last-h actions + state with the judge-
                                # then-re-enact instructions. Nothing is
                                # auto-executed; credit comes from the
                                # adoption check above on later steps.
                                _t_log = list(
                                    getattr(agents[_t_idx], "_step_log", []) or []
                                )[-_h:]
                                _t_actions = [a for a, _r in _t_log if a]
                                agent_social_returns[_init_idx].append(
                                    _sacts.render_imitation_payload(
                                        f"agent_{_t_idx}", _t_actions,
                                        agents[_t_idx].auto_curriculum.current_task,
                                        environment.get_position_text(_t_idx),
                                        environment.get_chamber(_t_idx),
                                    )
                                )
                                _sa_metrics["imitation_delivered_actions"] += (
                                    len(_t_actions)
                                )
                                if _t_actions:
                                    _pending_imitations[agent_id] = (
                                        _sacts.PendingImitation(
                                            _t_idx, _t_actions, step,
                                        )
                                    )
                                    if args.social_bidirectional:
                                        # Notice #1 (trigger): the target
                                        # learns it is being studied. Notice
                                        # #2 (adoption, above) fires only if
                                        # the imitator re-enacts — Hebbian
                                        # credit stays on adoption only.
                                        agent_social_returns[_t_idx].append(
                                            _sacts.render_imitation_started_notice(
                                                agent.name, len(_t_actions)
                                            )
                                        )
                                    # Paid at delivery — symmetric to a
                                    # message being paid when sent.
                                    if act_reward_tracker is not None:
                                        _act_events_this_step.append(
                                            (_init_idx, "imit",
                                             _sa_routing != "model")
                                        )
                                if _social_acts_path:
                                    _append_jsonl(_social_acts_path, {
                                        "ep": episode + 1, "step": step,
                                        "agent": agent.name, "act": "imitate",
                                        "target": f"agent_{_t_idx}",
                                        "horizon": _h, "gate": "delivered",
                                        "n_actions": len(_t_actions),
                                        "routing_source": _sa_routing,
                                    })
                            else:
                                # "communicate" (message handled by the comm
                                # block above) or "none" — log for act-mix.
                                if _social_acts_path:
                                    _append_jsonl(_social_acts_path, {
                                        "ep": episode + 1, "step": step,
                                        "agent": agent.name, "act": _act,
                                        "target": (
                                            content.get("communication_target")
                                            or None
                                        ),
                                    })

                time.sleep(sleep_time)

            # ── Phase 1b: Communication rewards + cooperation metrics + step logging ──
            positions = []
            for i in range(num_agents):
                try:
                    pos = environment.env.env._positions[i]
                except (AttributeError, IndexError):
                    pos = None
                positions.append(pos)

            # Capture task rewards before comm bonus is added (for decomposition)
            _task_rewards_this_step = {i: step_rewards_raw[i] for i in range(num_agents)}
            _chat_this_step = {}
            _agent_pos_map = {}
            _actions_this_step = {}
            # Speakers whose model-emitted communication_target was self /
            # "all" / unparseable. Routing rescues their message but the
            # comm-reward layer will skip BASE_MSG_REWARD for them so the
            # policy doesn't get +0.5 for talking to itself.
            _bad_target_speakers: set = set()
            for _msg_meta in _messages_this_step:
                if _msg_meta.get("routing", "model").startswith(
                        ("hebbian_fallback:", "random_fallback:")):
                    try:
                        _bad_target_speakers.add(
                            int(_msg_meta["sender"].split("_")[-1]))
                    except (KeyError, ValueError, IndexError):
                        pass
            for _i in range(num_agents):
                _c = step_contents[_i]
                msg = _c.get("communication", "") if _c else ""
                if msg and msg != "None":
                    _chat_this_step[_i] = msg
                _actions_this_step[_i] = (_c.get("action", "NoOp") if _c else "NoOp")
                _env_pos = environment.get_agent_position(_i)
                _agent_pos_map[_i] = _env_pos if _env_pos is not None else positions[_i]

            # ── Pitch-cap "futile" penalty ───────────────────────────────
            # Action-repetition penalty was REMOVED — too harsh; even at
            # -0.5/step it crushed cumulative returns deeper than milestone
            # spikes could compensate, and the policy learned "every action
            # is bad" rather than "this specific repeat is bad". Loops
            # break naturally now that hunger doesn't kill, the curriculum
            # only proposes chamber-feasible tasks, and Ch2 is reachable.
            #
            # The pitch-cap futile penalty is kept (it's narrower — only
            # fires when env.step() redirects LookUp/LookDown to NoOp due
            # to the camera being at its physical limit). Drain on these
            # specifically prevents the LookUp-against-ceiling spam
            # without penalising legitimate repeats of useful actions.
            _PITCH_FUTILE_PENALTY = 1.0
            for _i in range(num_agents):
                if step_contents[_i] is None:
                    continue
                _futile_n = environment.consume_futile(_i)
                if _futile_n > 0:
                    _pen = _PITCH_FUTILE_PENALTY * _futile_n
                    step_rewards_raw[_i] -= _pen
                    _step_pitch_penalty[_i] -= _pen

            _comm_rewards_this_step: dict = {}
            _comm_milestones: list = []
            _valid_speakers: set = set()
            if communication:
                _comm_rewards_this_step, _comm_milestones, _valid_speakers = comm_tracker.process_step(
                    step, _chat_this_step, _agent_pos_map,
                    bad_target_speakers=_bad_target_speakers,
                )
                for _aid, _bonus in _comm_rewards_this_step.items():
                    step_rewards_raw[_aid] += _bonus
                for _aid, _mid, _rw in _comm_milestones:
                    _comm_ev = {
                        "step": step,
                        "milestone": _mid,
                        "contributors": [f"agent_{_aid}"],
                        "reward": _rw,
                    }
                    metric.record_milestone_event(_comm_ev)
                    coop_metric.observe_milestone(step, _mid, [f"agent_{_aid}"])
                    ep_logger.log_event({"step": step, "type": "comm_milestone",
                                         "milestone": _mid, "agent": f"agent_{_aid}",
                                         "reward": _rw})

            # ── Act-reward symmetry: pay obs/imit acts like messages ──────
            _act_rewards_this_step: dict = {}
            _act_milestones: list = []
            if act_reward_tracker is not None and _act_events_this_step:
                _act_rewards_this_step, _act_milestones = (
                    act_reward_tracker.process_step(
                        step, _act_events_this_step, _agent_pos_map,
                    )
                )
                for _aid, _bonus in _act_rewards_this_step.items():
                    step_rewards_raw[_aid] += _bonus
                for _aid, _mid, _rw in _act_milestones:
                    metric.record_milestone_event({
                        "step": step, "milestone": _mid,
                        "contributors": [f"agent_{_aid}"], "reward": _rw,
                    })
                    coop_metric.observe_milestone(step, _mid, [f"agent_{_aid}"])
                    ep_logger.log_event({"step": step, "type": "act_milestone",
                                         "milestone": _mid,
                                         "agent": f"agent_{_aid}",
                                         "reward": _rw})

            # ── Flush per-message records to messages.jsonl ──
            # Stamp reward fields now that CommunicationTracker has run.
            # Note: the tracker bundles base + milestone in one float per
            # speaker; we split using the milestone events list.
            _msg_milestone_per_agent = {}
            for _aid, _mid, _rw in _comm_milestones:
                _msg_milestone_per_agent[_aid] = _msg_milestone_per_agent.get(_aid, 0.0) + _rw
            for _msg in _messages_this_step:
                _sid = int(_msg["sender"].split("_")[1])
                _comm_total = float(_comm_rewards_this_step.get(_sid, 0.0))
                _ms = float(_msg_milestone_per_agent.get(_sid, 0.0))
                _msg["valid"] = _sid in _valid_speakers
                _msg["rewarded_base"] = max(0.0, _comm_total - _ms)
                _msg["rewarded_milestone"] = _ms
                _msg["chamber"] = environment.get_chamber(_sid)
                # Receiver's chamber at receive time. Asymmetric with sender's
                # chamber whenever agents are in different rooms — useful for
                # post-hoc analysis of cross-chamber communication.
                try:
                    _rid = int(str(_msg.get("receiver", "")).split("_")[-1])
                    _msg["receiver_chamber"] = environment.get_chamber(_rid)
                except (ValueError, IndexError):
                    _msg["receiver_chamber"] = None
                ep_logger.log_message(_msg)

            # ── Orchestrator: message events + per-step compliance log ──
            # Uses the routed metadata the loop already built; complied is
            # whether the actual receiver matches the standing directive.
            if orchestrator_state is not None:
                for _msg in _messages_this_step:
                    orchestrator_state.add_event(
                        _orch_events.message_event(
                            step, _msg["sender"], _msg["receiver"],
                            _msg["text"],
                        )
                    )
                    # Villager issues task assignments, not comm directives —
                    # a compliance stream would be all-None noise rows.
                    if orchestrator_config.variant != "villager":
                        _o_directed = _orch_core.directive_comm_target(
                            orchestrator_state, _msg["sender"]
                        )
                        orch_logger.log_compliance({
                            "episode": episode + 1,
                            "t": step,
                            "agent": _msg["sender"],
                            "directed_comm_target": _o_directed,
                            "actual_comm_target": _msg["receiver"],
                            "complied": (_o_directed is not None
                                         and _o_directed == _msg["receiver"]),
                        })

            # ── Orchestrator plan/villager variants: task-compliance log ──
            # One record per task CHANGE, carrying the guidance that was
            # standing when the new task was generated (plan: the advisory
            # note; villager: the HARD assigned objective) — the raw
            # material for scoring whether central guidance shapes plans.
            if (orchestrator_state is not None
                    and orchestrator_config.variant in ("plan", "villager")):
                for _i in range(num_agents):
                    _cur_task = agents[_i].auto_curriculum.current_task
                    if _cur_task != _orch_prev_tasks.get(_i):
                        _active_note = (
                            _orch_plan_notes.get(_i)
                            if orchestrator_config.variant == "plan"
                            else _orch_assigned_objectives.get(_i)
                        )
                        orch_logger.log_task_compliance({
                            "episode": episode + 1,
                            "t": step,
                            "agent": f"agent_{_i}",
                            "active_note": _active_note or "",
                            "old_task": _orch_prev_tasks.get(_i),
                            "new_task": _cur_task,
                        })
                        _orch_prev_tasks[_i] = _cur_task

            coop_metric.observe_step(
                step,
                positions=_agent_pos_map,
                actions=_actions_this_step,
                messages=_chat_this_step,
                task_rewards=_task_rewards_this_step,
                # Hand the resolved sender→receiver routing to the metric so
                # its per-episode pair_messages matrix is populated. The
                # run-level matrix in comm_metrics is recomputed from
                # messages.jsonl post-hoc, but the per-episode summary
                # otherwise stays all-zeros without this.
                infos={"routed_messages": [
                    {"sender": _s, "receiver": _r}
                    for (_s, _r) in comm_events
                ]},
            )
            ep_logger.log_step(
                step,
                positions=_agent_pos_map,
                actions=_actions_this_step,
                messages=_chat_this_step,
                task_rewards=_task_rewards_this_step,
                comm_rewards=_comm_rewards_this_step,
                infos={"chambers": {i: environment.get_chamber(i) for i in range(num_agents)}},
            )

            # ── Phase 1c: Drain Five-chambers milestone rewards ──
            # Lua's emit_milestone() writes to milestone_events.jsonl and also
            # calls craftium.reward() as a backup, but the latter does not
            # reach env.step()'s rewards channel in the multi-agent
            # five-chambers context — so the JSONL is the authoritative reward
            # source. We must drain into step_rewards_raw HERE (before Hebbian
            # diffusion + record_reward) so the +N points propagate through
            # the graph and into cumulative_returns. The events are saved into
            # _milestone_events_this_step and re-consumed in Phase 3b below
            # for metrics / logging without re-reading the JSONL file.
            _milestone_events_this_step = environment.poll_milestone_events()
            # Anvil coop-detected diagnostic events (NO reward attached —
            # purely for analysis). Forward to metric so they end up in
            # final_metrics.json + the milestones.png plot's lower-priority
            # marker row. Lua emits one event per anvil per ACTIVE_WINDOW
            # tick gap.
            for _coop_ev in environment.poll_anvil_coop_events():
                metric.record_anvil_coop_event(_coop_ev)
            for _ev in _milestone_events_this_step:
                _rw = float(_ev.get("reward", 0))
                if _rw == 0.0:
                    continue
                for _name in _ev.get("contributors", []):
                    # The Lua side emits contributors as 'agent0' (no
                    # underscore — Craftium's player-name convention),
                    # while Python often uses 'agent_0'. The previous
                    # `split('_')[-1]` parser worked only for the
                    # underscored form: 'agent0'.split('_') = ['agent0'],
                    # int('agent0') raised ValueError, and the contributor
                    # was silently skipped. Net effect: every Lua-fired
                    # milestone reward was DROPPED — m1_move_5 / m2_dig_3
                    # / m_door1_open contributors never reached
                    # step_rewards_raw. Strip both prefix shapes:
                    _s = str(_name).removeprefix("agent_").removeprefix("agent")
                    try:
                        _aid = int(_s)
                    except ValueError:
                        continue
                    if 0 <= _aid < num_agents:
                        step_rewards_raw[_aid] += _rw
                        _step_milestone_drain[_aid] += _rw

            # ── Orchestrator: milestone events for the digest ──
            if orchestrator_state is not None:
                for _ev in _milestone_events_this_step:
                    orchestrator_state.add_event(
                        _orch_events.milestone_event(
                            step, _ev.get("milestone", ""),
                            _ev.get("contributors", []),
                        )
                    )

            # ── Phase 1d: Drain death / would-die penalties ──
            # deaths.lua emits these to death_events.jsonl; like milestones, the
            # server-side craftium.reward() it also fires does NOT reach
            # env.step()'s reward channel in multi-agent five-chambers, so this
            # JSONL is the authoritative source. Each event carries a NEGATIVE
            # reward for exactly one agent (−10 would-die in Ch1–4, −50 real Ch5
            # death). Drain into step_rewards_raw HERE (before Hebbian diffusion
            # + record_reward) so the penalty propagates through the graph and
            # into episode_return. Folded into the `task` decomposition stream.
            # NOTE: poll_death_events() already RATE-LIMITS would-die to one per
            # agent per EPISODE (the Lua callback fires per damage event, so an
            # attacked agent would otherwise stack dozens of −10s) — so this loop
            # sees at most one would-die per agent per episode. Deaths uncapped.
            _death_events_this_step = environment.poll_death_events()
            for _ev in _death_events_this_step:
                _rw = float(_ev.get("reward", 0))
                if _rw == 0.0:
                    continue
                _s = str(_ev.get("agent", "")).removeprefix("agent_").removeprefix("agent")
                try:
                    _aid = int(_s)
                except ValueError:
                    continue
                if 0 <= _aid < num_agents:
                    step_rewards_raw[_aid] += _rw
                    _step_death_drain[_aid] += _rw

            # ── Orchestrator: death events for the digest ──
            # Real Ch5 deaths only; would-die near-misses are penalties,
            # not deaths, and stay out of the event schema.
            if orchestrator_state is not None:
                for _ev in _death_events_this_step:
                    if _ev.get("kind") != "death":
                        continue
                    _s = str(_ev.get("agent", "")) \
                        .removeprefix("agent_").removeprefix("agent")
                    try:
                        _dead_name = f"agent_{int(_s)}"
                    except ValueError:
                        _dead_name = str(_ev.get("agent", "?"))
                    orchestrator_state.add_event(
                        _orch_events.death_event(step, _dead_name)
                    )

            # ── Phase 2: Hebbian update + reward diffusion ──

            # Per-agent one-step advantage δ_t = r_t - V(s_t).
            # V(s_t) was stored by select_action() in the pending transition.
            # We compute this before store_reward() so Hebbian sees the current
            # step's signal rather than a one-step-lagged value.
            # Agents without an active RL layer contribute None (falls back to
            # normalised reward for that agent inside _compute_modulator).
            step_advantages = []
            for _aid, _agent in enumerate(agents):
                v = _agent.rl_layer.get_pending_value() if _agent.rl_layer else None
                if v is not None:
                    step_advantages.append(step_rewards_raw[_aid] - v)
                else:
                    step_advantages.append(None)
            _any_advantage = any(a is not None for a in step_advantages)

            # Proximity collaboration bonus REMOVED — was +0.3 per pair per
            # step within interaction_radius, inflating returns by ~60 over
            # 100 steps when the team stayed close. Proximity still influences
            # learning indirectly through Hebbian co-activity (cij), which
            # gates LTP without paying out raw reward.

            # Gated-Hebbian variants (mode != "legacy") read structured,
            # chamber-gated reward streams instead of a single scalar:
            #   chambers      — per-agent chamber index 1..5 (0 ⇒ Ch1/solo,
            #                    gated OUT of every reward read).
            #   bond_rewards  — BONDABLE reward = milestone + comm + futile.
            #                    The drained death/would-die penalties
            #                    (_step_death_drain — −10 for a would-have-died
            #                    in Ch1–4, −50 for a real Ch5 death, via the
            #                    death_events.jsonl drain in Phase 1d) are NOT
            #                    summed here, so death is excluded from growth by
            #                    construction — Variant B never bonds on shared
            #                    deaths.
            #   total_rewards — full reward incl. death (= step_rewards_raw,
            #                    which now carries the drained death penalty);
            #                    used only for the neg_i decay gate, so a death
            #                    can still trip decay.
            # The legacy mode ignores these and uses step_rewards/advantages.
            _CHAMBER_TO_INT = {"ch1": 1, "ch2": 2, "ch3": 3, "ch4": 4, "ch5": 5}
            _chambers = [
                _CHAMBER_TO_INT.get(environment.get_chamber(_i) or "", 0)
                for _i in range(num_agents)
            ]
            _bond_rewards = [
                _step_milestone_drain[_i]
                + (float(_comm_rewards_this_step.get(_i, 0.0)) if communication else 0.0)
                # Act-reward symmetry: obs/imit pay enters the bondable
                # stream exactly like comm pay does (0.0 when the flag is off).
                + float(_act_rewards_this_step.get(_i, 0.0))
                + _step_pitch_penalty[_i]
                for _i in range(num_agents)
            ]
            # ── Orchestrator social/plan: feed the pair accumulator the SAME
            # per-step streams the Hebbian rule consumes (positions for
            # co-presence, comm pair events, bondable rewards, chambers).
            if _orch_pair_acc is not None:
                _orch_pair_acc.note_step(
                    positions,
                    comm_events if communication else [],
                    _bond_rewards,
                    _chambers,
                )

            hebbian_graph.update(
                positions=positions,
                step_rewards=step_rewards_raw,
                advantages=step_advantages if _any_advantage else None,
                comm_events=comm_events if communication else None,
                chambers=_chambers,
                bond_rewards=_bond_rewards,
                total_rewards=step_rewards_raw,
                social_events=(social_events
                               if social_act_mode == "choice" else None),
            )
            diffused_rewards = hebbian_graph.diffuse_rewards(step_rewards_raw)

            # ── Choice-mode co-firing event log + replay reward accounting ──
            if social_act_mode == "choice":
                if _cofiring_path is not None:
                    _cij_m = getattr(hebbian_graph, "_last_coactivity", None)
                    _c_terms = {
                        "spat": getattr(hebbian_graph, "_last_c_spat", None),
                        "comm": getattr(hebbian_graph, "_last_c_comm", None),
                        "obs": getattr(hebbian_graph, "_last_c_obs", None),
                        "imit": getattr(hebbian_graph, "_last_c_imit", None),
                    }
                    _eng = getattr(hebbian_graph, "_last_engagement", None)
                    _lg = getattr(hebbian_graph, "_last_growth", None)
                    _ld = getattr(hebbian_graph, "_last_decay", None)
                    if _cij_m is not None:
                        for _pi in range(num_agents):
                            for _pj in range(num_agents):
                                if _pi == _pj or float(_cij_m[_pi][_pj]) <= 0.0:
                                    continue
                                _dist = None
                                if (positions[_pi] is not None
                                        and positions[_pj] is not None):
                                    _dist = float(np.linalg.norm(
                                        np.asarray(positions[_pi][:3], dtype=float)
                                        - np.asarray(positions[_pj][:3], dtype=float)
                                    ))
                                _row = {
                                    "ep": episode + 1, "step": step,
                                    "i": _pi, "j": _pj,
                                    "c": float(_cij_m[_pi][_pj]),
                                    "dist": _dist,
                                    "g_i": (float(_eng[_pi])
                                            if _eng is not None else None),
                                    "g_j": (float(_eng[_pj])
                                            if _eng is not None else None),
                                    "growth": (float(_lg[_pi][_pj])
                                               if _lg is not None else None),
                                    "decay": (float(_ld[_pi][_pj])
                                              if _ld is not None else None),
                                    "W_after": float(hebbian_graph.W[_pi][_pj]),
                                }
                                _sa_metrics["cofire_pair_events"] += 1
                                for _cn, _cm in _c_terms.items():
                                    _v = (float(_cm[_pi][_pj])
                                          if _cm is not None else 0.0)
                                    _row[f"c_{_cn}"] = _v
                                    if _v > 0.0:
                                        _sa_metrics["cofire_events_by_channel"][_cn] += 1
                                _append_jsonl(_cofiring_path, _row)
                for _aid in range(num_agents):
                    if step_contents[_aid] is None:
                        continue
                    if _aid in _adopted_agents_this_step:
                        _sa_metrics["adopted_reward_sum"] += float(step_rewards_raw[_aid])
                    else:
                        _sa_metrics["nonadopted_steps"] += 1
                        _sa_metrics["nonadopted_reward_sum"] += float(step_rewards_raw[_aid])

            # ── Phase B+ §1: per-teammate reward-propagation cache ──
            # Disabled — rlvr.reward_propagation was removed (it provided
            # per_teammate_contributions + attribute_source_events). The
            # cache stays empty so the prompt-side propagation block is
            # always blank; downstream readers handle that fine.

            # ── Reward decomposition: split each agent's diffused reward into
            #    its source streams. Each stream is read directly from the
            #    per-source accumulator populated at the matching add-site:
            #      task            = env-step reward + pitch penalty + drained
            #                        milestones + drained death/would-die penalties
            #      comm_base       = BASE_MSG_REWARD per valid message
            #      comm_milestone  = Tier-2 per-chamber communication milestones
            #      proximity       = 0 (the +0.3/pair proximity bonus was removed;
            #                        the field is kept for back-compat with
            #                        downstream consumers that index the 5-tuple)
            #      hebbian_diffuse = diffused_rewards[i] - step_rewards_raw[i]
            #    The five streams sum to diffused_rewards[i] (== the value
            #    passed to record_reward).
            _comm_milestone_per_agent = {i: 0.0 for i in range(num_agents)}
            if communication:
                for _aid, _mid, _rw in (_comm_milestones or []):
                    _comm_milestone_per_agent[_aid] = (
                        _comm_milestone_per_agent.get(_aid, 0.0) + _rw
                    )
            _reward_decomp_this_step = {}
            for _aid in range(num_agents):
                _task = (_step_envstep_reward[_aid]
                         + _step_pitch_penalty[_aid]
                         + _step_milestone_drain[_aid]
                         + _step_death_drain[_aid])
                _comm_total = float(_comm_rewards_this_step.get(_aid, 0.0)) if communication else 0.0
                _comm_ms = float(_comm_milestone_per_agent.get(_aid, 0.0))
                _comm_base = _comm_total - _comm_ms
                _prox = 0.0
                _hebb = float(diffused_rewards[_aid]) - float(step_rewards_raw[_aid])
                _reward_decomp_this_step[_aid] = {
                    "task":            _task,
                    "comm_base":       _comm_base,
                    "comm_milestone":  _comm_ms,
                    "proximity":       _prox,
                    "hebbian_diffuse": _hebb,
                }
                # Act-reward symmetry streams (keys only exist when the flag
                # is on, so legacy decomposition records stay byte-identical).
                if act_reward_tracker is not None:
                    _act_total = float(_act_rewards_this_step.get(_aid, 0.0))
                    _act_ms = sum(_rw for _a, _m, _rw in _act_milestones
                                  if _a == _aid)
                    _reward_decomp_this_step[_aid]["social_base"] = (
                        _act_total - _act_ms
                    )
                    _reward_decomp_this_step[_aid]["social_milestone"] = _act_ms

            # ── Per-step reward log line ──────────────────────────────────
            # Surface every non-zero reward (task, comm, milestone, Hebbian
            # diffuse) into stdout so the SLURM .out / run.log shows the
            # signal stream as it accumulates. Previously only milestone
            # events ([MILESTONE] from Lua) and end-of-episode totals were
            # visible, so per-step comm-base / comm-milestone / Hebbian
            # contributions were silent — hard to audit "where did the
            # 50-pt reward in step 12 come from?" without parsing the
            # JSON step_log post-hoc. Compact format: show only the
            # streams that fired this step plus the final diffused total
            # the RL/metric layer actually saw.
            _row_strs = []
            for _aid in range(num_agents):
                _d = _reward_decomp_this_step[_aid]
                _total = float(diffused_rewards[_aid])
                _streams_nonzero = {k: v for k, v in _d.items() if abs(v) > 1e-6}
                if abs(_total) > 1e-6 or _streams_nonzero:
                    _comps = " ".join(
                        f"{k}={v:+.2f}" for k, v in _streams_nonzero.items()
                    ) or "0"
                    _row_strs.append(
                        f"agent_{_aid}[{_comps}]→{_total:+.2f}"
                    )
            if _row_strs:
                print(
                    f"[REWARD ep={episode+1} step={step+1}/{max_steps}] "
                    + "  ".join(_row_strs)
                )

            # ── Phase 3: Record (diffused) rewards for metrics + RL ──
            for agent_id, agent in enumerate(agents):
                agent_name = f"agent_{agent_id}"
                if environment._terminations.get(agent_name, False):
                    # Terminated (Ch5 permadeath). We must NOT keep accumulating
                    # ongoing per-step reward for a frozen dead client — but if a
                    # death penalty was drained for this agent THIS step, that
                    # terminal −50 still belongs in the episode return. Whether
                    # the engine flips termination on the death step is
                    # unreliable (see CraftiumEnvironmentInterface.episode_over),
                    # so without this the −50 is silently dropped whenever it
                    # does flip. Book it once — same diffused value + decomp as
                    # the live path, so no double-count — then skip the RL
                    # transition (a dead agent has no further actions to credit).
                    if abs(_step_death_drain[agent_id]) > 1e-6:
                        metric.record_reward(agent_id, diffused_rewards[agent_id])
                        metric.record_reward_decomposed(
                            agent_id, _reward_decomp_this_step[agent_id]
                        )
                        # Close the dead agent's last RL transition with the
                        # terminal reward and done=True, so the value head
                        # actually learns the death. store_reward() adds
                        # config.death_penalty on done — exactly as the live path
                        # would have, had the agent not been skipped — and the
                        # buffer safely no-ops if there was no pending
                        # transition. Gated on the death-drain so it fires once,
                        # only on the step the agent actually died.
                        if agent.rl_layer and agent.rl_layer.enabled:
                            agent.rl_layer.store_reward(
                                diffused_rewards[agent_id], done=True,
                                reward_task=_task_rewards_this_step.get(agent_id, 0.0),
                                reward_comm=_comm_rewards_this_step.get(agent_id, 0.0),
                            )
                    continue

                reward = diffused_rewards[agent_id]
                metric.record_reward(agent_id, reward)
                metric.record_reward_decomposed(agent_id, _reward_decomp_this_step[agent_id])

                # Feed reward to RL layer. (The macro-defer / macro-flush
                # branches that used to wrap this store_reward were
                # removed in T1.6 — agents always execute primitive
                # actions now, so the reward closes the pending
                # transition immediately.)
                if agent.rl_layer and agent.rl_layer.enabled:
                    content = step_contents[agent_id]
                    agent_done = environment._terminations.get(agent_name, False)
                    agent.rl_layer.store_reward(
                        reward, done=agent_done,
                        reward_task=_task_rewards_this_step.get(agent_id, 0.0),
                        reward_comm=_comm_rewards_this_step.get(agent_id, 0.0),
                    )

                    agent.rl_layer.record_context(
                        action=content.get("action", "NoOp") if content else "NoOp",
                        reward=reward,
                        task=agent.auto_curriculum.current_task or "Explore",
                    )

                    # MAPPO update when enough steps collected.
                    # Pass all agents' buffers so social replay (Eq. 7) can
                    # mix in neighbour transitions weighted by Hebbian bonds.
                    # Buffers are snapshotted once per UPDATE CYCLE (first
                    # updater takes it; cleared when every expected agent has
                    # consumed it) so neither agent order within a step nor
                    # --rl-update-stagger across steps decides who still has
                    # neighbours' experience.
                    if agent.rl_layer.should_update():
                        if _rl_replay_snapshot is None:
                            _rl_replay_snapshot = {
                                aid: agents[aid].rl_layer.buffer.snapshot()
                                for aid in range(num_agents)
                                if agents[aid].rl_layer
                                and agents[aid].rl_layer.enabled
                            }
                            _rl_replay_pending = set(_rl_replay_snapshot)
                        neighbour_buffers = {
                            aid: snap
                            for aid, snap in _rl_replay_snapshot.items()
                            if aid != agent_id
                        }
                        update_info = agent.rl_layer.update(
                            neighbour_buffers=neighbour_buffers,
                            hebbian_graph=hebbian_graph,
                        )
                        # Retire the snapshot once everyone expected has
                        # updated (agents that died mid-cycle are dropped so
                        # a permadeath cannot pin a stale snapshot forever).
                        _rl_replay_pending.discard(agent_id)
                        _rl_replay_pending = {
                            aid for aid in _rl_replay_pending
                            if not environment._terminations.get(
                                f"agent_{aid}", False)
                        }
                        if not _rl_replay_pending:
                            _rl_replay_snapshot = None
                        metric.record_rl_update(agent_id, update_info)
                        _wb.log_rl_update(agent_id, update_info, step=global_step)

                    # Agent-decided token-level optimisation
                    try:
                        token_info = await agent.rl_layer.maybe_token_optimize(
                            cancellation_token=CancellationToken(),
                            hebbian_graph=hebbian_graph,
                        )
                        if token_info:
                            metric.record_rl_token_opt(agent_id, token_info)
                            _wb.log_rl_token_opt(agent_id, token_info, step=global_step)

                        # Social propagation: when an agent trains, offer the
                        # same opportunity to strongly-bonded teammates.
                        if (token_info and token_info.get("decision") == "train"
                                and hebbian_config.enabled):
                            for j in range(num_agents):
                                if j == agent_id:
                                    continue
                                bond_w = float(hebbian_graph.W[agent_id, j])
                                if bond_w > 0.3 and agents[j].rl_layer.enabled:
                                    try:
                                        soc_info = await agents[j].rl_layer.maybe_token_optimize(
                                            cancellation_token=CancellationToken(),
                                            hebbian_graph=hebbian_graph,
                                        )
                                        if soc_info:
                                            logging.info(
                                                "[social token-opt] agent_%d triggered "
                                                "agent_%d (bond=%.3f) → decision=%s",
                                                agent_id, j, bond_w,
                                                soc_info.get("decision", "?"),
                                            )
                                            metric.record_rl_token_opt(j, soc_info)
                                            _wb.log_rl_token_opt(j, soc_info, step=global_step)
                                    except Exception as _soc_exc:
                                        logging.warning(
                                            "Social token-opt agent_%d failed: %s",
                                            j, _soc_exc,
                                        )
                    except Exception as _tok_exc:
                        logging.warning(f"Agent {agent_id} token_optimize failed: {_tok_exc}")

            # ── Phase 3a: Centralised critic — store team step, maybe update ──
            if centralized_critic is not None and joint_state_t is not None:
                _alive_rewards = [
                    float(diffused_rewards[_i])
                    for _i in range(num_agents)
                    if not environment._terminations.get(f"agent_{_i}", False)
                ]
                _team_reward = (
                    sum(_alive_rewards) / len(_alive_rewards) if _alive_rewards else 0.0
                )
                _team_done = any(
                    environment._terminations.get(f"agent_{_i}", False)
                    for _i in range(num_agents)
                )
                centralized_critic.store_step(
                    joint_state_t, _team_reward, v_global_t, _team_done,
                )
                if centralized_critic.should_update():
                    _critic_info = centralized_critic.update()
                    if _critic_info:
                        metric.record_rl_update(-1, _critic_info)
                        _wb.log_rl_update(-1, _critic_info, step=global_step)

            # ── Phase 3b: Five-chambers milestone events ──
            # Map kill-style milestones to a target-string so the cooperation
            # metric can credit pair_joint_kill / pair_boss_overlap. m21 fires
            # for the first Ch4 mob kill per agent, m22 fires once when all
            # Ch4 mobs are dead, m27 fires on boss death.
            _KILL_TARGETS = {
                "m21_first_mob_kill":  "ch4_zombie",
                "m22_all_mobs_killed": "ch4_zombie",
                "m27_boss_defeated":   "boss",
            }
            for _ev in _milestone_events_this_step:
                metric.record_milestone_event(_ev)
                _mid = _ev.get("milestone", "")
                # IMPORTANT: _ev["step"] is the Lua `step_counter` (Luanti
                # server ticks since mod load) — NOT the Python env-step
                # counter. Mixing the two in milestone_log breaks any
                # x-axis comparison against env-step-indexed series like
                # chamber_entry_steps. Always use the Python env step
                # (which is the polling step where the event surfaced)
                # for downstream metrics. Lua tick stays in the raw
                # event dict captured by record_milestone_event() for
                # debugging.
                _contribs_ev = _ev.get("contributors", [])
                coop_metric.observe_milestone(step, _mid, _contribs_ev)
                if _mid in _KILL_TARGETS:
                    # Credit the (first-listed) contributor as the killer; the
                    # joint-kill matrix is computed from recent damage events
                    # so the rest of the pair info is recovered there.
                    _killer = _contribs_ev[0] if _contribs_ev else None
                    coop_metric.observe_kill(step, _killer, _KILL_TARGETS[_mid])
                ep_logger.log_event({
                    "step": step,
                    "type": "milestone",
                    "id": _mid,
                    "contributors": _contribs_ev,
                    "reward": _ev.get("reward", 0),
                })
                # Surface milestone fires in the SLURM .out file. The Lua side
                # already writes "[SRV] [MILESTONE] ..." into stderr (tailed
                # by craftium), but parsing those lines is brittle — this is
                # the authoritative Python-side line, one per polled event.
                _contrib_str = ",".join(_contribs_ev) if _contribs_ev else "<none>"
                # Per-contributor chamber at fire time, so the milestone line
                # shows where each agent was when the event occurred.
                _contrib_chambers = []
                for _name in _contribs_ev:
                    try:
                        _aid = int(str(_name).split("_")[-1])
                        _contrib_chambers.append(
                            f"{_name}@{environment.get_chamber(_aid) or '?'}"
                        )
                    except (ValueError, IndexError):
                        _contrib_chambers.append(_name)
                _chamber_str = ",".join(_contrib_chambers) if _contrib_chambers else "<none>"
                print(
                    f"[MILESTONE] ep={episode+1} step={step} "
                    f"id={_mid or '?'} "
                    f"agents=[{_chamber_str}] "
                    f"reward={_ev.get('reward', 0)}",
                    flush=True,
                )

            # ── Phase 4: Graph metrics snapshot + SLURM log ──
            if hebbian_config.enabled and step % hebbian_config.log_graph_every == 0:
                graph_metrics = hebbian_graph.get_graph_metrics()
                metric.record_graph_snapshot(global_step, graph_metrics)
                metric.log(f"[Hebbian step {step}] {graph_metrics}")

                # Print a compact weight table to stdout so it lands in
                # the SLURM .out file alongside the reward/task summaries.
                W = hebbian_graph.get_all_weights()
                N = num_agents
                mean_bond = graph_metrics.get("mean_bond_strength", 0.0)
                top = graph_metrics.get("top_3_pairs", [])

                # Header row
                col_hdrs = "      " + "  ".join(f"ag{j:>2}" for j in range(N))
                rows = [col_hdrs, "      " + "------" * N]
                for i in range(N):
                    role_i = role_configs[i]["name"][:3].upper()
                    cells = "  ".join(f"{W[i, j]:5.3f}" for j in range(N))
                    rows.append(f"ag{i} {role_i}  {cells}")

                top_str = "  ".join(
                    f"({p['i']}→{p['j']})={p['w']:.3f}" for p in top
                ) or "none"

                print(
                    f"[{run_id}] [HEBBIAN] ep={episode+1} step={step+1} "
                    f"mean={mean_bond:.4f}  top3: {top_str}\n"
                    + "\n".join(f"  {r}" for r in rows)
                )

            metric.store_timestep(step_comm_count=step_comm_count)

            # ── Snapshot per-agent action/comm strings as previous-step
            # memory for next step's pre-step joint state encoding. The
            # centralised critic uses these as semantic features (last
            # action, last comm) that condition V(s_{t+1}).
            if centralized_critic is not None:
                for _i in range(num_agents):
                    _prev_step_actions[_i] = _actions_this_step.get(_i, "NoOp")
                    _prev_step_comms[_i]   = _chat_this_step.get(_i, "")

            # ── Periodic checkpoint (within episode) ──
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                _ep_ckpt_dir = os.path.join(checkpoint_dir, f"ep{episode+1}_step{step+1}")
                save_checkpoint(
                    checkpoint_dir=_ep_ckpt_dir,
                    episode=episode,
                    step=step + 1,
                    run_id=run_id,
                    args=args,
                    metric=metric,
                    agents=agents,
                    hebbian_graph=hebbian_graph,
                    frames_list=frames_list if args.checkpoint_frames else None,
                    save_frames=args.checkpoint_frames,
                    global_step=global_step,
                )

            # ── Graceful shutdown on signal ──
            if _shutdown_requested:
                _ep_ckpt_dir = os.path.join(checkpoint_dir, f"ep{episode+1}_step{step+1}_shutdown")
                save_checkpoint(
                    checkpoint_dir=_ep_ckpt_dir,
                    episode=episode,
                    step=step + 1,
                    run_id=run_id,
                    args=args,
                    metric=metric,
                    agents=agents,
                    hebbian_graph=hebbian_graph,
                    frames_list=frames_list if args.checkpoint_frames else None,
                    save_frames=args.checkpoint_frames,
                    global_step=global_step,
                )
                print(f"[CKPT] Shutdown checkpoint saved → {_ep_ckpt_dir}")
                # Close any open MP4 writers so the partial-episode video
                # is finalised (playable up to whatever frame was last
                # appended). Without this, the SLURM kill path leaves a
                # zero-byte / unfinalised MP4 file.
                if save_gif:
                    for _i in range(num_agents):
                        if _mp4_writers[_i] is not None:
                            try:
                                _mp4_writers[_i].close()
                            except Exception:
                                pass
                _wb.finish()
                environment.close()
                return

        # ── End-of-episode: finalize cooperation metrics + structured logs ──
        _hebb_W = hebbian_graph.snapshot().get("W") if hebbian_config.enabled else None
        _ep_final_step = (step + 1) if max_steps > 0 else 0
        _coop_summary = coop_metric.episode_summary(
            final_step=_ep_final_step,
            hebbian_weights=_hebb_W,
        )
        # Snapshot the per-episode return (delta since episode start) BEFORE
        # writing the episode summary so total_reward_per_agent reflects
        # only this episode, not the cumulative across all episodes so far.
        _ep_return_per_agent = {
            f"agent_{i}": float(metric.episode_returns[i])
            for i in range(num_agents)
        }
        # Snapshot per-episode track rewards / milestone count / comm count
        # BEFORE end_episode() resets them, so the W&B log uses fresh values.
        _ep_track_rewards = {
            i: dict(metric.track_rewards_episode[i])
            for i in range(num_agents)
        }
        _ep_milestone_count = {
            i: len(metric._agent_milestones_episode[i])
            for i in range(num_agents)
        }
        _ep_comm_count = list(metric.comm_count_episode)
        metric.end_episode(final_step=_ep_final_step)

        # W&B: per-episode headline numbers. step=global_step keeps the
        # x-axis consistent with the per-step throttled view above.
        _wb_episode_payload = {
            "ep/index": episode + 1,
            "ep/length": int(_ep_final_step),
            "ep/cooperation_score": float(_coop_summary.get("cooperation_score", 0.0)),
            "ep/total_reward": float(sum(_ep_return_per_agent.values())),
        }
        for i in range(num_agents):
            _wb_episode_payload[f"ep/return/agent_{i}"] = _ep_return_per_agent[f"agent_{i}"]
            # Final flush of the per-step return series so the coarse
            # (log_interval-sampled, default 10) step/episode_return curve always
            # ends on the TRUE episode total. Otherwise late-episode rewards —
            # e.g. Ch5 −50 deaths, which land in the last ~20% of steps, after
            # the final sampled step on short runs — never appear on that curve.
            _wb_episode_payload[f"step/episode_return/agent_{i}"] = (
                _ep_return_per_agent[f"agent_{i}"]
            )
            _wb_episode_payload[f"ep/milestones_reached/agent_{i}"] = _ep_milestone_count[i]
            _wb_episode_payload[f"ep/comm_count/agent_{i}"] = int(_ep_comm_count[i])
            for _track, _val in _ep_track_rewards[i].items():
                _wb_episode_payload[f"ep/track_reward/agent_{i}/{_track}"] = float(_val)
        if hebbian_config.enabled and _hebb_W is not None:
            try:
                import numpy as _np
                _W = _np.asarray(_hebb_W)
                # Off-diagonal mean = average bond strength between distinct agents.
                _mask = ~_np.eye(_W.shape[0], dtype=bool)
                _wb_episode_payload["ep/hebbian/mean_bond"] = float(_W[_mask].mean())
                _wb_episode_payload["ep/hebbian/max_bond"] = float(_W[_mask].max())
            except Exception:
                pass
        _wb.log(_wb_episode_payload, step=global_step)
        _ep_summary = {
            "episode": episode + 1,
            "final_step": _ep_final_step,
            "total_reward_per_agent": _ep_return_per_agent,
            "cooperation_metrics": _coop_summary,
        }
        ep_logger.finalize(_ep_summary)

        # Append Hebbian snapshot to run-level JSONL stream.
        # _hebb_W is a numpy.ndarray with float32 cells, which json.dumps
        # can't handle natively — convert to nested Python lists.
        _hebb_W_serialisable = (
            _hebb_W.tolist() if hasattr(_hebb_W, "tolist") else _hebb_W
        )
        _hebb_snapshot_path = os.path.join(metric.target_folder, "hebbian_snapshots.jsonl")
        with open(_hebb_snapshot_path, "a", encoding="utf-8") as _hf:
            _hf.write(_json.dumps({
                "episode": episode + 1,
                "final_step": _coop_summary["final_step"],
                "W": _hebb_W_serialisable,
                "cooperation_score": float(_coop_summary.get("cooperation_score", 0.0)),
                "reward_total": float(sum(_ep_return_per_agent.values())),
            }) + "\n")

        # ── End-of-episode checkpoint ──
        _ep_ckpt_dir = os.path.join(checkpoint_dir, f"ep{episode+1}_end")
        save_checkpoint(
            checkpoint_dir=_ep_ckpt_dir,
            episode=episode + 1,  # episode is complete
            step=0,
            run_id=run_id,
            args=args,
            metric=metric,
            agents=agents,
            hebbian_graph=hebbian_graph,
            global_step=global_step,
        )

        # ── End-of-episode media flush ──
        # The FULL-episode MP4 was streamed to disk one frame at a time
        # during the loop above. Close the writers now so the file is
        # finalised and playable. (Pre-refactor, this whole block also
        # re-encoded the MP4 from frames_list; that path is gone because
        # the streaming writer already produced the same artifact.)
        if save_gif:
            for _i in range(num_agents):
                if _mp4_writers[_i] is not None:
                    try:
                        _mp4_writers[_i].close()
                        _ep_mp4_path = (
                            f"{gif_dir}/{run_id}_{role_configs[_i]['agent_name']}"
                            f"_ep{episode+1}.mp4"
                        )
                        print(f"[{run_id}] Saved MP4: {_ep_mp4_path}")
                    except Exception as _exc:
                        logging.warning(
                            "[GIF] streaming MP4 close failed for agent %d: %s",
                            _i, _exc)

        # Also emit a final GIF for each agent — covers the LAST
        # gif_interval frames of the episode (the deque's max content).
        # The full-episode story lives in the MP4 above; this GIF is the
        # compact "what happened at the end" companion for quick review.
        if save_gif and frames_list:
            _final_window = list(frames_list)
            for i in range(num_agents):
                agent_frames = [
                    PIL.Image.fromarray(f[i])
                    for f in _final_window if f[i] is not None
                ]
                if agent_frames:
                    gif_path = (
                        f"{gif_dir}/{run_id}_{role_configs[i]['agent_name']}_ep{episode+1}.gif"
                    )
                    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                    agent_frames[0].save(
                        gif_path,
                        format="GIF",
                        append_images=agent_frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )
                    print(f"[{run_id}] Saved GIF: {gif_path}")

    print(f"[{run_id}] Experiment complete! Timesteps logged: {metric.timestep}")
    # Attach run config for reproducibility before saving
    metric.seed = seed
    metric.max_steps = max_steps
    metric.num_episodes = num_episodes
    metric.experiment_id = args.experiment_id
    metric.cli_args = vars(args)
    # Surface the env's action-health diagnostics in summary.txt /
    # final_metrics: why agents were rescued from NoOp (invalid action vs
    # explicit stall) and how many invented action names were recovered to a
    # valid primitive (e.g. 'Attack' -> 'Dig') instead of clamped.
    metric.idle_force_counts = environment.idle_force_summary()
    metric.action_recovered_counts = environment.action_recovered_summary()
    # Choice-mode social-act metrics (Experiment 2); {} in legacy runs.
    if social_act_mode == "choice":
        metric.social_act_metrics = {
            **_sa_metrics,
            "menu_channels": list(_social_menu_channels),
            "credit_channels": list(_cofire_channels),
            "cofire_attribution": hebbian_graph.get_channel_attribution(),
        }
    metric.save_run_metrics()

    # Save RL checkpoints
    for agent in agents:
        if agent.rl_layer and agent.rl_layer.enabled:
            agent.rl_layer.save()
            print(f"  Saved RL checkpoint for {agent.name}")

    # Save per-agent cognitive state (skills / episodes / curriculum) — the
    # durable JSON form of the job-local vector DBs. Unconditional: also
    # useful for non-Hebbian runs.
    from agent_modules.agent_state_io import export_agent_state as _export_as
    _export_as(
        agents, os.path.join(metric.target_folder, "agent_state"),
        run_id=run_id,
    )
    print(f"  Saved agent cognitive state: "
          f"{os.path.join(metric.target_folder, 'agent_state')}")

    # Save Hebbian graph state
    if hebbian_config.enabled:
        graph_path = os.path.join(metric.target_folder, "hebbian_graph_final.json")
        with open(graph_path, "w") as f:
            _json.dump(hebbian_graph.to_dict(), f, indent=2)
        print(f"  Saved final Hebbian graph: {graph_path}")

        # Force one final snapshot and re-save metrics to include it
        metric.record_graph_snapshot(metric.timestep, hebbian_graph.get_graph_metrics())
        metric.save_run_metrics()

    # ── W&B: final summary + optional artifact upload ─────────────────────
    try:
        _wb_final = {}
        for i in range(num_agents):
            _ep_returns = metric.per_episode_returns[i]
            if _ep_returns:
                _wb_final[f"final/mean_return/agent_{i}"] = (
                    sum(_ep_returns) / len(_ep_returns)
                )
            _wb_final[f"final/cumulative_return/agent_{i}"] = float(
                metric.cumulative_returns[i]
            )
            _ms_counts = [len(s) for s in metric.milestones_per_episode[i]]
            if _ms_counts:
                _wb_final[f"final/mean_milestones/agent_{i}"] = (
                    sum(_ms_counts) / len(_ms_counts)
                )
        if metric.episode_lengths:
            _wb_final["final/mean_episode_length"] = (
                sum(metric.episode_lengths) / len(metric.episode_lengths)
            )
            _wb_final["final/num_episodes"] = len(metric.episode_lengths)
        _wb.log(_wb_final, step=global_step)
        if args.wandb_upload_artifacts:
            _wb.upload_file(
                os.path.join(metric.target_folder, "final_metrics.json"),
                name=f"final_metrics__{run_id}",
                artifact_type="metrics",
            )
            _wb.upload_file(
                os.path.join(metric.target_folder, "summary.txt"),
                name=f"summary__{run_id}",
                artifact_type="summary",
            )
    except Exception as _e:
        logging.warning("[wandb] final summary failed: %s", _e)

    # Final cleanup with a hard watchdog. environment.close() can block on a
    # Minetest client's wait_close() (upstream craftium), and wandb's
    # finish() can stall on sync — either one leaves the SLURM job idling
    # until its wall-clock limit even though all results (metrics, gifs,
    # checkpoint) are already persisted above. Arm a SIGALRM: if cleanup
    # hangs past the timeout we force-exit, and SLURM's cgroup reaps any
    # surviving Minetest processes.
    import signal as _sig_done
    def _force_exit_on_hang(signum, frame):
        print("[shutdown] final cleanup exceeded 120s — forcing exit "
              "(results already saved).", flush=True)
        os._exit(0)
    try:
        _sig_done.signal(_sig_done.SIGALRM, _force_exit_on_hang)
        _sig_done.alarm(120)
    except (ValueError, AttributeError, OSError):
        pass  # SIGALRM unavailable (non-main-thread / non-Unix) — skip it

    _wb.finish()
    environment.close()

    try:
        _sig_done.alarm(0)  # cleanup finished in time — cancel the watchdog
    except (ValueError, AttributeError, OSError):
        pass


if __name__ == "__main__":
    # Force unbuffered stdout so prints appear immediately in SLURM logs
    import functools
    print = functools.partial(print, flush=True)
    args = parse_args()
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
    asyncio.run(run(args))
