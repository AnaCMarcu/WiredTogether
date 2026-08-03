"""Synthetic run-dir + llm_logs builders for analysis_qualitative tests.

Extends the on-disk-layout approach of ``test_coop_comm_eval._write_episode``
with a writer for llm_logs files that emits the EXACT line grammar of
``mindforge.agent_modules.llm_call`` (FileHandler formatter
"%(asctime)s %(levelname)s %(message)s").
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO / "analysis_qualitative"), str(REPO / "src"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

T0 = datetime(2026, 6, 26, 10, 0, 0)


class LogWriter:
    """Accumulates llm_log lines with a monotonically increasing clock."""

    def __init__(self):
        self.clock = T0
        self.files: dict[str, list[str]] = {}

    def tick(self, seconds: int = 1) -> str:
        self.clock += timedelta(seconds=seconds)
        return self.clock.strftime("%Y-%m-%d %H:%M:%S")

    def raw(self, fname: str, line: str):
        self.files.setdefault(fname, []).append(line)

    def call(self, fname: str, prefix: str, retry: int = 0, kwargs=("task",)):
        ts = self.tick()
        self.raw(fname,
                 f"{ts} INFO {prefix} call: sys_chars=100 user_chars=200 "
                 f"frame=True kwargs={sorted(kwargs)!r} retry={retry}")
        return ts

    def response(self, fname: str, prefix: str, payload, pretty=False):
        ts = self.tick()
        text = json.dumps(payload, indent=2 if pretty else None)
        self.raw(fname, f"{ts} INFO {prefix} Response: {text}")
        return ts

    def error(self, fname: str, prefix: str, what="Error parsing response (attempt 1): ValueError: x"):
        ts = self.tick()
        self.raw(fname, f"{ts} ERROR {prefix} {what}")
        return ts

    def dump(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        for fname, lines in self.files.items():
            (log_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


def action_prefix(agent: int) -> str:
    return f"Agent agent_{agent} on_messages: "


def write_episode(ep_dir: Path, steps_rows, messages, events, coop_metrics,
                  nested: bool = True):
    """Write one episode dir in the real EpisodeLogger layout."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    cols = ["step", "agent_id", "chamber", "pos_x", "pos_y", "pos_z", "action",
            "reward_task", "reward_comm", "wielded_item", "hp", "message"]
    with open(ep_dir / "step_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in steps_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    with open(ep_dir / "messages.jsonl", "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(ep_dir / "event_log.jsonl", "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    summary = (
        {"episode": 1, "final_step": len(steps_rows),
         "cooperation_metrics": coop_metrics}
        if nested else {"episode": 1, **coop_metrics}
    )
    (ep_dir / "episode_summary.json").write_text(json.dumps(summary),
                                                 encoding="utf-8")
    (ep_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def build_synth_run(root: Path, n_steps: int = 10, agent1_dies_at: int | None = 6):
    """Two-agent run: 1 episode, action log + anonymous module logs.

    Cadences (config.json): belief=2, critic=5, social=3. Agent 1 stops
    producing step rows at ``agent1_dies_at`` (death mid-episode).
    Returns (run_dir, expected dict).
    """
    run_dir = root / "exp99_test" / "seed_7"
    lw = LogWriter()
    steps_rows, messages = [], []

    for t in range(n_steps):
        for a in (0, 1):
            if a == 1 and agent1_dies_at is not None and t >= agent1_dies_at:
                continue
            turn = t + 1  # 1-based turn number for cadence rules
            pfx = action_prefix(a)
            # intra-turn anonymous modules BEFORE the action call
            if turn % 5 == 0:
                lw.call("critic.log", "Critic check_task_success: ")
                lw.response("critic.log", "Critic check_task_success: ",
                            {"reasoning": "r", "success": t % 2 == 0,
                             "critique": "c"}, pretty=True)
            if t == 0:
                lw.call("auto_curriculum.log", "Auto Curriculum get_new_task: ")
                lw.response("auto_curriculum.log", "Auto Curriculum get_new_task: ",
                            {"reasoning": "r", "task": f"task_{a}_v1"})
            if turn % 2 == 0:
                lw.call("belief_system.log",
                        "Belief System create_perception_beliefs: ")
                lw.response("belief_system.log",
                            "Belief System create_perception_beliefs: ",
                            {"beliefs": [f"see wall t{t}"]})
            if turn == 1 or turn % 3 == 0:
                spfx = f"SocialModule[agent_{a}]: "
                lw.call("social_module.log", spfx)
                lw.response("social_module.log", spfx,
                            {"reasoning": "bonds stable",
                             "referenced_bonds": {f"agent_{1-a}": 0.1},
                             "ask_target": f"agent_{1-a}" if turn == 3 else None,
                             "ask_message": "help me" if turn == 3 else None,
                             "respond_to": [], "confidence": 0.9,
                             "bond_change_explanation": {}})
            # the action call itself (one retry chain at t==2 for agent 0)
            msg_text = f"msg a{a} t{t}"
            payload = {"thoughts": f"think a{a} t{t} �", "action": "Dig",
                       "communication": msg_text,
                       "communication_target": f"agent_{1-a}"}
            lw.call("action_selection.log", pfx, retry=0)
            if a == 0 and t == 2:
                lw.raw("action_selection.log",
                       f"{lw.tick()} INFO {pfx} Response: not json at all")
                lw.error("action_selection.log", pfx)
                lw.call("action_selection.log", pfx, retry=1)
                lw.response("action_selection.log", pfx, payload, pretty=True)
            else:
                lw.response("action_selection.log", pfx, payload)

            steps_rows.append({
                "step": t, "agent_id": a, "chamber": "ch1",
                "pos_x": float(t), "pos_y": 11.5, "pos_z": float(a),
                "action": "Dig", "reward_task": 0.0, "reward_comm": 0.5,
                "wielded_item": "", "hp": 20, "message": msg_text,
            })
            messages.append({
                "t": t, "sender": f"agent_{a}", "receiver": f"agent_{1-a}",
                "text": msg_text, "tokens": 3, "routing": "model",
                "model_target": f"agent_{1-a}",
                "model_target_canonical": f"agent_{1-a}",
                "valid": True, "rewarded_base": 0.5, "rewarded_milestone": 0.0,
                "chamber": "ch1", "receiver_chamber": "ch1",
            })

    events = [{"step": 3, "type": "milestone", "id": "m2_dig_3_any",
               "contributors": ["agent0"], "reward": 30}]
    coop = {
        "pair_interaction": {
            "messages": [[0, 5], [5, 0]], "joint_dig": [[0, 2], [2, 0]],
            "proximity": [[0, 8], [8, 0]], "joint_kill": [[0, 0], [0, 0]],
            "ch5_damage_overlap": [[0, 0], [0, 0]],
        },
        "hebbian_W": [[0.0, 0.3], [0.2, 0.0]],
        "milestone_log": [{"step": 3, "milestone": "m2_dig_3_any",
                           "contributors": ["agent0"],
                           "comm_before_coop": True}],
    }
    write_episode(run_dir / "episodes" / "ep_0001", steps_rows, messages,
                  events, coop)
    lw.dump(run_dir / "llm_logs")
    (run_dir / "config.json").write_text(json.dumps({
        "cli_args": {"belief_interval": 2, "critic_interval": 5,
                     "social_interval": 3, "num_agents": 2}}),
        encoding="utf-8")
    (run_dir / "final_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "hebbian_snapshots.jsonl").write_text(
        json.dumps({"episode": 1, "final_step": n_steps,
                    "W": [[0.0, 0.3], [0.2, 0.0]],
                    "cooperation_score": 0.1, "reward_total": 5.0}) + "\n",
        encoding="utf-8")
    n_a0 = n_steps
    n_a1 = agent1_dies_at if agent1_dies_at is not None else n_steps
    return run_dir, {"n_a0": n_a0, "n_a1": n_a1}
