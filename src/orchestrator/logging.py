"""JSONL logging for the orchestrator.

Everything lands under ``<run_dir>/<orchestrator.log_dir_name>/``:

  calls.jsonl        one record per orchestrator LLM call
                     {episode, t, prompt_tokens, completion_tokens, changed,
                      why, ledger_snapshot, directives, failed,
                      leakage_filtered, map_path, ...}
  compliance.jsonl   one record per routed message while the orchestrator is
                     enabled {episode, t, agent, directed_comm_target,
                      actual_comm_target, complied}
  maps/              one schematic-map PNG per call (audit artifact)

Token counts also go to the run log as a tagged line
``[Orchestrator usage] prompt_tokens=... completion_tokens=...`` so the
run's FLOPs accounting (FLOPs = 2 * N_eff * tokens; scripts/compute_flops.py
parses log.txt) can attribute orchestrator calls separately. Note the calls
themselves still emit the standard ``[LocalModel usage]`` line inside the
shared client, so they are already included in the run-level aggregate —
the orchestrator tag only makes the split recoverable.
"""

from __future__ import annotations

import json
import logging as _stdlog
import os

logger = _stdlog.getLogger(__name__)


class OrchestratorLogger:
    def __init__(self, run_dir: str, dir_name: str = "orchestrator"):
        self.dir = os.path.join(str(run_dir), dir_name)
        self.maps_dir = os.path.join(self.dir, "maps")
        os.makedirs(self.maps_dir, exist_ok=True)
        self.calls_path = os.path.join(self.dir, "calls.jsonl")
        self.compliance_path = os.path.join(self.dir, "compliance.jsonl")
        # Plan variant only: one record per task CHANGE while a plan note
        # was standing — {episode, t, agent, active_note, old_task, new_task}.
        self.task_compliance_path = os.path.join(self.dir,
                                                 "task_compliance.jsonl")
        # Villager variant only: DAG snapshots (per change, with trigger)
        # and per-agent assignment lifecycle rows (allocate / freed_*).
        self.dag_path = os.path.join(self.dir, "dag.jsonl")
        self.assignments_path = os.path.join(self.dir, "assignments.jsonl")

    @staticmethod
    def _append(path: str, record: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("orchestrator log write failed (%s): %s", path, exc)

    def log_call(self, record: dict) -> None:
        self._append(self.calls_path, record)
        logger.info(
            "[Orchestrator usage] prompt_tokens=%d completion_tokens=%d "
            "tag=orchestrator failed=%s",
            int(record.get("prompt_tokens") or 0),
            int(record.get("completion_tokens") or 0),
            bool(record.get("failed")),
        )

    def log_compliance(self, record: dict) -> None:
        self._append(self.compliance_path, record)

    def log_task_compliance(self, record: dict) -> None:
        self._append(self.task_compliance_path, record)

    def log_dag(self, record: dict) -> None:
        self._append(self.dag_path, record)

    def log_assignment(self, record: dict) -> None:
        self._append(self.assignments_path, record)

    def map_path(self, episode: int, t: int) -> str:
        return os.path.join(self.maps_dir, f"ep{episode:04d}_t{t:06d}.png")
