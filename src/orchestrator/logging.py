"""JSONL logging for the orchestrator.

Everything lands under ``<run_dir>/<orchestrator.log_dir_name>/``:

  calls.jsonl            one record per decomposer / allocator LLM call
  dag.jsonl              DAG snapshot per change, with its trigger
  assignments.jsonl      per-agent assignment lifecycle (allocate / freed_*)
  task_compliance.jsonl  one record per curriculum task change while an
                         objective was assigned
  hmas2.jsonl            hmas2 variant: one record per step's plan -> check
                         -> revise protocol (rounds, objections, final plan,
                         message deliveries, latency)

Token counts also go to the run log as a tagged line
``[Orchestrator usage] prompt_tokens=... completion_tokens=...`` so the
run's FLOPs accounting (FLOPs = 2 * N_eff * tokens; analysis/compute_flops.py
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
        os.makedirs(self.dir, exist_ok=True)
        self.calls_path = os.path.join(self.dir, "calls.jsonl")
        # One record per task CHANGE while an objective was assigned —
        # {episode, t, agent, active_note, old_task, new_task}.
        self.task_compliance_path = os.path.join(self.dir,
                                                 "task_compliance.jsonl")
        # DAG snapshots (per change, with trigger) and per-agent assignment
        # lifecycle rows (allocate / freed_*).
        self.dag_path = os.path.join(self.dir, "dag.jsonl")
        self.assignments_path = os.path.join(self.dir, "assignments.jsonl")
        # hmas2 variant only: one record per step's planning protocol —
        # {episode, t, rounds, syntax_reprompts, objections, checked,
        # plan/check tokens + latency, final plan, reassigned, reports_in,
        # reports_dropped, messages{agent: delivery}, ...}.
        self.hmas2_path = os.path.join(self.dir, "hmas2.jsonl")

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

    def log_task_compliance(self, record: dict) -> None:
        self._append(self.task_compliance_path, record)

    def log_dag(self, record: dict) -> None:
        self._append(self.dag_path, record)

    def log_assignment(self, record: dict) -> None:
        self._append(self.assignments_path, record)

    def log_hmas2(self, record: dict) -> None:
        self._append(self.hmas2_path, record)
