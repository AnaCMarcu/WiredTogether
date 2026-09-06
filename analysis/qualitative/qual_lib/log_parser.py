"""Parser for llm_logs/*.log files written by mindforge llm_call.py.

Line grammar (FileHandler formatter "%(asctime)s %(levelname)s %(message)s",
datefmt "%Y-%m-%d %H:%M:%S"; the message starts with the caller's log_prefix):

    2026-06-26 10:00:39 INFO Agent agent_0 on_messages:  call: sys_chars=4813 user_chars=3308 frame=True kwargs=['agent_name', ...] retry=0
    2026-06-26 10:00:41 INFO Agent agent_0 on_messages:  Response: {"thoughts": ...}
    2026-06-26 10:00:41 ERROR Critic check_task_success:  Error parsing response (attempt 1): ValueError: Empty JSON response

Response payloads may span multiple lines (pretty-printed JSON): every line
that does not start with a timestamp header is a continuation of the previous
record. A retry chain (retry=0,1,2... of the same prefix) is folded into ONE
unit; the last parseable Response wins.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qual_lib import registry  # noqa: F401  (path bootstrap)
from qual_lib.json_robust import load_json  # vendored util.load_json (no autogen dep)

_HEADER = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>INFO|ERROR|WARNING) (?P<msg>.*)$"
)
# Split log_prefix from the event part of the message.
_EVENT = re.compile(
    r"^(?P<prefix>.*?)\s*(?P<kind>call:|Response:|Too many retries|"
    r"Missing prompt placeholder|Error calling LLM|Error parsing response)"
    r"(?P<rest>.*)$",
    re.DOTALL,
)
_RETRY = re.compile(r"retry=(\d+)\s*$")
_AGENT = re.compile(r"agent[_\s]?(\d+)", re.IGNORECASE)

# prefix start -> module name (mirrors llm_call._MODULE_PATTERNS routing)
_PREFIX_MODULE = (
    ("Agent ", None),  # method decides: on_messages / rl_thoughts / rl_comm
    ("Auto Curriculum", "auto_curriculum"),
    ("Belief System", "belief_system"),
    ("Critic", "critic"),
    ("EpisodicMemoryManager", "episodic_memory"),
    ("SocialModule", "social_module"),
    ("SkillManager", "skill_manager"),
)


def _classify_prefix(prefix: str):
    """(module, method, agent) from a log_prefix string."""
    prefix = prefix.strip().rstrip(":")
    agent = None
    m = _AGENT.search(prefix)
    if m:
        agent = f"agent_{m.group(1)}"
    module = "llm_other"
    for start, mod in _PREFIX_MODULE:
        if prefix.startswith(start):
            module = mod
            break
    # method = last whitespace-separated token that isn't the agent name
    tokens = [t for t in re.split(r"[\s\[\]]+", prefix) if t]
    method = tokens[-1] if tokens else ""
    if _AGENT.fullmatch(method or ""):
        method = tokens[-2] if len(tokens) >= 2 else ""
    if module is None:  # "Agent ..." prefixes
        module = {
            "on_messages": "action_selection",
            "rl_thoughts": "rl_thoughts",
            "rl_comm": "action_selection_comm",
        }.get(method, "llm_other")
    return module, method, agent


def _iter_raw_records(path: Path):
    """Yield (ts, level, message, line_start, line_end) with continuations joined."""
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            m = _HEADER.match(line)
            if m:
                if cur is not None:
                    yield cur
                cur = [m.group("ts"), m.group("level"), m.group("msg"), i, i]
            elif cur is not None:
                cur[2] += "\n" + line
                cur[4] = i
    if cur is not None:
        yield cur


def _parse_response_payload(rest: str):
    """(parsed_dict_or_None, raw_text). rest is everything after 'Response:'."""
    raw = rest.strip()
    if not raw:
        return None, raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, raw
    except json.JSONDecodeError:
        pass
    parsed = load_json(raw)
    if parsed:
        return parsed, raw
    return None, raw


def iter_llm_units(path: Path):
    """Yield one dict per retry-group unit in file order.

    Unit schema: {module, method, agent, ts (first call), ts_last, file,
    line_start, line_end, n_retries, n_errors, parse_ok, response, raw_chars}.
    """
    open_units: dict = {}  # prefix -> unit under construction
    order: list = []       # completed units in file order (by first line)

    def _close(prefix):
        u = open_units.pop(prefix, None)
        if u is not None:
            order.append(u)

    for ts, level, msg, l0, l1 in _iter_raw_records(path):
        ev = _EVENT.match(msg)
        if not ev:
            continue
        prefix, kind, rest = ev.group("prefix"), ev.group("kind"), ev.group("rest")
        module, method, agent = _classify_prefix(prefix)

        if kind == "call:":
            rm = _RETRY.search(rest.split("\n", 1)[0])
            retry = int(rm.group(1)) if rm else 0
            if retry == 0 or prefix not in open_units:
                _close(prefix)
                open_units[prefix] = {
                    "module": module, "method": method, "agent": agent,
                    "ts": ts, "ts_last": ts, "file": path.name,
                    "line_start": l0, "line_end": l1,
                    "n_retries": retry, "n_errors": 0,
                    "parse_ok": False, "response": None, "raw_chars": 0,
                }
            else:
                u = open_units[prefix]
                u["n_retries"] = max(u["n_retries"], retry)
                u["ts_last"] = ts
                u["line_end"] = l1
        elif kind == "Response:":
            u = open_units.get(prefix)
            if u is None:  # Response without a call line (truncated file head)
                u = {
                    "module": module, "method": method, "agent": agent,
                    "ts": ts, "ts_last": ts, "file": path.name,
                    "line_start": l0, "line_end": l1,
                    "n_retries": 0, "n_errors": 0,
                    "parse_ok": False, "response": None, "raw_chars": 0,
                }
                open_units[prefix] = u
            parsed, raw = _parse_response_payload(rest)
            u["ts_last"] = ts
            u["line_end"] = l1
            u["raw_chars"] = len(raw)
            if parsed is not None:
                u["parse_ok"] = True
                u["response"] = parsed
            elif u["response"] is None:
                u["response"] = {"_raw": raw[:4000]}
        else:  # error kinds
            u = open_units.get(prefix)
            if u is not None:
                u["n_errors"] += 1
                u["ts_last"] = ts
                u["line_end"] = l1
                if kind == "Too many retries":
                    _close(prefix)

    for prefix in list(open_units):
        _close(prefix)
    order.sort(key=lambda u: u["line_start"])
    return order


def parse_run_llm_logs(run) -> list:
    """All units across a run's llm_logs, tagged with file, in per-file order."""
    units = []
    for stem, path in run.llm_logs.items():
        for u in iter_llm_units(path):
            units.append(u)
    return units
