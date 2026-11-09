"""Check a finished chain run (``runs/<chain>/<run_id>``).

    python chains/check_chain_run.py runs/smithing_3/<run_id>

``run_chain.py`` runs these checks after the last step and stores the result
in ``chain_summary.json``. Four checks, each PASS/FAIL with the evidence:

usernames   every username in each step's trajectory task definition is
            ``<agent_id>_<role>`` (server-valid, lowercase, <= 32 chars) and
            matches the chain's slot map; no original username survives in it;
            team players seen in observations use the same names.
reset       each agent's round-0 observation (taken right after login and the
            task's initial setup) holds no item it ended the previous step with
            unless the task YAML gives it that item, and no item the YAML does
            not list beyond what the server adds to every fresh character.
state       step 1 creates every agent's state file; each later step loads the
            file saved by that agent's previous step; every file ends with one
            history entry per step the agent played.
verifiers   every step has a verifier message without errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from chain_spec import NamingError, TASK_PREFIX, parse_username  # noqa: E402

POSITION_TOLERANCE = 10   # tiles; apply_initial_state accepts anything this close


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory(obs: Optional[Dict[str, Any]]) -> Optional[Counter]:
    items = ((obs or {}).get("inventory") or {}).get("items")
    if items is None:
        return None
    c: Counter = Counter()
    for it in items:
        if isinstance(it, dict) and it.get("key"):
            c[str(it["key"]).lower()] += int(it.get("count", 0) or 0)
    return c


def _yaml_inventory(block: Dict[str, Any]) -> Counter:
    c: Counter = Counter()
    for it in block.get("inventory_items") or []:
        c[str(it["item"]).lower()] += int(it.get("count", 1))
    return c


def _round0(step_dir: Path, slot: str) -> Optional[Dict[str, Any]]:
    for f in sorted((step_dir / "observation_logs").glob(f"{slot}_observations_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("round_number") == 0:
                return rec.get("observation")
    return None


def _final_obs(traj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in traj.get("rounds") or []:
        for a in r.get("actions") or []:
            obs = a.get("observation")
            if isinstance(obs, dict) and "inventory" in obs:
                out[a.get("agent_name")] = obs
    return out


def _players(traj: Dict[str, Any]) -> set:
    names = set()
    for r in traj.get("rounds") or []:
        for a in r.get("actions") or []:
            for p in ((a.get("observation") or {}) if isinstance(a.get("observation"), dict)
                      else {}).get("players") or []:
                if isinstance(p, dict) and p.get("name"):
                    names.add(str(p["name"]))
    return names


def _task_yaml(run_dir: Path, s: Dict[str, Any]) -> Path:
    """The step's task file inside the run (absolute paths break once a run is copied)."""
    local = run_dir / "tasks" / s["dir"] / Path(s["task_yaml"].replace("\\", "/")).name
    return local if local.is_file() else Path(s["task_yaml"])


def _trajectory(step_dir: Path) -> Optional[Dict[str, Any]]:
    files = sorted(step_dir.glob("*_trajectory.json"))
    return _load(files[0]) if files else None


def check_usernames(steps: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    problems, seen = [], []
    for s in steps:
        traj = _trajectory(run_dir / s["dir"])
        if traj is None:
            problems.append(f"step {s['step']}: no trajectory")
            continue
        tdef = traj.get("task_definition") or {}
        team = set()
        for slot in s["slots"]:
            name = (tdef.get(slot["slot"]) or {}).get("username")
            seen.append(name)
            team.add(name)
            try:
                aid, role = parse_username(name or "")
            except NamingError as exc:
                problems.append(f"step {s['step']} {slot['slot']}: {exc}")
                continue
            if (aid, role, name) != (slot["agent_id"], slot["role"], slot["username"]):
                problems.append(f"step {s['step']} {slot['slot']}: {name} != {slot['username']}")
        text = json.dumps(tdef)
        for slot in s["slots"]:
            old = slot["original_username"]
            for form in {old, TASK_PREFIX.sub("", old, count=1)}:
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(form)}(?![A-Za-z0-9_])", text, re.I):
                    problems.append(f"step {s['step']}: old name {form!r} in the task definition")
        for p in _players(traj):
            low = p.lower()
            if low in {n.lower() for n in team if n} and low != p:
                problems.append(f"step {s['step']}: player shown as {p!r}")
    return {"ok": not problems, "problems": problems, "names": sorted({n for n in seen if n})}


def check_reset(steps: List[Dict[str, Any]], run_dir: Path, random_spawn: bool) -> Dict[str, Any]:
    problems, notes = [], []
    last_final: Dict[str, Counter] = {}          # agent id -> inventory at the end of its last step
    fresh_extras: Optional[Counter] = None
    for s in steps:
        step_dir = run_dir / s["dir"]
        task = yaml.safe_load(_task_yaml(run_dir, s).read_text(encoding="utf-8"))
        traj = _trajectory(step_dir) or {}
        finals = _final_obs(traj)
        for slot in s["slots"]:
            aid, sl = slot["agent_id"], slot["slot"]
            block = task.get(sl) or {}
            obs0 = _round0(step_dir, sl)
            inv0 = _inventory(obs0)
            if inv0 is None:
                problems.append(f"step {s['step']} {aid}: no round-0 observation with an inventory")
                continue
            want = _yaml_inventory(block)
            extra = inv0 - want
            missing = want - inv0
            carried = Counter({k: v for k, v in extra.items() if k in last_final.get(aid, {})})
            if carried:
                problems.append(f"step {s['step']} {aid}: carried over {dict(carried)}")
            other = extra - carried
            if other:
                if fresh_extras is None and s is steps[0]:
                    fresh_extras = other           # what the server gives a fresh character
                if fresh_extras is None or other - fresh_extras:
                    problems.append(f"step {s['step']} {aid}: unexpected items at start {dict(other)}")
                else:
                    notes.append(f"step {s['step']} {aid}: server starter items {dict(other)}")
            if missing:
                notes.append(f"step {s['step']} {aid}: YAML items missing at start {dict(missing)}")
            loc, want_loc = (obs0 or {}).get("location") or {}, block.get("location")
            if want_loc and not random_spawn and "x" in loc:
                d = abs(loc["x"] - want_loc["x"]) + abs(loc["y"] - want_loc["y"])
                if d > POSITION_TOLERANCE:
                    problems.append(f"step {s['step']} {aid}: starts {d} tiles from the YAML spawn")
            fin = _inventory(finals.get(sl))
            if fin is not None:
                last_final[aid] = fin
        if s["step"] == steps[0]["step"] and len(steps) == 1:
            notes.append("only one step: reset between steps not exercised")
    return {"ok": not problems, "problems": problems, "notes": notes}


def check_state(steps: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    problems: List[str] = []
    last_step: Dict[str, int] = {}           # agent id -> last step it played
    order: Dict[str, List[int]] = {}         # agent id -> every step it played
    for s in steps:
        loaded = (s.get("state") or {}).get("loaded") or {}
        saved = (s.get("state") or {}).get("saved") or {}
        for slot in s["slots"]:
            aid = slot["agent_id"]
            before = last_step.get(aid)
            info = loaded.get(aid)
            if info is None:
                problems.append(f"step {s['step']} {aid}: no load record")
            elif before is None and info.get("file_existed"):
                problems.append(f"step {s['step']} {aid}: first appearance but a state file existed")
            elif before is not None and (not info.get("file_existed")
                                         or info.get("saved_after_step") != before):
                problems.append(f"step {s['step']} {aid}: loaded state from step "
                                f"{info.get('saved_after_step')}, expected step {before}")
            if aid not in saved:
                problems.append(f"step {s['step']} {aid}: state not saved")
            order.setdefault(aid, []).append(s["step"])
            last_step[aid] = s["step"]
    for aid, steps_played in order.items():
        f = run_dir / "agent_state" / f"{aid}.json"
        if not f.is_file():
            problems.append(f"{aid}: {f} missing")
            continue
        hist = [h.get("step") for h in _load(f).get("history") or []]
        if hist != steps_played:
            problems.append(f"{aid}: history steps {hist}, played {steps_played}")
    return {"ok": not problems, "problems": problems,
            "files": sorted(str(p) for p in (run_dir / "agent_state").glob("*.json"))}


def check_verifiers(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    problems = []
    for s in steps:
        msg = s.get("verifier_message")
        if not msg:
            problems.append(f"step {s['step']}: no verifier message")
        elif "error" in msg.lower():
            problems.append(f"step {s['step']}: {msg}")
        if s.get("runner_verified") is not None and s.get("runner_verified") != s.get("success"):
            problems.append(f"step {s['step']}: runner said {s.get('runner_verified')}, "
                            f"verifier said {s.get('success')}")
    return {"ok": not problems, "problems": problems,
            "messages": {s["step"]: s.get("verifier_message") for s in steps}}


def run_checks(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    summary = _load(run_dir / "chain_summary.json")
    steps = [s for s in summary.get("steps") or [] if not s.get("error")]
    random_spawn = bool((summary.get("flags") or {}).get("random_spawn"))
    out = {"usernames": check_usernames(steps, run_dir),
           "reset": check_reset(steps, run_dir, random_spawn),
           "state": check_state(steps, run_dir),
           "verifiers": check_verifiers(steps)}
    out["ok"] = all(v["ok"] for v in out.values()) and bool(steps)
    return out


def report(result: Dict[str, Any]) -> str:
    lines = []
    for name in ("usernames", "reset", "state", "verifiers"):
        r = result[name]
        lines.append(f"{'PASS' if r['ok'] else 'FAIL'}  {name}")
        lines += [f"      - {p}" for p in r.get("problems", [])]
        lines += [f"      . {n}" for n in r.get("notes", [])]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    result = run_checks(args.run_dir)
    print(report(result))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
