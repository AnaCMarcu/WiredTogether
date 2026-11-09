"""Write one renamed task YAML (plus its verifier) per chain step.

    python chains/generate_tasks.py chains/smithing_3.yaml [--agentworld-root DIR]

For every step this writes ``chains/generated/<chain>/step_NN_task_MM/``:

* ``<original stem>.yaml`` — the source task with every old username replaced
  by ``<agent_id>_<role>``: the ``username`` fields and every mention in the
  text (both ``t06_smelter_agent`` and the unprefixed ``smelter_agent`` the
  task prose uses). Nothing else changes: the replacement runs on the raw
  text, and the result is checked to differ from the source only in those
  strings. ``new_character: true`` must already be set (fail otherwise), so
  every task starts from a clean character.
* ``<original stem>_success_criteria.py`` — the source task's verifier,
  copied byte for byte. ``agents/verification.py`` finds it next to the YAML;
  its ``from verifier_utils import ...`` resolves because that loader puts the
  benchmark's ``verifier_utils`` in ``sys.modules`` before loading a verifier.

The original stem is kept so ``task_id`` / ``task_key``, the trajectory
filename (``task_07_trajectory.json``) and AgentWorld's scorers keep working;
the step folder keeps a chain that repeats a task from overwriting itself.
The source files are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from chain_spec import (TASK_PREFIX, Chain, ChainError, ChainStep, load_chain,  # noqa: E402
                        resolve_root, slot_order)

GENERATED = HERE / "generated"
_WORD = r"A-Za-z0-9_"


@dataclass
class GeneratedStep:
    step: ChainStep
    directory: Path
    task_yaml: Path
    verifier: Path
    source_verifier: Path

    def manifest(self) -> Dict[str, Any]:
        return {"step": self.step.index, "dir": self.directory.name,
                "task_key": self.step.task_stem, "source": str(self.step.source),
                "task_yaml": str(self.task_yaml), "verifier": str(self.verifier),
                "source_verifier": str(self.source_verifier),
                "slots": [a.to_dict() for a in self.step.assignments]}


# ── renaming ─────────────────────────────────────────────────────────────────
def name_variants(original: str) -> List[str]:
    """Forms a task uses for one agent: ``t06_smelter_agent`` and ``smelter_agent``."""
    out = [original]
    bare = TASK_PREFIX.sub("", original, count=1)
    if bare != original:
        out.append(bare)
    return out


def replacement_table(step: ChainStep) -> Dict[str, str]:
    """Lower-cased old form -> new username, for every slot of the step."""
    table: Dict[str, str] = {}
    for a in step.assignments:
        for form in name_variants(a.original_username):
            key = form.lower()
            if key in table and table[key] != a.username:
                raise ChainError(f"step {step.index}: {form!r} would name two agents")
            table[key] = a.username
    return table


def _pattern(forms) -> re.Pattern:
    # Longest first, one pass: "weapon_smith_agent" wins over "smith_agent", and a
    # replacement is never rewritten again. Word-character guards stop
    # "smith_agent" from matching inside "weapon_smith_agent".
    alts = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<![{_WORD}])(?:{alts})(?![{_WORD}])", re.IGNORECASE)


def rewrite_text(text: str, table: Dict[str, str]) -> str:
    if not table:
        return text
    return _pattern(table).sub(lambda m: table[m.group(0).lower()], text)


def _same_shape(orig: Any, new: Any, table: Dict[str, str], path: str = "") -> None:
    """``new`` must equal ``orig`` with the replacements applied inside strings only."""
    if isinstance(orig, dict):
        if not isinstance(new, dict) or list(orig) != list(new):
            raise ChainError(f"renaming changed the YAML keys at {path or '/'}")
        for k in orig:
            _same_shape(orig[k], new[k], table, f"{path}/{k}")
    elif isinstance(orig, list):
        if not isinstance(new, list) or len(orig) != len(new):
            raise ChainError(f"renaming changed a list at {path}")
        for i, (a, b) in enumerate(zip(orig, new)):
            _same_shape(a, b, table, f"{path}[{i}]")
    elif isinstance(orig, str):
        if new != rewrite_text(orig, table):
            raise ChainError(f"renaming changed {path} beyond the username replacements")
    elif orig != new or type(orig) is not type(new):
        raise ChainError(f"renaming changed a non-text value at {path}: {orig!r} -> {new!r}")


def rename_task_text(raw: str, step: ChainStep) -> str:
    """The renamed YAML text, checked; raises ChainError on anything unexpected."""
    table = replacement_table(step)
    new_raw = rewrite_text(raw, table)
    orig = yaml.safe_load(raw)
    new = yaml.safe_load(new_raw)
    _same_shape(orig, new, table)
    for a in step.assignments:
        block = new.get(a.slot) or {}
        if block.get("username") != a.username:
            raise ChainError(f"step {step.index}: {a.slot}.username is {block.get('username')!r}, "
                             f"expected {a.username!r}")
        if block.get("new_character") is not True:
            raise ChainError(
                f"step {step.index}: {a.slot} in {step.source.name} does not set "
                "new_character: true, so the character would not be reset between tasks")
    leftover = _pattern(table).search(new_raw)
    if leftover:
        raise ChainError(f"step {step.index}: old username {leftover.group(0)!r} still present")
    if slot_order(new) != [a.slot for a in step.assignments]:
        raise ChainError(f"step {step.index}: slots changed while renaming")
    return new_raw


# ── verifiers ────────────────────────────────────────────────────────────────
def load_verification(root: Path):
    """AgentWorld's own ``agents/verification.py`` (its verifier-selection rules)."""
    import importlib.util
    path = root / "agents" / "verification.py"
    if not path.is_file():
        raise ChainError(f"{root} is not an AgentWorld checkout (no agents/verification.py)")
    spec = importlib.util.spec_from_file_location("agentworld_verification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_step(step: ChainStep, out_dir: Path, verification) -> GeneratedStep:
    raw = step.source.read_bytes().decode("utf-8")      # keeps the file's own line endings
    new_raw = rename_task_text(raw, step)
    step_dir = out_dir / step.dir_name
    step_dir.mkdir(parents=True, exist_ok=True)
    task_yaml = step_dir / step.source.name
    task_yaml.write_bytes(new_raw.encode("utf-8"))

    source_verifier = Path(verification.verifier_for_task(step.source))
    verifier = step_dir / f"{step.task_stem}_success_criteria.py"
    shutil.copyfile(source_verifier, verifier)
    found = Path(verification.verifier_for_task(task_yaml))
    if found.resolve() != verifier.resolve():
        raise ChainError(f"AgentWorld would load {found} for {task_yaml}, not the copy {verifier}")
    # Executes the copy's imports through AgentWorld's loader (verifier_utils preloaded).
    saved_flag, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        verification._load_verifier(verifier)
    finally:
        sys.dont_write_bytecode = saved_flag
    return GeneratedStep(step=step, directory=step_dir, task_yaml=task_yaml,
                         verifier=verifier, source_verifier=source_verifier)


def generate_chain(chain: Chain, root: Path, out_dir: Optional[Path] = None) -> List[GeneratedStep]:
    """Write every step into ``out_dir`` (default ``chains/generated/<chain>``), replacing it."""
    verification = load_verification(root)
    out_dir = Path(out_dir) if out_dir is not None else GENERATED / chain.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    steps = [generate_step(s, out_dir, verification) for s in chain.steps]
    manifest = {"chain": chain.name, "chain_file": str(chain.path), "agentworld_root": str(root),
                "agent_ids": chain.agent_ids, "steps": [g.manifest() for g in steps]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return steps


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chain", type=Path, help="chain YAML, e.g. chains/smithing_3.yaml")
    ap.add_argument("--agentworld-root", default=None, help="default: $AGENTWORLD_ROOT")
    ap.add_argument("--out", type=Path, default=GENERATED,
                    help="parent folder; the chain goes in <out>/<chain> (default: chains/generated)")
    args = ap.parse_args(argv)
    root = resolve_root(args.agentworld_root)
    if root is None:
        ap.error("set --agentworld-root or $AGENTWORLD_ROOT")
    chain = load_chain(args.chain, root)
    for g in generate_chain(chain, root, args.out / chain.name):
        names = ", ".join(f"{a.slot}={a.username}" for a in g.step.assignments)
        print(f"step {g.step.index}: {g.task_yaml}  [{names}]")


if __name__ == "__main__":
    main()
