"""Shared orchestrator plumbing: agent-name normalisation, tolerant JSON
parsing, client construction, and the world/task snapshots the villager
controller feeds to its decomposer and allocator prompts.

Heavy imports (autogen, agent_modules) happen lazily inside functions so the
pure logic here stays importable and unit-testable without the runtime stack.
"""

from __future__ import annotations

import json
import re
from typing import Optional

_AGENT_ID_RE = re.compile(r"^\s*agent_?(\d+)\s*$", re.IGNORECASE)


def _normalize_agent(s) -> Optional[str]:
    """Canonicalize 'agent2' / 'Agent_2' / ' agent_2 ' -> 'agent_2'; else None."""
    if not isinstance(s, str):
        return None
    m = _AGENT_ID_RE.match(s)
    if m is None:
        return None
    return f"agent_{int(m.group(1))}"


# ── Response parsing ─────────────────────────────────────────────────────
# The shared load_json() salvage regex only matches ONE level of braces, and
# the backbone reliably closes a nested object one brace early and then keeps
# going — json.loads stops at the first complete document ("Extra data")
# even though both halves are individually valid. So: decode successive
# top-level chunks and merge them, rather than demanding one well-formed
# document.

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _decode_wrapped(decoder, fragment: str) -> Optional[dict]:
    """Decode a continuation fragment like ``"directives": {...}, "why": "x"}``
    by re-opening the object the model closed too early."""
    for candidate in ("{" + fragment, "{" + fragment + "}"):
        try:
            obj, _ = decoder.raw_decode(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_orchestrator_json(raw: str) -> dict:
    """Parse a response into one dict, tolerating a premature outer close.

    Well-formed output takes the fast path unchanged: the first raw_decode
    consumes the whole string. Otherwise the trailing remainder is decoded
    as a continuation and merged. Earlier keys win, so the first complete
    object stays authoritative.
    """
    text = _strip_fences(raw or "")
    if not text:
        return {}
    decoder = json.JSONDecoder()
    merged: dict = {}
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in ", \t\r\n":
            idx += 1
        if idx >= n:
            break
        if text[idx] != "{":
            if merged:
                obj = _decode_wrapped(decoder, text[idx:])
                if obj:
                    for k, v in obj.items():
                        merged.setdefault(k, v)
                break
            nxt = text.find("{", idx)  # leading prose before the first object
            if nxt < 0:
                break
            idx = nxt
            continue
        try:
            obj, end = decoder.raw_decode(text, idx)
        except ValueError:
            break
        if isinstance(obj, dict):
            for k, v in obj.items():
                merged.setdefault(k, v)
        if end <= idx:
            break
        idx = end
    return merged


# ── Client construction ──────────────────────────────────────────────────

def create_orchestrator_client(cfg, response_format,
                               max_tokens: Optional[int] = None,
                               raw_text: bool = False):
    """Build the orchestrator's LLM client.

    ``cfg.model is None`` (default) reuses the agents' backbone via the same
    ``create_model_client`` factory every other module uses. A model override
    is only supported on the HTTP-client path — the local in-process client
    holds ONE shared model, and loading a second would clobber the agents'.

    ``response_format`` is the pydantic schema injected/enforced by the
    client — the villager controller passes its decompose / allocate
    schemas, one client each. ``max_tokens`` raises the local client's
    generation cap (default 1024) for calls whose output grows with N (the
    hmas2 planner writes one entry per agent). ``response_format=None`` +
    ``raw_text=True`` is a plain client: no JSON-schema instruction injected
    and no brace slicing of the output — the hmas2 variant ports HMAS-2's
    prompts verbatim and parses their output itself. The defaults leave the
    client exactly as before.
    """
    import os

    from mindforge.agent_modules.util import create_model_client

    if cfg.model is None:
        if max_tokens is None and not raw_text:
            return create_model_client(response_format=response_format)
        return create_model_client(response_format=response_format,
                                   max_tokens=max_tokens, raw_text=raw_text)

    if os.environ.get("LLM_MODEL_PATH", ""):
        raise ValueError(
            "orchestrator.model overrides are not supported with a local "
            "in-process backbone (LLM_MODEL_PATH is set): the local client "
            "holds one shared model. Unset --orchestrator-model to reuse "
            "the agents' backbone."
        )

    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from mindforge.agent_modules.util import _resolve_api_key, base_url

    return OpenAIChatCompletionClient(
        model=cfg.model,
        base_url=base_url,
        api_key=_resolve_api_key("api.key"),
        response_format=response_format,
        model_info={
            "vision": True,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )



# ── Environment snapshot (data already produced/logged by the loop) ─────

def _parse_hp(status_text: str) -> Optional[float]:
    if not status_text or "Health:" not in status_text:
        return None
    try:
        return float(status_text.split("Health:")[1].split("/")[0].strip())
    except (ValueError, IndexError):
        return None


def collect_env_state(environment, num_agents: int, t: int,
                      recent_messages: Optional[list] = None) -> dict:
    """Build the plain-dict world snapshot the map renderer + text fallback
    consumes. Reads only state the loop already reads elsewhere (positions,
    chambers, status text, the door/anvil/cell state files)."""
    agents = {}
    for i in range(num_agents):
        name = f"agent_{i}"
        try:
            pos = environment.get_agent_position(i)
        except Exception:
            pos = None
        try:
            chamber = environment.get_chamber(i)
        except Exception:
            chamber = None
        try:
            hp = _parse_hp(environment.get_player_status_text(i) or "")
        except Exception:
            hp = None
        try:
            alive = not environment._terminations.get(name, False)
        except Exception:
            alive = True
        agents[name] = {"pos": pos, "chamber": chamber, "hp": hp,
                        "alive": alive}

    doors = {}
    for door, fname in (("door1", "door1_state.txt"),
                        ("door2", "door2_state.txt"),
                        ("door3", "door3_state.txt"),
                        ("door4", "door4_state.txt")):
        try:
            doors[door] = bool(environment._door_state_file_exists(fname))
        except Exception:
            doors[door] = False

    anvils = []
    cell_doors_open = []
    try:
        import os

        world_path = environment._get_world_path()
        try:
            with open(os.path.join(world_path, "anvils.txt"), "r") as f:
                for line in f.read().strip().splitlines():
                    fields = line.split("|")
                    if len(fields) < 2:
                        continue
                    try:
                        hp_val = int(fields[1])
                    except (TypeError, ValueError):
                        continue
                    if hp_val > 0:  # unbroken only
                        anvils.append({"kind": fields[0], "hp": hp_val})
        except (FileNotFoundError, OSError):
            pass
        try:
            with open(os.path.join(world_path, "cell_doors_state.txt"),
                      "r") as f:
                for line in f.read().strip().splitlines():
                    head = line.split(":", 1)[0].strip()
                    try:
                        cell_doors_open.append(int(head))
                    except ValueError:
                        continue
        except (FileNotFoundError, OSError):
            pass
    except Exception:
        pass

    return {
        "step": t,
        "agents": agents,
        "doors": doors,
        "anvils": anvils,
        "cell_doors_open": cell_doors_open,
        "recent_messages": list(recent_messages or []),
    }



# ── Parsing entry point ──────────────────────────────────────────────────

def _default_parse_json():
    """Tolerant parser first, the repo's shared load_json as a backstop.

    The fast path returns early only for responses carrying a top-level
    ``ledger`` key — a schema no current prompt produces, so decompose /
    allocate responses always go through load_json first and fall back to
    the tolerant parse when it yields nothing. This is exactly the behaviour
    the reported orchestrator runs had; keep it unchanged so reruns parse
    identically.
    """
    def _parse(raw: str) -> dict:
        parsed = parse_orchestrator_json(raw)
        if isinstance(parsed, dict) and "ledger" in parsed:
            return parsed
        from mindforge.agent_modules.util import load_json

        fallback = load_json(raw)
        return fallback if fallback else parsed

    return _parse



# ── Task snapshot ────────────────────────────────────────────────────────

def collect_task_table(agents, num_agents: int) -> str:
    """Per-agent task snapshot for the decomposer and allocator: current
    auto-curriculum task + recent completed/failed. Read-only over
    loop-owned objects."""
    def _clip(s, n=90):
        s = str(s or "")
        return s if len(s) <= n else s[:n - 1] + "..."

    lines = []
    for i in range(num_agents):
        try:
            cur = agents[i].auto_curriculum
        except (IndexError, AttributeError):
            continue
        done = [_clip(x) for x in (getattr(cur, "completed_tasks", []) or [])[-3:]]
        failed = [_clip(x) for x in (getattr(cur, "failed_tasks", []) or [])[-3:]]
        cur_task = _clip(cur.current_task) or "(none yet)"
        lines.append(
            f'  agent_{i}: current="{cur_task}"'
            f" | recently completed: {'; '.join(done) or '(none)'}"
            f" | recently failed: {'; '.join(failed) or '(none)'}"
        )
    return "\n".join(lines) or "  (no task information available)"

