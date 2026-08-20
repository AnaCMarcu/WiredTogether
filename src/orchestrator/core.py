"""Orchestrator call orchestration: scheduling, the LLM call, validation,
state persistence, and the two coupling surfaces (directive text for the
``{social_directive}`` prompt slot; a canonical comm_target for the routing
bias) — mirroring the SocialModule couplings exactly.

Heavy imports (autogen, agent_modules) happen lazily inside functions so the
pure logic here stays importable and unit-testable without the runtime stack.
"""

from __future__ import annotations

import json
import logging as _stdlog
import re
from typing import Optional

from pydantic import BaseModel

from orchestrator import events as _events
from orchestrator import map_render as _map_render
from orchestrator import prompt as _prompt
from orchestrator.config import OrchestratorConfig
from orchestrator.state import OrchestratorState

logger = _stdlog.getLogger(__name__)

_AGENT_ID_RE = re.compile(r"^\s*agent_?(\d+)\s*$", re.IGNORECASE)

# ── Relational-leakage filter ────────────────────────────────────────────
# The ledger must contain task/progress facts only, never relational quality
# judgments — this exclusion is scientifically load-bearing (the orchestrator
# must not become a slow Hebbian graph). Coarse keyword/regex guard; dropped
# entries are logged under ``leakage_filtered`` so leakage is quantifiable.
_LEAKAGE_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"work(s|ed|ing)?\s+(well|great|better|best)",
    r"good\s+(team|pair|partner|teammate|duo)",
    r"great\s+(team|pair|partner|teammate|duo)",
    r"trust",
    r"prefer",
    r"synerg",
    r"bond",
    r"reliab",
    r"(coordinate|cooperate)[sd]?\s+(well|better|best)",
))


def is_relational_leakage(fact: str) -> bool:
    text = str(fact)
    return any(p.search(text) for p in _LEAKAGE_PATTERNS)


class OrchestratorResponse(BaseModel):
    """Response schema handed to the shared client factory, mirroring how
    every other module enforces its JSON shape (SocialThought etc.)."""
    ledger: dict
    directives: dict
    changed: bool = True
    why: str = ""


def _normalize_agent(s) -> Optional[str]:
    """Canonicalize 'agent2' / 'Agent_2' / ' agent_2 ' -> 'agent_2'; else None."""
    if not isinstance(s, str):
        return None
    m = _AGENT_ID_RE.match(s)
    if m is None:
        return None
    return f"agent_{int(m.group(1))}"


# ── Response parsing ─────────────────────────────────────────────────────
# The shared load_json() cannot handle this schema's nesting depth. Its
# salvage regex only matches ONE level of braces, so on a 4-deep response it
# returns an inner fragment (literally {"comm_target": ..., "help": ...}),
# which then fails validation as "missing/invalid top-level 'ledger'".
#
# That matters because of a failure the backbone makes reliably here: it
# closes `ledger` one brace early, emitting stall_counter as a SIBLING of
# ledger, closing the outer object, and then continuing anyway —
#     {"ledger": {...}, "stall_counter": N}, "directives": {...}, "why": "..."}
# json.loads stops at the first complete document ("Extra data"). In the
# first smoke run this cost 14 of 27 attempts (52%), rising to 6 of the last
# 8 calls once the ledger reached its facts cap and responses got longer —
# even though BOTH halves were individually valid and complete.
#
# So: decode successive top-level chunks and merge them, rather than
# demanding one well-formed document.

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

def create_orchestrator_client(cfg: OrchestratorConfig):
    """Build the orchestrator's LLM client.

    ``cfg.model is None`` (default) reuses the agents' backbone via the same
    ``create_model_client`` factory every other module uses. A model override
    is only supported on the HTTP-client path — the local in-process client
    holds ONE shared model, and loading a second would clobber the agents'.
    """
    import os

    from agent_modules.util import create_model_client

    if cfg.model is None:
        return create_model_client(response_format=OrchestratorResponse)

    if os.environ.get("LLM_MODEL_PATH", ""):
        raise ValueError(
            "orchestrator.model overrides are not supported with a local "
            "in-process backbone (LLM_MODEL_PATH is set): the local client "
            "holds one shared model. Unset --orchestrator-model to reuse "
            "the agents' backbone."
        )

    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from agent_modules.util import _resolve_api_key, base_url

    return OpenAIChatCompletionClient(
        model=cfg.model,
        base_url=base_url,
        api_key=_resolve_api_key("api.key"),
        response_format=OrchestratorResponse,
        model_info={
            "vision": True,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )


def _client_supports_vision(client) -> bool:
    try:
        info = getattr(client, "model_info", None)
        if info is None:
            info = getattr(client, "capabilities", None)
        if info is None:
            return False
        if isinstance(info, dict):
            return bool(info.get("vision", False))
        return bool(getattr(info, "vision", False))
    except Exception:
        return False


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
    consume. Reads only state the loop already reads elsewhere (positions,
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


# ── Scheduling ───────────────────────────────────────────────────────────

def should_call(state: OrchestratorState, t: int, cfg: OrchestratorConfig,
                step_events: Optional[list] = None) -> bool:
    """True on the first call of an episode, when the cadence elapses, or
    (with event_triggers) when a milestone / chamber_change / death sits in
    the buffer since the last call.

    Only events NEWER than last_call_step count as triggers: a failed call
    keeps its events in the buffer (so the next call's digest still shows
    them) but has consumed them as triggers — otherwise a persistently
    failing model would be re-called every single step."""
    if state.last_call_step < 0:
        return True
    if t - state.last_call_step >= cfg.cadence:
        return True
    if cfg.event_triggers:
        pending = state.event_buffer + list(step_events or [])
        if any(ev.get("type") in _events.TRIGGER_TYPES
               and ev.get("t", t) > state.last_call_step
               for ev in pending):
            return True
    return False


# ── Validation ───────────────────────────────────────────────────────────

def validate_response(parsed: dict, living_agents: list, t: int) -> dict:
    """Validate + clean one parsed orchestrator response.

    Returns ``{"ok", "error", "ledger", "directives", "leakage_filtered",
    "warnings"}``. Structural problems (missing keys, missing living agents,
    self/"all"/dead comm_targets) fail the response; relational task_facts
    are FILTERED (dropped + reported), not failed — the filter is an audit
    guard, not a retry trigger.
    """
    result = {"ok": False, "error": None, "ledger": None, "directives": None,
              "leakage_filtered": [], "warnings": []}
    if not isinstance(parsed, dict):
        result["error"] = "response is not a JSON object"
        return result

    ledger = parsed.get("ledger")
    directives = parsed.get("directives")
    if not isinstance(ledger, dict):
        result["error"] = "missing/invalid top-level 'ledger'"
        return result
    if not isinstance(directives, dict):
        result["error"] = "missing/invalid top-level 'directives'"
        return result

    # The backbone reliably emits stall_counter as a SIBLING of ledger rather
    # than inside it (the same brace-nesting slip parse_orchestrator_json
    # recovers from). Accept either placement instead of discarding the value.
    if "stall_counter" not in ledger and "stall_counter" in parsed:
        ledger = {**ledger, "stall_counter": parsed["stall_counter"]}
        result["warnings"].append(
            "stall_counter arrived at the top level; hoisted into ledger")

    living = [_normalize_agent(a) or a for a in living_agents]

    # ── Directives: exactly the living agents ──
    cleaned_directives = {}
    for raw_name, entry in directives.items():
        name = _normalize_agent(raw_name)
        if name is None or name not in living:
            result["warnings"].append(
                f"stripped directive for non-living/unknown agent "
                f"{raw_name!r}")
            continue
        if not isinstance(entry, dict):
            result["error"] = f"directive for {name} is not an object"
            return result
        target = _normalize_agent(entry.get("comm_target"))
        if target is None:
            result["error"] = (
                f"directive for {name}: comm_target "
                f"{entry.get('comm_target')!r} is not a valid agent name "
                f"(never 'all')")
            return result
        if target == name:
            result["error"] = f"directive for {name}: comm_target is itself"
            return result
        if target not in living:
            result["error"] = (
                f"directive for {name}: comm_target {target} is not a "
                f"living teammate")
            return result
        cleaned_directives[name] = {
            "comm_target": target,
            "help": str(entry.get("help") or ""),
        }
    missing = [a for a in living if a not in cleaned_directives]
    if missing:
        result["error"] = f"directives missing for living agents: {missing}"
        return result

    # ── Ledger ──
    raw_facts = ledger.get("task_facts")
    if raw_facts is None:
        raw_facts = []
    if not isinstance(raw_facts, list):
        result["error"] = "ledger.task_facts is not a list"
        return result
    facts = []
    for fact in raw_facts:
        text = fact if isinstance(fact, str) else str(fact)
        if is_relational_leakage(text):
            result["leakage_filtered"].append(text)
            continue
        facts.append(text)

    progress = ledger.get("progress")
    if progress is not None and not isinstance(progress, dict):
        result["error"] = "ledger.progress is neither null nor an object"
        return result
    if progress is None:
        progress = {}
    progress = dict(progress)
    progress.setdefault("current_stage_goal", "")
    progress.setdefault("expected_signal", "")
    # assignments = copy of the (cleaned) directives; issued_at_step = now.
    progress["assignments"] = dict(cleaned_directives)
    try:
        progress["issued_at_step"] = int(progress.get("issued_at_step", t))
    except (TypeError, ValueError):
        progress["issued_at_step"] = t

    try:
        stall = int(ledger.get("stall_counter", 0))
    except (TypeError, ValueError):
        stall = 0

    result["ok"] = True
    result["ledger"] = {"task_facts": facts, "progress": progress,
                        "stall_counter": stall}
    result["directives"] = cleaned_directives
    return result


# ── The call ─────────────────────────────────────────────────────────────

def _default_parse_json():
    """Tolerant parser first, the repo's shared load_json as a backstop.

    parse_orchestrator_json handles this schema's depth (which load_json's
    one-level salvage regex cannot); load_json still covers the malformations
    it was written for — missing commas, single quotes, prose around the JSON.
    """
    def _parse(raw: str) -> dict:
        parsed = parse_orchestrator_json(raw)
        if isinstance(parsed, dict) and "ledger" in parsed:
            return parsed
        try:
            from agent_modules.util import load_json
        except ImportError:
            from mindforge.agent_modules.util import load_json
        fallback = load_json(raw)
        return fallback if fallback else parsed

    return _parse


async def orchestrate(
    state: OrchestratorState,
    env_state: dict,
    llm,
    cfg: OrchestratorConfig,
    *,
    living_agents: list,
    episode: int,
    t: int,
    orch_logger=None,
    parse_json=None,
    num_agents: Optional[int] = None,
) -> OrchestratorState:
    """Run one orchestrator call and persist the result into ``state``.

    On validation failure after the single retry, the previous directives
    and ledger are kept unchanged (``failed_calls`` incremented, raw output
    logged) — malformed content is never injected.
    """
    if parse_json is None:
        parse_json = _default_parse_json()
    n_total = num_agents if num_agents is not None else len(living_agents)

    # 1. Map (image when possible, else the text fallback block).
    map_path = None
    map_image = None
    map_text = ""
    if cfg.use_map_image and _client_supports_vision(llm):
        out_path = (orch_logger.map_path(episode, t) if orch_logger
                    else f"orchestrator_map_ep{episode}_t{t}.png")
        map_path = _map_render.render_map(env_state, out_path,
                                          num_agents=n_total)
        if map_path is not None:
            try:
                import PIL.Image as _PILImage
                from autogen_core import Image as _AutogenImage

                map_image = _AutogenImage.from_pil(_PILImage.open(map_path))
            except Exception as exc:
                logger.warning("Orchestrator: map image attach failed (%s); "
                               "falling back to text block", exc)
                map_image = None
    if map_image is None:
        map_text = _map_render.render_map_text(env_state, num_agents=n_total)

    # 2. Digest + prompt.
    digest = _events.build_digest(state.event_buffer, cfg.max_digest_events)
    filled = _prompt.format_prompt(
        n_agents=len(living_agents),
        agent_names=living_agents,
        last_call_step=state.last_call_step,
        current_step=t,
        digest=digest,
        ledger=state.ledger,
        directives=state.directives,
        stall_threshold=cfg.stall_threshold,
        map_text_fallback=map_text,
    )

    # 3-5. Call + parse + validate; max 1 retry, then keep previous state.
    from autogen_core import CancellationToken
    from autogen_core.models import UserMessage

    content = [filled] if map_image is None else [filled, map_image]
    prompt_tokens = 0
    completion_tokens = 0
    verdict = None
    raw_tail = None
    for attempt in range(2):  # initial + max 1 retry
        try:
            response = await llm.create(
                [UserMessage(content=content, source="user")],
                cancellation_token=CancellationToken(),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            logger.error("Orchestrator: LLM call failed (attempt %d): %s",
                         attempt + 1, str(exc)[:300])
            continue
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens += int(
                getattr(usage, "completion_tokens", 0) or 0)
        raw = response.content if isinstance(response.content, str) \
            else str(response.content)
        # 6000, not 2000: at 2000 the first smoke run's failed responses were
        # clipped mid-object, so the log could not distinguish "model was
        # truncated" from "model closed a brace early" without a re-run.
        raw_tail = raw[:6000]
        # The repo's load_json returns {} on garbage, but tolerate parsers
        # that raise instead — either way it's a failed attempt, not a crash.
        try:
            parsed = parse_json(raw)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Orchestrator: JSON parse raised (attempt %d): %s",
                           attempt + 1, exc)
            parsed = None
        verdict = validate_response(parsed, living_agents, t)
        if verdict["ok"]:
            break
        logger.warning("Orchestrator: response failed validation "
                       "(attempt %d): %s", attempt + 1, verdict["error"])

    record = {
        "episode": episode,
        "t": t,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "map_path": map_path,
        "digest_events": len(state.event_buffer),
    }

    if verdict is not None and verdict["ok"]:
        parsed_changed = bool(parsed.get("changed", True)) \
            if isinstance(parsed, dict) else True
        why = str(parsed.get("why") or "") if isinstance(parsed, dict) else ""
        state.apply_success(verdict["ledger"], verdict["directives"], t,
                            cfg.max_task_facts)
        record.update({
            "failed": False,
            "changed": parsed_changed,
            "why": why,
            "ledger_snapshot": state.ledger,
            "directives": state.directives,
            "leakage_filtered": verdict["leakage_filtered"],
            "warnings": verdict["warnings"],
        })
        if verdict["leakage_filtered"]:
            logger.warning("Orchestrator: filtered %d relational task_facts: "
                           "%s", len(verdict["leakage_filtered"]),
                           verdict["leakage_filtered"])
    else:
        state.record_failure(t)
        record.update({
            "failed": True,
            "changed": False,
            "why": (verdict["error"] if verdict is not None
                    else "LLM call failed"),
            "ledger_snapshot": state.ledger,
            "directives": state.directives,
            "leakage_filtered": [],
            "raw_output": raw_tail,
        })
        logger.error("Orchestrator: keeping previous ledger/directives "
                     "(failed call #%d). Raw: %s",
                     state.failed_calls, (raw_tail or "")[:300])

    if orch_logger is not None:
        orch_logger.log_call(record)
    return state


# ── Coupling surfaces ────────────────────────────────────────────────────

def render_agent_directive(agent_name: str, state: OrchestratorState) -> str:
    """Render this agent's standing directive for the ``{social_directive}``
    slot in the action prompt (advisory coupling — same slot, same tone as
    SocialModule.render_directive)."""
    entry = state.directives.get(_normalize_agent(agent_name) or agent_name)
    if not entry:
        return ("Coordinator directive: (none yet — the team coordinator has "
                "not issued directives)")
    target = entry.get("comm_target", "")
    help_text = entry.get("help", "")
    return (
        "Coordinator directive (from the non-embodied team coordinator; "
        "it sees a top-down view of the whole team):\n"
        f"  Talk to: {target} — put {target} in your communication_target "
        f"field when you communicate this step.\n"
        f"  Help: {help_text or '(no specific help suggestion)'}\n"
        "  You may deviate if your local view clearly contradicts this."
    )


def directive_comm_target(state: OrchestratorState,
                          agent_name: str) -> Optional[str]:
    """Canonical 'agent_N' the orchestrator directed ``agent_name`` to
    message, or None. Used by the bias coupling at the routing site and by
    the per-step compliance log."""
    entry = state.directives.get(_normalize_agent(agent_name) or agent_name)
    if not entry:
        return None
    return _normalize_agent(entry.get("comm_target"))
