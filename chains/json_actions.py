"""Actions as one JSON reply instead of native tool calls (``wired_agent:WiredJSONAgent``).

AgentWorld's OpenAI agent sends its tools through the API's function calling
with ``tool_choice="required"``, so the server must turn the model's output
into a structured tool call. With Gemma 4 on our vLLM (0.30, ``gemma4``
parser) that failed on about half of all turns in the metallurgy_4 pilot:
0 of 108 first turns but ~48% of later turns came back with no tool call and
no text at all, and AgentWorld counts such a turn as "thinking". Our own
AgentWorld harness never had empty turns: the model writes plain text, one
JSON object whose ``action`` holds the call as a string
(``harvest_resource(targetInstance="645004284")``), and our parser turns it
into the call (``src/agentworld/actions.py``).

:class:`JSONActionMixin` replaces only ``_make_api_call``:

* the request carries no ``tools``: the tools offered this turn (AgentWorld's
  schemas, with their full descriptions, plus ``direct_chat``) are written
  into the system prompt as calls with placeholders, and the model is asked
  for ``{"thoughts": ..., "action": ...}``;
* AgentWorld's conversation history (its assistant tool calls and tool
  results) is sent as plain text: a past call becomes the JSON reply that
  would have produced it, a tool result becomes a user message, empty
  assistant turns are dropped and same-role neighbours are merged so roles
  alternate;
* the reply is parsed against the offered tools' schemas (types coerced:
  ids and usernames written bare become strings, as AgentWorld's tools compare
  them with the observation's string ids); a call the literal parser rejects
  (quotes or apostrophes inside a message, a missing closing bracket, trailing
  punctuation) is read by the tool's own parameter names (:func:`lenient_args`).
  Of the 125 unusable replies in the metallurgy_4 JSON pilot, 40 were bare
  usernames and 71 calls the literal parser could not match, nearly all
  direct_chat; 8 named tools that do not exist. The call is returned in the OpenAI response shape, so
  AgentWorld's ``_extract_tool_calls``, history and ``[TOOL_CALL_INFO]``
  records work unchanged;
* an unusable reply is retried once in the same turn with the error shown;
  if that fails too the turn has no tool call, as in AgentWorld.

Sampling follows our harness's vLLM client: temperature 0.7, top_p 0.9,
max_tokens 1024, Gemma thinking off. Everything else (runner, rounds, tools,
game server, verifiers, scoring) is AgentWorld's. Report it as a harness
change when comparing with AgentWorld's published numbers.
"""

from __future__ import annotations

import ast
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

ANSWER_FORMAT = ('{"thoughts": "<one or two sentences of reasoning>", '
                 '"action": "<exactly one tool call, e.g. move_character(x=120, y=40)>"}')
NEXT_TURN = "Choose your next action. Answer with ONE JSON object."

#: Placeholder per argument name; others fall back to the schema type.
ARG_HINTS = {
    "x": "<x>", "y": "<y>", "index": "<inventory slot number>",
    "targetInstance": "<instance id from your observation>",
    "targetPlayer": "<exact player name>", "itemKey": "<item key>",
    "skill": "<skill, e.g. Smithing>", "count": "<number>", "seconds": "<number>",
    "message": "<text>", "response": "<what your team finished>",
    "target": "<teammate username>",
}

_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_HEAD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_INSTANCE_ID = re.compile(r"\s*(?:instance|id)?\s*[:#]?\s*(\d+)\s*", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CONTEXT_ERROR = re.compile(r"maximum context length|max_tokens|max_completion_tokens", re.IGNORECASE)


# ── tool schemas -> prompt text ──────────────────────────────────────────────
def _function(tool: Dict[str, Any]) -> Dict[str, Any]:
    return tool.get("function", tool)


def tool_specs(tools: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """name -> {"properties": {...} (schema order), "required": [...], "description": str}."""
    out: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        f = _function(tool)
        params = f.get("parameters") or {}
        props = dict(params.get("properties") or {})
        out[f.get("name", "")] = {"properties": props,
                                  "required": [r for r in params.get("required") or [] if r in props],
                                  "description": " ".join(str(f.get("description") or "").split())}
    return out


def _arg_shape(name: str, prop: Dict[str, Any]) -> str:
    hint = ARG_HINTS.get(name, f"<{prop.get('type', 'value')}>")
    return f'{name}="{hint}"' if prop.get("type") == "string" else f"{name}={hint}"


def call_shape(name: str, spec: Dict[str, Any]) -> str:
    """The call with every required argument as a placeholder."""
    req = spec["required"]
    return f"{name}({', '.join(_arg_shape(a, spec['properties'][a]) for a in req)})"


def tool_reference(tools: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for name, spec in tool_specs(tools).items():
        line = f"- {call_shape(name, spec)}"
        optional = [a for a in spec["properties"] if a not in spec["required"]]
        if optional:
            line += "   optional: " + ", ".join(_arg_shape(a, spec["properties"][a]) for a in optional)
        if spec["description"]:
            line += f"\n    {spec['description']}"
        lines.append(line)
    return "\n".join(lines)


def instructions(tools: Sequence[Dict[str, Any]]) -> str:
    return (
        "=== HOW TO ACT ===\n"
        "You act by writing ONE tool call as text in the \"action\" field of a JSON reply "
        "(there is no separate function-calling interface). The tools you can use now "
        "(write the call exactly like these, with keyword arguments, replacing each <...> "
        "with a real value):\n"
        f"{tool_reference(tools)}\n\n"
        "Rules:\n"
        "- Fill in every argument a tool shows. A tool name on its own does nothing.\n"
        "- Use exact instance ids, item keys and usernames from your observation and the task.\n"
        "- Exactly one tool call per reply.\n\n"
        f"Respond with ONE JSON object and nothing else:\n{ANSWER_FORMAT}"
    )


# ── reply -> tool call ───────────────────────────────────────────────────────
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """The first JSON object in ``text`` (code fences, prose and <think> blocks allowed)."""
    text = _THINK.sub("", text or "")
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):          # a bare word: target=a1_gold_miner
        return node.id
    value = ast.literal_eval(node)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("only plain values are allowed as arguments")


def lenient_args(rest: str, spec: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Keyword arguments read by the tool's own parameter names, for calls the literal
    parser rejects. ``rest`` is everything after ``name(``. Each value runs to the next
    ``, <param>=`` (or the end); one pair of surrounding quotes is removed. Values stay
    strings (coerced to the schema type later). None if the text has no keyword layout."""
    params = list(spec["properties"])
    s = re.sub(r"[\s;.,]+$", "", rest)
    if s.endswith(")"):
        s = s[:-1]
    if not params:
        return {} if not s.strip() else None
    pat = re.compile(r"(?:^|,)\s*(" + "|".join(map(re.escape, params)) + r")\s*=\s*")
    hits = list(pat.finditer(s))
    if not hits or s[:hits[0].start()].strip():
        return None
    out: Dict[str, str] = {}
    for i, h in enumerate(hits):
        value = s[h.end(): hits[i + 1].start() if i + 1 < len(hits) else len(s)].strip()
        if value[:1] in ("'", '"'):
            q, value = value[0], value[1:]
            if value.endswith(q):
                value = value[:-1]
            value = value.replace("\\" + q, q)
        out[h.group(1)] = value
    return out


def _coerce(name: str, value: Any, prop: Dict[str, Any]) -> Any:
    kind = prop.get("type")
    if kind == "integer":
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a whole number")
        if isinstance(value, float) and value.is_integer():
            return int(value)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a whole number") from None
    if kind == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
    if kind == "boolean":
        return value if isinstance(value, bool) else str(value).strip().lower() == "true"
    if kind == "string":
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        value = "" if value is None else str(value)
        if name == "targetInstance":            # "[instance 645004284]" copied from the observation
            m = _INSTANCE_ID.fullmatch(value)
            if m:
                value = m.group(1)
        return value
    return value


def parse_call(action: Any, tools: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(call, None)`` with call = {"name", "arguments"}, or ``(None, error)``. Never raises."""
    specs = tool_specs(tools)
    if isinstance(action, dict):                       # {"name": ..., "arguments"/"args": {...}}
        name = str(action.get("name") or "")
        args = dict(action.get("arguments") or action.get("args") or {})
    else:
        raw = str(action or "").strip().strip("`").strip()
        if not raw:
            return None, "the action is empty"
        if raw in specs:
            raw += "()"
        head = _HEAD.match(raw)
        if not head:
            return None, f"the action is not a tool call: {raw[:80]!r}"
        name = head.group(1)
        if name not in specs:
            return None, f"unknown tool {name!r}; available: {', '.join(specs)}"
        args, err = None, None
        m = _CALL.match(raw.splitlines()[0]) or _CALL.match(raw)
        if m:
            try:
                call = ast.parse(f"_f({m.group(2)})", mode="eval").body
                assert isinstance(call, ast.Call)
                params = list(specs[name]["properties"])
                if len(call.args) > len(params):
                    raise ValueError("too many arguments")
                args = {p: _literal(node) for p, node in zip(params, call.args)}
                for kw in call.keywords:
                    if kw.arg is None:
                        raise ValueError("**kwargs are not allowed")
                    args[kw.arg] = _literal(kw.value)
            except (SyntaxError, ValueError, AssertionError) as exc:
                args, err = None, f"could not read the arguments of {name}: {exc}"
        else:
            err = f"the action is not a tool call: {raw[:80]!r}"
        if args is None:
            # Free text in a message breaks the literal parser: quotes or apostrophes
            # inside it, a missing closing bracket, trailing punctuation.
            args = lenient_args(raw[head.end():], specs[name])
            if args is None:
                return None, err
    if name not in specs:
        return None, f"unknown tool {name!r}; available: {', '.join(specs)}"
    spec = specs[name]
    missing = [a for a in spec["required"] if a not in args]
    if missing:
        return None, f"{name} is missing {', '.join(missing)}; write it as {call_shape(name, spec)}"
    try:
        clean = {k: _coerce(k, v, spec["properties"][k]) for k, v in args.items()
                 if k in spec["properties"]}
    except ValueError as exc:
        return None, f"{name}: {exc}"
    return {"name": name, "arguments": clean}, None


def parse_reply(text: str, tools: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """``(call or None, thoughts, error or None)`` for one model reply."""
    obj = extract_json(text)
    if obj is None:
        bare, _ = parse_call(_THINK.sub("", text or "").strip(), tools)   # a bare call is fine too
        if bare is not None:
            return bare, "", None
        return None, "", ("the reply was empty" if not (text or "").strip()
                          else "the reply was not one JSON object")
    thoughts = str(obj.get("thoughts") or "")
    if "action" not in obj:
        return None, thoughts, 'the JSON has no "action" field'
    call, err = parse_call(obj["action"], tools)
    return call, thoughts, err


def call_string(name: str, args: Dict[str, Any]) -> str:
    """``name(k="v", n=3)``: the form the model is asked to write."""
    parts = [f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v, str) else v}"
             for k, v in args.items()]
    return f"{name}({', '.join(parts)})"


# ── AgentWorld history -> plain text conversation ───────────────────────────
def text_messages(history: Sequence[Dict[str, Any]], how_to_act: str) -> List[Dict[str, str]]:
    """System prompt + instructions, then alternating user/assistant text ending with a user turn."""
    out: List[Dict[str, str]] = []
    names: Dict[str, str] = {}

    def add(role: str, content: str) -> None:
        if out and out[-1]["role"] == role and role != "system":
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})

    for m in history:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            if not out:
                out.append({"role": "system", "content": f"{content}\n\n{how_to_act}"})
            else:
                add("user", content)
        elif role == "user":
            add("user", content)
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                f = calls[0].get("function") or {}
                args = f.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {}
                names[calls[0].get("id") or ""] = f.get("name") or "your action"
                add("assistant", json.dumps({"thoughts": content,
                                             "action": call_string(f.get("name") or "", args)},
                                            ensure_ascii=False))
            elif content.strip():
                add("assistant", content)
        elif role == "tool":
            add("user", f"Result of {names.get(m.get('tool_call_id') or '', 'your last action')}: "
                        f"{content}")
    if not out or out[0]["role"] != "system":
        out.insert(0, {"role": "system", "content": how_to_act})
    if len(out) == 1 or out[1]["role"] != "user":
        out.insert(1, {"role": "user", "content": NEXT_TURN})
    if out[-1]["role"] != "user":
        out.append({"role": "user", "content": NEXT_TURN})
    return out


# ── the agent mixin ──────────────────────────────────────────────────────────
class JSONActionMixin:
    """Put before AgentWorld's ``OpenAIAgent`` (uses its ``session``, ``base_url``, ``model``)."""

    json_sampling: Dict[str, Any] = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024}
    json_thinking: bool = False
    json_retries: int = 1           # extra attempts within one turn after an unusable reply
    json_http_attempts: int = 5     # as AgentWorld's OpenAIAgent
    json_backoff_s: float = 10.0

    def _make_api_call(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        tools = list(getattr(self, "tools", None) or [])
        convo = text_messages(messages, instructions(tools))
        thoughts, text, err = "", "", None
        for attempt in range(1, self.json_retries + 2):
            resp = self._json_post(convo)
            if "error" in resp:
                return resp
            choices = resp.get("choices") or []
            if not choices:
                return {"error": "no choices in the model response"}
            text = (choices[0].get("message") or {}).get("content") or ""
            call, thoughts, err = parse_reply(text, tools)
            if call is not None:
                if attempt > 1:
                    print(f"[JSON] usable reply on attempt {attempt}", flush=True)
                return {"choices": [{"message": {
                    "role": "assistant", "content": thoughts,
                    "tool_calls": [{"id": f"call_{uuid.uuid4().hex[:16]}", "type": "function",
                                    "function": {"name": call["name"],
                                                 "arguments": json.dumps(call["arguments"],
                                                                         ensure_ascii=False)}}]}}],
                        "usage": resp.get("usage")}
            print(f"[JSON] unusable reply (attempt {attempt}/{self.json_retries + 1}): {err} | "
                  f"raw: {text[:300]!r}", flush=True)
            convo = convo + [
                {"role": "assistant", "content": text.strip() or "(empty reply)"},
                {"role": "user", "content": f"Your reply could not be used: {err}. "
                                            f"Answer again with ONE JSON object:\n{ANSWER_FORMAT}"}]
        # No usable call: AgentWorld logs this turn as one without a tool call. The
        # reason stays in the history, so the next turn's prompt shows it.
        said = (thoughts or text.strip())[:500]
        return {"choices": [{"message": {"role": "assistant",
                                         "content": f"{said}\n[no action taken: {err}]".strip()}}]}

    def _json_post(self, convo: List[Dict[str, str]]) -> Dict[str, Any]:
        import requests
        url = f"{self.base_url}/chat/completions"
        data: Dict[str, Any] = {"model": self.model, "messages": convo, **self.json_sampling,
                                "chat_template_kwargs": {"enable_thinking": self.json_thinking}}
        last_error: Any = None
        for cnt in range(1, self.json_http_attempts + 1):
            try:
                r = self.session.post(url, json=data, timeout=120)
                if (r.status_code == 400 and "max_tokens" in data
                        and _CONTEXT_ERROR.search(r.text or "")):
                    # A long history leaves less than max_tokens of context: let the server cap it.
                    print("[JSON] prompt near the context limit; retrying without max_tokens", flush=True)
                    data = {k: v for k, v in data.items() if k != "max_tokens"}
                    r = self.session.post(url, json=data, timeout=120)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as exc:
                last_error = exc
                body = getattr(getattr(exc, "response", None), "text", "") or ""
                print(f"[API ERROR] Attempt {cnt}/{self.json_http_attempts} failed: "
                      f"{type(exc).__name__}: {exc} | Body: {body[:500]}", flush=True)
                if cnt < self.json_http_attempts:
                    time.sleep(self.json_backoff_s * (cnt + 1))
        return {"error": f"API call failed after {self.json_http_attempts} attempts: {last_error}"}
