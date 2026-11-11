"""chains/json_actions.py: JSON replies instead of native tool calls (WiredJSONAgent).

Tests marked ``needs_aw`` build the real agent from an AgentWorld checkout
($AGENTWORLD_ROOT) and are skipped without one.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "chains"))

from json_actions import (ANSWER_FORMAT, NEXT_TURN, JSONActionMixin,  # noqa: E402
                          instructions, parse_reply, text_messages, tool_reference)
from wired_agent import DIRECT_CHAT_TOOL  # noqa: E402

AW = os.environ.get("AGENTWORLD_ROOT")
needs_aw = pytest.mark.skipif(not (AW and (Path(AW) / "agents" / "run.py").is_file()),
                              reason="needs an AgentWorld checkout in $AGENTWORLD_ROOT")


def _tool(name, props, required, desc=""):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


# Shapes as in AgentWorld's agents/tool_definitions.py (descriptions shortened).
TOOLS = [
    _tool("move_character", {"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"],
          "Move the character. Maximum movement distance is 120 tiles."),
    _tool("harvest_resource", {"targetInstance": {"type": "string"}}, ["targetInstance"]),
    _tool("craft_item", {"skill": {"type": "string"}, "itemKey": {"type": "string"},
                         "count": {"type": "integer"}}, ["skill", "itemKey"]),
    _tool("transfer_items", {"targetPlayer": {"type": "string"}, "itemKey": {"type": "string"},
                             "count": {"type": "integer"}}, ["targetPlayer", "itemKey", "count"]),
    _tool("enter_portal", {}, []),
    DIRECT_CHAT_TOOL,
]


# ── prompt text ──────────────────────────────────────────────────────────────
def test_tool_reference_writes_calls_out_with_quoted_strings():
    ref = tool_reference(TOOLS)
    assert "- move_character(x=<x>, y=<y>)\n    Move the character. Maximum movement" in ref
    assert '- harvest_resource(targetInstance="<instance id from your observation>")' in ref
    assert ('- craft_item(skill="<skill, e.g. Smithing>", itemKey="<item key>")'
            '   optional: count=<number>') in ref
    assert '- direct_chat(target="<teammate username>", message="<text>")' in ref
    assert "- enter_portal()" in ref
    how = instructions(TOOLS)
    assert ANSWER_FORMAT in how and "chat(" not in how.replace("direct_chat(", "")


# ── parsing ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("reply,name,args", [
    ('{"thoughts": "go", "action": "move_character(x=120, y=40)"}', "move_character",
     {"x": 120, "y": 40}),
    ('```json\n{"thoughts": "t", "action": "move_character(x=\\"120\\", y=40.0)"}\n```',
     "move_character", {"x": 120, "y": 40}),
    # bare ids become strings (AgentWorld compares them with the observation's string ids)
    ('{"thoughts": "", "action": "harvest_resource(targetInstance=645004284)"}',
     "harvest_resource", {"targetInstance": "645004284"}),
    ('{"action": "harvest_resource(targetInstance=\\"instance 645004284\\")"}',
     "harvest_resource", {"targetInstance": "645004284"}),
    ('{"action": "craft_item(\\"Smithing\\", \\"bronzebar\\")"}', "craft_item",
     {"skill": "Smithing", "itemKey": "bronzebar"}),
    ('{"action": "transfer_items(targetPlayer=\\"a3_smith\\", itemKey=\\"tinore\\", count=\\"1\\")"}',
     "transfer_items", {"targetPlayer": "a3_smith", "itemKey": "tinore", "count": 1}),
    ('{"action": "direct_chat(target=\\"a2_smelter\\", message=\\"send coal\\")"}', "direct_chat",
     {"target": "a2_smelter", "message": "send coal"}),
    ('{"action": {"name": "move_character", "arguments": {"x": 3, "y": 4}}}', "move_character",
     {"x": 3, "y": 4}),
    ('<think>hmm</think>{"action": "enter_portal"}', "enter_portal", {}),
    ("I will go. move_character(x=1, y=2)", None, None),      # prose + call, no JSON: rejected
    ("move_character(x=1, y=2)", "move_character", {"x": 1, "y": 2}),   # a bare call is fine
    ('{"action": "craft_item(skill=\\"Smithing\\", itemKey=\\"axe\\", colour=\\"red\\")"}',
     "craft_item", {"skill": "Smithing", "itemKey": "axe"}),            # unknown args dropped
    # failures seen in the metallurgy_4 JSON pilot, now read:
    ('{"action": "direct_chat(target=a1_gold_miner, message=\\"hi\\")"}', "direct_chat",
     {"target": "a1_gold_miner", "message": "hi"}),                     # bare username
    ('{"action": "direct_chat(target=\\"a1_miner\\", message=\\"I have logs (5) ready\\""}',
     "direct_chat", {"target": "a1_miner", "message": "I have logs (5) ready"}),   # no closing bracket
    ('{"action": "direct_chat(target=\\"a2_smelter\\", message=\\"say \\"go\\" now\\")"}',
     "direct_chat", {"target": "a2_smelter", "message": 'say "go" now'}),          # quotes inside
    ('{"action": "direct_chat(target=\'a2_smelter\', message=\'I\'m at (200, 280)\')"}',
     "direct_chat", {"target": "a2_smelter", "message": "I'm at (200, 280)"}),     # apostrophe
    ('{"action": "move_character(x=1, y=2)."}', "move_character", {"x": 1, "y": 2}),  # trailing dot
    # broken JSON around a readable action (pendant_staff_4 run, memory arm):
    ('{"thoughts": "Following up.", "action": "direct_chat(target=\\"a1_lumberjack\\", '
     'message=\\"Just following up on the oak logs!\\"}', "direct_chat",
     {"target": "a1_lumberjack", "message": "Just following up on the oak logs!"}),   # call never closed
    ('{"thoughts": "I\\\'m here", "action": "move_character(x=1, y=2)"}', "move_character",
     {"x": 1, "y": 2}),                                                               # invalid \' escape
])
def test_parse_reply(reply, name, args):
    call, _, err = parse_reply(reply, TOOLS)
    if name is None:
        assert call is None and err
    else:
        assert err is None and call == {"name": name, "arguments": args}


@pytest.mark.parametrize("reply,fragment", [
    ("", "empty"),
    ("   ", "empty"),
    ("thinking about it", "not one JSON object"),
    ('{"thoughts": "x"}', 'no "action"'),
    ('{"action": ""}', "empty"),
    ('{"action": "chat(message=\\"hi all\\")"}', "unknown tool 'chat'"),
    ('{"action": "transfer_items"}',
     'write it as transfer_items(targetPlayer="<exact player name>", itemKey="<item key>", '
     'count=<number>)'),
    ('{"action": "move_character(x=\\"north\\", y=2)"}', "x must be a whole number"),
    ('{"action": "move_character(1, 2, 3)"}', "too many arguments"),
    # the lenient reading needs a keyword layout from the first argument on
    ('{"action": "direct_chat(hello there, message=\\"x\\")"}', "could not read the arguments of direct_chat"),
    ('{"action": "direct_chat(target=\\"a2_smelter\\""}', "direct_chat is missing message"),
    ('{"thoughts": "x", "action": "move_char', "not a tool call"),       # salvaged but cut off
])
def test_parse_reply_errors(reply, fragment):
    call, _, err = parse_reply(reply, TOOLS)
    assert call is None and fragment in err


def test_salvage_keeps_the_thoughts():
    from json_actions import salvage_action
    assert salvage_action('{"thoughts": "a \\"b\\" c", "action": "sleep(seconds=2)"') == ('a "b" c',
                                                                                           "sleep(seconds=2)")
    assert salvage_action("no json here") is None


# ── history -> text ─────────────────────────────────────────────────────────
def _tc(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_text_messages_alternate_and_end_with_the_user():
    history = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": "walk", "tool_calls": [_tc("c1", "move_character", {"x": 1, "y": 2})]},
        {"role": "tool", "tool_call_id": "c1", "content": "moved"},
        {"role": "assistant"},                                   # an empty turn: dropped
        {"role": "assistant", "tool_calls": [_tc("c2", "harvest_resource", {"targetInstance": "77"})]},
        {"role": "tool", "tool_call_id": "c2", "content": "got ore"},
    ]
    out = text_messages(history, "HOW")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert out[0]["content"] == "SYS\n\nHOW"
    assert json.loads(out[2]["content"]) == {"thoughts": "walk", "action": "move_character(x=1, y=2)"}
    assert out[3]["content"] == "Result of move_character: moved"
    assert json.loads(out[4]["content"])["action"] == 'harvest_resource(targetInstance="77")'
    assert out[5]["content"] == "Result of harvest_resource: got ore"
    # a past reply that was not used is shown, then the model is asked again
    out2 = text_messages(history[:2] + [{"role": "assistant", "content": "x\n[no action taken: e]"}], "HOW")
    assert [m["role"] for m in out2] == ["system", "user", "assistant", "user"]
    assert out2[-1]["content"] == NEXT_TURN


# ── the mixin, with a fake HTTP session ──────────────────────────────────────
class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._payload


def _reply(content):
    return _Resp(200, {"choices": [{"message": {"role": "assistant", "content": content}}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


class _Session:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def post(self, url, json=None, timeout=None):
        self.requests.append(json)
        return self.responses.pop(0)


class _Agent(JSONActionMixin):
    json_backoff_s = 0.0

    def __init__(self, responses):
        self.base_url, self.model, self.tools = "http://x/v1", "wt", TOOLS
        self.session = _Session(responses)


HISTORY = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]


def _calls(resp):
    """What AgentWorld's OpenAIAgent._extract_tool_calls reads from the message."""
    msg = resp["choices"][0]["message"]
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in msg.get("tool_calls", [])]


def test_usable_reply_becomes_one_openai_tool_call():
    agent = _Agent([_reply('{"thoughts": "mine", "action": "harvest_resource(targetInstance=12)"}')])
    resp = agent._make_api_call(HISTORY)
    assert _calls(resp) == [("harvest_resource", {"targetInstance": "12"})]
    assert resp["choices"][0]["message"]["content"] == "mine"
    body = agent.session.requests[0]
    assert "tools" not in body and "tool_choice" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.7, 0.9, 1024)
    assert body["messages"][0]["content"].startswith("SYS\n\n=== HOW TO ACT ===")


def test_unusable_reply_is_retried_once_with_the_error():
    agent = _Agent([_reply(""), _reply('{"action": "move_character(x=5, y=6)"}')])
    resp = agent._make_api_call(HISTORY)
    assert _calls(resp) == [("move_character", {"x": 5, "y": 6})]
    retry = agent.session.requests[1]["messages"]
    assert [m["role"] for m in retry[-2:]] == ["assistant", "user"]
    assert "could not be used: the reply was empty" in retry[-1]["content"]


def test_two_unusable_replies_leave_the_turn_without_a_call():
    agent = _Agent([_reply("hmm"), _reply('{"action": "transfer_items"}')])
    resp = agent._make_api_call(HISTORY)
    assert _calls(resp) == []
    assert "[no action taken: transfer_items is missing" in resp["choices"][0]["message"]["content"]


def test_context_limit_retries_without_max_tokens():
    err = _Resp(400, text="This model's maximum context length is 32768 tokens")
    agent = _Agent([err, _reply('{"action": "enter_portal()"}')])
    assert _calls(agent._make_api_call(HISTORY)) == [("enter_portal", {})]
    assert "max_tokens" in agent.session.requests[0] and "max_tokens" not in agent.session.requests[1]


def test_http_failure_returns_agentworlds_error_shape():
    agent = _Agent([_Resp(503, text="down")] * 5)
    assert "API call failed after 5 attempts" in agent._make_api_call(HISTORY)["error"]


# ── the real agent ───────────────────────────────────────────────────────────
@needs_aw
def test_real_json_agent_runs_one_agentworld_turn():
    from run_chain import bootstrap
    bootstrap(Path(AW).resolve())
    from agent_factory import AgentFactory
    agent = AgentFactory.create_agent(
        provider="custom", api_key="x", model="wt", username="a1_copper_miner", password="p",
        base_url="http://127.0.0.1:9",
        llm_params={"agent_class": "wired_agent:WiredJSONAgent", "base_url": "http://127.0.0.1:9/v1"})
    mro = [c.__name__ for c in type(agent).__mro__]
    assert mro.index("JSONActionMixin") < mro.index("OpenAIAgent")
    assert "direct_chat" in [t["function"]["name"] for t in agent.tools]
    agent._get_current_environment_observation = lambda: "OBS"
    agent.session = _Session([_reply('{"thoughts": "go", "action": "move_character(x=1, y=2)"}')])
    executed = []
    agent.game_tools.move_character = lambda args: executed.append(args) or "moved"
    out = agent.execute_single_tool_call("TASK")
    assert executed == [{"x": 1, "y": 2}]
    assert "move_character" in out
    roles = [m["role"] for m in agent.conversation_history]
    assert roles[-2:] == ["assistant", "tool"]
    sent = agent.session.requests[0]
    assert "tools" not in sent and "direct_chat(target=" in sent["messages"][0]["content"]
