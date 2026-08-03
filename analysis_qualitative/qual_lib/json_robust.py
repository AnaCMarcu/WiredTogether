"""Robust model-output JSON parsing.

Vendored verbatim from ``src/mindforge/agent_modules/util.py`` (functions
``_fix_common_json_errors``, ``_strip_markdown_fences``, ``_try_parse``,
``load_json``) because importing that module requires ``autogen_core``/
``autogen_ext``, which are not installed on the analysis machine. Keep in
sync with the source if it ever changes (it is pinned by run data anyway:
these logs were produced by the version copied here).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional


def _fix_common_json_errors(text: str) -> str:
    # <value>\n<key>: → <value>,\n<key>:
    text = re.sub(
        r'("(?:[^"\\]|\\.)*"|true|false|null|\d+\.?\d*|\]|\})'
        r'(\s*\n\s*)'
        r'("(?:[^"\\]|\\.)*"\s*:)',
        r'\1,\2\3',
        text,
    )
    text = re.sub(r',\s*([\}\]])', r'\1', text)  # trailing commas
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def _strip_markdown_fences(response: str) -> str:
    """Remove ```json / ``` wrappers and {{...}} double-brace escaping."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    if response.startswith("{{") and response.endswith("}}"):
        response = response[1:-1]
    return response


def _try_parse(candidate: str) -> Optional[dict]:
    for attempt in (candidate, _fix_common_json_errors(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def load_json(response: str) -> dict:
    """Parse a model response into a dict; {} when nothing parses cleanly."""
    response = _strip_markdown_fences(response)

    parsed = _try_parse(response)
    if parsed is not None:
        return parsed

    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
    for match in reversed(matches):
        parsed = _try_parse(match)
        if parsed is not None:
            return parsed

    first = response.find('{')
    last = response.rfind('}')
    if first != -1 and last > first:
        parsed = _try_parse(response[first:last + 1])
        if parsed is not None:
            return parsed

    logging.debug("qual load_json: failed to decode: %s", response[:200])
    return {}
