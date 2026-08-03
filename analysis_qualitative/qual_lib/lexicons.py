"""Rule lexicons: message categories, impossible objects, thought-verb map.

The impossible-object table is the machine-readable derivative of
``_CHAMBER_OBJECT_WHITELIST`` in src/mindforge/custom_agent.py (prose).
Deliberately conservative: only objects that are unambiguous evidence of a
wrong-chamber reference are listed (e.g. "stone"/"door" occur legitimately
everywhere, so they are absent).
"""

from __future__ import annotations

import re

# ── message categories (multi-label, word-boundary regexes) ─────────────
MESSAGE_CATEGORIES = {
    "request": re.compile(
        r"\b(please|can you|could you|help me|need (you|help)|assist|"
        r"come (here|to|over)|join me|wait for me|free me|let me out)\b", re.I),
    "directive": re.compile(
        r"^(dig|press|move|turn|attack|go|check|look|break|hit|open|collect|"
        r"pick|equip|follow|stay|keep|stop)\b|"
        r"\b(you should|you need to|your turn)\b", re.I),
    "commitment": re.compile(
        r"\b(i('| a|w)?ll|i am going|i'm going|about to|i will|i plan|"
        r"next i|going to (dig|press|move|attack|check))\b", re.I),
    "status_report": re.compile(
        r"\b(i see|i'm (digging|moving|attacking|pressing|heading|in)|"
        r"i am (digging|moving|attacking|pressing|heading|in)|i broke|"
        r"i (collected|got|found|reached|entered|pressed)|door is|"
        r"switch (is|was)|anvil (is|hp)|zombie|reward|milestone)\b", re.I),
    "acknowledgment": re.compile(
        r"\b(ok(ay)?|got it|confirmed?|roger|understood|on it|will do|"
        r"sounds good|thanks|good work|nice|great)\b", re.I),
    "question": re.compile(
        r"\?|\b(where (are|is)|what (is|are)|did you|have you|are you|"
        r"can we|status\b.*\?)", re.I),
}

# ── chamber-impossible objects ───────────────────────────────────────────
# object-group -> (regex, set of chambers where it CAN be seen)
_CH3 = {"ch3", "ch3_communal"}
IMPOSSIBLE_OBJECTS = {
    "tree": (re.compile(r"\b(tree|trunk|log|sapling|leaves)\b", re.I), {"ch1"}),
    "animal": (re.compile(r"\b(chicken|sheep|cow|pig|animal)\b", re.I), {"ch1"}),
    "anvil": (re.compile(r"\banvils?\b", re.I), {"ch2"}),
    "switch": (re.compile(r"\b(blue )?switch(es)?\b|\bcell [abc]\b", re.I), _CH3),
    "zombie": (re.compile(r"\bzombies?\b", re.I), {"ch4", "ch5"}),
    "boss": (re.compile(r"\bboss\b", re.I), {"ch5"}),
}


def impossible_mentions(text: str, chamber: str) -> list:
    """Object groups mentioned in text that cannot exist in this chamber."""
    if not text or not chamber:
        return []
    out = []
    ch = str(chamber).lower()
    for name, (rx, allowed) in IMPOSSIBLE_OBJECTS.items():
        if ch not in allowed and rx.search(text):
            out.append(name)
    return out


# ── Ch3-specific coordination lexicon ────────────────────────────────────
SWITCH_TALK = re.compile(
    r"\b(switch|press|cell|locked|unlock|open( the|s)? door|free (me|you)|"
    r"my door|your door)\b", re.I)
DOOR_CONFUSION = re.compile(
    r"\b(open my own door|my switch opens my|press my (own )?switch to open my|"
    r"i (can|will) open my (own )?(cell )?door)\b", re.I)

# ── env-object naming (referential density) ──────────────────────────────
ENV_OBJECT = re.compile(
    r"\b(tree|trunk|log|stone|anvil|switch|door|zombie|boss|chicken|sheep|"
    r"sword|chestplate|block|cell|drop)\b", re.I)
TEAMMATE_NAME = re.compile(r"\bagent[_ ]?\d\b", re.I)

# ── thought-verb -> expected env action (coarse) ─────────────────────────
ACTION_VERB_MAP = [
    (re.compile(r"\b(dig|mine|break|attack|hit|punch|swing)\b", re.I), "Dig"),
    (re.compile(r"\bturn(ing)? left\b", re.I), "TurnLeft"),
    (re.compile(r"\bturn(ing)? right\b|\bturn around\b", re.I), "TurnRight"),
    (re.compile(r"\blook(ing)? up\b", re.I), "LookUp"),
    (re.compile(r"\blook(ing)? down\b", re.I), "LookDown"),
    (re.compile(r"\bjump\b", re.I), "Jump"),
    (re.compile(r"\b(move|walk|go|head|advance|approach|chase|forward)\b", re.I),
     "MoveForward"),
    (re.compile(r"\b(back up|backward|retreat)\b", re.I), "MoveBackward"),
    (re.compile(r"\bplace\b", re.I), "Place"),
    (re.compile(r"\bdrop\b", re.I), "Drop"),
    (re.compile(r"\b(wait|idle|stay still|no.?op)\b", re.I), "NoOp"),
]


def intended_actions(thought: str) -> set:
    """Coarse set of env actions a thought announces (order-independent)."""
    if not thought:
        return set()
    return {act for rx, act in ACTION_VERB_MAP if rx.search(thought)}


def categorize_message(text: str) -> list:
    if not text:
        return []
    return [name for name, rx in MESSAGE_CATEGORIES.items() if rx.search(text)]
