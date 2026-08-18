#!/usr/bin/env python3
"""Mine transplant Phase-B runs for appendix-ready interaction excerpts.

For each expB arm (transplant / shuffled) x seed x episode this emits:
  1. per-agent dominant message target (the re-pairing readout),
  2. history-referencing messages (partner/anvil/shared-past keywords),
  3. seatmate exchanges in the window leading up to each milestone.

Everything is verbatim from episodes/ep_*/messages.jsonl and event_log.jsonl;
output is one markdown file per run under paper_assets_transplant/qualitative_excerpts/.
Candidates only — the paper appendix quotes are hand-picked from these files.

Usage:
    python make_transplant_excerpts.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = REPO / "runs_from_daic" / "pair_bonding"
OUT = REPO / "paper_assets_transplant" / "qualitative_excerpts"

ARMS = {
    "transplant": ROOT / "expB_merged_transplant",
    "shuffled": ROOT / "expB_merged_shuffled",
}
SEATMATE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

# Words that suggest the message leans on (real or fabricated) shared history.
HISTORY_RE = re.compile(
    r"\b(anvil|together again|last time|remember|as before|like before|"
    r"previous|earlier|we broke|we did|our (?:old|usual)|partner|reunite|"
    r"back with|again)\b",
    re.IGNORECASE,
)

MILESTONE_WINDOW = 12  # steps of context before each milestone
MAX_HISTORY_HITS = 60  # per episode, keep the file readable


def agent_idx(name: str) -> int | None:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else None


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def fmt_msg(m: dict) -> str:
    s, r = agent_idx(m.get("sender")), agent_idx(m.get("receiver"))
    return (f"| {m.get('t')} | a{s}->a{r} | {m.get('chamber', '?')} | "
            f"{(m.get('text') or '').strip()} |")


def episode_sections(ep_dir: Path) -> list[str]:
    msgs = list(read_jsonl(ep_dir / "messages.jsonl"))
    events = list(read_jsonl(ep_dir / "event_log.jsonl"))
    milestones = [e for e in events if e.get("type") == "milestone"]
    lines: list[str] = []

    # 1. dominant message target per agent (re-pairing readout)
    targets: dict[int, Counter] = defaultdict(Counter)
    for m in msgs:
        s, r = agent_idx(m.get("sender")), agent_idx(m.get("receiver"))
        if s is not None and r is not None:
            targets[s][r] += 1
    lines.append("### Dominant message targets")
    lines.append("")
    lines.append("| agent | dominant target | share | seatmate share | n msgs |")
    lines.append("|---|---|---|---|---|")
    for a in sorted(targets):
        c = targets[a]
        total = sum(c.values())
        dom, dom_n = c.most_common(1)[0]
        mate_share = c.get(SEATMATE.get(a), 0) / total if total else 0.0
        lines.append(f"| a{a} | a{dom} | {dom_n / total:.2f} | "
                     f"{mate_share:.2f} | {total} |")
    lines.append("")

    # 2. history-referencing messages
    hits = [m for m in msgs if HISTORY_RE.search(m.get("text") or "")]
    lines.append(f"### History-referencing messages ({len(hits)} hits, "
                 f"showing up to {MAX_HISTORY_HITS})")
    lines.append("")
    lines.append("| t | pair | chamber | text |")
    lines.append("|---|---|---|---|")
    lines.extend(fmt_msg(m) for m in hits[:MAX_HISTORY_HITS])
    lines.append("")

    # 3. seatmate exchanges leading up to each milestone
    lines.append("### Milestone lead-ups (seatmate messages only)")
    lines.append("")
    by_t: dict[int, list[dict]] = defaultdict(list)
    for m in msgs:
        if isinstance(m.get("t"), int):
            by_t[m["t"]].append(m)
    for ev in milestones:
        step = ev.get("step")
        if not isinstance(step, int):
            continue
        lines.append(f"**{ev.get('id')} (+{ev.get('reward')}) at t={step}, "
                     f"contributors={ev.get('contributors')}**")
        lines.append("")
        lines.append("| t | pair | chamber | text |")
        lines.append("|---|---|---|---|")
        for t in range(max(0, step - MILESTONE_WINDOW), step + 2):
            for m in by_t.get(t, []):
                s, r = agent_idx(m.get("sender")), agent_idx(m.get("receiver"))
                if s is not None and SEATMATE.get(s) == r:
                    lines.append(fmt_msg(m))
        lines.append("")
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, arm_dir in ARMS.items():
        for seed_dir in sorted(arm_dir.glob("seed_*")):
            lines = [f"# {arm} / {seed_dir.name}", ""]
            for ep_dir in sorted((seed_dir / "episodes").glob("ep_*")):
                lines.append(f"## {ep_dir.name}")
                lines.append("")
                lines.extend(episode_sections(ep_dir))
            out_path = OUT / f"{arm}_{seed_dir.name}.md"
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
