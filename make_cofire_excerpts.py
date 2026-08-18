#!/usr/bin/env python3
"""Join co-firing channel events with milestones for appendix excerpts.

For selected cofiring_final arms this emits, per run and episode:
  1. total bond growth split by channel (spat/comm/obs/imit),
  2. for each milestone: the co-firing growth and social acts (observe /
     imitate / communicate) in the preceding window,
  3. the top growth spikes with their channel decomposition.

Sources (all verbatim, per run dir): cofiring_events.jsonl, social_acts.jsonl,
episodes/ep_*/{event_log.jsonl,messages.jsonl}. Output: one markdown file per
run under paper_assets_cofiring_v2/qualitative_excerpts/. Candidates only —
appendix quotes are hand-picked from these files.

Usage:
    python make_cofire_excerpts.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = REPO / "runs_from_daic" / "cofiring_final"
OUT = REPO / "paper_assets_cofiring_v2" / "qualitative_excerpts"

# imitation arm and communication arm — the RQ2 contrast in the paper
ARMS = ["exp22_cofire_pri", "exp20_cofire_prc"]
CHANNELS = ["c_spat", "c_comm", "c_obs", "c_imit"]
MILESTONE_WINDOW = 15  # steps of context before each milestone
TOP_SPIKES = 25


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


def run_sections(run_dir: Path) -> list[str]:
    cof = list(read_jsonl(run_dir / "cofiring_events.jsonl"))
    acts = list(read_jsonl(run_dir / "social_acts.jsonl"))
    lines: list[str] = []
    if not cof:
        lines.append("*(no cofiring_events.jsonl in this run)*")
        return lines

    cof_by_ep: dict[int, list[dict]] = defaultdict(list)
    for r in cof:
        cof_by_ep[r.get("ep")].append(r)
    acts_by_ep_step: dict[tuple, list[dict]] = defaultdict(list)
    for a in acts:
        acts_by_ep_step[(a.get("ep"), a.get("step"))].append(a)

    for ep_dir in sorted((run_dir / "episodes").glob("ep_*")):
        ep = int(ep_dir.name.split("_")[-1])
        rows = cof_by_ep.get(ep, [])
        lines.append(f"## ep_{ep:04d}")
        lines.append("")

        # 1. growth by channel: attribute each row's growth to its channels
        # proportionally to the channel contributions c_*
        growth_by_ch = dict.fromkeys(CHANNELS, 0.0)
        total_growth = 0.0
        for r in rows:
            g = r.get("growth") or 0.0
            total_growth += g
            csum = sum(r.get(ch) or 0.0 for ch in CHANNELS)
            if csum > 0:
                for ch in CHANNELS:
                    growth_by_ch[ch] += g * (r.get(ch) or 0.0) / csum
        lines.append(f"**Total growth {total_growth:.4f}**, by channel: "
                     + ", ".join(f"{ch}={growth_by_ch[ch]:.4f}"
                                 for ch in CHANNELS))
        lines.append("")

        # 2. milestones with preceding co-firing + social acts
        events = list(read_jsonl(ep_dir / "event_log.jsonl"))
        milestones = [e for e in events if e.get("type") == "milestone"]
        rows_by_step: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            rows_by_step[r.get("step")].append(r)
        msgs_by_step: dict[int, list[dict]] = defaultdict(list)
        for m in read_jsonl(ep_dir / "messages.jsonl"):
            if isinstance(m.get("t"), int):
                msgs_by_step[m["t"]].append(m)

        for ev in milestones:
            step = ev.get("step")
            if not isinstance(step, int):
                continue
            lines.append(f"### {ev.get('id')} (+{ev.get('reward')}) at "
                         f"t={step}, contributors={ev.get('contributors')}")
            lines.append("")
            lines.append("| t | detail |")
            lines.append("|---|---|")
            for t in range(max(0, step - MILESTONE_WINDOW), step + 2):
                for a in acts_by_ep_step.get((ep, t), []):
                    if a.get("act") not in (None, "none"):
                        lines.append(
                            f"| {t} | act: {a.get('agent')} "
                            f"{a.get('act')} -> {a.get('target')} |")
                for m in msgs_by_step.get(t, []):
                    s, r_ = agent_idx(m.get("sender")), agent_idx(
                        m.get("receiver"))
                    lines.append(f"| {t} | msg a{s}->a{r_}: "
                                 f"{(m.get('text') or '').strip()} |")
                for r in rows_by_step.get(t, []):
                    if (r.get("growth") or 0.0) > 0:
                        ch = ", ".join(f"{c}={r.get(c)}" for c in CHANNELS
                                       if (r.get(c) or 0.0) > 0)
                        lines.append(
                            f"| {t} | cofire a{r.get('i')}~a{r.get('j')}: "
                            f"growth={r.get('growth'):.4f} ({ch}) "
                            f"W_after={r.get('W_after'):.3f} |")
            lines.append("")

        # 3. top growth spikes
        top = sorted(rows, key=lambda r: r.get("growth") or 0.0,
                     reverse=True)[:TOP_SPIKES]
        lines.append(f"### Top {len(top)} growth spikes")
        lines.append("")
        lines.append("| t | pair | growth | channels | W_after |")
        lines.append("|---|---|---|---|---|")
        for r in top:
            ch = ", ".join(f"{c}={r.get(c)}" for c in CHANNELS
                           if (r.get(c) or 0.0) > 0)
            lines.append(f"| {r.get('step')} | a{r.get('i')}~a{r.get('j')} | "
                         f"{r.get('growth'):.4f} | {ch} | "
                         f"{r.get('W_after'):.3f} |")
        lines.append("")
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for seed_dir in sorted((ROOT / arm).glob("seed_*")):
            lines = [f"# {arm} / {seed_dir.name}", ""]
            lines.extend(run_sections(seed_dir))
            out_path = OUT / f"{arm}_{seed_dir.name}.md"
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
