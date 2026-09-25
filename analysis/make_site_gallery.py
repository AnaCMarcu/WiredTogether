#!/usr/bin/env python
"""Render `site/gallery.html`: every reviewed candidate clip, filterable.

The index page shows the dozen scenes the paper annotates; this page holds the
wider set that `analysis/scan_clip_candidates.py --cut` mined and rendered into
`site/assets/videos/candidates/`. It reads that directory's `manifest.json`.

Thinning the set is a text file: `site/assets/videos/candidates/keep.txt`, one
clip name per line. When it exists, only those clips are published and every
other candidate's files are left in place but unlisted. Without it, the whole
manifest is published.

`--review` adds the scanner's score, note and chat excerpt under each clip —
the version to look at while deciding what to keep. Never deploy that one.

Every card also carries the clip's transcript: the messages its agents sent
or received over the window, as recorded in the manifest, which the page
lights up in step with playback. `--inject-index` writes the same transcripts
into the hand-written `index.html`, between `<!-- chat:NAME -->` markers.

Usage:
  python analysis/make_site_gallery.py                 # public gallery
  python analysis/make_site_gallery.py --review        # with scores and notes
  python analysis/make_site_gallery.py --inject-index  # transcripts into index.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path

from paths import REPO  # noqa: E402  (also puts siblings on sys.path)

SITE = REPO / "site"
CANDIDATES = SITE / "assets" / "videos" / "candidates"
MANIFEST = CANDIDATES / "manifest.json"
KEEP = CANDIDATES / "keep.txt"
OUT = SITE / "gallery.html"
INDEX = SITE / "index.html"
SCENES_MANIFEST = SITE / "assets" / "videos" / "manifest.json"

ARM_ORDER = ["qwen9b_base", "qwen9b_heb", "gemma_base", "gemma_heb", "gemma_3f", "orch",
             "qwen2b_mappo", "qwen2b_mappo_inst", "qwen2b_mappo_trace",
             "qwen2b_ippo", "qwen2b_ippo_inst", "qwen2b_ippo_trace",
             "gemma_mappo", "gemma_mappo_inst", "gemma_ippo", "gemma_ippo_inst", "gemma_ippo_trace",
             "base_n6", "heb_n6", "orch_n6", "transplant", "transplant_shuffled"]
ARM_LABEL = {
    "qwen9b_base": ("Qwen3.5-9B", "no social layer"),
    "qwen9b_heb": ("Qwen3.5-9B", "+inst."),
    "gemma_base": ("Gemma-E4B", "no social layer"),
    "gemma_heb": ("Gemma-E4B", "+inst."),
    "gemma_3f": ("Gemma-E4B", "+trace"),
    "orch": ("Gemma-E4B", "+Central Orch."),
    "qwen2b_mappo": ("Qwen3.5-2B MAPPO", "no graph"),
    "qwen2b_mappo_inst": ("Qwen3.5-2B MAPPO", "+inst."),
    "qwen2b_mappo_trace": ("Qwen3.5-2B MAPPO", "+trace"),
    "qwen2b_ippo": ("Qwen3.5-2B IPPO", "no graph"),
    "qwen2b_ippo_inst": ("Qwen3.5-2B IPPO", "+inst."),
    "qwen2b_ippo_trace": ("Qwen3.5-2B IPPO", "+trace"),
    "gemma_mappo": ("Gemma-E4B MAPPO", "no graph"),
    "gemma_mappo_inst": ("Gemma-E4B MAPPO", "+inst."),
    "gemma_ippo": ("Gemma-E4B IPPO", "no graph"),
    "gemma_ippo_inst": ("Gemma-E4B IPPO", "+inst."),
    "gemma_ippo_trace": ("Gemma-E4B IPPO", "+trace"),
    "base_n6": ("Six agents", "no graph"),
    "heb_n6": ("Six agents", "+inst."),
    "orch_n6": ("Six agents", "+Central Orch."),
    "transplant": ("Transplant, six agents", "co-fired partners"),
    "transplant_shuffled": ("Transplant, six agents", "re-paired"),
}
RL_ARMS = {k for k in ARM_ORDER if "mappo" in k or "ippo" in k}
SIX_ARMS = {"base_n6", "heb_n6", "orch_n6", "transplant", "transplant_shuffled"}
FAMILY_TITLE = {
    "zs": "Zero-shot VLM agents — the graph enters inference",
    "rl": "RL fine-tuned agents — the graph enters learning",
    "six": "Six-agent teams — orchestration at scale, and partner transfer",
}
KIND_LABEL = {
    "anvil": "Anvil broken together",
    "door": "Door opened on a partner's switch",
    "kill": "Kill with a teammate present",
    "ch5": "Reached the boss chamber",
    "ch2_stall": "Chamber 2 timed out — no anvil",
    "ch3_stall": "Chamber 3 — a door never opened",
    "ch4_stall": "Chamber 4 — no kill",
    "lost": "Turning in place",
}
CHAMBER = {"ch1": "Ch1", "ch2": "Ch2", "ch3": "Ch3", "ch4": "Ch4", "ch5": "Ch5"}


def load() -> list[dict]:
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if KEEP.exists():
        keep = {l.strip() for l in KEEP.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")}
        items = [it for it in items if it["name"] in keep]
    return [it for it in items if (SITE / it["video"]).exists()]


PREFIX = ""   # set by main(): path from the output page to site/


def transcript(it: dict) -> str:
    """The clip's messages as a step-synced list; empty when none were logged."""
    msgs = it.get("messages") or []
    if not msgs:
        return ""
    agents = set(it["agents"])
    pair = len(agents) >= 2
    rows, n_between = [], 0
    for m in msgs:
        snd, rcv = m["sender"], m.get("receiver")
        between = pair and snd in agents and rcv in agents
        n_between += between
        cls = f"msg a{snd}" + (" partner" if between else "")
        to = f'<span class="to">→ a{rcv}</span>' if rcv is not None else '<span class="to">→ all</span>'
        rows.append(f'<li class="{cls}" data-t="{m["t"]}"><span class="t">t{m["t"]}</span>'
                    f'<span><span class="who">a{snd}</span>{to}{html.escape(m["text"])}</span></li>')
    chips = ""
    if pair and n_between:
        # Default to the exchange between the two shown agents when there is one.
        chips = (f'<div class="chips"><button class="chip" data-chat="all">All ({len(msgs)})</button>'
                 f'<button class="chip active" data-chat="pair">Between the two ({n_between})</button></div>')
    head = f'<div class="chat-head"><span>Messages, step-synced — click one to jump</span>{chips}</div>'
    return f'<div class="chat">{head}<ol>{"".join(rows)}</ol></div>'


def viewtags(agents, layout=None, kind="pair", hidden=False) -> str:
    """One label per panel, placed on the strip or grid, in the agent's colour."""
    n = len(agents)
    cols, rows = layout or (n, 1)
    spans = []
    for i, a in enumerate(agents):
        left = (i % cols) / cols * 100
        top = (i // cols) / rows * 100
        spans.append(f'<span class="view a{a}" style="left:calc({left:.3f}% + 8px);'
                     f'top:calc({top:.3f}% + 8px)">agent {a}</span>')
    return (f'<div class="viewtags {kind}" aria-hidden="true"{" hidden" if hidden else ""}>'
            + "".join(spans) + '</div>')


LAZY_POSTERS = False   # the gallery sets this: posters load as cards approach


def family_of(arm: str) -> str:
    return "rl" if arm in RL_ARMS else "six" if arm in SIX_ARMS else "zs"


def video_tag(it: dict) -> str:
    """The clip as first shown: every agent in the episode when that variant exists."""
    src, poster = (it["video_all"], it["poster_all"]) if it.get("video_all") else (it["video"], it["poster"])
    attr = "data-poster" if LAZY_POSTERS else "poster"
    return (f'<video src="{PREFIX}{src}" {attr}="{PREFIX}{poster}" muted loop playsinline '
            f'preload="none"></video>')


def stage(it: dict, video_html: str | None = None) -> str:
    """The video with its label sets, in a positioned wrapper.

    All agents are shown first — that is what lets the views be compared in
    parallel — and the two (or one) the moment names are behind the button.
    """
    video_html = video_html or video_tag(it)
    has_all = bool(it.get("agents_all"))
    tags = viewtags(it["agents"], kind="pair", hidden=has_all)
    if has_all:
        tags += viewtags(it["agents_all"], it.get("layout_all"), kind="all")
    return f'<div class="stage" data-stage>{video_html}{tags}</div><!-- /stage -->'


def all_attrs(it: dict) -> str:
    """The focus (pair) variant and the agent count, for the toggle."""
    if not it.get("video_all"):
        return ""
    return (f' data-pair-video="{PREFIX}{it["video"]}" data-pair-poster="{PREFIX}{it["poster"]}"'
            f' data-all-n="{len(it["agents_all"])}" data-pair-n="{len(it["agents"])}"')


def card(it: dict, review: bool) -> str:
    badge = {"success": "ok", "failure": "bad", "partial": "mid"}[it["outcome"]]
    kind = KIND_LABEL.get(it.get("kind", ""), it.get("kind", ""))
    steps = it["steps"]
    sides = (f"about agents {it['agents'][0]} &amp; {it['agents'][1]}" if len(it["agents"]) > 1
             else f"about agent {it['agents'][0]}")
    prov = f"seed {it['seed']} · episode {it['episode']} · steps {steps[0]}–{steps[1]} · {sides}"
    extra = ""
    if review:
        chat = html.escape(it.get("chat", "")).replace(" | ", "<br>")
        extra = (f'<p class="review"><strong>score {it.get("score", "")}</strong> · '
                 f'{html.escape(it.get("caption", ""))}</p>'
                 f'<p class="review chat">{chat}</p>'
                 f'<p class="review mono">{it["name"]}</p>')
    return f'''
      <figure class="card" data-outcome="{it["outcome"]}" data-kind="{it.get("kind", "")}" data-arm="{it["run"]}" data-family="{family_of(it["run"])}" data-clip="{it["name"]}" data-t0="{steps[0]}" data-fps="{it.get("fps", 6)}"{all_attrs(it)}>
        {stage(it)}
        <div class="body">
          <figcaption>
            <h3>{html.escape(kind)} <span class="badge {badge}">{it["outcome"]}</span></h3>
            <p class="prov">{prov}</p>{extra}
          </figcaption>{transcript(it)}
        </div>
      </figure>'''


def render(items: list[dict], review: bool) -> str:
    by_arm = defaultdict(list)
    for it in items:
        by_arm[it["run"]].append(it)
    blocks = []
    family_done = set()
    for arm in ARM_ORDER:
        rows = by_arm.get(arm)
        if not rows:
            continue
        fam = "rl" if arm in RL_ARMS else "six" if arm in SIX_ARMS else "zs"
        if fam not in family_done:
            family_done.add(fam)
            blocks.append(f'\n    <h2 class="family" id="family-{fam}">{FAMILY_TITLE[fam]}</h2>')
        model, variant = ARM_LABEL[arm]
        tag = ("orch" if "Orch" in variant else "heb" if variant.startswith("+") or "co-fired" in variant else "")
        rows.sort(key=lambda r: (r["outcome"] != "success", -float(r.get("score", 0))))
        blocks.append(f'''
    <h3 class="armhead" data-arm="{arm}">{model} <span class="tag {tag}">{html.escape(variant)}</span>
      <span class="count">{len(rows)} clips</span></h3>
    <div class="grid videos" data-arm="{arm}">{"".join(card(it, review) for it in rows)}
    </div>''')

    kinds_present = sorted({it.get("kind", "") for it in items}, key=list(KIND_LABEL).index)
    kind_chips = "".join(f'<button class="chip" data-filter-kind="{k}">{html.escape(KIND_LABEL[k])}</button>'
                         for k in kinds_present)
    n_ok = sum(it["outcome"] == "success" for it in items)
    n_bad = len(items) - n_ok
    title = "Wired Together — Moments" + (" (review)" if review else "")
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="robots" content="{'noindex' if review else 'index'}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500&display=swap">
<link rel="stylesheet" href="{PREFIX}assets/css/style.css?v=20260923b">
</head>
<body class="subpage">

<nav id="navbar"><ul class="nav-buttons">
  <li><a href="{PREFIX}index.html">← Project page</a></li>
  <li><a href="{PREFIX}index.html#wire">WIRE</a></li>
  <li><a href="{PREFIX}index.html#method">Method</a></li>
  <li><a href="{PREFIX}index.html#results">Results</a></li>
  <li><a href="{PREFIX}index.html#moments">Moments</a></li>
  <li><a href="#moments" class="active">All moments</a></li>
  <li><a href="{PREFIX}index.html#data">Data</a></li>
</ul></nav>

<section class="main" id="moments">
  <header>
    <div class="container">
      <h2>Every moment, both ways</h2>
      <p class="lede">{len(items)} clips across the zero-shot, RL fine-tuned and six-agent arms — {n_ok} where cooperation visibly worked and {n_bad} where it visibly did not — mined from the milestone and message logs of every episode, then cut to the step window around the event. Every clip opens with all agents' first-person views side by side; the button on it narrows to the agents contributing to the action, the first named being the one the event credits. Recordings replay at six environment steps per second.</p>
    </div>
  </header>
  <div class="content">
    <div class="container">
      <div class="controls">
        <div class="chips" role="group" aria-label="Filter by family">
          <span class="lbl">Agents</span>
          <button class="chip active" data-filter-family="all">All</button>
          <button class="chip" data-filter-family="zs">Zero-shot</button>
          <button class="chip" data-filter-family="rl">RL fine-tuned</button>
          <button class="chip" data-filter-family="six">Six-agent</button>
        </div>
        <div class="chips" role="group" aria-label="Filter by outcome">
          <span class="lbl">Outcome</span>
          <button class="chip active" data-filter="all">All</button>
          <button class="chip" data-filter="success">Successes</button>
          <button class="chip" data-filter="failure">Failures</button>
        </div>
        <div class="chips" role="group" aria-label="Filter by kind">
          <span class="lbl">Kind</span>
          <button class="chip active" data-filter-kind="all">Any</button>{kind_chips}
        </div>
        <button class="chip ghost" id="playall">▶ Play all visible</button>
      </div>
      {"".join(blocks)}
    </div>
  </div>
</section>

<footer id="footer"><div class="container">
  <p>Recordings are unedited agent views from WIRE, built on <a href="https://github.com/mikelma/craftium">Craftium</a> and <a href="https://www.luanti.org/">Luanti</a>.</p>
</div></footer>

<script defer src="{PREFIX}assets/js/main.js?v=20260923b"></script>
</body>
</html>
'''


def inject_index() -> None:
    """Fill the `<!-- chat:NAME -->` slots in index.html from both manifests."""
    import re

    items = {}
    if SCENES_MANIFEST.exists():
        m = json.loads(SCENES_MANIFEST.read_text(encoding="utf-8"))
        for section in ("tour", "scenes"):
            for it in m.get(section, []):
                items[it["name"]] = it
    if MANIFEST.exists():
        for it in json.loads(MANIFEST.read_text(encoding="utf-8")):
            items[it["name"]] = it
    s = INDEX.read_text(encoding="utf-8")
    filled, missing = 0, []

    def fill(match):
        nonlocal filled
        name = match.group(1)
        it = items.get(name)
        if not it:
            missing.append(name)
            return match.group(0)
        filled += 1
        return f"<!-- chat:{name} -->{transcript(it)}<!-- /chat:{name} -->"

    s = re.sub(r"<!-- chat:([^ ]+) -->.*?<!-- /chat:\1 -->", fill, s, flags=re.S)
    # The sync needs the window start and rate on the card itself.
    def attrs(match):
        name = match.group(2)
        it = items.get(name)
        if not it:
            return match.group(0)
        head = re.sub(r'\s*data-(t0|fps)="[^"]*"', "", match.group(1))
        return f'{head} data-clip="{name}" data-t0="{it["steps"][0]}" data-fps="{it.get("fps", 6)}">'
    s = re.sub(r'(<figure class="card[^>]*?)\s*data-clip="([^"]+)"(?:\s*data-(?:t0|fps)="[^"]*")*>', attrs, s)

    # Panel labels and the all-views variant: rebuild the stage around each
    # card's video. A previous stage (or bare label block) is collapsed first.
    s = re.sub(r'<div class="stage" data-stage>\s*(<video [^>]*></video>).*?<!-- /stage -->',
               lambda m: m.group(1), s, flags=re.S)
    s = re.sub(r'(<video [^>]*></video>)(?:\s*<div class="viewtags[^"]*"[^>]*>.*?</div>)+',
               lambda m: m.group(1), s, flags=re.S)

    def restage(match):
        name = match.group(1)
        it = items.get(name)
        if not it:
            return match.group(0)
        head = re.sub(r'\s*data-(?:all|pair)-(?:video|poster|n)="[^"]*"', "",
                      match.group(0)[:match.start(2) - match.start(0)])
        head = re.sub(r'\s*data-family="[^"]*"', "", head)
        tag, _, gap = head.rstrip().rpartition(">")          # "<figure ...", ">", ""
        gap = head[len(head.rstrip()):]                       # the whitespace before <video>
        head = f'{tag} data-family="{family_of(it["run"])}"{all_attrs(it)}>{gap}'   # index: eager posters
        return head + stage(it)
    s = re.sub(r'(?s)<figure class="card[^"]*"[^>]*data-clip="([^"]+)"[^>]*>\s*(<video [^>]*></video>)',
               restage, s)
    INDEX.write_text(s, encoding="utf-8")
    print(f"{INDEX.relative_to(REPO)}: {filled} transcripts injected"
          + (f"; no manifest entry for {missing}" if missing else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", action="store_true", help="show scores, notes and chat")
    ap.add_argument("--inject-index", action="store_true",
                    help="write transcripts into index.html and stop")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.inject_index:
        inject_index()
        return
    if not MANIFEST.exists():
        raise SystemExit(f"no {MANIFEST.relative_to(REPO)} — run scan_clip_candidates.py --cut")
    items = load()
    global PREFIX, LAZY_POSTERS
    LAZY_POSTERS = True
    rel = Path(os.path.relpath(SITE, args.out.resolve().parent)).as_posix()
    PREFIX = "" if rel == "." else rel + "/"
    args.out.write_text(render(items, args.review), encoding="utf-8")
    kept = "keep.txt" if KEEP.exists() else "no keep.txt (all)"
    print(f"{args.out.resolve().relative_to(REPO)}: {len(items)} clips ({kept})"
          + ("  [REVIEW — do not deploy]" if args.review else ""))


if __name__ == "__main__":
    main()
