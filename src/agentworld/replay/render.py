"""Frame renderer for replay videos.

Layout (default 1280 × 720): the map view on the left, a side panel on the
right. One round becomes ``frames_per_round`` frames with agents moving
linearly from their previous position; rounds that carry messages are held a
little longer so the text is readable.

Colour follows the job (reference palette, see the dataviz skill):
agents of the focused team take the categorical slots in a fixed order and
always carry a name label (identity is never colour alone); other teams'
agents are neutral grey; DMs are blue, transfers aqua, board posts neutral
ink; a kill uses the reserved critical red and always has a text label.
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from agentworld.replay.mapdata import TILE, PlainMap

# Reference palette (light): categorical slots in fixed order, ink, surfaces.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#9a9893"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
DM_COLOR = SLOTS[0]
XFER_COLOR = SLOTS[2]
POST_COLOR = INK_2
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"


def _rgb(hex_: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    h = hex_.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _font(size: int, bold: bool = False):
    try:
        import matplotlib
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)
    except Exception:
        return ImageFont.load_default()


def clean(text: Any) -> str:
    """Drop glyphs the font cannot draw (emoji, variation selectors)."""
    s = str(text or "")
    return "".join(ch for ch in s if ord(ch) < 0x1F000 and ord(ch) != 0xFE0F
                   and not (0x2600 <= ord(ch) <= 0x27BF and ch not in "✓✕→"))


def short_name(name: str, multi_team: bool) -> str:
    """'t02_fletcher_agent' → 'fletcher' ('t03·fletcher' in multi-team worlds)."""
    parts = name.split("_")
    team = parts[0] if multi_team and parts and parts[0].startswith("t") and parts[0][1:].isdigit() else ""
    core = [p for p in parts if not (p.startswith("t") and p[1:].isdigit()) and p != "agent"]
    label = "_".join(core) or name
    return f"{team}·{label}" if team else label


def load_states(episode_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(episode_dir) / "replay" / "state.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class RenderOptions:
    width: int = 1280
    height: int = 720
    panel_width: int = 420
    fps: int = 12
    frames_per_round: int = 8
    hold_frames: int = 14           # extra frames on rounds with messages
    camera: str = "team"            # "team" | "agent" | "world"
    team: Optional[str] = None      # for camera="team" (default: first team)
    agent: Optional[int] = None     # for camera="agent"; also the panel's reader view
    reader_view: bool = False       # panel shows what `agent` READ (gated inbox)
    show_bonds: bool = True
    margin_tiles: int = 6
    min_view_tiles: int = 28
    rounds: Optional[Tuple[int, int]] = None
    title: str = ""


@dataclass
class _Placed:
    rects: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def free(self, r: Tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = r
        return all(x1 <= a or x0 >= c or y1 <= b or y0 >= d for a, b, c, d in self.rects)


class ReplayRenderer:
    def __init__(self, states: List[Dict[str, Any]], map_: Any = None,
                 options: Optional[RenderOptions] = None):
        if not states:
            raise ValueError("no replay states")
        self.states = states
        self.map = map_ if map_ is not None else PlainMap()
        self.o = options or RenderOptions()
        self.f = {s: _font(s) for s in (11, 12, 13, 15)}
        self.fb = {s: _font(s, bold=True) for s in (11, 12, 13, 16)}
        first = states[0]["agents"]
        self.names = [a["name"] for a in first]
        self.teams = [a["team"] for a in first]
        self.multi = len(set(self.teams)) > 1
        self.labels = [short_name(n, self.multi) for n in self.names]
        self.focus_team = self.o.team or self.teams[0]
        self.msgs: Dict[int, Dict[str, Any]] = {}
        for s in states:
            for m in s.get("messages", []):
                self.msgs[m["msg_id"]] = m
        self.max_round = max(s["round"] for s in states)
        self.color: Dict[int, str] = {}
        team_order = list(dict.fromkeys(self.teams))
        if self.o.camera == "world" and 1 < len(team_order) <= len(SLOTS):
            # World view: colour is the TEAM (fixed slot order); labels carry the prefix.
            for i, t in enumerate(self.teams):
                self.color[i] = SLOTS[team_order.index(t)]
        else:
            members = [i for i, t in enumerate(self.teams) if t == self.focus_team]
            for k, i in enumerate(members):
                self.color[i] = SLOTS[k] if k < len(SLOTS) else OTHER

    # ── camera ─────────────────────────────────────────────────────────────
    def _focus(self) -> List[int]:
        if self.o.camera == "agent" and self.o.agent is not None:
            return [self.o.agent]
        if self.o.camera == "world":
            return list(range(len(self.names)))
        return [i for i, t in enumerate(self.teams) if t == self.focus_team]

    def _view(self, state: Dict[str, Any]) -> Tuple[int, int, int, int]:
        mw = self.o.width - self.o.panel_width
        aspect = mw / self.o.height
        pts = [(a["x"], a["y"]) for a in state["agents"]
               if a["i"] in self._focus() and a["x"] is not None]
        if not pts:
            pts = [(a["x"], a["y"]) for a in state["agents"] if a["x"] is not None] or [(0, 0)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w = max(max(xs) - min(xs) + 2 * self.o.margin_tiles, self.o.min_view_tiles)
        h = max(max(ys) - min(ys) + 2 * self.o.margin_tiles, w / aspect)
        w = max(w, h * aspect)
        w_t, h_t = int(math.ceil(w)), int(math.ceil(w / aspect))
        return int(cx - w_t / 2), int(cy - h_t / 2), w_t, h_t

    # ── frames ─────────────────────────────────────────────────────────────
    def frames(self) -> Iterator[Image.Image]:
        prev = None
        lo, hi = self.o.rounds or (1, self.max_round)
        for state in self.states:
            if not (lo <= state["round"] <= hi):
                prev = state
                continue
            view = self._view(state)
            base = self._base(view)
            k = self.o.frames_per_round
            n_frames = k + (self.o.hold_frames if state.get("messages") else 0)
            for f in range(n_frames):
                alpha = min(1.0, (f + 1) / k)
                yield self._frame(state, prev, alpha, view, base)
            prev = state

    def _base(self, view) -> Image.Image:
        x0, y0, w, h = view
        img = self.map.region(x0, y0, w, h)
        return img.resize((self.o.width - self.o.panel_width, self.o.height), Image.BILINEAR)

    def _to_px(self, view, x: float, y: float) -> Tuple[float, float]:
        x0, y0, w, h = view
        sx = (self.o.width - self.o.panel_width) / (w * TILE)
        return ((x - x0 + 0.5) * TILE * sx, (y - y0 + 0.5) * TILE * sx)

    def _frame(self, state, prev, alpha, view, base) -> Image.Image:
        canvas = Image.new("RGB", (self.o.width, self.o.height), SURFACE)
        canvas.paste(base, (0, 0))
        over = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        pos = self._positions(state, prev, alpha, view)
        placed = _Placed()
        for (x, y) in pos.values():  # labels and bubbles never cover an agent
            placed.rects.append((int(x - 10), int(y - 18), int(x + 10), int(y + 10)))
        self._draw_mobs(d, state, view)
        self._draw_transfers(d, state, pos, placed)
        self._draw_agents(d, state, pos, placed)
        self._draw_kills(d, state, pos, view)
        self._draw_bubbles(d, state, pos, placed)
        merged = Image.alpha_composite(canvas.crop((0, 0) + base.size).convert("RGBA"), over)
        canvas.paste(merged.convert("RGB"), (0, 0))
        self._draw_panel(canvas, state)
        return canvas

    def _positions(self, state, prev, alpha, view) -> Dict[int, Tuple[float, float]]:
        out = {}
        before = {a["i"]: a for a in prev["agents"]} if prev else {}
        for a in state["agents"]:
            if a["x"] is None:
                continue
            x, y = a["x"], a["y"]
            p = before.get(a["i"])
            if p and p.get("x") is not None:
                x = p["x"] + (x - p["x"]) * alpha
                y = p["y"] + (y - p["y"]) * alpha
            out[a["i"]] = self._to_px(view, x, y)
        return out

    def _in_view(self, xy) -> bool:
        return 0 <= xy[0] < self.o.width - self.o.panel_width and 0 <= xy[1] < self.o.height

    def _halo_text(self, d, xy, text, font, fill=INK):
        d.text(xy, text, font=font, fill=fill, stroke_width=2, stroke_fill=(255, 255, 255, 230))

    def _draw_mobs(self, d, state, view):
        detail = self.o.camera != "world"
        for m in state.get("mobs", []):
            if m.get("x") is None:
                continue
            x, y = self._to_px(view, m["x"], m["y"])
            if not self._in_view((x, y)):
                continue
            r = 5
            d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=_rgb("#6b2f2f", 220),
                      outline=(255, 255, 255, 200))
            if detail:
                self._halo_text(d, (x + 7, y - 7), clean(m.get("name", "")), self.f[11], INK_2)

    def _agent_color(self, i: int) -> str:
        return self.color.get(i, OTHER)

    def _place(self, placed: "_Placed", w: int, h: int, candidates) -> Tuple[int, int]:
        """First candidate top-left whose box is free; else the first one."""
        mw = self.o.width - self.o.panel_width
        for (cx, cy) in candidates:
            cx = int(min(max(2, cx), mw - w - 2))
            rect = (cx, int(cy), cx + w, int(cy) + h)
            if placed.free(rect):
                placed.rects.append(rect)
                return cx, int(cy)
        cx, cy = candidates[0]
        cx = int(min(max(2, cx), mw - w - 2))
        placed.rects.append((cx, int(cy), cx + w, int(cy) + h))
        return cx, int(cy)

    def _draw_agents(self, d, state, pos, placed):
        world = self.o.camera == "world"
        acts = {}
        for a in state.get("actions", []):
            acts.setdefault(a["i"], a)
        big = world and len(state["agents"]) > 30
        r = 4 if big else 8
        for a in state["agents"]:
            i = a["i"]
            if i not in pos:
                continue
            x, y = pos[i]
            col = _rgb(self._agent_color(i))
            d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255, 235))
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
            if a.get("hp") is not None and a.get("maxhp"):
                frac = max(0.0, min(1.0, a["hp"] / max(1, a["maxhp"])))
                bw = 2 * r + 6
                d.rectangle([x - bw / 2, y - r - 7, x + bw / 2, y - r - 4], fill=(40, 40, 40, 160))
                d.rectangle([x - bw / 2, y - r - 7, x - bw / 2 + bw * frac, y - r - 4],
                            fill=_rgb(GOOD if frac > 0.35 else CRITICAL))
        # Labels after all markers, placed to avoid each other.
        order = sorted((a for a in state["agents"] if a["i"] in pos), key=lambda a: pos[a["i"]][1])
        for a in order:
            i = a["i"]
            if big and i not in self._speakers(state):
                continue
            x, y = pos[i]
            name = self.labels[i] + (" …" if a.get("busy") else "")
            act = acts.get(i)
            act_text = clean(self._act_label(act)) if act and not world else ""
            w = int(max(d.textlength(name, font=self.fb[11]),
                        d.textlength(act_text, font=self.f[11]) if act_text else 0)) + 4
            h = 28 if act_text else 14
            cands = [(x - w / 2, y + r + 3), (x + r + 4, y - 6), (x - w - r - 4, y - 6),
                     (x - w / 2, y + r + 3 + h), (x + r + 4, y + 10), (x - w - r - 4, y + 10)]
            lx, ly = self._place(placed, w, h, cands)
            self._halo_text(d, (lx + 2, ly), name, self.fb[11])
            if act_text:
                self._halo_text(d, (lx + 2, ly + 14), act_text, self.f[11],
                                INK_2 if act.get("ok") else INK_3)

    @staticmethod
    def _act_label(act: Dict[str, Any]) -> str:
        call = act["call"]
        name = call.split("(", 1)[0]
        args = call[len(name) + 1:-1]
        if name == "craft_item":
            item = next((p.split("=", 1)[1] for p in args.split(", ") if p.startswith("itemKey=")), "")
            return f"craft {item}" + ("" if act.get("ok") else " (failed)")
        if name == "harvest_resource":
            return "harvest"
        if name == "attack_entity":
            return "attack"
        if name == "move_character":
            return "move"
        if name in ("wait", "sleep"):
            return ""
        if name == "transfer_items":
            return "" if act.get("ok") else "transfer (failed)"
        return name.replace("_", " ")

    def _speakers(self, state) -> set:
        return {m["sender"] for m in state.get("messages", [])}

    def _arrow(self, d, a, b, color, width=2, head=8, dashed=False):
        (x0, y0), (x1, y1) = a, b
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        x1s, y1s = x1 - ux * 10, y1 - uy * 10
        px, py = -uy, ux
        tip = [(x1s + ux * head, y1s + uy * head),
               (x1s + px * head * 0.5, y1s + py * head * 0.5),
               (x1s - px * head * 0.5, y1s - py * head * 0.5)]
        halo = (255, 255, 255, 220)
        segs = []
        if dashed:
            n = max(1, int(L // 10))
            for k in range(0, n, 2):
                t0, t1 = k / n, min(1.0, (k + 1) / n)
                segs.append([(x0 + (x1s - x0) * t0, y0 + (y1s - y0) * t0),
                             (x0 + (x1s - x0) * t1, y0 + (y1s - y0) * t1)])
        else:
            segs.append([(x0, y0), (x1s, y1s)])
        for seg in segs:  # white under-stroke keeps the line legible on any tile
            d.line(seg, fill=halo, width=width + 3)
        d.polygon(tip, fill=halo, outline=halo, width=3)
        for seg in segs:
            d.line(seg, fill=color, width=width)
        d.polygon(tip, fill=color)

    def _draw_transfers(self, d, state, pos, placed):
        for e in state.get("events", []):
            if e["kind"] != "xfer" or e["src"] not in pos or e.get("dst") not in pos:
                continue
            a, b = pos[e["src"]], pos[e["dst"]]
            bend = max(28.0, 0.25 * math.hypot(b[0] - a[0], b[1] - a[1]))
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + bend   # arcs dip below
            pts = [((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * mx + t ** 2 * b[0],
                    (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * my + t ** 2 * b[1])
                   for t in [k / 16 for k in range(17)]]
            d.line(pts[:-2], fill=(255, 255, 255, 220), width=7, joint="curve")
            d.line(pts[:-2], fill=_rgb(XFER_COLOR), width=4, joint="curve")
            self._arrow(d, pts[-3], pts[-1], _rgb(XFER_COLOR), width=4, head=10)
            label = clean(f"{e['data'].get('count', '')}x "
                          f"{e['data'].get('itemKey') or e['data'].get('item', '')}")
            w = int(d.textlength(label, font=self.fb[11])) + 6
            apex_y = (a[1] + b[1]) / 4 + my / 2
            lx, ly = self._place(placed, w, 15, [((a[0] + b[0]) / 2 - w / 2, apex_y + 2),
                                                 ((a[0] + b[0]) / 2 - w / 2, apex_y + 18)])
            self._halo_text(d, (lx + 3, ly), label, self.fb[11], INK)

    def _draw_kills(self, d, state, pos, view):
        for e in state.get("events", []):
            if e["kind"] == "kill" and e["src"] in pos:
                x, y = pos[e["src"]]
                d.line([(x + 10, y - 22), (x + 20, y - 12)], fill=_rgb(CRITICAL), width=3)
                d.line([(x + 10, y - 12), (x + 20, y - 22)], fill=_rgb(CRITICAL), width=3)
                self._halo_text(d, (x + 24, y - 26), clean(f"killed {e['data'].get('mob', '')}"),
                                self.fb[11], CRITICAL)
            elif e["kind"] == "death" and e["src"] in pos:
                x, y = pos[e["src"]]
                self._halo_text(d, (x - 14, y - 32), "died", self.fb[12], CRITICAL)

    def _wrap(self, text: str, width: int = 30, lines: int = 3) -> List[str]:
        out = textwrap.wrap(clean(text), width=width) or [""]
        if len(out) > lines:
            out = out[:lines]
            out[-1] = out[-1][: width - 1] + "…"
        return out

    def _draw_bubbles(self, d, state, pos, placed):
        mw = self.o.width - self.o.panel_width
        world = self.o.camera == "world" and len(state["agents"]) > 30
        for m in sorted(state.get("messages", []), key=lambda m: (m["kind"] != "dm", m["msg_id"])):
            s = m["sender"]
            if s not in pos:
                continue
            is_dm = m["kind"] == "dm"
            color = _rgb(DM_COLOR if is_dm else POST_COLOR)
            if is_dm:
                r = m.get("receiver")
                header = f"→ {self.labels[r]}" if r is not None else "→ ?"
            else:
                header = f"[board · {m.get('post_kind', 'status')}]"
            body = self._wrap(m["text"], 22 if world else 30, 1 if world else 3)
            font, hfont = self.f[12], self.fb[12]
            tw = max([d.textlength(header, font=hfont)] + [d.textlength(b, font=font) for b in body])
            bw, bh = int(tw) + 14, 18 + 15 * len(body)
            x, y = pos[s]
            bx, by = int(min(max(4, x - bw / 2), mw - bw - 4)), int(y - 30 - bh)
            for _ in range(12):
                rect = (bx, by, bx + bw, by + bh)
                if placed.free(rect) and by >= 2:
                    break
                by -= bh // 2 + 4
                if by < 2:
                    by = int(y + 30)
            placed.rects.append((bx, by, bx + bw, by + bh))
            d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=6, fill=(255, 255, 255, 240),
                                outline=color, width=2)
            d.text((bx + 7, by + 3), header, font=hfont, fill=color)
            for k, line in enumerate(body):
                d.text((bx + 7, by + 18 + 15 * k), line, font=font, fill=INK)
            tail = (x, y - 10)
            anchor = (bx + bw / 2, by + bh if by + bh < y else by)
            d.line([anchor, tail], fill=color, width=1)
            if is_dm and m.get("receiver") is not None:
                r = m["receiver"]
                if r in pos and self._in_view(pos[r]):
                    self._arrow(d, (bx + bw, by + bh / 2), pos[r], color, width=3, dashed=True)
                else:
                    self._halo_text(d, (bx + bw + 4, by + 2), "(off-screen)", self.f[11], INK_3)

    # ── panel ──────────────────────────────────────────────────────────────
    def _draw_panel(self, canvas: Image.Image, state: Dict[str, Any]) -> None:
        x0 = self.o.width - self.o.panel_width
        d = ImageDraw.Draw(canvas)
        d.rectangle([x0, 0, self.o.width, self.o.height], fill=SURFACE)
        d.line([(x0, 0), (x0, self.o.height)], fill=GRID, width=1)
        x, y = x0 + 16, 12
        title = self.o.title or "AgentWorld replay"
        d.text((x, y), clean(title), font=self.fb[16], fill=INK)
        y += 22
        sub = f"Round {state['round']} / {self.max_round}"
        if self.o.camera == "team":
            sub += f"  ·  team {self.focus_team}"
        elif self.o.camera == "agent" and self.o.agent is not None:
            sub += f"  ·  following {self.labels[self.o.agent]}"
        else:
            sub += f"  ·  {len(self.names)} agents"
        d.text((x, y), sub, font=self.f[13], fill=INK_2)
        y += 24
        y = self._panel_progress(d, state, x, y)
        y += 6
        if self.o.reader_view and self.o.agent is not None:
            y = self._panel_reader(d, state, x, y)
        else:
            y = self._panel_chat(d, state, x, y)
        if self.o.show_bonds and state.get("bonds_top"):
            self._panel_bonds(d, state, x0 + 16, max(y + 8, self.o.height - 200))

    def _panel_progress(self, d, state, x, y) -> int:
        teams = state.get("teams", {})
        shown = [self.focus_team] if self.o.camera != "world" else sorted(teams)[:6]
        bw = self.o.panel_width - 140
        for t in shown:
            info = teams.get(t, {})
            p = info.get("progress") or 0.0
            d.text((x, y), t, font=self.f[12], fill=INK_2)
            d.rounded_rectangle([x + 40, y + 3, x + 40 + bw, y + 11], radius=4, fill=GRID)
            if p > 0:
                d.rounded_rectangle([x + 40, y + 3, x + 40 + max(8, bw * min(1, p)), y + 11],
                                    radius=4, fill=SLOTS[0])
            label = "solved ✓" if info.get("solved") else f"{int(round(100 * p))}%"
            d.text((x + 48 + bw, y - 1), label, font=self.f[12],
                   fill=GOOD if info.get("solved") else INK_2)
            y += 18
        if self.o.camera == "world" and len(teams) > 6:
            solved = sum(1 for v in teams.values() if v.get("solved"))
            d.text((x, y), f"{solved}/{len(teams)} teams solved", font=self.f[12], fill=INK_2)
            y += 18
        return y

    def _relevant(self, m: Dict[str, Any]) -> bool:
        focus = set(self._focus())
        return m["sender"] in focus or m.get("receiver") in focus

    def _panel_chat(self, d, state, x, y) -> int:
        d.text((x, y), "Direct messages", font=self.fb[13], fill=INK)
        y += 20
        history = [m for m in self.msgs.values()
                   if m["round"] <= state["round"] and m["kind"] == "dm" and self._relevant(m)]
        lines = []
        for m in sorted(history, key=lambda m: m["msg_id"])[-9:]:
            head = f"r{m['round']} {self.labels[m['sender']]} → " \
                   f"{self.labels[m['receiver']] if m.get('receiver') is not None else '?'}: "
            lines.append((m, head, self._wrap(m["text"], 40, 2)))
        if not lines:
            d.text((x, y), "none yet", font=self.f[12], fill=INK_3)
            y += 18
        for m, head, body in lines:
            fresh = m["round"] == state["round"]
            d.ellipse([x, y + 4, x + 7, y + 11], fill=self._agent_color(m["sender"]))
            d.text((x + 12, y), clean(head), font=self.fb[12] if fresh else self.f[12],
                   fill=INK if fresh else INK_2)
            y += 15
            for line in body:
                d.text((x + 12, y), line, font=self.f[12], fill=INK if fresh else INK_2)
                y += 15
            y += 3
        y += 6
        d.text((x, y), "Board", font=self.fb[13], fill=INK)
        y += 20
        posts = [m for m in self.msgs.values()
                 if m["kind"] == "post" and state["round"] - 2 <= m["round"] <= state["round"]
                 and (self.o.camera == "world" or m["sender"] in set(self._focus()))]
        for m in sorted(posts, key=lambda m: m["msg_id"])[-4:]:
            d.text((x, y), clean(f"r{m['round']} {self.labels[m['sender']]} [{m.get('post_kind')}]"),
                   font=self.f[12], fill=INK_2)
            y += 15
            d.text((x + 12, y), self._wrap(m["text"], 44, 1)[0], font=self.f[12], fill=INK_2)
            y += 17
        return y

    def _panel_reader(self, d, state, x, y) -> int:
        i = self.o.agent
        reads = (state.get("reads") or {}).get(str(i)) or (state.get("reads") or {}).get(i)
        d.text((x, y), f"What {self.labels[i]} read this round", font=self.fb[13], fill=INK)
        y += 20
        if not reads:
            d.text((x, y), "busy or done — nothing read", font=self.f[12], fill=INK_3)
            return y + 18
        for label, ids in (("Inbox (DMs)", reads.get("dms", [])), ("Board", reads.get("board", []))):
            d.text((x, y), f"{label}: {len(ids)}", font=self.fb[12], fill=INK_2)
            y += 16
            for mid in ids[:5]:
                m = self.msgs.get(mid)
                if not m:
                    continue
                d.ellipse([x, y + 4, x + 7, y + 11], fill=self._agent_color(m["sender"]))
                d.text((x + 12, y), clean(f"{self.labels[m['sender']]}: ")
                       + self._wrap(m["text"], 40, 1)[0], font=self.f[12], fill=INK)
                y += 16
            y += 4
        contacts = ", ".join(self.labels[j] for j in reads.get("contacts", []))
        d.text((x, y), "Contacts:", font=self.fb[12], fill=INK_2)
        y += 16
        for line in self._wrap(contacts, 48, 3):
            d.text((x, y), line, font=self.f[12], fill=INK_2)
            y += 15
        return y

    def _panel_bonds(self, d, state, x, y) -> None:
        focus = self._focus()[:10]
        if len(focus) < 2:
            return
        d.text((x, y), "Strongest bonds", font=self.fb[13], fill=INK)
        d.text((x + 130, y + 2), "line width = bond W", font=self.f[11], fill=INK_3)
        cx, cy, R = x + 190, y + 105, 62
        pts = {i: (cx + R * math.cos(2 * math.pi * k / len(focus) - math.pi / 2),
                   cy + R * math.sin(2 * math.pi * k / len(focus) - math.pi / 2))
               for k, i in enumerate(focus)}
        top = state["bonds_top"]
        for i in focus:
            for j, w in top.get(str(i), top.get(i, [])):
                if j in pts and w > 0.12:
                    d.line([pts[i], pts[j]], fill=INK_2, width=max(1, int(round(1 + 5 * w))))
        for i, (px, py) in pts.items():
            d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=self._agent_color(i),
                      outline=SURFACE, width=2)
            tw = d.textlength(self.labels[i], font=self.f[11])
            lx = px + 10 if px >= cx else px - 10 - tw   # labels point outward
            d.text((lx, py - 7), self.labels[i], font=self.f[11], fill=INK_2)

    # ── output ─────────────────────────────────────────────────────────────
    def render_to(self, path: str | Path) -> Path:
        import imageio_ffmpeg
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w, h = self.o.width - self.o.width % 2, self.o.height - self.o.height % 2
        writer = imageio_ffmpeg.write_frames(
            str(path), (w, h), fps=self.o.fps, codec="libx264", pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p", macro_block_size=1,
            output_params=["-crf", "20", "-preset", "medium"])
        writer.send(None)
        try:
            for frame in self.frames():
                if frame.size != (w, h):
                    frame = frame.crop((0, 0, w, h))
                writer.send(frame.tobytes())
        finally:
            writer.close()
        return path

    def snapshot(self, round_num: int) -> Image.Image:
        """The last (fully settled) frame of one round, e.g. for figures."""
        prev = None
        for state in self.states:
            if state["round"] == round_num:
                view = self._view(state)
                return self._frame(state, prev, 1.0, view, self._base(view))
            prev = state
        raise KeyError(round_num)
