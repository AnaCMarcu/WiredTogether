"""Map backgrounds for replay videos.

:class:`KaetramMap` draws the real AgentWorld map from the server's
``packages/server/data/map/world.json`` (tile ids per cell) and the client's
``public/img/tilesets/tilesheet-*.png`` (64 × 64 tiles of 16 px each). The
lookup follows the client renderer (packages/client/src/renderer/canvas.ts):
a stored value ``t`` is drawn as tile ``t − 1`` of the tileset whose
[firstGid, lastGid] range contains it, at column ``rel % 64``, row
``rel // 64``. A cell holds 0 (empty), one id, or a list of layered ids; flipped
tiles are dicts ``{tileId, h, v, d}``.

Regions are assembled from 32 × 32-tile chunks cached in memory, so a camera
that drifts slowly re-uses almost everything.

:class:`PlainMap` is a neutral grid used when no checkout is available (and in
tests).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

TILE = 16
CHUNK = 32


class PlainMap:
    """Neutral checker grid; same interface as :class:`KaetramMap`."""

    def __init__(self, width: int = 1056, height: int = 768):
        self.width, self.height = width, height

    def region(self, x0: int, y0: int, w: int, h: int) -> Image.Image:
        img = Image.new("RGB", (w * TILE, h * TILE), (226, 228, 222))
        d = ImageDraw.Draw(img)
        for ty in range(h):
            for tx in range(w):
                if (x0 + tx + y0 + ty) % 2 == 0:
                    d.rectangle([tx * TILE, ty * TILE, (tx + 1) * TILE - 1, (ty + 1) * TILE - 1],
                                fill=(214, 217, 209))
        return img


class KaetramMap:
    def __init__(self, agentworld_root: str | Path):
        root = Path(agentworld_root)
        world = json.loads((root / "packages/server/data/map/world.json").read_text(encoding="utf-8"))
        client = json.loads((root / "packages/client/data/maps/map.json").read_text(encoding="utf-8"))
        self.width = int(world["width"])
        self.height = int(world["height"])
        self.data: List[Any] = world["data"]
        self.tilesets = sorted(client["tilesets"], key=lambda t: t["firstGid"])
        self.tileset_dir = root / "packages/client/public/img/tilesets"
        self._sheets: Dict[str, Image.Image] = {}
        self._chunks: Dict[Tuple[int, int], Image.Image] = {}

    def _sheet(self, path: str) -> Image.Image:
        if path not in self._sheets:
            self._sheets[path] = Image.open(self.tileset_dir / path).convert("RGBA")
        return self._sheets[path]

    @lru_cache(maxsize=8192)
    def _tile(self, tile_id: int, h: bool = False, v: bool = False, d: bool = False
              ) -> Optional[Image.Image]:
        tid = tile_id - 1
        for ts in self.tilesets:
            if ts["firstGid"] <= tid <= ts["lastGid"]:
                sheet = self._sheet(ts["path"])
                cols = sheet.width // TILE
                rel = tid - ts["firstGid"]
                x, y = (rel % cols) * TILE, (rel // cols) * TILE
                tile = sheet.crop((x, y, x + TILE, y + TILE))
                if d:
                    tile = tile.transpose(Image.Transpose.TRANSPOSE)
                if h:
                    tile = tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if v:
                    tile = tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                return tile
        return None

    def _draw_cell(self, img: Image.Image, cell: Any, px: int, py: int) -> None:
        layers = cell if isinstance(cell, list) else [cell]
        for t in layers:
            if isinstance(t, dict):
                tile = self._tile(int(t.get("tileId", 0)), bool(t.get("h")), bool(t.get("v")),
                                  bool(t.get("d")))
            elif t:
                tile = self._tile(int(t))
            else:
                continue
            if tile is not None:
                img.alpha_composite(tile, (px, py))

    def _chunk(self, cx: int, cy: int) -> Image.Image:
        key = (cx, cy)
        if key not in self._chunks:
            img = Image.new("RGBA", (CHUNK * TILE, CHUNK * TILE), (24, 24, 22, 255))
            for ty in range(CHUNK):
                y = cy * CHUNK + ty
                if not (0 <= y < self.height):
                    continue
                for tx in range(CHUNK):
                    x = cx * CHUNK + tx
                    if not (0 <= x < self.width):
                        continue
                    idx = y * self.width + x
                    if idx < len(self.data):
                        self._draw_cell(img, self.data[idx], tx * TILE, ty * TILE)
            if len(self._chunks) > 256:
                self._chunks.pop(next(iter(self._chunks)))
            self._chunks[key] = img
        return self._chunks[key]

    def region(self, x0: int, y0: int, w: int, h: int) -> Image.Image:
        out = Image.new("RGBA", (w * TILE, h * TILE), (24, 24, 22, 255))
        for cy in range(y0 // CHUNK, (y0 + h) // CHUNK + 1):
            for cx in range(x0 // CHUNK, (x0 + w) // CHUNK + 1):
                ch = self._chunk(cx, cy)
                out.paste(ch, ((cx * CHUNK - x0) * TILE, (cy * CHUNK - y0) * TILE))
        return out.convert("RGB")


def load_map(agentworld_root: Optional[str | Path]) -> Any:
    """The real map if the checkout has it, else :class:`PlainMap`."""
    if agentworld_root:
        root = Path(agentworld_root)
        if (root / "packages/server/data/map/world.json").is_file():
            return KaetramMap(root)
    return PlainMap()
