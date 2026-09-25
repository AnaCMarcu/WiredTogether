#!/usr/bin/env python
"""Is a hostile mob actually in frame? A colour gate for the Chamber 4/5 clips.

A combat clip in which no zombie is ever visible is not a combat clip, however
good the messages read. The VoxeLibre zombie is the only blue thing in WIRE's
combat chambers — cyan-green head (hue ~176 deg), blue shirt and legs (hue
~235 deg) — against brown stone, brown floors, red villagers and warm torch
light, so a hue mask on the agent's own view is a reliable presence test.

The hotbar at the bottom of the frame is dark navy and is masked out, and so
is the bottom-right corner where the agent's own held item is drawn: from
Chamber 2 on that is a diamond sword, bright cyan, squarely in the teal band.
The purple anvils sit above 250 deg and are excluded by the hue band; a bright
cyan cap keeps sword glints elsewhere in frame from counting as a head.

    python analysis/mob_visibility.py site/assets/videos/candidates/*.mp4
    python analysis/mob_visibility.py --recording <run> --agent 1 --ep 2 --t 683 701

Both entry points return the per-frame fraction of zombie-coloured pixels and
the verdict that `scan_clip_candidates.py` uses to drop Ch4/Ch5 candidates
where nothing hostile was ever on screen.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from paths import REPO  # noqa: E402  (also puts siblings on sys.path)

# Hue bands in degrees, saturation/value in [0, 1]. Blue = shirt and legs,
# teal = head. Both are far from anything else in the combat chambers.
BLUE = (205, 248)     # the purple anvils' edge pixels start at ~252
TEAL = (160, 195)
MIN_S, MIN_V = 0.30, 0.20
TEAL_MAX_V = 0.72        # a zombie head is dim; a diamond sword is not
HUD_FROM = 0.72          # fraction of frame height where the hotbar starts
HELD_X, HELD_Y = 0.60, 0.42   # the held item occupies x >= 60% and y >= 42%

#: A frame counts as "mob visible" above this fraction of scored pixels.
FRAME_THRESH = 0.0015
#: A clip passes when this share of sampled frames show a mob, or any single
#: frame shows a large one (a zombie point-blank fills a lot of the view).
CLIP_MIN_SHARE = 0.15
CLIP_BIG_FRAME = 0.012


def measure(img: Image.Image) -> tuple[float, float]:
    """(share of non-HUD pixels in the zombie hue bands, mean brightness).

    A rendered side-by-side clip (aspect > 3:1) is two agent views; each panel
    carries its own hotbar and held item, so score the panels separately and
    let the more mob-filled one speak for the frame.
    """
    if img.width > 3 * img.height:
        half = img.width // 2
        left = _measure_view(img.crop((0, 0, half, img.height)))
        right = _measure_view(img.crop((half, 0, img.width, img.height)))
        return max(left[0], right[0]), (left[1] + right[1]) / 2
    return _measure_view(img)


def _measure_view(img: Image.Image) -> tuple[float, float]:
    hsv = np.asarray(img.convert("RGB").convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * (360 / 255), hsv[..., 1] / 255, hsv[..., 2] / 255
    H_, W_ = hsv.shape[:2]
    keep = np.ones((H_, W_), dtype=bool)
    keep[int(H_ * HUD_FROM):, :] = False                       # hotbar
    keep[int(H_ * HELD_Y):, int(W_ * HELD_X):] = False         # held item
    sat = (s >= MIN_S) & (v >= MIN_V) & keep
    blue = sat & (h >= BLUE[0]) & (h <= BLUE[1])
    teal = sat & (h >= TEAL[0]) & (h <= TEAL[1]) & (v <= TEAL_MAX_V)
    return float((blue | teal).sum() / keep.sum()), float(v[keep].mean())


def fraction(img: Image.Image) -> float:
    """Share of non-HUD pixels in the zombie hue bands."""
    return measure(img)[0]


def _frames(exe: str, src: Path, tmp: Path, start: float | None, dur: float | None,
            fps: float) -> list[Path]:
    cmd = [exe, "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vf", f"fps={fps}", "-f", "image2", str(tmp / "f_%04d.png")]
    subprocess.run(cmd, check=True)
    return sorted(tmp.glob("f_*.png"))


def verdict(samples: list[tuple[float, float]]) -> tuple[bool, dict]:
    if not samples:
        return False, dict(frames=0, share=0.0, peak=0.0, brightness=0.0)
    fractions = [f for f, _ in samples]
    share = sum(f >= FRAME_THRESH for f in fractions) / len(fractions)
    peak = max(fractions)
    ok = share >= CLIP_MIN_SHARE or peak >= CLIP_BIG_FRAME
    bright = sum(b for _, b in samples) / len(samples)
    return ok, dict(frames=len(fractions), share=round(share, 2), peak=round(peak, 4),
                    brightness=round(bright, 3))


def check_clip(exe: str, clip: Path, fps: float = 3.0) -> tuple[bool, dict]:
    """Rendered clip (either view counts): sample `fps` frames per second."""
    with tempfile.TemporaryDirectory() as td:
        frames = _frames(exe, clip, Path(td), None, None, fps)
        return verdict([measure(Image.open(f)) for f in frames])


def check_recording(exe: str, src: Path, t0: int, t1: int, src_fps: int = 2,
                    every: int = 2) -> tuple[bool, dict]:
    """Source recording over an episode-step window, sampling every `every` steps."""
    start, dur = t0 / src_fps, (t1 - t0 + 1) / src_fps
    with tempfile.TemporaryDirectory() as td:
        frames = _frames(exe, src, Path(td), start, dur, src_fps / every)
        return verdict([measure(Image.open(f)) for f in frames])


def main() -> None:
    import make_media_clips as clips  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", type=Path, help="rendered clips to check")
    ap.add_argument("--recording", help="run key in make_media_clips.RUNS")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--agent", type=int)
    ap.add_argument("--ep", type=int)
    ap.add_argument("--t", nargs=2, type=int, metavar=("T0", "T1"))
    args = ap.parse_args()
    exe = clips.ffmpeg_exe()

    if args.recording:
        src = clips.recording(args.recording, args.agent, args.ep, args.seed)
        ok, st = check_recording(exe, src, *args.t)
        print(f"{'MOB' if ok else '---'}  {src.name} t{args.t[0]}-{args.t[1]}  {st}")
        return
    if not args.clips:
        sys.exit("nothing to check")
    for c in args.clips:
        ok, st = check_clip(exe, c)
        rel = c.resolve().relative_to(REPO) if c.resolve().is_relative_to(REPO) else c
        print(f"{'MOB' if ok else '---'}  {str(rel):<70} share={st['share']:<5} "
              f"peak={st['peak']:<7} bright={st['brightness']}")


if __name__ == "__main__":
    main()
