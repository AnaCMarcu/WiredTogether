#!/usr/bin/env python
"""Build the project page's hero loop: a short montage of best moments.

The recordings are 320x180 at one frame per environment step. This takes
single-agent views of the moments the paper singles out — the agent the event
credits — and joins them into one 16:9 loop: the same step rate as the clips on
the page, a crisp nearest-neighbour upscale (the voxels stay voxels), short
cross-fades, a light grade and a gentle vignette. No motion interpolation and
no camera moves: an earlier version had both, and the warped, drifting frames
read as motion sickness rather than cinema. Every frame is a real observation.

Writes site/assets/videos/hero.mp4 and hero.jpg.

Usage:
  python analysis/make_hero.py            # ~20 s
  python analysis/make_hero.py --audit    # per-shot brightness / mob presence
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from paths import REPO  # noqa: E402  (also puts siblings on sys.path)
import make_media_clips as clips  # noqa: E402

OUT = REPO / "site" / "assets" / "videos" / "hero.mp4"

#: (run key, seed, episode, agent, first step, last step): one agent's own
#: view. Chosen with `--audit` for brightness and, in Chambers 4/5, for a mob
#: actually in frame (analysis/mob_visibility.py).
SHOTS = [
    ("gemma_3f",    123, 1, 0, 36, 66),      # Ch1: first steps, digging, the door
    ("qwen9b_base", 789, 2, 1, 196, 222),    # Ch2: anvil broken by agreement
    ("qwen9b_heb",   42, 2, 2, 289, 312),    # Ch2: anvil with a bonded partner
    ("gemma_3f",    123, 1, 0, 436, 458),    # Ch3: the blue switch, pressed
    ("gemma_3f",    123, 1, 0, 620, 650),    # Ch4: first contact
    ("qwen9b_heb",   42, 2, 0, 683, 701),    # Ch4: three agents on one mob
    ("gemma_heb",  1011, 2, 0, 679, 705),    # Ch4: a zombie point-blank
    ("qwen9b_base",1213, 1, 1, 758, 782),    # Ch4: fighting in the doorway
    ("gemma_3f",    123, 1, 0, 870, 900),    # Ch5: the boss chamber
]

SRC_FPS = clips.SRC_FPS
STEP_RATE = 6          # env steps per second, the same as the clips on the page
OUT_FPS = 24           # container rate; each step is held for four frames
W, H = 1280, 720
XFADE = 0.45           # seconds of cross-fade between shots


def build(exe: str, out: Path) -> None:
    inputs, chains, durations = [], [], []
    for i, (run, seed, ep, agent, t0, t1) in enumerate(SHOTS):
        src = clips.recording(run, agent, ep, seed)
        inputs += ["-i", str(src)]
        start, end = t0 / SRC_FPS, (t1 + 1) / SRC_FPS
        durations.append((t1 - t0 + 1) / STEP_RATE)
        k = SRC_FPS / STEP_RATE
        chains.append(
            f"[{i}:v]trim=start={start:.3f}:end={end:.3f},setpts=(PTS-STARTPTS)*{k:.6f},"
            f"fps={OUT_FPS},scale={W}:{H}:flags=neighbor,setsar=1,format=yuv420p[s{i}]")

    # Chain the cross-fades: each offset is the running length minus the fades.
    prev, total = "s0", durations[0]
    for i in range(1, len(SHOTS)):
        offset = total - XFADE
        chains.append(f"[{prev}][s{i}]xfade=transition=fade:duration={XFADE}:offset={offset:.3f}[x{i}]")
        prev = f"x{i}"
        total = offset + durations[i]
    # The rooms are dark by nature; lift the mids a little so the page's
    # overlay has something to sit on.
    chains.append(f"[{prev}]eq=contrast=1.06:saturation=1.08:gamma=1.2:brightness=0.02,"
                  f"vignette=angle=PI/6,fps={OUT_FPS}[v]")

    cmd = [exe, "-y", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(chains), "-map", "[v]", "-an",
           "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-r", str(OUT_FPS), str(out)]
    subprocess.run(cmd, check=True)
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(out), "-ss", "6",
                    "-frames:v", "1", "-q:v", "3", str(out.with_suffix(".jpg"))], check=True)
    print(f"{out.relative_to(REPO)}  {total:.1f}s  {out.stat().st_size / 1e6:.1f} MB")


def audit(exe: str, shots=None) -> None:
    """Per-shot brightness and mob presence, to choose windows that read on screen."""
    import mob_visibility  # noqa: E402

    for run, seed, ep, agent, t0, t1 in (shots or SHOTS):
        src = clips.recording(run, agent, ep, seed)
        ok, st = mob_visibility.check_recording(exe, src, t0, t1, every=3)
        print(f"  {run:<12} s{seed:<5} e{ep} a{agent} t{t0:>4}-{t1:<4} "
              f"bright={st['brightness']:.2f}  mob={'Y' if ok else '-'} peak={st['peak']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="print per-shot brightness/mob stats")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    exe = clips.ffmpeg_exe()
    if args.audit:
        audit(exe)
        return
    build(exe, args.out)


if __name__ == "__main__":
    main()
