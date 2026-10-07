"""CLI: render replay videos for one episode.

    python -m agentworld.replay runs/.../episode_1 --team t00
    python -m agentworld.replay runs/.../episode_1 --agent 3 --reader-view
    python -m agentworld.replay runs/.../episode_1 --world --rounds 10:30

The real map is drawn when ``--agentworld-root`` (or $AGENTWORLD_ROOT) points
at an AgentWorld checkout; otherwise a plain grid is used.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agentworld.replay.mapdata import load_map
from agentworld.replay.render import RenderOptions, ReplayRenderer, load_states


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir", type=Path)
    cam = ap.add_mutually_exclusive_group()
    cam.add_argument("--team", help="follow one team (default: the first)")
    cam.add_argument("--agent", type=int, help="follow one agent (global index)")
    cam.add_argument("--world", action="store_true", help="fit every agent")
    ap.add_argument("--reader-view", action="store_true",
                    help="panel shows what --agent actually read (its gated inbox)")
    ap.add_argument("--rounds", help="first:last round, e.g. 10:30")
    ap.add_argument("--no-bonds", action="store_true")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--frames-per-round", type=int, default=8)
    ap.add_argument("--title", default="")
    ap.add_argument("--agentworld-root", default=os.environ.get("AGENTWORLD_ROOT"))
    ap.add_argument("--out", type=Path, help="output .mp4 (default: <episode>/videos/...)")
    ap.add_argument("--png", type=int, metavar="ROUND", help="write one still frame instead")
    args = ap.parse_args(argv)

    camera = "world" if args.world else "agent" if args.agent is not None else "team"
    rounds = tuple(int(v) for v in args.rounds.split(":")) if args.rounds else None
    opts = RenderOptions(fps=args.fps, frames_per_round=args.frames_per_round, camera=camera,
                         team=args.team, agent=args.agent, reader_view=args.reader_view,
                         show_bonds=not args.no_bonds, rounds=rounds, title=args.title)
    renderer = ReplayRenderer(load_states(args.episode_dir), load_map(args.agentworld_root), opts)
    tag = camera if camera == "world" else f"{camera}_{args.team or args.agent or renderer.focus_team}"
    if args.png is not None:
        out = args.out or args.episode_dir / "videos" / f"{tag}_r{args.png}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        renderer.snapshot(args.png).save(out)
    else:
        out = args.out or args.episode_dir / "videos" / f"{tag}.mp4"
        renderer.render_to(out)
    print(out)


if __name__ == "__main__":
    main()
