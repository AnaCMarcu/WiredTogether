"""Episode video output."""

from __future__ import annotations

import logging

import numpy as np


def _frames_to_mp4(pil_frames: list, mp4_path: str, fps: int = 2) -> None:
    """Write PIL frames directly to MP4 using imageio[ffmpeg] (bundled binary, no system ffmpeg)."""
    try:
        import imageio
        with imageio.get_writer(mp4_path, fps=fps, macro_block_size=1) as writer:
            for frame in pil_frames:
                writer.append_data(np.array(frame))
        print(f"  Saved MP4: {mp4_path}")
    except Exception as exc:
        logging.warning("MP4 save failed (%s): %s", mp4_path, exc)
