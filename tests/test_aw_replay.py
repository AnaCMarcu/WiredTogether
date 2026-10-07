"""Replay renderer: frames from a scheduler run and synthetic large worlds."""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agentworld.replay.mapdata import PlainMap, load_map
from agentworld.replay.render import (DM_COLOR, ReplayRenderer, RenderOptions, clean,
                                      load_states, short_name)
from test_aw_scheduler import _build


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    out = tmp_path_factory.mktemp("aw_replay")
    sched, world, fake, ex = _build(out)
    asyncio.run(sched.run_episode(1))
    ex.shutdown()
    return out / "episode_1"


def _rgb(hex_):
    h = hex_.lstrip("#")
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def _has_color(img: Image.Image, rgb, box=None, tol=12) -> bool:
    a = np.asarray(img.crop(box) if box else img).astype(int)
    return bool((np.abs(a - np.array(rgb)).sum(axis=2) <= tol).any())


def test_short_names_and_clean():
    assert short_name("t02_fletcher_agent", False) == "fletcher"
    assert short_name("t03_t02_fletcher_agent", True) == "t03·fletcher"
    assert clean("done ✅ ok ✓") == "done  ok ✓"


def test_load_map_falls_back_to_plain(tmp_path):
    assert isinstance(load_map(None), PlainMap)
    assert isinstance(load_map(tmp_path), PlainMap)


def test_frames_have_constant_size_and_hold_on_messages(episode):
    states = load_states(episode)
    opts = RenderOptions(frames_per_round=3, hold_frames=2)
    frames = list(ReplayRenderer(states, PlainMap(), opts).frames())
    with_msgs = sum(1 for s in states if s["messages"])
    assert len(frames) == 3 * len(states) + 2 * with_msgs
    assert {f.size for f in frames} == {(1280, 720)}


def test_dm_bubble_drawn_in_dm_colour(episode):
    states = load_states(episode)
    r = ReplayRenderer(states, PlainMap(), RenderOptions())
    dm_round = next(s["round"] for s in states if any(m["kind"] == "dm" for m in s["messages"]))
    quiet_round = next((s["round"] for s in states if not s["messages"]), None)
    img = r.snapshot(dm_round)
    map_box = (0, 0, 1280 - 420, 720)
    assert _has_color(img, _rgb(DM_COLOR), map_box)
    if quiet_round is not None:
        assert not _has_color(r.snapshot(quiet_round), _rgb(DM_COLOR), map_box, tol=6)


def test_reader_view_lists_only_what_the_agent_read(episode):
    states = load_states(episode)
    fletcher = 2
    s = next(s for s in states if s["reads"].get(str(fletcher), {}).get("dms"))
    r = ReplayRenderer(states, PlainMap(), RenderOptions(camera="agent", agent=fletcher,
                                                         reader_view=True))
    assert r.snapshot(s["round"]).size == (1280, 720)
    read_ids = s["reads"][str(fletcher)]["dms"]
    assert all(r.msgs[m]["receiver"] == fletcher for m in read_ids)


def _synthetic_world(n=100, rounds=3):
    rng = np.random.default_rng(0)
    states = []
    for rnd in range(1, rounds + 1):
        agents = [{"i": i, "name": f"t{i // 4:02d}_role{i % 4}_agent", "team": f"t{i // 4:02d}",
                   "x": float(200 + (i % 10) * 6 + rnd), "y": float(150 + (i // 10) * 6),
                   "hp": 50, "maxhp": 69, "inv": {}, "busy": False, "reward": 0.0,
                   "reward_parts": {}, "thoughts": ""} for i in range(n)]
        msgs = [{"round": rnd, "sender": int(s), "text": "need 3 logs, who has them?",
                 "kind": "dm", "receiver": int((s + 1) % n), "post_kind": "none",
                 "msg_id": rnd * 1000 + k} for k, s in enumerate(rng.choice(n, 8, replace=False))]
        states.append({"round": rnd, "t_wall": 0.0, "agents": agents, "actions": [],
                       "events": [], "messages": msgs, "reads": {}, "mobs": [],
                       "teams": {f"t{k:02d}": {"progress": 0.1 * k % 1, "solved": 0}
                                 for k in range(n // 4)},
                       "bonds_top": {str(i): [[(i + 1) % n, 0.5]] for i in range(n)}})
    return states


def test_world_view_at_n100_is_fast(tmp_path):
    states = _synthetic_world()
    r = ReplayRenderer(states, PlainMap(), RenderOptions(camera="world", frames_per_round=4,
                                                         hold_frames=0))
    t0 = time.time()
    frames = list(r.frames())
    assert len(frames) == 12
    assert time.time() - t0 < 20


def test_render_to_mp4(episode, tmp_path):
    states = load_states(episode)[:3]
    out = ReplayRenderer(states, PlainMap(), RenderOptions(frames_per_round=2, hold_frames=1)
                         ).render_to(tmp_path / "v.mp4")
    assert out.is_file() and out.stat().st_size > 1000


def test_cli_writes_png(episode, tmp_path):
    from agentworld.replay.__main__ import main
    main([str(episode), "--team", "t00", "--png", "1", "--out", str(tmp_path / "f.png")])
    assert Image.open(tmp_path / "f.png").size == (1280, 720)
