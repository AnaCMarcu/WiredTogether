"""CLI end to end: real run() wiring, fake game server and fake LLM."""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import social_stubs  # noqa: F401
from agentworld import cli
from agentworld.agent import ModelClients
from fake_kaetram import FakeGameTools, FakeWorld
from test_aw_agent import FakeLLM

FIX = Path(__file__).parent / "fixtures" / "agentworld" / "v1.3_benchmark"


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    root = tmp_path / "agentworld"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "game_tools.py").write_text("# placeholder\n")
    suite = root / "data_v0.1_multi" / "v1.3_benchmark"
    shutil.copytree(FIX, suite)
    world = FakeWorld()
    vendor = SimpleNamespace(root=root, master_password="pw",
                             make_tools=lambda url: FakeGameTools(world, url))
    monkeypatch.setattr("agentworld.vendor.load_game_tools", lambda root=None: vendor)
    llm = FakeLLM()
    monkeypatch.setattr(ModelClients, "from_env",
                        classmethod(lambda cls: ModelClients(llm, llm, llm, llm)))
    return root, llm


def _args(root, out, *extra):
    return cli.build_parser().parse_args([
        "--agentworld-root", str(root), "--tasks", "task_02_arrow_production",
        "--server-url", "http://fake", "--embedder", "hash", "--rounds", "12",
        "--setup-sleep-scale", "0",
        "--out", str(out), *extra])


@pytest.mark.slow
def test_cli_runs_two_episodes_with_videos(fake_root, tmp_path):
    root, llm = fake_root
    out = tmp_path / "run"
    args = _args(root, out, "--replicas", "2", "--episodes", "2", "--arm", "hebbian",
                 "--video", "both")
    summaries = asyncio.run(cli.run(args))
    assert [s["episode"] for s in summaries] == [1, 2]
    assert all(t["success"] == 1 for s in summaries for t in s["teams"].values())
    for ep in (1, 2):
        d = out / f"episode_{ep}"
        assert (d / "videos" / "team_t00.mp4").stat().st_size > 1000
        assert (d / "videos" / "world.mp4").is_file()
        W = np.load(d / "hebbian_W.npy")
        assert W.shape == (6, 6)
        assert json.loads((d / "summary.json").read_text())["tokens"]["per_agent"]
    assert json.loads((out / "config.json").read_text())["arm"] == "hebbian"


@pytest.mark.parametrize("arm,expect_bonds", [("base", False), ("shuffled", True),
                                              ("prompt_only", True), ("oracle", True)])
def test_arms_differ_only_in_reading_and_prompt(fake_root, tmp_path, arm, expect_bonds):
    root, llm = fake_root
    args = _args(root, tmp_path / arm, "--replicas", "2", "--arm", arm, "--video", "none")
    summaries = asyncio.run(cli.run(args))
    assert all(t["success"] == 1 for t in summaries[0]["teams"].values())
    prompts = [u for _, u in llm.prompts if "Decide now" in u]
    has_bonds = any("Your strongest bonds" in u for u in prompts)
    assert has_bonds == expect_bonds
    # Every arm still records W (the base arm as a passive observer).
    assert np.load(tmp_path / arm / "episode_1" / "hebbian_W.npy").shape == (6, 6)
