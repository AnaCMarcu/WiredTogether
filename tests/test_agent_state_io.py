"""agent_state_io: export/import round-trip against fake vector DBs.

The real stores are ChromaDB collections; here they are duck-typed fakes
(same _ensure_initialized/_collection surface), so the tests run without
chromadb/autogen/the ST embedding model — matching how the module itself
avoids heavy imports.
"""

import json
from types import SimpleNamespace

import pytest

from mindforge.agent_modules.agent_state_io import (
    export_agent_state,
    import_agent_state,
)


class FakeCollection:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []

    def add(self, documents, metadatas, ids):
        assert len(documents) == len(metadatas) == len(ids)
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def get(self):
        return {
            "ids": list(self.ids),
            "documents": list(self.documents),
            "metadatas": list(self.metadatas),
        }

    def count(self):
        return len(self.ids)


class FakeVectorDB:
    def __init__(self):
        self._collection = FakeCollection()

    def _ensure_initialized(self):
        pass


def _fake_agent(name="agent_0", skills=None, episodes=None):
    agent = SimpleNamespace(
        name=name,
        skill_manager=SimpleNamespace(
            skills=dict(skills or {}), vectordb=FakeVectorDB()
        ),
        episode_manager=SimpleNamespace(vectordb=FakeVectorDB()),
        auto_curriculum=SimpleNamespace(
            current_task=None,
            current_context="",
            completed_tasks=[],
            failed_tasks=[],
        ),
        _initialized=False,
        _episode_summary_dirty=False,
    )
    for k, (text, success) in enumerate(episodes or []):
        agent.episode_manager.vectordb._collection.add(
            documents=[text],
            metadatas=[{"episode": k, "success": success}],
            ids=[f"episode_{k}"],
        )
    return agent


SKILLS = {
    "dig": {"code": "Dig", "description": "Action: Dig. Context: dug wood."},
    "turn_right": {"code": "Turn Right", "description": "Action: Turn Right."},
}
EPISODES = [("Task: break anvil with agent_1.", 1), ("Task: explore.", 0)]


# ── export ───────────────────────────────────────────────────────────────────

def test_export_writes_schema(tmp_path):
    agent = _fake_agent(skills=SKILLS, episodes=EPISODES)
    agent.auto_curriculum.current_context = "ctx"
    agent.auto_curriculum.completed_tasks = ["align with agent_1"]
    agent.auto_curriculum.failed_tasks = ["failed thing"]

    export_agent_state([agent], str(tmp_path / "agent_state"),
                       run_id="run_x", episode=2)

    with open(tmp_path / "agent_state" / "agent_0.json") as f:
        state = json.load(f)
    assert state["agent_name"] == "agent_0"
    assert state["slot"] == 0
    assert state["run_id"] == "run_x"
    assert state["episode"] == 2
    assert state["skills"] == SKILLS
    assert [e["text"] for e in state["episodes"]] == [t for t, _ in EPISODES]
    assert [e["success"] for e in state["episodes"]] == [1, 0]
    assert state["curriculum"] == {
        "current_context": "ctx",
        "completed_tasks": ["align with agent_1"],
        "failed_tasks": ["failed thing"],
    }


def test_export_sorts_episodes_by_numeric_id(tmp_path):
    agent = _fake_agent()
    coll = agent.episode_manager.vectordb._collection
    # Insert out of order, with a 2-digit id to defeat lexicographic sorting.
    for k in (10, 2, 0):
        coll.add(documents=[f"ep {k}"],
                 metadatas=[{"episode": k, "success": 0}],
                 ids=[f"episode_{k}"])
    export_agent_state([agent], str(tmp_path))
    with open(tmp_path / "agent_0.json") as f:
        state = json.load(f)
    assert [e["id"] for e in state["episodes"]] == \
        ["episode_0", "episode_2", "episode_10"]


def test_export_never_raises_on_broken_agent(tmp_path):
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    # Must log-and-continue, not raise (runs inside checkpoint paths).
    export_agent_state([Exploding()], str(tmp_path))
    assert not (tmp_path / "agent_0.json").exists()


def test_export_handles_missing_managers(tmp_path):
    bare = SimpleNamespace(name="agent_0")
    export_agent_state([bare], str(tmp_path))
    with open(tmp_path / "agent_0.json") as f:
        state = json.load(f)
    assert state["skills"] == {} and state["episodes"] == []


# ── import ───────────────────────────────────────────────────────────────────

def _manifest(tmp_path, agents_dict):
    path = tmp_path / "merged_manifest.json"
    with open(path, "w") as f:
        json.dump({"condition": "transplant", "agents": agents_dict}, f)
    return str(path)


def test_import_round_trip(tmp_path):
    source = _fake_agent(skills=SKILLS, episodes=EPISODES)
    source.auto_curriculum.completed_tasks = ["align with agent_1"]
    export_agent_state([source], str(tmp_path / "exported"))
    with open(tmp_path / "exported" / "agent_0.json") as f:
        state = json.load(f)

    fresh = _fake_agent()
    import_agent_state([fresh], _manifest(tmp_path, {"0": state}))

    assert fresh.skill_manager.skills == SKILLS
    scoll = fresh.skill_manager.vectordb._collection
    assert set(scoll.ids) == set(SKILLS)
    # `code` must land in DB metadata (a later --resume rebuilds skills from
    # the DB and would otherwise lose it).
    meta_by_id = dict(zip(scoll.ids, scoll.metadatas))
    assert meta_by_id["dig"]["code"] == "Dig"

    ecoll = fresh.episode_manager.vectordb._collection
    assert ecoll.ids == ["episode_0", "episode_1"]
    assert ecoll.documents == [t for t, _ in EPISODES]
    # add_episode ids continue from count() → next id would be episode_2.
    assert ecoll.count() == 2

    assert fresh.auto_curriculum.completed_tasks == ["align with agent_1"]
    assert fresh.auto_curriculum.current_task is None
    # The _initialized guard is THE critical bit: without it the first
    # on_messages call wipes everything just imported.
    assert fresh._initialized is True
    assert fresh._episode_summary_dirty is True


def test_import_renumbers_episode_ids(tmp_path):
    state = {
        "skills": {},
        "episodes": [
            {"id": "episode_7", "text": "late", "success": 0},
            {"id": "episode_3", "text": "early", "success": 1},
        ],
        "curriculum": {},
    }
    fresh = _fake_agent()
    import_agent_state([fresh], _manifest(tmp_path, {"0": state}))
    ecoll = fresh.episode_manager.vectordb._collection
    assert ecoll.ids == ["episode_0", "episode_1"]
    assert ecoll.documents == ["early", "late"]
    assert [m["episode"] for m in ecoll.metadatas] == [0, 1]


def test_import_missing_slot_warns_but_continues(tmp_path):
    state = {"skills": SKILLS, "episodes": [], "curriculum": {}}
    a0, a1 = _fake_agent("agent_0"), _fake_agent("agent_1")
    import_agent_state([a0, a1], _manifest(tmp_path, {"0": state}))
    assert a0._initialized is True
    assert a1._initialized is False  # untouched — starts fresh


def test_import_no_slots_at_all_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no agent slots"):
        import_agent_state(
            [_fake_agent()], _manifest(tmp_path, {"5": {"skills": {}}})
        )
