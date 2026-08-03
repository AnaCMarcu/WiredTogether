"""JSON export/import of per-agent cognitive state.

The per-agent vector DBs (skills, episodic memory) live on job-local storage
under SLURM (/tmp/mindforge_$SLURM_JOB_ID, see skill_manager._chromadb_base_dir)
and the episodic DB is wiped on every construction — so none of this state
survives a job on its own. This module gives it a durable, portable form:

  export_agent_state()  — dump skills / episodic docs / curriculum lists to
                          <out_dir>/agent_{i}.json. Pure output, never raises;
                          hooked into save_checkpoint() and the end-of-run
                          path of multi_agent_craftium.py.
  import_agent_state()  — load a merged manifest (mindforge/tools/
                          merge_pair_runs.py) into freshly-constructed agents,
                          re-inserting docs into the fresh ChromaDBs
                          (embeddings are recomputed at add time).

Everything is duck-typed against CustomAgent's attribute surface
(skill_manager / episode_manager / auto_curriculum) so this module imports
without ChromaDB, torch, or the ST embedding model — pytest can exercise it
with fakes.
"""

import json
import logging
import os


def _episode_sort_key(entry):
    """Order episode records by the numeric suffix of their id."""
    id_ = str(entry.get("id", ""))
    try:
        return (0, int(id_.rsplit("_", 1)[-1]))
    except ValueError:
        return (1, 0)


def _collect_agent_state(agent) -> dict:
    """Snapshot one agent's transplantable cognitive state as a JSON dict."""
    state = {
        "agent_name": getattr(agent, "name", None),
        "skills": {},
        "episodes": [],
        "curriculum": {},
    }

    sm = getattr(agent, "skill_manager", None)
    if sm is not None:
        # The in-memory dict is authoritative (the DB drops `code` from its
        # metadata — see _persist_skill) and needs no DB round-trip.
        for name, payload in (getattr(sm, "skills", None) or {}).items():
            state["skills"][name] = {
                "code": payload.get("code", ""),
                "description": payload.get("description", ""),
            }

    em = getattr(agent, "episode_manager", None)
    vectordb = getattr(em, "vectordb", None)
    if vectordb is not None:
        vectordb._ensure_initialized()
        coll = vectordb._collection
        if coll is not None and coll.count() > 0:
            data = coll.get()
            for id_, doc, meta in zip(
                data["ids"], data["documents"], data["metadatas"]
            ):
                meta = meta or {}
                state["episodes"].append({
                    "id": id_,
                    "text": doc,
                    "episode": meta.get("episode"),
                    "success": int(meta.get("success", 0)),
                })
            state["episodes"].sort(key=_episode_sort_key)

    ac = getattr(agent, "auto_curriculum", None)
    if ac is not None:
        state["curriculum"] = {
            "current_context": getattr(ac, "current_context", "") or "",
            "completed_tasks": list(getattr(ac, "completed_tasks", None) or []),
            "failed_tasks": list(getattr(ac, "failed_tasks", None) or []),
        }

    return state


def export_agent_state(agents, out_dir, run_id=None, episode=None) -> None:
    """Write <out_dir>/agent_{i}.json for every agent.

    Best-effort by design: export runs inside checkpoint/end-of-run paths and
    must never be the reason a training run dies. Failures are logged per
    agent and swallowed.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:
        logging.warning("agent_state_io: cannot create %s: %s", out_dir, exc)
        return

    for i, agent in enumerate(agents):
        try:
            state = _collect_agent_state(agent)
            state["slot"] = i
            if run_id is not None:
                state["run_id"] = run_id
            if episode is not None:
                state["episode"] = episode
            path = os.path.join(out_dir, f"agent_{i}.json")
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, path)
        except Exception as exc:
            logging.warning(
                "agent_state_io: export failed for agent_%d: %s", i, exc
            )


def import_agent_state(agents, manifest_path) -> None:
    """Load a merged agent-state manifest into freshly-constructed agents.

    Must run AFTER agent construction (which wipes/recreates the per-agent
    DBs) and BEFORE the first on_messages call. Raises on structural problems
    — a broken transplant should kill the run at startup, not 24h in.

    For each slot present in the manifest:
      * skills   → re-added to the skill ChromaDB (metadata carries `code` so
                   a later --resume keeps it) and to skill_manager.skills;
      * episodes → re-added with sequential ids episode_0..n-1, so
                   add_episode()'s count()-based ids continue at episode_n;
      * curriculum lists restored; current_task left None so a fresh,
        chamber-appropriate task is proposed at step 0;
      * agent._initialized set True — otherwise the first on_messages call
        (current_task is None and not _initialized) calls clear_data() and
        silently destroys everything just imported;
      * agent._episode_summary_dirty set True so the first prompt's episode
        summary is regenerated from the imported docs.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    agent_states = manifest.get("agents", manifest)

    imported_any = False
    for i, agent in enumerate(agents):
        st = agent_states.get(str(i))
        if st is None:
            logging.warning(
                "agent_state_io: manifest has no entry for slot %d — "
                "agent_%d starts fresh", i, i
            )
            continue

        skills = st.get("skills", {}) or {}
        sm = agent.skill_manager
        sm.vectordb._ensure_initialized()
        coll = sm.vectordb._collection
        if coll is None:
            raise RuntimeError(
                f"agent_{i}: skill ChromaDB failed to initialize — cannot "
                "import transplanted skills"
            )
        if skills:
            names = list(skills.keys())
            coll.add(
                documents=[skills[n].get("description", "") for n in names],
                metadatas=[
                    {"name": n, "code": skills[n].get("code", "")}
                    for n in names
                ],
                ids=names,
            )
            for n in names:
                sm.skills[n] = {
                    "code": skills[n].get("code", ""),
                    "description": skills[n].get("description", ""),
                }

        episodes = sorted(st.get("episodes", []) or [], key=_episode_sort_key)
        em = agent.episode_manager
        em.vectordb._ensure_initialized()
        ecoll = em.vectordb._collection
        if ecoll is None:
            raise RuntimeError(
                f"agent_{i}: episodic ChromaDB failed to initialize — cannot "
                "import transplanted episodes"
            )
        if episodes:
            ecoll.add(
                documents=[ep.get("text", "") for ep in episodes],
                metadatas=[
                    {"episode": k, "success": int(ep.get("success", 0))}
                    for k, ep in enumerate(episodes)
                ],
                ids=[f"episode_{k}" for k in range(len(episodes))],
            )

        cur = st.get("curriculum", {}) or {}
        ac = agent.auto_curriculum
        ac.current_context = cur.get("current_context", "") or ""
        ac.completed_tasks = list(cur.get("completed_tasks", []) or [])
        ac.failed_tasks = list(cur.get("failed_tasks", []) or [])

        agent._initialized = True
        agent._episode_summary_dirty = True
        imported_any = True
        print(
            f"[TRANSPLANT] agent_{i}: {len(skills)} skills, "
            f"{len(episodes)} episodes, "
            f"{len(ac.completed_tasks)} completed tasks "
            f"(source: {st.get('source', 'n/a')})"
        )

    if not imported_any:
        raise RuntimeError(
            f"--agent-state-init: no agent slots imported from "
            f"{manifest_path} (manifest keys: {sorted(agent_states)[:10]})"
        )
