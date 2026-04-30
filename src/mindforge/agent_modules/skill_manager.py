import logging
import os
import shutil
from pathlib import Path

from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

from agent_modules.llm_call import llm_call
from agent_modules.util import (
    SkillResponse,
    create_model_client,
    safe_format,
    ST_MODEL_NAME,
)


# ─── Module helpers ────────────────────────────────────────────────────

def _chromadb_base_dir() -> str:
    """Return a fast-local directory for ChromaDB persistence.

    SQLite (used by ChromaDB) requires POSIX file locks that Lustre/GPFS
    (DelftBlue /scratch) does not reliably support → SQLITE_IOERR (code 5898).
    Use /tmp on the compute node (local SSD) when running under SLURM.
    Each job gets an isolated subdirectory via $SLURM_JOB_ID.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        base = f"/tmp/mindforge_{job_id}"
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "chromadb_autogen")
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "chromadb_autogen")
    return os.path.join(str(Path.home()), "chromadb_autogen")


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r") as f:
        return f.read()


def _require_keys(*keys):
    """Build an llm_call parse_check that asserts all keys are present."""
    def check(content):
        for k in keys:
            assert k in content
        return content
    return check


# ─── Prompt loading ────────────────────────────────────────────────────

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

skill_description_prompt = _load_prompt("skill_description_prompt.txt")
skill_description_info   = _load_prompt("skill_description_info.txt")
skill_construct_query    = _load_prompt("skill_construct_query.txt")


# ─── Main class ────────────────────────────────────────────────────────

class SkillManager:
    def __init__(
        self,
        retrieval_top_k=5,
        skill_model_client=None,
        override_skill_prompt=None,
        override_skill_info_prompt=None,
        reset=True,
        agent_name="agent_0",
    ):
        self.skill_model_client = skill_model_client or create_model_client(
            response_format=SkillResponse
        )
        self.skills = {}

        self.skill_prompt = (
            override_skill_prompt or safe_format(skill_description_prompt)
        )
        self._skill_info_prompt = override_skill_info_prompt or skill_description_info

        self.vectordb = self._init_vectordb(agent_name, retrieval_top_k, reset)
        if not reset:
            self._restore_skills_from_db()

    @staticmethod
    def _init_vectordb(agent_name: str, top_k: int, reset: bool) -> ChromaDBVectorMemory:
        """Create (or reset) a per-agent ChromaDB collection for skill storage."""
        db_path = os.path.join(_chromadb_base_dir(), f"skills_vectodb_{agent_name}")
        if reset and os.path.exists(db_path):
            # Wipe the directory entirely — avoids corrupted SQLite files from prior runs.
            shutil.rmtree(db_path, ignore_errors=True)
        return ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="skill_vectordb_nvidia",
                persistence_path=db_path,
                k=top_k,
                score_threshold=0.4,
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                    model_name=ST_MODEL_NAME, device="cuda"
                ),
            ),
        )

    def _restore_skills_from_db(self):
        """Resume: rebuild self.skills from what's already persisted in the vector DB."""
        try:
            self.vectordb._ensure_initialized()
            if self.vectordb._collection is None:
                return
            existing = self.vectordb._collection.get()
            for id_, doc, meta in zip(
                existing["ids"], existing["documents"], existing["metadatas"]
            ):
                self.skills[id_] = {
                    "code": meta.get("code", ""),
                    "description": doc,
                }
        except Exception as e:
            logging.warning("SkillManager: failed to load persisted skills: %s", e)

    # ─── Public API ────────────────────────────────────────────────

    async def add_skill(
        self, action: str, cancellation_token, agent_thoughts: str = None,
    ):
        """Persist a discrete-action skill (templated, no LLM call).

        For free-form code skills use generate_skill_description() instead.
        Returns (skill_name, skill_description, already_existed).
        """
        skill_name = self._action_to_skill_name(action)
        skill_description = self._format_skill_description(action, agent_thoughts)

        if skill_name == "unknown":
            logging.warning("Skipping skill save: empty/invalid action.")
            return skill_name, skill_description, False

        already_exists = self._delete_existing(skill_name)
        self._persist_skill(skill_name, skill_description, action)
        self._verify_db_sync(skill_name)
        return skill_name, skill_description, already_exists

    async def generate_skill_description(
        self, action: str, cancellation_token, agent_thoughts: str = None,
    ):
        response = await llm_call(
            self.skill_model_client,
            system_prompt=self.skill_prompt,
            user_prompt=self._skill_info_prompt,
            cancellation_token=cancellation_token,
            parse_check=_require_keys("name", "description"),
            log_prefix="SkillManager generate_skill_description: ",
            agent_thoughts=agent_thoughts,
            action=action,
        )
        return response.get("name", "unknown"), response.get("description", "")

    async def get_skills(self, query: str):
        if self.vectordb._collection.count() == 0:
            return None
        k = min(self.vectordb._collection.count(), self.vectordb._config.k)
        logging.info("Querying vector db for skills with query: %s", query)
        data = self.vectordb._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        logging.info("Skills found: %s", data)
        ids = data["ids"][0] if data.get("ids") and data["ids"] else []
        return [self.skills[s] for s in ids if s in self.skills]

    async def construct_query(self, task, frame, cancellation_token):
        """Build a description query for skill retrieval, conditioned on task+frame."""
        response = await llm_call(
            self.skill_model_client,
            user_prompt=skill_construct_query,
            cancellation_token=cancellation_token,
            parse_check=_require_keys("description"),
            log_prefix="SkillManager construct_query: ",
            task=task,
            frame=frame,
        )
        return response.get("description", "")

    async def clear_data(self):
        await self.vectordb.clear()
        return True

    # ─── add_skill internals ───────────────────────────────────────

    @staticmethod
    def _action_to_skill_name(action: str) -> str:
        if not action:
            return "unknown"
        return action.lower().replace(" ", "_")

    @staticmethod
    def _format_skill_description(action: str, agent_thoughts: str | None) -> str:
        return f"Action: {action}. Context: {agent_thoughts or 'No additional context.'}"

    def _delete_existing(self, skill_name: str) -> bool:
        if skill_name not in self.skills:
            return False
        print(f"Skill '{skill_name}' already exists.")
        self.vectordb._collection.delete(ids=[skill_name])
        return True

    def _persist_skill(self, skill_name: str, description: str, action: str):
        self.vectordb._ensure_initialized()
        if self.vectordb._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB")
        self.vectordb._collection.add(
            documents=[description],
            metadatas=[{"name": skill_name}],
            ids=[skill_name],
        )
        self.skills[skill_name] = {
            "code": action,
            "description": description,
        }

    def _verify_db_sync(self, skill_name: str):
        db_count = self.vectordb._collection.count()
        if db_count != len(self.skills):
            logging.warning(
                "SkillManager: vectordb count (%d) != skills dict (%d) after add_skill('%s')",
                db_count, len(self.skills), skill_name,
            )
