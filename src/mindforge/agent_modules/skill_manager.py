import json
import logging
import shutil
from agent_modules.llm_call import llm_call
from agent_modules.util import SkillResponse, create_model_client, safe_format, ST_MODEL_NAME
import os
from pathlib import Path


def _chromadb_base_dir() -> str:
    """Return a fast-local directory for ChromaDB persistence.

    On HPC, $HOME is NFS and causes SQLite I/O errors.
    Prefer $SCRATCH or a sibling directory of the project.
    """
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "chromadb_autogen")
    # Fallback: use a directory next to the project
    return os.path.join(str(Path.home()), "chromadb_autogen")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    ChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig
)

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage, SystemMessage


_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

with open(os.path.join(_PROMPT_DIR, "skill_description_prompt.txt"), "r") as f:
    skill_description_prompt = f.read()
with open(os.path.join(_PROMPT_DIR, "skill_description_info.txt"), "r") as f:
    skill_description_info = f.read()
with open(os.path.join(_PROMPT_DIR, "skill_construct_query.txt"), "r") as f:
    skill_construct_query = f.read()


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
        self.skill_model_client = (
            skill_model_client
            if skill_model_client
            else create_model_client(resonse_format=SkillResponse)
        )
        self.skills = {}
        # self.vectordb = ChromaDBVectorMemory(
        #     config=PersistentChromaDBVectorMemoryConfig(
        #         collection_name="skills_vectordb_nvidia",
        #         persistence_path=os.path.join(
        #             str(Path.home()),
        #             f"chromadb_autogen/skills_vectodb_{agent_name}",
        #         ),
        #         k=retrieval_top_k,  # Return top  k results
        #         score_threshold=0.4,  # Minimum similarity score
        #         embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
        #             model_name="all-MiniLM-L6-v2", device="cuda"
        #         ),
        #     ),
        # )
        db_path = os.path.join(_chromadb_base_dir(), f"skills_vectodb_{agent_name}")

        # On reset, delete the directory entirely to avoid corrupted SQLite files
        if reset and os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)

        self.vectordb = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="skill_vectordb_nvidia",
                persistence_path=db_path,
                k=retrieval_top_k,  # Return top  k results
                score_threshold=0.4,  # Minimum similarity score
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(model_name=ST_MODEL_NAME,
                                                                                     device="cuda")
            ),
        )
        if not reset:
            # Resume: rebuild skills dict from what's persisted
            try:
                self.vectordb._ensure_initialized()
                if self.vectordb._collection is not None:
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

        self.skill_prompt = (
            override_skill_prompt
            if override_skill_prompt
            else safe_format(skill_description_prompt)
        )
        self._skill_info_prompt = override_skill_info_prompt or skill_description_info

    async def add_skill(
        self, action: str, cancellation_token, agent_thoughts: str = None
    ):

        # generate skill description and name
        skill_name, skill_description = await self.generate_skill_description(
            action, cancellation_token, agent_thoughts
        )

        # Skip saving when description generation failed
        if skill_name == "unknown" or not skill_description:
            logging.warning(
                f"Skipping skill save: name='{skill_name}', description='{skill_description}'"
            )
            return skill_name, skill_description, False

        # does skill already exist?
        already_exists = False
        if skill_name in self.skills:
            print(f"Skill '{skill_name}' already exists.")
            self.vectordb._collection.delete(ids=[skill_name])
            already_exists = True

        self.vectordb._ensure_initialized()
        if self.vectordb._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB")

        # add skill to vectordb
        self.vectordb._collection.add(
            documents=[skill_description],
            metadatas=[{"name": skill_name}],
            ids=[skill_name],
        )
        # add skill to skills dict
        self.skills[skill_name] = {
            "code": action,
            "description": skill_description,
        }
        # Verify sync (warn instead of crash — persistence edge cases are possible)
        db_count = self.vectordb._collection.count()
        if db_count != len(self.skills):
            logging.warning(
                "SkillManager: vectordb count (%d) != skills dict (%d) after add_skill('%s')",
                db_count, len(self.skills), skill_name,
            )
        return skill_name, skill_description, already_exists

    async def generate_skill_description(
        self,
        action: str,
        cancellation_token,
        agent_thoughts: str = None,
    ):
        def parse_check(content):
            assert "name" in content and "description" in content
            return content

        response = await llm_call(
            self.skill_model_client,
            system_prompt=self.skill_prompt,
            user_prompt=self._skill_info_prompt,
            cancellation_token=cancellation_token,
            parse_check=parse_check,
            log_prefix="SkillManager generate_skill_description: ",
            agent_thoughts=agent_thoughts,
            action=action,
        )
        return response.get("name", "unknown"), response.get("description", "")

    async def get_skills(self, query: str):
        if self.vectordb._collection.count() == 0:
            return None
        k = min(self.vectordb._collection.count(), self.vectordb._config.k)
        # query vector db
        logging.info(f"Querying vector db for skills with query: {query}")
        data = self.vectordb._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        logging.info(f"Skills found: {data}")
        # format skills
        skill_memory = []
        ids = data["ids"][0] if data.get("ids") and data["ids"] else []
        for s in ids:
            if s in self.skills:
                skill_memory.append(self.skills[s])

        return skill_memory

    # construct a query for fetching relevant skills based on the task and frame
    async def construct_query(self, task, frame, cancellation_token):

        user_message_prompt = skill_construct_query

        def parse_check(content):
            assert "description" in content
            return content

        response = await llm_call(
            self.skill_model_client,
            user_prompt=user_message_prompt,
            cancellation_token=cancellation_token,
            parse_check=parse_check,
            log_prefix="SkillManager construct_query: ",
            task=task,
            frame=frame,
        )
        return response.get("description", "")

    async def clear_data(self):
        await self.vectordb.clear()
        return True


# await self.vectordb.add(
#             MemoryContent(
#                 content=content,
#                 mime_type=MemoryMimeType.TEXT,
#                 metadata={"category": catergory, "type": type},
#             )
#         )
