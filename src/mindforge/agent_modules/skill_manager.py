import json
import logging
from agent_modules.llm_call import llm_call
from agent_modules.util import SkillResponse, create_model_client
import os
from pathlib import Path

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


with open("prompts/skill_description_prompt.txt", "r") as f:
    skill_description_prompt = f.read()
with open("prompts/skill_description_info.txt", "r") as f:
    skill_description_info = f.read()
with open("prompts/skill_construct_query.txt", "r") as f:
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
        self.vectordb = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="skill_vectordb_nvidia",
                persistence_path=os.path.join(
                    str(Path.home()), f"chromadb_autogen/skills_vectodb_{agent_name}"
                ),
                k=retrieval_top_k,  # Return top  k results
                score_threshold=0.4,  # Minimum similarity score
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(model_name="all-MiniLM-L6-v2",
                                                                                     device="cuda")
            ),
        )
        self.skill_prompt = (
            override_skill_prompt
            if override_skill_prompt
            else eval(f"f'''{skill_description_prompt}'''")
        )
        self._skill_info_prompt = override_skill_info_prompt or skill_description_info

    async def add_skill(
        self, action: str, cancellation_token, agent_thoughts: str = None
    ):

        # generate skill description and name
        skill_name, skill_description = await self.generate_skill_description(
            action, cancellation_token, agent_thoughts
        )

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
        # assert skills dict and vectordb are in sync
        assert self.vectordb._collection.count() == len(
            self.skills
        ), "vectordb is not synced with skills"
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
        return response["name"], response["description"]

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
        for s in data["ids"][0]:
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
        return response["description"]

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
