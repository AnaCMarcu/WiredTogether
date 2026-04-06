import logging
import os
import shutil
from pathlib import Path
from agent_modules.llm_call import llm_call
from agent_modules.util import EpisodeResponse, create_model_client, safe_format, ST_MODEL_NAME
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    ChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
from autogen_core.models import UserMessage, SystemMessage

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

with open(os.path.join(_PROMPT_DIR, "episode_summary_prompt.txt"), "r") as f:
    episode_summary_prompt = f.read()


class EpisodicMemoryManager:
    def __init__(
        self,
        retrieval_top_k=5,
        episode_model_client=None,
        override_episode_prompt=None,
        agent_name="agent_0",
    ):
        self.episode_model_client = (
            episode_model_client
            if episode_model_client
            else create_model_client(response_format=EpisodeResponse)
        )
        # self.vectordb = ChromaDBVectorMemory(
        #     config=PersistentChromaDBVectorMemoryConfig(
        #         collection_name="episodes_vectordb_nvidia",
        #         persistence_path=os.path.join(
        #             str(Path.home()),
        #             f"chromadb_autogen/episodes_vectodb_{agent_name}",
        #         ),
        #         k=retrieval_top_k,  # Return top  k results
        #         score_threshold=0.4,  # Minimum similarity score
        #         embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
        #             model_name="all-MiniLM-L6-v2", device="cuda"
        #         ),
        #     ),
        # )
        from agent_modules.skill_manager import _chromadb_base_dir
        db_path = os.path.join(_chromadb_base_dir(), f"episodes_vectodb_{agent_name}")

        # Wipe stale/corrupted SQLite files from previous runs
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)

        self.vectordb = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="episode_vectordb_nvidia",
                persistence_path=db_path,
                k=retrieval_top_k,  # Return top  k results
                score_threshold=0.4,  # Minimum similarity score
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(model_name=ST_MODEL_NAME, device="cuda")
            )
        )
        self.episode_prompt = (
            override_episode_prompt
            if override_episode_prompt
            else safe_format(episode_summary_prompt)
        )

    # add new episode
    def add_episode(
        self, task, last_action, last_thoughts, critique, task_beliefs="", perception_beliefs="",success=False
    ):
        # construct episode
        episode = f"""
        Task: {task}
        Task Beliefs: {task_beliefs}
        Thoughts: {last_thoughts}
        Last Action: {last_action}
        Last action was a Success: {success}
        Critique: {critique}
        Perception beliefs: {perception_beliefs}
        """
        # save episode to vector database
        self.vectordb._ensure_initialized()
        if self.vectordb._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB")
        self.vectordb._collection.add(
            documents=[episode],
            metadatas=[{"episode": self.vectordb._collection.count()}],
            ids=[f"episode_{self.vectordb._collection.count()}"],
        )
        return episode




    # generate summary of the episode
    async def generate_episode_summary(
        self, episodes: list[str], cancellation_token=None
    ):
        if not episodes or episodes == []:
            return "There are no past episodes."
        combined_episodes = "\n\n".join(episodes)

        def parse_check(content):
            assert "summary" in content
            return content

        response = await llm_call(
            self.episode_model_client,
            system_prompt=self.episode_prompt,
            user_prompt=combined_episodes,
            cancellation_token=cancellation_token,
            parse_check=parse_check,
            log_prefix="EpisodicMemoryManager generate_episode_summary: ",
        )
        return response

    # retrieve episodes
    def retrieve_episodes(self, query: str):
        self.vectordb._ensure_initialized()
        if self.vectordb._collection.count() == 0:
            return None
        k = min(self.vectordb._collection.count(), self.vectordb._config.k)
        # query vector db
        logging.info(f"Querying vector db for episodes with query: {query}")
        data = self.vectordb._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        logging.info(f"episodes found: {data}")

        # format episodes
        formatted_episodes = data["documents"][0] if data.get("documents") and data["documents"] else []

        return formatted_episodes

    async def clear_data(self):
        await self.vectordb.clear()
        return True
