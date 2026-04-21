import json
import logging
import re
import shutil
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, UserMessage, SystemMessage
from autogen_agentchat.messages import TextMessage

# Keywords that indicate a task maps to a real game action.
# Tasks without any of these are replaced with a role-biased default.
_ACHIEVABLE_KEYWORDS = frozenset({
    "dig", "mine", "break", "chop", "punch",        # resource gathering
    "kill", "attack", "fight", "hit", "hunt",        # combat
    "find", "move", "walk", "go", "run", "approach", # navigation
    "place", "build", "craft",                       # construction
    "collect", "gather", "get", "pick",              # generic gathering
})

# Words that indicate a task is stuck on a map fixture or UI artifact.
# Any task containing one of these is replaced with the role default even if
# it also contains an achievable keyword (e.g. "move away from the golden block").
_TASK_BLOCKLIST = frozenset({
    "golden",   # spawn platform block — agents fixate on it
    "spawn",    # any reference to the spawn point itself
    "platform", # explicit platform references
    "hud",      # UI confusion
    "interface",
    "inventory screen",
    "leave the golden",
    "clear the golden",
    "collect the golden",
    "break the golden",
})

_ROLE_DEFAULT_TASKS = {
    "agent": "Explore by moving forward and turning to survey the room",
}


def _is_achievable_task(task: str) -> bool:
    """Check if a task describes something the agent can actually do.

    Returns False if the task references a map fixture or UI element that agents
    cannot meaningfully interact with (blocklist), even if it contains an action
    keyword — e.g. "move away from the golden block" passes the keyword check but
    describes a non-game-progress loop.
    """
    task_lower = task.lower()
    if any(kw in task_lower for kw in _TASK_BLOCKLIST):
        return False
    return any(kw in task_lower for kw in _ACHIEVABLE_KEYWORDS)
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    ChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig
)
import os
from pathlib import Path

from agent_modules.llm_call import llm_call
from agent_modules.util import (
    CurriculumAnswerResponse,
    CurriculumQuestionResponse,
    CurruliculumResponse,
    create_model_client,
    safe_format,
    ST_MODEL_NAME,
)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# load in prompts
with open(os.path.join(_PROMPT_DIR, "curriculum_prompt.txt"), "r") as f:
    curriculum_prompt = f.read()
with open(os.path.join(_PROMPT_DIR, "curriculum_info.txt"), "r") as f:
    curriculum_info = f.read()
with open(os.path.join(_PROMPT_DIR, "curriculum_questions.txt"), "r") as f:
    curriculum_questions = f.read()
with open(os.path.join(_PROMPT_DIR, "curriculum_answer.txt"), "r") as f:
    curriculum_answer = f.read()


class AutoCurriculum:
    def __init__(
        self,
        initial_task: TextMessage = None,
        initial_context="",
        override_curriculum_prompt=None,
        override_questions_prompt=None,
        task_model_client=None,
        question_model_client=None,
        answer_model_client=None,
        retrieval_top_k=5,
        agent_name="agent_0",
    ):
        self.current_task = initial_task
        self.current_context = initial_context
        self.curriculum_prompt = (
            override_curriculum_prompt
            if override_curriculum_prompt
            else safe_format(curriculum_prompt)
        )
        self._questions_prompt = override_questions_prompt or curriculum_questions
        self.task_model_client = (
            task_model_client
            if task_model_client
            else create_model_client(response_format=CurruliculumResponse)
        )
        self.question_model_client = (
            question_model_client
            if question_model_client
            else create_model_client(response_format=CurriculumQuestionResponse)
        )
        self.answer_model_client = (
            answer_model_client
            if answer_model_client
            else create_model_client(response_format=CurriculumAnswerResponse)
        )
        self.completed_tasks = []
        self.failed_tasks = []
        self._role = "agent"
        # self.vectordb = ChromaDBVectorMemory(
        #     config=PersistentChromaDBVectorMemoryConfig(
        #         collection_name="autocurriculum_vectordb_nvidia",
        #         persistence_path=os.path.join(
        #             str(Path.home()),
        #             f"chromadb_autogen/autocurriculum_vectodb_{agent_name}",
        #         ),
        #         k=retrieval_top_k,  # Return top  k results
        #         score_threshold=0.4,  # Minimum similarity score
        #         embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
        #             model_name="all-MiniLM-L6-v2", device="cuda"
        #         ),
        #     ),
        # )
        from agent_modules.skill_manager import _chromadb_base_dir
        db_path = os.path.join(_chromadb_base_dir(), f"autocurriculum_vectodb_{agent_name}")

        # Wipe stale/corrupted SQLite files from previous runs
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)

        self.vectordb = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="autocurriculum_vectordb_nvidia",
                persistence_path=db_path,
                k=retrieval_top_k,  # Return top  k results
                score_threshold=0.4,  # Minimum similarity score
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(model_name=ST_MODEL_NAME,
                                                                                     device="cuda")
            ),
        )

    async def get_new_task(
        self,
        frame,
        last_action,
        last_thoughts,
        cancellation_token: CancellationToken,
        success=None,
        critique=None,
        communications=None,
        do_question_answers=True,
        picked_object=None,
        position_text=None,
        player_status_text=None,
    ):
        # gather information for the curriculum prompt
        # used in eval, do not remove !!!
        completed_tasks = self.get_completed_tasks()
        failed_tasks = self.get_failed_tasks()
        if do_question_answers:
            question_answers = await self.get_question_answers(
                frame, cancellation_token, completed_tasks, failed_tasks, communications
            )
        else:
            question_answers = ""

        def parse_check(content):
            assert "task" in content and content["task"] is not None
            return content

        response = await llm_call(
            self.task_model_client,
            system_prompt=self.curriculum_prompt,
            user_prompt=curriculum_info,
            cancellation_token=cancellation_token,
            # frame=frame,
            parse_check=parse_check,
            log_prefix="Auto Curriculum get_new_task: ",
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            last_task=self.current_task,
            question_answers=question_answers,
            communications=communications,
            last_action=last_action,
            last_thoughts=last_thoughts,
            success=success,
            critique=critique,
            picked_object=picked_object,
            position_text=position_text or "Unknown",
            player_status_text=player_status_text or "Health: ?/20 | Hunger: ?/20 | Time: Unknown",
        )
        task = response.get("task", self.current_task or "Explore") or "Explore"
        task = task.strip()
        if not _is_achievable_task(task):
            fallback = _ROLE_DEFAULT_TASKS.get(self._role, "Dig a nearby tree to get wood")
            logging.warning(
                "Non-actionable task '%s' for %s, replacing with '%s'",
                task, self._role, fallback,
            )
            # Feed the blocked task back as a failed task so the curriculum LLM
            # learns not to generate it again (without this, it repeats the same
            # invalid tasks because it never sees them as rejected).
            self.failed_tasks.append(f"{task} [INVALID: not achievable in this environment]")
            task = fallback
        self.current_task = task
        context = await self.get_task_context(
            frame, cancellation_token, communications=communications
        )
        self.current_context = context
        return self.current_task, self.current_context

    # This function should return a string of question-answer pairs
    async def get_question_answers(
        self,
        frame,
        cancellation_token: CancellationToken,
        completed_tasks,
        failed_tasks,
        communications,
    ):

        # get questions from the model
        questions, concepts = await self.get_questions(
            frame, cancellation_token, completed_tasks, failed_tasks, communications
        )
        # get answers from the model
        answers = []
        for question in questions:
            # TODO: MindForge had a vectordb search here based on embedding similarity of questions to stop repetitions
            answer = await self.get_answer(
                question,
                frame,
                cancellation_token,
            )
            answers.append(answer)
        # format the question-answer pairs
        formatted_question_answers = "\n".join(
            ["question: " + q + "\n" + a for q, a in zip(questions, answers)]
        )
        logging.info(
            f"Auto Curriculum question-answer pairs: {formatted_question_answers}"
        )
        return formatted_question_answers

    # This function should return a string of questions
    async def get_questions(
        self,
        frame,
        cancellation_token: CancellationToken,
        completed_tasks,
        failed_tasks,
        communications,
    ):
        def parse_check(content):
            assert "questions" in content
            return content

        response = await llm_call(
            self.question_model_client,
            user_prompt=self._questions_prompt,
            cancellation_token=cancellation_token,
            # frame=frame,
            parse_check=parse_check,
            log_prefix="Auto Curriculum get_questions: ",
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            communications=communications,
        )

        # parse response to get questions and concepts
        questions = []
        concepts = []
        try:
            question_lines = response.get("questions", [])

            for i in range(0, len(question_lines) - 1, 2):
                q_line = question_lines[i]
                if i + 1 >= len(question_lines):
                    break
                c_line = question_lines[i + 1]

                q_match = re.match(r"Question \d+: (.+)", q_line)
                c_match = re.match(r"Concept \d+: (.+)", c_line)

                if q_match and c_match:
                    questions.append(q_match.group(1))
                    concepts.append(c_match.group(1))
        except (KeyError, IndexError, TypeError) as e:
            logging.error("Error in auto curriculum when parsing questions: %s", e)

        return questions, concepts

    # This function should return a the answer to the question
    async def get_answer(
        self,
        question,
        frame,
        cancellation_token: CancellationToken,
        relevant_past_context="",
    ):
        def parse_check(content):
            assert "answer" in content
            return content

        response = await llm_call(
            self.answer_model_client,
            user_prompt=curriculum_answer,
            cancellation_token=cancellation_token,
            # frame=frame,
            parse_check=parse_check,
            log_prefix="Auto Curriculum get_answer: ",
            question=question,
            relevant_past_context=relevant_past_context,
        )
        answer = response.get("answer", "")

        return answer.strip()

    def get_completed_tasks(self):
        # This function should return a string of completed tasks
        # TODO: link to memory
        if not self.completed_tasks:
            return "None"
        return "\n- " + "\n- ".join(self.completed_tasks)

    def get_failed_tasks(self):
        # This function should return a string of failed tasks
        # TODO: link to memory
        if not self.failed_tasks:
            return "None"
        return "\n- " + "\n- ".join(self.failed_tasks)

    async def get_task_context(self, frame, cancellation_token, communications=""):
        question = f"How to {self.current_task}" f" in this environment?"
        # retrieve past context from memory
        relevant_past_context = self.retrieve_context(question)
        answer = await self.get_answer(
            question,
            frame,
            cancellation_token,
            relevant_past_context=relevant_past_context,
        )
        context = f"Question: {question}\n{answer}"
        logging.info(f"Auto Curriculum context: {context}")
        return context

    def save_context(self, context):
        self.vectordb._ensure_initialized()
        if self.vectordb._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB")
        self.vectordb._collection.add(
            documents=[context],
            metadatas=[{"name": self.current_task}],
            ids=[self.current_task],
        )
        return context

    def retrieve_context(self, query):
        self.vectordb._ensure_initialized()
        if self.vectordb._collection is None:
            return None
        if self.vectordb._collection.count() == 0:
            return None
        k = min(self.vectordb._collection.count(), self.vectordb._config.k)
        logging.info(f"Querying vector db for contexts with query: {query}")
        data = self.vectordb._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        logging.info(f"Contexts found: {data}")
        # format skills
        semantic_memory = []
        for s in data["documents"][0]:
            semantic_memory.append(s)
        return semantic_memory

    async def clear_data(self):
        self.current_task = None
        self.current_context = ""
        self.completed_tasks = []
        self.failed_tasks = []
        await self.vectordb.clear()  # Clear the vector database collection
        logging.info("Auto Curriculum reset.")
        return True
