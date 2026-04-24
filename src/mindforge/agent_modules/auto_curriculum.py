import logging
import os
import re
import shutil

from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

from agent_modules.llm_call import llm_call
from agent_modules.skill_manager import _chromadb_base_dir
from agent_modules.util import (
    CurriculumAnswerResponse,
    CurriculumQuestionResponse,
    CurruliculumResponse,
    create_model_client,
    safe_format,
    ST_MODEL_NAME,
)


# ─── Task validity rules ───────────────────────────────────────────────

# Tasks without any of these keywords map to nothing the agent can actually do.
_ACHIEVABLE_KEYWORDS = frozenset({
    "dig", "mine", "break", "chop", "punch",         # resource gathering
    "kill", "attack", "fight", "hit", "hunt",        # combat
    "find", "move", "walk", "go", "run", "approach", # navigation
    "place", "build", "craft",                       # construction
    "collect", "gather", "get", "pick",              # generic gathering
})

# Tasks containing these get replaced even if they match a keyword — they
# describe map fixtures / UI artefacts that agents fixate on unproductively.
_TASK_BLOCKLIST = frozenset({
    "golden", "spawn", "platform",
    "hud", "interface", "inventory screen",
    "leave the golden", "clear the golden",
    "collect the golden", "break the golden",
})

_ROLE_DEFAULT_TASKS = {
    "agent": "Explore by moving forward and turning to survey the room",
}


def _is_achievable_task(task: str) -> bool:
    """True if `task` describes something the agent can do in-world."""
    task_lower = task.lower()
    if any(kw in task_lower for kw in _TASK_BLOCKLIST):
        return False
    return any(kw in task_lower for kw in _ACHIEVABLE_KEYWORDS)


# ─── Prompt loading ────────────────────────────────────────────────────

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r") as f:
        return f.read()


curriculum_prompt = _load_prompt("curriculum_prompt.txt")
curriculum_info = _load_prompt("curriculum_info.txt")
curriculum_questions = _load_prompt("curriculum_questions.txt")
curriculum_answer = _load_prompt("curriculum_answer.txt")


# ─── Parse-check helpers for llm_call ──────────────────────────────────

def _require_key(key):
    """Build a parse_check that asserts `key` is present and non-null."""
    def check(content):
        assert key in content and content[key] is not None
        return content
    return check


# ─── Main class ────────────────────────────────────────────────────────

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
        self.completed_tasks = []
        self.failed_tasks = []
        self._role = "agent"

        self.curriculum_prompt = (
            override_curriculum_prompt or safe_format(curriculum_prompt)
        )
        self._questions_prompt = override_questions_prompt or curriculum_questions

        self.task_model_client = task_model_client or create_model_client(
            response_format=CurruliculumResponse
        )
        self.question_model_client = question_model_client or create_model_client(
            response_format=CurriculumQuestionResponse
        )
        self.answer_model_client = answer_model_client or create_model_client(
            response_format=CurriculumAnswerResponse
        )

        self.vectordb = self._init_vectordb(agent_name, retrieval_top_k)

    @staticmethod
    def _init_vectordb(agent_name: str, top_k: int) -> ChromaDBVectorMemory:
        """Create (or reset) a per-agent ChromaDB collection for task context."""
        db_path = os.path.join(
            _chromadb_base_dir(), f"autocurriculum_vectodb_{agent_name}"
        )
        # Wipe stale/corrupted SQLite files from previous runs
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
        return ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="autocurriculum_vectordb_nvidia",
                persistence_path=db_path,
                k=top_k,
                score_threshold=0.4,
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                    model_name=ST_MODEL_NAME, device="cuda"
                ),
            ),
        )

    # ─── Public API: get a new task ────────────────────────────────────

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
        current_chamber=None,
        completed_milestones=None,
    ):
        completed = self.get_completed_tasks()
        failed = self.get_failed_tasks()

        question_answers = ""
        if do_question_answers:
            question_answers = await self.get_question_answers(
                frame, cancellation_token, completed, failed, communications
            )

        response = await llm_call(
            self.task_model_client,
            system_prompt=self.curriculum_prompt,
            user_prompt=curriculum_info,
            cancellation_token=cancellation_token,
            parse_check=_require_key("task"),
            log_prefix="Auto Curriculum get_new_task: ",
            completed_tasks=completed,
            failed_tasks=failed,
            last_task=self.current_task,
            question_answers=question_answers,
            communications=communications,
            last_action=last_action,
            last_thoughts=last_thoughts,
            success=success,
            critique=critique,
            picked_object=picked_object,
            position_text=position_text or "Unknown",
            player_status_text=player_status_text
                or "Health: ?/20 | Hunger: ?/20 | Time: Unknown",
            current_chamber=current_chamber or "Unknown",
            completed_milestones=self._format_milestones(completed_milestones),
            inventory=picked_object or "empty",
        )

        task = (response.get("task") or self.current_task or "Explore").strip()
        task = self._validate_or_fallback(task)
        self.current_task = task

        self.current_context = await self.get_task_context(
            frame, cancellation_token, communications=communications
        )
        return self.current_task, self.current_context

    def _validate_or_fallback(self, task: str) -> str:
        """Replace unachievable/blocklisted tasks with a role default."""
        if _is_achievable_task(task):
            return task
        fallback = _ROLE_DEFAULT_TASKS.get(self._role, "Dig a nearby tree to get wood")
        logging.warning(
            "Non-actionable task '%s' for %s, replacing with '%s'",
            task, self._role, fallback,
        )
        # Record the rejected task so the curriculum LLM stops proposing it.
        self.failed_tasks.append(f"{task} [INVALID: not achievable in this environment]")
        return fallback

    @staticmethod
    def _format_milestones(completed_milestones) -> str:
        if not completed_milestones:
            return "none"
        return ", ".join(sorted(completed_milestones))

    # ─── Question / answer sub-pipeline ────────────────────────────────

    async def get_question_answers(
        self,
        frame,
        cancellation_token: CancellationToken,
        completed_tasks,
        failed_tasks,
        communications,
    ) -> str:
        """Return a formatted question-answer string for the curriculum prompt."""
        questions, _ = await self.get_questions(
            frame, cancellation_token, completed_tasks, failed_tasks, communications
        )
        answers = [
            await self.get_answer(q, frame, cancellation_token) for q in questions
        ]
        formatted = "\n".join(
            f"question: {q}\n{a}" for q, a in zip(questions, answers)
        )
        logging.info("Auto Curriculum question-answer pairs: %s", formatted)
        return formatted

    async def get_questions(
        self,
        frame,
        cancellation_token: CancellationToken,
        completed_tasks,
        failed_tasks,
        communications,
    ):
        response = await llm_call(
            self.question_model_client,
            user_prompt=self._questions_prompt,
            cancellation_token=cancellation_token,
            parse_check=_require_key("questions"),
            log_prefix="Auto Curriculum get_questions: ",
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            communications=communications,
        )
        return self._parse_questions_and_concepts(response.get("questions", []))

    @staticmethod
    def _parse_questions_and_concepts(lines):
        """Parse interleaved `Question N: ...` / `Concept N: ...` lines."""
        questions, concepts = [], []
        try:
            for i in range(0, len(lines) - 1, 2):
                q_match = re.match(r"Question \d+: (.+)", lines[i])
                c_match = re.match(r"Concept \d+: (.+)", lines[i + 1])
                if q_match and c_match:
                    questions.append(q_match.group(1))
                    concepts.append(c_match.group(1))
        except (KeyError, IndexError, TypeError) as e:
            logging.error("Error parsing curriculum questions: %s", e)
        return questions, concepts

    async def get_answer(
        self,
        question,
        frame,
        cancellation_token: CancellationToken,
        relevant_past_context="",
    ) -> str:
        response = await llm_call(
            self.answer_model_client,
            user_prompt=curriculum_answer,
            cancellation_token=cancellation_token,
            parse_check=_require_key("answer"),
            log_prefix="Auto Curriculum get_answer: ",
            question=question,
            relevant_past_context=relevant_past_context,
        )
        return response.get("answer", "").strip()

    # ─── Task/failure log + context retrieval ──────────────────────────

    def get_completed_tasks(self) -> str:
        if not self.completed_tasks:
            return "None"
        return "\n- " + "\n- ".join(self.completed_tasks)

    def get_failed_tasks(self) -> str:
        if not self.failed_tasks:
            return "None"
        return "\n- " + "\n- ".join(self.failed_tasks)

    async def get_task_context(self, frame, cancellation_token, communications=""):
        question = f"How to {self.current_task} in this environment?"
        relevant_past_context = self.retrieve_context(question)
        answer = await self.get_answer(
            question, frame, cancellation_token,
            relevant_past_context=relevant_past_context,
        )
        context = f"Question: {question}\n{answer}"
        logging.info("Auto Curriculum context: %s", context)
        return context

    # ─── Vector DB operations ──────────────────────────────────────────

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
        logging.info("Querying vector db for contexts with query: %s", query)
        data = self.vectordb._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        logging.info("Contexts found: %s", data)
        return list(data["documents"][0])

    async def clear_data(self):
        self.current_task = None
        self.current_context = ""
        self.completed_tasks = []
        self.failed_tasks = []
        await self.vectordb.clear()
        logging.info("Auto Curriculum reset.")
        return True
