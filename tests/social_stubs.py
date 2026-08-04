"""Module-local stubs for the Experiment-2 social-act tests.

The choice-mode tests import ``mindforge`` modules whose import chains touch
autogen / chromadb, which are not installed on the dev machine (and conftest
deliberately does not stub them globally — global stubs would mask accidental
heavy imports). Importing THIS module installs minimal stand-ins, only for
packages that are actually absent, following the test_gemma4_compat pattern.

Also puts ``src/mindforge`` on sys.path so the runtime-style absolute imports
(``from agent_modules.x import ...``) resolve, matching how
multi_agent_craftium runs in production.
"""

import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_MINDFORGE = REPO / "src" / "mindforge"
for _p in (str(REPO / "src"), str(_MINDFORGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _absent(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


class _Msg:
    def __init__(self, content=None, source="test", **kwargs):
        self.content = content
        self.source = source
        self.__dict__.update(kwargs)


def _install() -> None:
    if _absent("autogen_core"):
        core = types.ModuleType("autogen_core")

        class Image:
            @staticmethod
            def from_pil(img):
                return img

            @staticmethod
            def to_base64(img):
                return img

        core.CancellationToken = type("CancellationToken", (), {})
        core.Image = Image
        core.EVENT_LOGGER_NAME = "autogen_core.events"

        models = types.ModuleType("autogen_core.models")

        class RequestUsage:
            def __init__(self, prompt_tokens=0, completion_tokens=0):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens

        class CreateResult:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        models.ChatCompletionClient = type("ChatCompletionClient", (), {})
        models.CreateResult = CreateResult
        models.LLMMessage = _Msg
        models.ModelInfo = dict
        models.RequestUsage = RequestUsage
        models.SystemMessage = type("SystemMessage", (_Msg,), {})
        models.UserMessage = type("UserMessage", (_Msg,), {})
        core.models = models
        sys.modules["autogen_core"] = core
        sys.modules["autogen_core.models"] = models

    if _absent("autogen_ext"):
        ext = types.ModuleType("autogen_ext")
        ext.__path__ = []
        ext_models = types.ModuleType("autogen_ext.models")
        ext_models.__path__ = []
        ext_openai = types.ModuleType("autogen_ext.models.openai")

        class OpenAIChatCompletionClient:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

        ext_openai.OpenAIChatCompletionClient = OpenAIChatCompletionClient

        ext_memory = types.ModuleType("autogen_ext.memory")
        ext_memory.__path__ = []
        ext_chroma = types.ModuleType("autogen_ext.memory.chromadb")

        class _MemoryStub:
            def __init__(self, *args, **kwargs):
                pass

            async def add(self, *args, **kwargs):
                pass

            async def query(self, *args, **kwargs):
                return types.SimpleNamespace(results=[])

            async def clear(self):
                pass

            async def close(self):
                pass

        ext_chroma.ChromaDBVectorMemory = _MemoryStub
        ext_chroma.PersistentChromaDBVectorMemoryConfig = _MemoryStub
        ext_chroma.SentenceTransformerEmbeddingFunctionConfig = _MemoryStub
        # PEP 562: any other PUBLIC name the memory modules import resolves
        # to the same permissive stub (the tests never exercise these
        # classes). Dunders must still raise — import machinery and inspect
        # probe modules for __file__/__path__ etc., and a class object there
        # poisons unrelated imports (torch).
        def _chroma_getattr(name):
            if name.startswith("_"):
                raise AttributeError(name)
            return _MemoryStub

        ext_chroma.__getattr__ = _chroma_getattr

        sys.modules["autogen_ext"] = ext
        sys.modules["autogen_ext.models"] = ext_models
        sys.modules["autogen_ext.models.openai"] = ext_openai
        sys.modules["autogen_ext.memory"] = ext_memory
        sys.modules["autogen_ext.memory.chromadb"] = ext_chroma

    if _absent("autogen_agentchat"):
        chat = types.ModuleType("autogen_agentchat")
        chat.__path__ = []

        agents = types.ModuleType("autogen_agentchat.agents")

        class BaseChatAgent:
            def __init__(self, name, description=""):
                self.name = name
                self.description = description

        agents.BaseChatAgent = BaseChatAgent

        messages = types.ModuleType("autogen_agentchat.messages")
        messages.BaseChatMessage = _Msg
        messages.TextMessage = type("TextMessage", (_Msg,), {})
        messages.MultiModalMessage = type("MultiModalMessage", (_Msg,), {})

        chat.agents = agents
        chat.messages = messages
        sys.modules["autogen_agentchat"] = chat
        sys.modules["autogen_agentchat.agents"] = agents
        sys.modules["autogen_agentchat.messages"] = messages

    if _absent("chromadb"):
        chroma = types.ModuleType("chromadb")

        class _Collection:
            def __init__(self):
                self._docs = {}

            def add(self, *args, **kwargs):
                pass

            def query(self, *args, **kwargs):
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            def get(self, *args, **kwargs):
                return {"documents": [], "metadatas": [], "ids": []}

            def count(self):
                return 0

            def delete(self, *args, **kwargs):
                pass

        class PersistentClient:
            def __init__(self, *args, **kwargs):
                pass

            def get_or_create_collection(self, *args, **kwargs):
                return _Collection()

            def delete_collection(self, *args, **kwargs):
                pass

        chroma.PersistentClient = PersistentClient
        chroma.Client = PersistentClient
        utils_mod = types.ModuleType("chromadb.utils")
        ef_mod = types.ModuleType("chromadb.utils.embedding_functions")

        class SentenceTransformerEmbeddingFunction:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, texts):
                return [[0.0] * 8 for _ in texts]

        ef_mod.SentenceTransformerEmbeddingFunction = (
            SentenceTransformerEmbeddingFunction
        )
        utils_mod.embedding_functions = ef_mod
        chroma.utils = utils_mod
        sys.modules["chromadb"] = chroma
        sys.modules["chromadb.utils"] = utils_mod
        sys.modules["chromadb.utils.embedding_functions"] = ef_mod

    if _absent("wandb"):
        wandb_mod = types.ModuleType("wandb")
        wandb_mod.log = lambda *a, **k: None
        wandb_mod.init = lambda *a, **k: None
        sys.modules["wandb"] = wandb_mod


_install()
