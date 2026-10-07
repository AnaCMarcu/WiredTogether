"""Skill and episode memory shared by all agents of a run.

MindForge's WIRE agent gives every agent three ChromaDB stores, each loading
its own SentenceTransformer on the GPU — 300 model copies at N = 100. Here one
embedder serves the whole run and each agent's stores are plain in-process
arrays (cosine top-k); that is all the retrieval the agent needs.

:class:`HashEmbedder` (bag of hashed words) is the dependency-free fallback,
used in tests and whenever no sentence-transformer model can be loaded.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    dim = 256

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for w in _WORD.findall(str(text).lower()):
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little")
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-8)


class SentenceEmbedder:
    """One sentence-transformer model for the whole process (lazy, thread-safe).

    The model defaults to ``$ST_MODEL_NAME`` (the cluster launchers point it at
    the local copy under models/, since compute nodes are offline) and runs on
    ``$ST_DEVICE`` (default CPU: a few short queries per round, and the GPU
    belongs to the vLLM server).
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        import os
        self.model_name = model_name or os.environ.get("ST_MODEL_NAME", "all-MiniLM-L6-v2")
        self.device = device or os.environ.get("ST_DEVICE", "cpu")
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(kind: str = "auto", model_name: Optional[str] = None,
                  device: Optional[str] = None):
    if kind == "hash":
        return HashEmbedder()
    if kind == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return HashEmbedder()
    return SentenceEmbedder(model_name, device)


@dataclass
class VectorStore:
    embedder: Any
    capacity: int = 500
    texts: List[str] = field(default_factory=list)
    metas: List[Dict[str, Any]] = field(default_factory=list)
    _vecs: Optional[np.ndarray] = None

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None, dedupe: bool = True) -> bool:
        if dedupe and text in self.texts:
            return False
        vec = self.embedder.encode([text])
        self.texts.append(text)
        self.metas.append(meta or {})
        self._vecs = vec if self._vecs is None else np.vstack([self._vecs, vec])
        if len(self.texts) > self.capacity:
            self.texts.pop(0)
            self.metas.pop(0)
            self._vecs = self._vecs[1:]
        return True

    def query(self, text: str, k: int = 3, min_score: float = 0.2
              ) -> List[Tuple[float, str, Dict[str, Any]]]:
        if not self.texts:
            return []
        q = self.embedder.encode([text])[0]
        scores = self._vecs @ q
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.texts[i], self.metas[i]) for i in order
                if scores[i] >= min_score]

    def clear(self) -> None:
        self.texts.clear()
        self.metas.clear()
        self._vecs = None

    def __len__(self) -> int:
        return len(self.texts)


class SharedMemory:
    """Per-agent stores over one embedder."""

    def __init__(self, embedder: Any = None, capacity: int = 500):
        self.embedder = embedder or HashEmbedder()
        self.capacity = capacity
        self._stores: Dict[str, VectorStore] = {}

    def store(self, name: str) -> VectorStore:
        if name not in self._stores:
            self._stores[name] = VectorStore(self.embedder, self.capacity)
        return self._stores[name]
