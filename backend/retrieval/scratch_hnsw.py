"""Rori's from-scratch HNSW engine (github.com/roridemonslayer/vector-search-engine) as a retriever.

That engine is pure Python with no numpy or FAISS, so it's here to compare a hand-built
approximate index against exact dense search on the same MiniLM embeddings: same vectors,
different way of finding the nearest ones.

The engine is imported from a sibling checkout (../vector-search-engine) or from the
folder in the KESTREL_HNSW_PATH environment variable. If it isn't there, this retriever
simply isn't offered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from retrieval.base import Retriever, SearchResult, cache_path
from retrieval.dense import DenseRetriever

ENGINE_PATH = Path(os.environ.get("KESTREL_HNSW_PATH", Path(__file__).resolve().parents[3] / "vector-search-engine"))


def load_engine():
    """Import the engine's hnsw module, or return None if it isn't checked out."""
    if not (ENGINE_PATH / "hnsw.py").exists():
        return None
    if str(ENGINE_PATH) not in sys.path:
        sys.path.append(str(ENGINE_PATH))
    import hnsw  # imported here because Python can only find it once the path is added

    return hnsw


class ScratchHNSWRetriever(Retriever):
    name = "hnsw"

    def __init__(self, ef: int = 100):
        self.ef = ef  # beam width on the ground floor: higher is more accurate and slower
        self.dense = DenseRetriever()  # reuses the cached MiniLM embeddings and the shared model
        self.engine = load_engine()
        self.graph: dict = {}
        self.layers: dict = {}
        self.entry: str | None = None

    def index(self, docs: dict[str, str], cache_key: str | None = None) -> None:
        if self.engine is None:
            raise RuntimeError(f"vector-search-engine not found at {ENGINE_PATH}")
        self.dense.index(docs, cache_key)
        path = cache_path(self.name, cache_key, docs, ".pkl") if cache_key else None
        if path and path.exists():
            self.graph, self.layers = self.engine.load_index(str(path))
        else:
            self.graph, self.layers = {}, {}
            # The engine works on plain Python lists, one insert at a time, like its own benchmarks.
            for doc_id, vector in zip(self.dense.doc_ids, self.dense.embeddings, strict=True):
                self.engine.insert(self.graph, self.layers, doc_id, vector.tolist())
            if path:
                self.engine.save_index(self.graph, self.layers, str(path))
        self.entry = max(self.layers, key=self.layers.get)  # search starts on the highest floor

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self.entry is None:
            raise RuntimeError("Call index() before search()")
        query_vector = self.dense._encode([query])[0].tolist()
        found = self.engine.layered_search(self.graph, query_vector, self.layers, start=self.entry, k=k, ef=self.ef)
        return [SearchResult(doc_id, float(score)) for doc_id, score in found]
