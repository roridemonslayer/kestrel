"""Name -> retriever. Adding a retriever means implementing Retriever and adding one line here."""

from __future__ import annotations

from typing import Callable

from retrieval.base import Retriever
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever

REGISTRY: dict[str, tuple[str, Callable[[], Retriever]]] = {
    "bm25": ("BM25 keyword search (rank_bm25)", BM25Retriever),
    "dense": ("Dense embeddings, all-MiniLM-L6-v2, cosine similarity", DenseRetriever),
    "hybrid": ("BM25 + dense fused with Reciprocal Rank Fusion", lambda: HybridRetriever([BM25Retriever(), DenseRetriever()])),
}


def build_retriever(name: str) -> Retriever:
    if name not in REGISTRY:
        raise ValueError(f"Unknown retriever {name!r}. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name][1]()
