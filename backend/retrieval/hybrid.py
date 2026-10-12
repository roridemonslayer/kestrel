"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

BM25 scores and cosine similarities are on different scales, so adding them
directly doesn't work. RRF ignores the raw scores and uses only positions:
a document at rank r in one list earns 1 / (c + r), and its points from every
list are added up. Documents that both retrievers rank highly float to the top.

c (60 by convention, from the original RRF paper) keeps a single #1 ranking
from dominating: rank 1 earns 1/61 and rank 2 earns 1/62, a small gap.
"""

from __future__ import annotations

from retrieval.base import Retriever, SearchResult


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], c: int = 60, depth: int = 100):
        self.retrievers = retrievers
        self.c = c
        self.depth = depth  # how many results to pull from each retriever before fusing

    def index(self, docs: dict[str, str], cache_key: str | None = None) -> None:
        for retriever in self.retrievers:
            retriever.index(docs, cache_key)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        fused: dict[str, float] = {}
        for retriever in self.retrievers:
            for rank, result in enumerate(retriever.search(query, self.depth), start=1):
                fused[result.doc_id] = fused.get(result.doc_id, 0.0) + 1 / (self.c + rank)
        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [SearchResult(doc_id, score) for doc_id, score in best]
