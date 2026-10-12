"""Reranking: re-score a retriever's top results with a cross-encoder.

The dense retriever embeds the query and each document separately, then compares
the vectors. A cross-encoder instead reads the query and one document together,
word by word, and outputs a single relevance score. That's much more accurate,
but far too slow to run over every document, so it only re-scores the top
`depth` candidates another retriever already found ("retrieve, then rerank").
"""

from __future__ import annotations

from functools import cache

from retrieval.base import Retriever, SearchResult

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@cache
def load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class RerankRetriever(Retriever):
    def __init__(self, first_stage: Retriever, model_name: str = DEFAULT_MODEL, depth: int = 50):
        self.first_stage = first_stage
        self.model_name = model_name
        self.depth = depth  # how many first-stage candidates the cross-encoder re-scores
        self.texts: dict[str, str] = {}
        self.name = f"{first_stage.name}+rerank"

    def index(self, docs: dict[str, str], cache_key: str | None = None) -> None:
        self.first_stage.index(docs, cache_key)
        self.texts = docs  # the cross-encoder needs the raw text of each candidate

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        candidates = self.first_stage.search(query, max(k, self.depth))
        if not candidates:
            return []
        model = load_cross_encoder(self.model_name)
        scores = model.predict([(query, self.texts[c.doc_id]) for c in candidates], show_progress_bar=False)
        reranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)[:k]
        return [SearchResult(c.doc_id, float(score)) for c, score in reranked]
