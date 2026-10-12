"""BM25: the classic keyword retriever.

It scores a document by how many query words it contains, giving more weight to
rare words and less to words repeated many times or found in very long documents.
"""

from __future__ import annotations

import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.base import Retriever, SearchResult, cache_path

# Very common words carry almost no signal for matching, so they're dropped.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with".split()
)
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # how quickly repeated words stop adding score
        self.b = b  # how much long documents are penalized
        self.doc_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def index(self, docs: dict[str, str], cache_key: str | None = None) -> None:
        path = cache_path(f"{self.name}-k{self.k1}-b{self.b}", cache_key, docs, ".pkl") if cache_key else None
        if path and path.exists():
            with path.open("rb") as f:
                self.doc_ids, self.bm25 = pickle.load(f)
            return
        self.doc_ids = list(docs)
        self.bm25 = BM25Okapi([tokenize(docs[d]) for d in self.doc_ids], k1=self.k1, b=self.b)
        if path:
            with path.open("wb") as f:
                pickle.dump((self.doc_ids, self.bm25), f)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self.bm25 is None:
            raise RuntimeError("Call index() before search()")
        scores = self.bm25.get_scores(tokenize(query))
        # Highest score first; equal scores are broken by document order, so the ranking is the
        # same on every machine (argpartition/argsort order ties differently across numpy builds).
        top = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [SearchResult(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]
