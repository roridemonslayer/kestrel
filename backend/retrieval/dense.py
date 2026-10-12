"""Dense retrieval: compare meaning instead of exact words.

A sentence-transformers model turns every document and every query into a vector
(384 numbers for MiniLM). Texts with similar meaning get vectors pointing in
similar directions, so the best matches are the documents whose vectors have
the highest cosine similarity with the query's vector.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from retrieval.base import Retriever, SearchResult, cache_path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=None)
def load_model(model_name: str):
    """Load a model once per process, so every DenseRetriever (including the one inside hybrid) shares it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.doc_ids: list[str] = []
        self.embeddings: np.ndarray | None = None  # one row per document, each row unit length

    @property
    def model(self):
        # Loading the model takes a few seconds, so it only happens when first needed.
        return load_model(self.model_name)

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        # normalize_embeddings=True makes every vector unit length, so a plain dot product IS cosine similarity.
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def index(self, docs: dict[str, str], cache_key: str | None = None) -> None:
        tag = f"{self.name}-{self.model_name.split('/')[-1]}"
        path = cache_path(tag, cache_key, docs, ".npy") if cache_key else None
        if path and path.exists():
            self.embeddings = np.load(path)
            self.doc_ids = json.loads(path.with_suffix(".ids.json").read_text())
            return
        self.doc_ids = list(docs)
        self.embeddings = self._encode([docs[d] for d in self.doc_ids], show_progress=True)
        if path:
            np.save(path, self.embeddings)
            path.with_suffix(".ids.json").write_text(json.dumps(self.doc_ids))

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Call index() before search()")
        query_vector = self._encode([query])[0]
        scores = self.embeddings @ query_vector  # cosine similarity with every document at once
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchResult(self.doc_ids[i], float(scores[i])) for i in top]
