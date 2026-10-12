"""The one interface every retriever implements.

A retriever does two things:
  index(docs)        build whatever it needs from a {doc_id: text} mapping
  search(query, k)   return the k best doc ids for a query, best first, with scores

Scores are only comparable within one retriever (BM25 scores and cosine
similarities live on different scales), which is why hybrid fusion works on
ranks rather than raw scores.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "indexes"


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def index(self, docs: dict[str, str], cache_key: str | None = None) -> None:
        """Build the index. With a cache_key, reuse an index saved on disk for the same documents."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return up to k results, highest score first."""


def cache_path(retriever_name: str, cache_key: str, docs: dict[str, str], suffix: str) -> Path:
    """Where to cache an index.

    The path includes a fingerprint of the documents, so a changed corpus never reuses a stale index.
    """
    fingerprint = hashlib.sha1()
    for doc_id in sorted(docs):
        fingerprint.update(doc_id.encode())
        fingerprint.update(b"\0")
        fingerprint.update(docs[doc_id].encode())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{cache_key}-{retriever_name}-{fingerprint.hexdigest()[:12]}{suffix}"
