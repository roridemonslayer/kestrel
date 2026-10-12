"""Run every query of a dataset through every chosen retriever and score the results."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

from data.loaders import Dataset, documents_for_indexing, load_dataset
from eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from retrieval.base import Retriever
from retrieval.registry import build_retriever


@dataclass
class QueryScore:
    query_id: str
    recall: float
    rr: float
    ndcg: float
    latency_ms: float
    top_ids: list[str]


@dataclass
class RetrieverReport:
    retriever: str
    recall: float
    mrr: float
    ndcg: float
    p50_ms: float
    p95_ms: float
    per_query: list[QueryScore] = field(default_factory=list)


# Finished evaluations are saved here so a restart doesn't re-run them (the reranker takes minutes).
# Bump EVAL_VERSION whenever retriever or metric code changes in a way that changes the numbers.
EVAL_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "evals"
EVAL_VERSION = 1


def _saved_report(dataset_name: str, retriever_name: str, k: int) -> Path:
    return EVAL_CACHE / f"v{EVAL_VERSION}-{dataset_name}-{retriever_name}-k{k}.json"


def load_report(dataset_name: str, retriever_name: str, k: int) -> RetrieverReport | None:
    path = _saved_report(dataset_name, retriever_name, k)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["per_query"] = [QueryScore(**q) for q in data["per_query"]]
    return RetrieverReport(**data)


def save_report(dataset_name: str, k: int, report: RetrieverReport) -> None:
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)
    _saved_report(dataset_name, report.retriever, k).write_text(json.dumps(asdict(report)))


@cache
def dataset(name: str) -> Dataset:
    return load_dataset(name)


_index_lock = threading.Lock()


@cache
def _indexed(dataset_name: str, retriever_name: str) -> Retriever:
    retriever = build_retriever(retriever_name)
    retriever.index(documents_for_indexing(dataset(dataset_name)), cache_key=dataset_name)
    # The first search can pay one-time costs (loading a model), so pay them here, not in a timed search.
    retriever.search("warm up", 1)
    return retriever


def indexed_retriever(dataset_name: str, retriever_name: str) -> Retriever:
    """Build (or load from the disk cache) a retriever's index once per process.

    The lock stops two web requests from building the same index at the same time.
    """
    with _index_lock:
        return _indexed(dataset_name, retriever_name)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the value below which pct% of the measurements fall."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def evaluate(dataset_name: str, retriever_names: list[str], k: int = 10) -> list[RetrieverReport]:
    ds = dataset(dataset_name)
    reports = []
    for name in retriever_names:
        retriever = indexed_retriever(dataset_name, name)
        per_query = []
        for query_id, query in ds.queries.items():
            start = time.perf_counter()
            results = retriever.search(query, k)
            latency_ms = (time.perf_counter() - start) * 1000
            ranked = [r.doc_id for r in results]
            relevant = ds.qrels[query_id]
            per_query.append(
                QueryScore(
                    query_id=query_id,
                    recall=recall_at_k(ranked, relevant, k),
                    rr=reciprocal_rank(ranked, relevant, k),
                    ndcg=ndcg_at_k(ranked, relevant, k),
                    latency_ms=latency_ms,
                    top_ids=ranked,
                )
            )
        latencies = [q.latency_ms for q in per_query]
        reports.append(
            RetrieverReport(
                retriever=name,
                recall=mean([q.recall for q in per_query]),
                mrr=mean([q.rr for q in per_query]),
                ndcg=mean([q.ndcg for q in per_query]),
                p50_ms=percentile(latencies, 50),
                p95_ms=percentile(latencies, 95),
                per_query=per_query,
            )
        )
    return reports
