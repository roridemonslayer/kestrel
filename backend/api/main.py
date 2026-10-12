"""HTTP API for the web UI. Run from backend/:  uvicorn api.main:app --reload"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.loaders import DATASETS
from eval.runner import RetrieverReport, dataset, evaluate, indexed_retriever, load_report, save_report
from retrieval.registry import REGISTRY

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the default dataset's indexes and model in the background, so the first request isn't slow.
    def warm():
        for name in REGISTRY:
            indexed_retriever("scifact", name)
        # Pre-run the default leaderboard so the UI's first load is instant.
        run_evaluation(EvaluateRequest())

    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="kestrel", description="Compare retrievers on BEIR datasets", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# A full evaluation takes a while, so results are kept per (dataset, retriever, k) for the life of the server.
_eval_cache: dict[tuple[str, str, int], RetrieverReport] = {}
_eval_lock = threading.Lock()  # one evaluation at a time, so concurrent requests never redo the same work


def _check(dataset_name: str, retriever_names: list[str]) -> None:
    if dataset_name not in DATASETS:
        raise HTTPException(404, f"Unknown dataset {dataset_name!r}")
    unknown = [r for r in retriever_names if r not in REGISTRY]
    if unknown:
        raise HTTPException(404, f"Unknown retriever(s): {', '.join(unknown)}")
    if not retriever_names:
        raise HTTPException(400, "Pick at least one retriever")


@app.get("/datasets")
def list_datasets():
    return [{"name": name, "description": description} for name, description in DATASETS.items()]


@app.get("/retrievers")
def list_retrievers():
    return [{"name": name, "description": description} for name, (description, _) in REGISTRY.items()]


@app.get("/datasets/{dataset_name}/queries")
def list_queries(dataset_name: str):
    _check(dataset_name, ["bm25"])
    ds = dataset(dataset_name)
    return [{"id": qid, "text": text, "relevant": list(ds.qrels[qid])} for qid, text in ds.queries.items()]


@app.get("/datasets/{dataset_name}/docs")
def get_docs(dataset_name: str, ids: str):
    """Titles and snippets for a comma-separated list of doc ids."""
    _check(dataset_name, ["bm25"])
    ds = dataset(dataset_name)
    wanted = [i for i in ids.split(",") if i in ds.corpus][:100]
    return {i: {"title": ds.corpus[i]["title"], "snippet": _snippet(ds.corpus[i]["text"])} for i in wanted}


class EvaluateRequest(BaseModel):
    dataset: str = "scifact"
    retrievers: list[str] = Field(default_factory=lambda: list(REGISTRY))
    k: int = Field(10, ge=1, le=100)


@app.post("/evaluate")
def run_evaluation(req: EvaluateRequest):
    _check(req.dataset, req.retrievers)
    with _eval_lock:
        for r in req.retrievers:
            if (req.dataset, r, req.k) not in _eval_cache and (saved := load_report(req.dataset, r, req.k)):
                _eval_cache[(req.dataset, r, req.k)] = saved
        missing = [r for r in req.retrievers if (req.dataset, r, req.k) not in _eval_cache]
        if missing:
            for report in evaluate(req.dataset, missing, req.k):
                _eval_cache[(req.dataset, report.retriever, req.k)] = report
                save_report(req.dataset, req.k, report)
    ds = dataset(req.dataset)
    reports = [_eval_cache[(req.dataset, r, req.k)] for r in req.retrievers]
    return {
        "dataset": req.dataset,
        "k": req.k,
        "queries": {qid: {"text": text, "relevant": list(ds.qrels[qid])} for qid, text in ds.queries.items()},
        "retrievers": [
            {
                "name": r.retriever,
                "recall": r.recall,
                "mrr": r.mrr,
                "ndcg": r.ndcg,
                "p50_ms": r.p50_ms,
                "p95_ms": r.p95_ms,
                "per_query": [asdict(q) for q in r.per_query],
            }
            for r in reports
        ],
    }


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    dataset: str = "scifact"
    retrievers: list[str] = Field(default_factory=lambda: list(REGISTRY))
    k: int = Field(10, ge=1, le=50)
    query_id: str | None = None  # set when the query comes from the dataset, so relevant docs can be marked


def _snippet(text: str, limit: int = 280) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


@app.post("/search")
def search(req: SearchRequest):
    _check(req.dataset, req.retrievers)
    ds = dataset(req.dataset)
    query_id = req.query_id
    if query_id is None:
        # A query typed exactly like one in the dataset still gets its judgments.
        query_id = next((qid for qid, text in ds.queries.items() if text.strip() == req.query.strip()), None)
    relevant = ds.qrels.get(query_id, {}) if query_id else {}

    columns = []
    for name in req.retrievers:
        retriever = indexed_retriever(req.dataset, name)
        start = time.perf_counter()
        results = retriever.search(req.query, req.k)
        latency_ms = (time.perf_counter() - start) * 1000
        columns.append(
            {
                "retriever": name,
                "latency_ms": latency_ms,
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "score": r.score,
                        "title": ds.corpus[r.doc_id]["title"],
                        "snippet": _snippet(ds.corpus[r.doc_id]["text"]),
                        "relevant": r.doc_id in relevant if query_id else None,
                    }
                    for r in results
                ],
            }
        )
    return {"query": req.query, "query_id": query_id, "relevant": list(relevant), "columns": columns}
