# kestrel

![Leaderboard](docs/leaderboard.png)

A test bench for search quality. Pick a dataset, run its queries through several retrievers (BM25, dense embeddings, hybrid), and compare them with standard IR metrics (recall@k, MRR, nDCG@k) and latency.

It's also the start of a shared retrieval layer: the code in `backend/retrieval/` is plain Python with no web code, so later projects can reuse it.

## Status

- [x] BEIR dataset loader (SciFact)
- [x] Retriever interface + BM25 (index cached to disk)
- [x] Metrics (recall@k, MRR, nDCG@k in plain Python) + hand-checkable tests
- [x] Eval runner + CLI
- [x] Dense retriever (all-MiniLM-L6-v2, embeddings cached to disk)
- [x] Hybrid (Reciprocal Rank Fusion)
- [x] API (FastAPI)
- [x] Web UI: leaderboard, query explorer, live search

## Results (SciFact, 300 test queries, k = 10)

| Retriever | recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| Hybrid + cross-encoder rerank | **0.822** | **0.657** | **0.691** | 1467 | 3205 |
| Hybrid (BM25 + dense, RRF) | 0.815 | 0.655 | 0.689 | 39.8 | 119.0 |
| BM25 | 0.786 | 0.635 | 0.665 | 17.4 | 42.0 |
| Dense (all-MiniLM-L6-v2) | 0.790 | 0.606 | 0.648 | 12.8 | 65.9 |

BM25's nDCG@10 of 0.665 matches the BM25 baseline in the BEIR paper, and MiniLM's 0.648 matches its published SciFact score, which is a good check that the loader and metrics are right. Hybrid wins on every quality metric because keyword and semantic search fail on different queries. Latency is measured on a MacBook Air (M2); hybrid pays for running both retrievers.

Reranking hybrid's top 50 with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) barely helps here (+0.002 nDCG@10) while making each search about 45× slower. That model was trained on web search questions, not scientific claims, so on SciFact it mostly agrees with hybrid's order; a domain-matched reranker is the next thing to try.

### NFCorpus (323 test queries, k = 10)

| Retriever | recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| Hybrid (BM25 + dense, RRF) | **0.167** | **0.547** | **0.344** | 19.5 | 139.9 |
| Dense (all-MiniLM-L6-v2) | 0.159 | 0.506 | 0.318 | 9.3 | 47.2 |
| BM25 | 0.152 | 0.515 | 0.308 | 3.3 | 15.9 |

NFCorpus queries have dozens of relevant documents each, so recall@10 is low for every retriever by design; nDCG uses its graded (1 or 2) judgments.

## The app

- **Leaderboard**: recall@10, MRR@10, nDCG@10, p50/p95 latency per retriever, with charts.
- **Query explorer**: every query's score per retriever. Pick two retrievers to find where one wins and the other fails, then open a query to see each one's actual top results, with relevant documents marked and any that every retriever missed.
- **Live search**: type a query and see the retrievers side by side; dataset queries highlight the documents a human judged relevant.

![Query explorer: BM25 misses the paper that dense and hybrid rank first](docs/explorer.png)

![Live search](docs/search.png)

## How it fits together

```
backend/
  retrieval/   Retriever interface (index, search) + BM25, dense, hybrid. No web code, reusable elsewhere.
  data/        BEIR loader: corpus, queries, qrels
  eval/        metrics (plain Python), runner, CLI
  api/         FastAPI app
frontend/      React + Vite + Recharts
```

`retrieval/rerank.py` wraps any retriever with a cross-encoder second stage (retrieve, then rerank). Evaluations are saved under `.cache/evals/`, so restarts don't re-run them.

Adding a retriever means implementing `index(docs)` and `search(query, k)` from `retrieval/base.py` and registering it in `retrieval/registry.py`. Indexes and embeddings are cached under `.cache/`, keyed by a fingerprint of the corpus.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt
cd backend
../.venv/bin/python -m data.loaders scifact   # downloads SciFact once, prints its size
../.venv/bin/python -m pytest
../.venv/bin/python -m eval.cli --dataset scifact --k 10   # prints the results table (add --format markdown or json)
```

Run the app (two terminals):

```bash
cd backend && ../.venv/bin/uvicorn api.main:app --port 8077   # API; warms indexes on startup
cd frontend && npm install && npm run dev                      # UI at http://localhost:5173
```
