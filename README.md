# kestrel

A test bench for search quality. Pick a dataset, run its queries through several retrievers (BM25, dense embeddings, hybrid), and compare them with standard IR metrics (recall@k, MRR, nDCG@k) and latency.

It's also the start of a shared retrieval layer: the code in `backend/retrieval/` is plain Python with no web code, so later projects can reuse it.

## Status

- [x] BEIR dataset loader (SciFact)
- [x] Retriever interface + BM25 (index cached to disk)
- [x] Metrics (recall@k, MRR, nDCG@k in plain Python) + hand-checkable tests
- [x] Eval runner + CLI
- [ ] Dense retriever
- [ ] Hybrid (RRF)
- [ ] API
- [ ] Web UI

## Results (SciFact, 300 test queries, k = 10)

| Retriever | recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| BM25 | 0.786 | 0.635 | 0.665 | 11.5 | 27.9 |

BM25's nDCG@10 of 0.665 matches the BM25 baseline reported in the BEIR paper, which is a good check that the loader and metrics are right.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt
cd backend
../.venv/bin/python -m data.loaders scifact   # downloads SciFact once, prints its size
../.venv/bin/python -m pytest
../.venv/bin/python -m eval.cli --dataset scifact --k 10   # prints the results table
```
