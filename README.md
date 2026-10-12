# kestrel

A test bench for search quality. Pick a dataset, run its queries through several retrievers (BM25, dense embeddings, hybrid), and compare them with standard IR metrics (recall@k, MRR, nDCG@k) and latency.

It's also the start of a shared retrieval layer: the code in `backend/retrieval/` is plain Python with no web code, so later projects can reuse it.

## Status

- [x] BEIR dataset loader (SciFact)
- [x] Retriever interface + BM25 (index cached to disk)
- [ ] Metrics + tests
- [ ] Eval runner + CLI
- [ ] Dense retriever
- [ ] Hybrid (RRF)
- [ ] API
- [ ] Web UI

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt
cd backend
../.venv/bin/python -m data.loaders scifact   # downloads SciFact once, prints its size
../.venv/bin/python -m pytest
```
