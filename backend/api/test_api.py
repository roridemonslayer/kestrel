from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_lists_datasets_and_retrievers():
    assert client.get("/datasets").json()[0]["name"] == "scifact"
    names = [r["name"] for r in client.get("/retrievers").json()]
    assert names[:3] == ["bm25", "dense", "hybrid"]
    assert "hybrid+rerank" in names


def test_evaluate_bm25_matches_cli_numbers():
    body = client.post("/evaluate", json={"dataset": "scifact", "retrievers": ["bm25"], "k": 10}).json()
    bm25 = body["retrievers"][0]
    assert round(bm25["ndcg"], 3) == 0.665
    assert len(bm25["per_query"]) == 300


def test_search_marks_relevant_docs_for_dataset_queries():
    queries = client.get("/datasets/scifact/queries").json()
    q = next(q for q in queries if q["id"] == "3")
    body = client.post("/search", json={"query": q["text"], "retrievers": ["bm25"], "k": 10}).json()
    assert body["query_id"] == "3"
    results = body["columns"][0]["results"]
    assert all(r["relevant"] in (True, False) for r in results)
    assert any(r["relevant"] for r in results) == bool(set(q["relevant"]) & {r["doc_id"] for r in results})


def test_unknown_retriever_is_a_404():
    assert client.post("/search", json={"query": "x", "retrievers": ["nope"]}).status_code == 404
