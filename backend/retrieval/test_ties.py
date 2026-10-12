from retrieval.bm25 import BM25Retriever


def test_tied_scores_rank_in_document_order():
    # Both documents contain the query word once and have the same length, so their BM25 scores tie.
    bm25 = BM25Retriever()
    bm25.index({"first": "apple pie recipe", "second": "apple tart recipe", "third": "banana bread"})
    results = bm25.search("apple", k=2)
    assert results[0].score == results[1].score
    assert [r.doc_id for r in results] == ["first", "second"]
