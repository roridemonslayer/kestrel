from data.loaders import Dataset
from eval.runner import percentile, search_dataset
from retrieval.base import SearchResult


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile([7.0], 95) == 7.0


class _FixedRetriever:
    """Returns the same ranking for every query: doc ids q1, d1, d2, ..."""

    def search(self, query, k=10):
        ids = ["q1", "d1", "d2", "d3"]
        return [SearchResult(doc_id=doc_id, score=1.0 / (rank + 1)) for rank, doc_id in enumerate(ids)][:k]


def test_search_dataset_drops_the_query_document_only_where_needed():
    retriever = _FixedRetriever()
    arguana = Dataset(name="arguana", corpus={}, queries={}, qrels={})
    scifact = Dataset(name="scifact", corpus={}, queries={}, qrels={})
    # ArguAna: the query's own document goes, and there are still k results.
    assert [r.doc_id for r in search_dataset(arguana, retriever, "text", 2, "q1")] == ["d1", "d2"]
    # A typed query with no id, or a dataset without self-matches, is left alone.
    assert [r.doc_id for r in search_dataset(arguana, retriever, "text", 2, None)] == ["q1", "d1"]
    assert [r.doc_id for r in search_dataset(scifact, retriever, "text", 2, "q1")] == ["q1", "d1"]
