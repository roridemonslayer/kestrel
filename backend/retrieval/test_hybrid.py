from retrieval.base import Retriever, SearchResult
from retrieval.hybrid import HybridRetriever


class Fixed(Retriever):
    """A fake retriever that always returns the same ranking, for testing fusion."""

    def __init__(self, ranking):
        self.ranking = ranking

    def index(self, docs, cache_key=None):
        pass

    def search(self, query, k=10):
        return [SearchResult(d, 1.0) for d in self.ranking[:k]]


def test_rrf_prefers_documents_both_lists_agree_on():
    # "b" is 2nd in both lists: 1/62 + 1/62 = 0.0323
    # "a" is 1st in one list only: 1/61 = 0.0164
    hybrid = HybridRetriever([Fixed(["a", "b", "c"]), Fixed(["d", "b", "e"])])
    results = hybrid.search("anything", k=3)
    assert results[0].doc_id == "b"
    assert abs(results[0].score - 2 / 62) < 1e-12
