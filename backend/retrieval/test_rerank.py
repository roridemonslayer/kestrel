from retrieval import rerank
from retrieval.base import Retriever, SearchResult
from retrieval.rerank import RerankRetriever


class Fixed(Retriever):
    name = "fixed"

    def __init__(self, ranking):
        self.ranking = ranking

    def index(self, docs, cache_key=None):
        pass

    def search(self, query, k=10):
        return [SearchResult(d, 1.0) for d in self.ranking[:k]]


class LengthScorer:
    """Stands in for the cross-encoder: scores a (query, text) pair by the text's length."""

    def predict(self, pairs, show_progress_bar=False):
        return [len(text) for _, text in pairs]


def test_reranker_reorders_first_stage_candidates(monkeypatch):
    monkeypatch.setattr(rerank, "load_cross_encoder", lambda name: LengthScorer())
    reranker = RerankRetriever(Fixed(["short", "longest", "medium"]))
    reranker.index({"short": "ab", "longest": "abcdef", "medium": "abcd"})
    assert [r.doc_id for r in reranker.search("q", k=2)] == ["longest", "medium"]
    assert reranker.name == "fixed+rerank"
