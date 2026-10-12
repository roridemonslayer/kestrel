from data.loaders import load_dataset


def test_scifact_loads_with_known_sizes():
    ds = load_dataset("scifact")
    assert len(ds.corpus) == 5183
    assert len(ds.queries) == 300
    # Every query has at least one relevant document, and every judged document exists in the corpus.
    for query_id, judged in ds.qrels.items():
        assert query_id in ds.queries
        assert judged
        assert all(doc_id in ds.corpus for doc_id in judged)
